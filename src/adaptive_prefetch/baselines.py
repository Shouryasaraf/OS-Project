"""Causal per-request prefetch baselines.

Predictors observe the current request and propose blocks for *later* reads.
The simulator applies the same cache, warm-up, and cost rules to every mode.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Protocol

from .trace import Request


class CandidatePredictor(Protocol):
    def next_candidates(self, request: Request) -> list[int]:
        """Update state from this request and return future block candidates."""


class StridePrefetcher:
    def __init__(self, degree: int = 2, min_repetitions: int = 2):
        if degree < 1 or min_repetitions < 2:
            raise ValueError("degree >= 1 and min_repetitions >= 2 are required")
        self.degree = degree
        self.min_repetitions = min_repetitions
        self.last_lba: int | None = None
        self.last_delta: int | None = None
        self.repetitions = 0

    def next_candidates(self, request: Request) -> list[int]:
        if request.operation != "R":
            return []
        if self.last_lba is None:
            self.last_lba = request.lba
            return []
        delta = request.lba - self.last_lba
        self.repetitions = self.repetitions + 1 if delta == self.last_delta else 1
        self.last_lba = request.lba
        self.last_delta = delta
        if delta <= 0 or self.repetitions < self.min_repetitions:
            return []
        end = request.lba + request.size_blocks
        return [request.lba + k * delta for k in range(1, self.degree + 1)
                if request.lba + k * delta >= end]


class MarkovPrefetcher:
    """First- or second-order Markov transitions between address deltas."""

    def __init__(self, order: int = 1, top_k: int = 2,
                 confidence_threshold: float = 0.3, max_delta: int = 4096):
        if order not in {1, 2} or top_k < 1 or not 0 <= confidence_threshold <= 1:
            raise ValueError("order must be 1 or 2; top_k >= 1; threshold in [0,1]")
        self.order = order
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.max_delta = max_delta
        self.last_lba: int | None = None
        # Only the last `order` deltas are ever read, so the buffer is trimmed
        # rather than grown: on a multi-million-request trace an unbounded
        # list would retain one int per read for no benefit.
        self.deltas: list[int] = []
        self.transitions: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)

    def next_candidates(self, request: Request) -> list[int]:
        if request.operation != "R":
            return []
        if self.last_lba is None:
            self.last_lba = request.lba
            return []
        delta = request.lba - self.last_lba
        self.last_lba = request.lba
        if len(self.deltas) >= self.order and abs(delta) <= self.max_delta:
            self.transitions[tuple(self.deltas)][delta] += 1
        self.deltas.append(delta)
        if len(self.deltas) > self.order:
            del self.deltas[0]
        if len(self.deltas) < self.order:
            return []
        counts = self.transitions.get(tuple(self.deltas))
        if not counts:
            return []
        total = sum(counts.values())
        predicted = counts.most_common(self.top_k)
        if predicted[0][1] / total < self.confidence_threshold:
            return []
        end = request.lba + request.size_blocks
        return [request.lba + next_delta for next_delta, _ in predicted
                if request.lba + next_delta >= end]


#: ``linux_ra`` window ceiling in 512-B blocks: 256 blocks is the 128 KiB
#: ``read_ahead_kb`` default every block device on the test host reports.
DEFAULT_LINUX_RA_MAX_BLOCKS = 256
#: Blocks per 4 KiB page; the sequential test compares page indices, as the
#: kernel does, not block addresses.
PAGE_BLOCKS = 8


class LinuxReadahead:
    """An emulation of the Linux on-demand read-ahead **policy** (mm/readahead.c).

    It is not the kernel and does not claim to reproduce it. It models the
    policy's shape in 512-B blocks, with one read-ahead state for the whole
    stream (as for buffered reads of one block device):

    * **sync path**, on a request with any missed block: a sequential read
      (it starts within one page of the previous read's end), a read at
      block 0, or an oversize read opens a fresh window sized from the
      request (``get_init_ra_size``). Otherwise it looks back for a run of
      resident blocks just before the request (``page_cache_prev_miss``) and
      treats the read as a stream only if that run is longer than the
      request. A read that is neither leaves the state alone: no prefetch.
    * **async path**, on a fully-hit request that touches the marker block
      (``PG_readahead``) while it is still an unused prefetch: the window
      advances and ramps x4 / x2 / max (``get_next_ra_size``).

    It reads cache state, so it cannot be a ``next_candidates`` predictor:
    the replay loop drives it through :meth:`observe` with the request's
    missed blocks and a read-only :class:`~adaptive_prefetch.simulator.CacheView`,
    after the demand reads and before any prefetch. Ported from the
    validated scratch prototype (99.83% hit with 628 prefetch I/Os on a
    sequential scan, nothing issued on random input).

    ``warmup`` requests are observed without touching the window state (the
    previous-read position is still tracked), matching the prototype and the
    simulator's first-window rule. Write requests are ignored, as the kernel's
    read-ahead state ignores them.
    """

    #: Tells the replay loop to call ``observe`` instead of ``next_candidates``.
    cache_aware = True

    def __init__(self, max_blocks: int = DEFAULT_LINUX_RA_MAX_BLOCKS,
                 warmup: int = 0):
        if max_blocks < 1 or warmup < 0:
            raise ValueError("max_blocks >= 1 and warmup >= 0 are required")
        self.max_blocks = max_blocks
        self.warmup = warmup
        self.start = 0
        self.size = 0
        self.async_size = 0
        self.marker: int | None = None
        self.prev_end: int | None = None
        self.seen = 0

    def _init_size(self, request_blocks: int) -> int:
        size = 1
        while size < request_blocks:
            size <<= 1
        if size <= self.max_blocks // 32:
            return size * 4
        if size <= self.max_blocks // 4:
            return size * 2
        return self.max_blocks

    def _next_size(self, current: int) -> int:
        if current < self.max_blocks // 16:
            return current * 4
        if current <= self.max_blocks // 2:
            return current * 2
        return self.max_blocks

    def observe(self, request: Request, missed_blocks: list[int],
                cache_view) -> list[int]:
        """Update the window state from this request; return blocks to prefetch."""
        index = self.seen
        self.seen += 1
        if request.operation != "R":
            return []
        lo = request.lba
        hi = lo + request.size_blocks
        proposed: list[int] = []
        if index >= self.warmup:
            marker = self.marker
            windows: list[tuple[int, int]] = []
            if missed_blocks:
                req = request.size_blocks
                prev = self.prev_end
                sequential = (prev is not None and
                              0 <= lo // PAGE_BLOCKS - (prev - 1) // PAGE_BLOCKS <= 1)
                if lo == 0 or req > self.max_blocks or sequential:
                    self.start, self.size = lo, self._init_size(req)
                    self.async_size = (self.size - req if self.size > req
                                       else self.size >> 1)
                    windows.append((self.start, self.start + self.size))
                else:
                    # page_cache_prev_miss: how much of what precedes this
                    # read is already resident, i.e. was read as a stream.
                    contiguous = 0
                    block = lo - 1
                    while (block >= 0 and contiguous < self.max_blocks
                           and block in cache_view):
                        contiguous += 1
                        block -= 1
                    if contiguous > req:
                        self.start = lo
                        self.size = min(contiguous + req, self.max_blocks)
                        self.async_size = PAGE_BLOCKS
                        windows.append((self.start, self.start + self.size))
                    else:
                        # A standalone random read: no read-ahead.
                        self.size = 0
            elif (marker is not None and lo <= marker < hi
                  and marker in cache_view.prefetch_hits):
                self.start += self.size
                self.size = self._next_size(self.size)
                self.async_size = self.size
                windows.append((self.start, self.start + self.size))
            for first, last in windows:
                proposed.extend(range(max(first, hi), last))
                marker = last - self.async_size if self.async_size else None
                if marker is not None and marker < hi:
                    marker = None
            self.marker = marker
        self.prev_end = hi
        return proposed

"""The decision source: the simulator's own replay, recorded per request.

``record_plan`` runs ``simulator.replay_stream`` once with a
:class:`RecordingLRU` passed as ``cache_factory`` and turns the logged cache
operations into a per-request plan: the hit/miss flag of every demanded block
and the runs of blocks newly prefetched after the request. Every simulator
mode therefore works on the real device with **no policy re-implementation**,
and the plan is the simulator's decision sequence by construction.

:class:`ShadowLRU` is the user-space cache Backend B keeps. It has exactly
``LRUCache``'s rules, and :func:`verify_plan` replays a plan through it with
no I/O, which is how the tests pin plan identity for every mode.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, NamedTuple

from ..simulator import (DEFAULT_READ_AHEAD_DEPTH, LRUCache, LatencyModel,
                         MODES_SUPPORTED, Metrics, replay_stream)
from . import DEFAULT_CAPACITY

READ, WRITE, PREFETCH = 0, 1, 2

#: Realio pseudo-modes with no simulator plan: the kernel's own read-ahead.
KERNEL_MODES = ("kernel_ra", "kernel_seq")
ONLINE_PREFIX = "online-"


class CacheDivergence(AssertionError):
    """A backend's cache disagreed with the simulator plan it was executing."""


class RecordingLRU(LRUCache):
    """``LRUCache`` that logs every operation it performs and changes nothing.

    Writes are logged too, so a trace with writes still segments correctly;
    unsuccessful prefetches (block already resident) are not, because they
    did nothing.
    """

    def __init__(self, capacity: int, metrics: Metrics):
        super().__init__(capacity, metrics)
        self.log: list[tuple[int, int, bool]] = []

    def read(self, block: int) -> bool:
        hit = super().read(block)
        self.log.append((READ, block, hit))
        return hit

    def write(self, block: int) -> None:
        super().write(block)
        self.log.append((WRITE, block, False))

    def prefetch(self, block: int) -> bool:
        issued = super().prefetch(block)
        if issued:
            self.log.append((PREFETCH, block, True))
        return issued


class PlanStep(NamedTuple):
    """One request's slice of the plan.

    ``hit_flags`` has one byte per block of a read (1 = hit) and is empty for
    a write. ``prefetch_runs`` are ``(first_block, count)`` runs of
    consecutive newly prefetched blocks, in issue order.
    """
    hit_flags: bytes
    prefetch_runs: tuple[tuple[int, int], ...]


def parse_mode(label: str) -> tuple[str, dict]:
    """``deep@64`` -> ``("deep", {"read_ahead_depth": 64})``; others unchanged.

    The label is what a results row carries; the mode and keyword arguments
    are what ``replay`` takes. Kernel and online modes are not simulator
    modes and are rejected here (the runner dispatches them first).
    """
    base, _, arg = label.partition("@")
    if base not in MODES_SUPPORTED:
        raise ValueError(f"unknown simulator mode {label!r}")
    if not arg:
        return base, {}
    if base == "deep":
        depth = int(arg)
        if depth < 0:
            raise ValueError("deep@N needs N >= 0")
        return base, {"read_ahead_depth": depth}
    if base == "linux_ra":
        return base, {"linux_ra_max_blocks": int(arg)}
    raise ValueError(f"mode {base!r} takes no @ parameter")


def _runs(blocks: list[int]) -> tuple[tuple[int, int], ...]:
    runs: list[list[int]] = []
    for block in blocks:
        if runs and block == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([block, 1])
    return tuple((first, count) for first, count in runs)


def _segment(log: list, requests: list, plan: list[PlanStep]) -> None:
    """Append ``requests``' plan steps from ``log``, which must hold exactly
    their cache operations, then empty ``log``."""
    i = 0
    for request in requests:
        n = request.size_blocks
        ops = log[i:i + n]
        expected = READ if request.operation == "R" else WRITE
        if len(ops) != n or any(op != expected for op, _, _ in ops):
            raise CacheDivergence(f"plan log does not segment at request {len(plan)}")
        i += n
        prefetched = []
        while i < len(log) and log[i][0] == PREFETCH:
            prefetched.append(log[i][1])
            i += 1
        flags = bytes(int(hit) for _, _, hit in ops) if expected == READ else b""
        plan.append(PlanStep(flags, _runs(prefetched)))
    if i != len(log):
        raise CacheDivergence(f"{len(log) - i} cache operations left after request "
                              f"{len(plan) - 1}")
    log.clear()


def record_plan(chunks: Iterable, mode: str, *, capacity: int = DEFAULT_CAPACITY,
                window_size: int = 32,
                read_ahead_depth: int = DEFAULT_READ_AHEAD_DEPTH,
                latency_model: LatencyModel | None = None,
                segment: int = 4096,
                **replay_kw) -> tuple[Metrics, list[PlanStep]]:
    """Run the simulator once and return its metrics and per-request plan.

    ``chunks`` is an iterable of request lists, as ``replay_stream`` takes;
    pass ``[requests]`` for one list. ``mode`` is a simulator mode or a
    ``deep@N`` / ``linux_ra@N`` label. Other keywords go to
    ``replay_stream`` unchanged (``model`` for ``adaptive``, a trained
    artifact path for an artifact-backed mode).

    The log is segmented by construction of the replay loop: a request
    performs exactly one cache read (or write) per block, then only
    prefetches. Anything else raises :class:`CacheDivergence`.

    The stream is fed to ``replay_stream`` in pieces of at most ``segment``
    requests, and each piece's log is turned into plan steps and dropped
    before the next is replayed (``replay_stream`` is chunk-invariant). A
    whole-run log costs ~100 bytes per block operation: 20k requests of
    msr_web_2 under ``deep@64`` peaked at 386 MiB, past the page-cache
    child's scope limit, which killed it.
    """
    if segment <= 0:
        raise ValueError("segment must be positive")
    base, extra = parse_mode(mode)
    if "read_ahead_depth" in extra:
        read_ahead_depth = extra.pop("read_ahead_depth")
    replay_kw.update(extra)
    caches: list[RecordingLRU] = []
    plan: list[PlanStep] = []

    def factory(cap: int, metrics: Metrics) -> RecordingLRU:
        caches.append(RecordingLRU(cap, metrics))
        return caches[-1]

    def pieces():
        # replay_stream asks for the next piece only after it has finished
        # every request of the previous one, prefetches included.
        previous: list = []
        for chunk in chunks:
            for start in range(0, len(chunk), segment):
                if previous:
                    _segment(caches[0].log, previous, plan)
                previous = chunk[start:start + segment]
                yield previous
        if previous:
            _segment(caches[0].log, previous, plan)

    metrics = replay_stream(pieces(), capacity=capacity, window_size=window_size,
                            mode=base, latency_model=latency_model,
                            read_ahead_depth=read_ahead_depth,
                            cache_factory=factory, **replay_kw)
    if caches and caches[0].log:
        raise CacheDivergence(f"{len(caches[0].log)} cache operations left after "
                              "the last request")
    return metrics, plan


class ShadowLRU:
    """A user-space LRU with exactly ``LRUCache``'s semantics.

    Entries are ``[pending, prefetched]``: ``pending`` is whatever the backend
    uses to wait for an in-flight read (``None`` once resident), and
    ``prefetched`` is the still-unused-prefetch flag that decides usefulness.
    An evicted entry's read is simply left to finish.
    """

    __slots__ = ("capacity", "blocks", "useful", "prefetches")

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.blocks: OrderedDict[int, list] = OrderedDict()
        self.useful = 0
        self.prefetches = 0

    def _insert(self, block: int, entry: list) -> None:
        if len(self.blocks) == self.capacity:
            self.blocks.popitem(last=False)
        self.blocks[block] = entry

    def read(self, block: int):
        """The hit entry, or ``None`` after inserting the missed block."""
        entry = self.blocks.get(block)
        if entry is not None:
            if entry[1]:
                self.useful += 1
                entry[1] = False
            self.blocks.move_to_end(block)
            return entry
        self._insert(block, [None, False])
        return None

    def write(self, block: int) -> None:
        entry = self.blocks.get(block)
        if entry is not None:
            entry[1] = False
            self.blocks.move_to_end(block)
        else:
            self._insert(block, [None, False])

    def prefetch(self, block: int, pending=None) -> bool:
        if block < 0 or block in self.blocks:
            return False
        self._insert(block, [pending, True])
        self.prefetches += 1
        return True


def verify_plan(requests: list, plan: list[PlanStep], capacity: int) -> dict:
    """Replay ``plan`` through a :class:`ShadowLRU` with no I/O.

    Raises :class:`CacheDivergence` at the first block whose hit/miss differs
    from the plan, or at a planned prefetch the shadow would not issue.
    Returns the counters Backend B reports, for comparison with ``Metrics``.
    """
    cache = ShadowLRU(capacity)
    read_blocks = hits = 0
    for index, (request, step) in enumerate(zip(requests, plan)):
        if request.operation != "R":
            for block in range(request.lba, request.lba + request.size_blocks):
                cache.write(block)
        else:
            for offset, flag in enumerate(step.hit_flags):
                hit = cache.read(request.lba + offset) is not None
                if hit != bool(flag):
                    raise CacheDivergence(
                        f"request {index} block {request.lba + offset}: shadow "
                        f"{'hit' if hit else 'miss'}, plan {'hit' if flag else 'miss'}")
                read_blocks += 1
                hits += hit
        for first, count in step.prefetch_runs:
            for block in range(first, first + count):
                if not cache.prefetch(block):
                    raise CacheDivergence(f"request {index}: planned prefetch of "
                                          f"resident block {block}")
    return {"read_blocks": read_blocks, "read_hits": hits,
            "prefetches": cache.prefetches, "useful_prefetches": cache.useful}

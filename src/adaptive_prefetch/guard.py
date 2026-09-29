"""Feedback-gated prefetching: decide from observed precision, not prediction.

The oracle experiment (see docs/DECISIONS.md D11) showed that a *perfect*
window-routing classifier beats fixed read-ahead by less than half a
percentage point on most real traces, because the candidate policies perform
almost identically. The one large, real win is the opposite decision: knowing
when prefetching **hurts**.

That decision does not need to be predicted. ``LRUCache`` already counts
``useful_prefetches``, so the realised precision of our own speculative reads
is directly observable during replay, causally, with no future knowledge and
no trained model. A controller that compares realised precision against the
cost model's break-even point can throttle itself.

Three policies, all in this module:

* ``guard``          - prefetch only while realised precision clears break-even.
* ``depth_adaptive`` - read-ahead depth scaled by the precision margin.
* ``correlate``      - read-ahead only when a repeated offset pattern is
                       confirmed over a longer history than one window.
"""

from __future__ import annotations

from collections import Counter, deque

from .trace import Request


#: Floor on the rolling window, in intervals. Small windows make the
#: controller twitchy on the first few intervals; it is a real clamp, not a
#: no-op, so tests must use it explicitly rather than asking for less.
MIN_WINDOW = 16


class PrecisionMonitor:
    """Rolling realised precision of issued prefetches.

    A prefetch is useful when the block it brought in is later read. That
    outcome is observable at read time, so this is causal: it uses only what a
    real system would already know, with no future access.

    Precision is measured over a **rolling window**, not the whole run. A
    cumulative average cannot react when the workload changes, which is
    exactly when the decision matters most.
    """

    __slots__ = ("window", "_useful", "_issued", "history")

    def __init__(self, window: int = 512):
        self.window = max(MIN_WINDOW, window)
        self._useful = 0
        self._issued = 0
        # Each entry is (useful, issued) for one accounting interval.
        self.history: deque[tuple[int, int]] = deque(maxlen=self.window)

    def record(self, useful: int, issued: int) -> None:
        """Account one interval.

        Recorded **unconditionally**, including intervals that issued nothing.
        Usefulness from blocks prefetched several intervals ago often lands in
        an interval that issued nothing, and dropping those entries discards
        exactly the signal the controller needs.
        """
        self._useful += useful
        self._issued += issued
        self.history.append((useful, issued))

    @property
    def issued(self) -> int:
        return self._issued

    @property
    def useful(self) -> int:
        return self._useful

    @property
    def precision(self) -> float:
        """Realised precision over the recent window.

        Returns ``None``-equivalent (0.0) only when nothing was ever issued.
        Callers should consult :attr:`has_data` before reading it, so "never
        measured" is distinguishable from "measured as zero".
        """
        if not self.history or self._issued == 0:
            return 0.0
        useful = sum(u for u, _ in self.history)
        issued = sum(i for _, i in self.history)
        if issued <= 0:
            return 0.0
        # Clamped at 1.0. Usefulness from a prefetch issued in an earlier
        # interval can land inside this window, so a bounded window can pair
        # more usefulness with fewer issuances and report a ratio above 1.
        # A prefetch cannot be used more than once, so 1.0 is the ceiling.
        return min(1.0, useful / issued)

    @property
    def has_data(self) -> bool:
        """True once at least one prefetch has been issued.

        It is deliberately *issued*, not *judged*: usefulness may not have
        landed yet, and reporting 0.0 precision in that case is correct -- a
        prefetch nobody has read yet has realised no value.
        """
        return self._issued > 0

    def margin(self, break_even: float) -> float:
        """How far precision sits above break-even, in probability points.

        Negative means speculative reads are destroying value at the current
        cost model, and the controller should stop issuing them.

        A break-even of exactly 0 means a prefetch is *free*, so the margin is
        the full precision. Only an infinite break-even (a hit saves nothing)
        is a hard veto.
        """
        if break_even == float("inf"):
            return -1.0
        return self.precision - break_even

    def exploring(self, minimum: int) -> bool:
        """True until enough prefetches have been issued to judge them.

        Without this the controller deadlocks: it starts with no measured
        precision, concludes it cannot prefetch, issues nothing, and therefore
        never measures any. A small exploration budget breaks the cycle.
        """
        return self._issued < minimum


def read_ahead(request: Request, depth: int, stride: int | None = None,
               use_stride: bool = False) -> list[int]:
    """Depth-controlled read-ahead, optionally along a learned stride.

    Returns ``depth`` **blocks**, in both branches. The stride branch used to
    return one block per future request and then drop any candidate inside
    this request's own extent, which meant it issued nothing at all whenever
    ``stride < size_blocks`` and only one block per request otherwise -- so
    ``correlate`` scored *half* of plain read-ahead on a contiguous scan, the
    most common workload shape there is.

    Plain read-ahead starts at the first block past this request's extent: a
    request at lba 1000 of size 8 prefetches 1008, 1009, ... at depth 2.

    Strided read-ahead starts ``stride`` blocks ahead and walks the predicted
    stream for ``depth`` blocks. When ``stride`` equals the request size that
    is exactly the next request; when it is larger it is the leading part of a
    request further away. When it is *smaller* than the size the stream
    overlaps this request, so the first few candidates are already resident
    and ``cache.prefetch`` declines them -- harmless, and much better than
    refusing to prefetch at all.
    """
    if request.operation != "R" or depth <= 0:
        return []
    if use_stride and stride:
        start = request.lba + stride
        return list(range(start, start + depth))
    # Plain read-ahead: the first block past this request's extent, onward.
    start = request.lba + request.size_blocks
    return list(range(start, start + depth))


class CorrelateDetector:
    """Detect a confirmed repeated offset over a longer history.

    ``dominant_stride`` looks inside one window and returns None on most real
    traces. This keeps a longer delta history and requires the same offset to
    recur several times before proposing anything, which is stricter about
    what counts as a pattern.
    """

    __slots__ = ("order", "min_repeat", "max_delta", "_deltas", "table",
                 "_last_lba", "capacity")

    def __init__(self, order: int = 3, min_repeat: int = 3,
                 max_delta: int = 1 << 20, capacity: int = 4096):
        self.order = order
        self.min_repeat = min_repeat
        self.max_delta = max_delta
        self._deltas: deque[int] = deque(maxlen=order)
        # Bounded. A plain Counter over every offset history seen grows without
        # limit on a narrow-address workload (~150 B/request, a third of the
        # cost of the Request objects the streaming design exists to avoid) and
        # makes confirmed_stride() O(len(table)) per decision.
        self.capacity = max(64, capacity)
        self.table: dict[tuple[int, ...], int] = {}
        self._last_lba: int | None = None

    def observe(self, delta: int) -> None:
        if abs(delta) <= self.max_delta:
            key = tuple(self._deltas)
            self.table[key] = self.table.get(key, 0) + 1
            if len(self.table) > self.capacity:
                # Drop the least-seen entries; a stride worth acting on is
                # seen many times, so eviction cannot discard it.
                for old in sorted(self.table, key=self.table.get)[:len(self.table) // 2]:
                    del self.table[old]
        self._deltas.append(delta)

    def confirmed_stride(self) -> int | None:
        """The most repeated positive delta in the history, if confirmed.

        Recomputed on every call rather than memoised. An earlier version
        cached on ``len(table)``, which did not change when an *existing*
        key's count was incremented, so after a phase change the detector
        kept reporting the old stride indefinitely. The scan is over a table
        capped at :attr:`capacity` entries, called once per decision
        interval, so it is not worth caching incorrectly.
        """
        counts: Counter[int] = Counter()
        for deltas, seen in self.table.items():
            if seen < self.min_repeat or len(deltas) < self.order:
                continue
            counts[deltas[-1]] += seen
        if not counts:
            return None
        stride, seen = counts.most_common(1)[0]
        return stride if seen >= self.min_repeat and stride > 0 else None

    def update(self, request: Request) -> None:
        if request.operation != "R":
            return
        if self._last_lba is not None:
            self.observe(request.lba - self._last_lba)
        self._last_lba = request.lba


class GateController:
    """Shared state machine for the feedback-gated modes.

    ``replay`` and ``replay_stream`` both drive this rather than each
    reimplementing the decision, so the two can never diverge again.
    """

    #: Consecutive closed decisions before forcing a one-interval re-probe.
    #: Without this a gate that closes can never reopen: it stops issuing, so
    #: it never gathers new evidence, so it stays closed for the rest of the
    #: trace even if the workload changes completely.
    REPROBE_AFTER = 8

    def __init__(self, mode: str, break_even: float,
                 exploration: int = 256, min_depth: int = 2):
        self.mode = mode
        self.break_even = break_even
        self.exploration = exploration
        self.min_depth = min_depth
        self.monitor = PrecisionMonitor()
        self.correlator = CorrelateDetector() if mode == "correlate" else None
        self.active_depth = 0
        self.switches = 0
        self.stride: int | None = None
        self._previous_depth = 0
        self._issued_since = 0
        self._useful_total = 0
        self._useful_accounted = 0
        self._closed_streak = 0
        self._probes_left = 0

    def note_prefetch(self) -> None:
        """Call once per issued prefetch."""
        self._issued_since += 1

    def note_useful(self, total_useful: int) -> None:
        """Call after each request with the cumulative useful-prefetch count."""
        self._useful_total = total_useful

    def observe(self, request: Request) -> None:
        """Feed one request to the correlator. No-op unless mode is correlate."""
        if self.correlator is not None:
            self.correlator.update(request)

    def decide(self) -> int:
        """Account the interval and set the depth for the next one."""
        if self.break_even == float("inf"):
            # A prefetch that saves nothing can never be worth issuing, so
            # exploration and re-probing must not override it. Skipping them
            # here is what stops the controller burning 1000+ reads on a cost
            # model where every single one is provably wasted.
            self.active_depth = 0
            self._closed_streak = 0
            return 0
        # ``note_useful`` supplies a *cumulative* count, so the delta since the
        # last decision is what belongs in this interval's record. Recording
        # the cumulative value instead double-counts every earlier interval's
        # usefulness and inflates the ratio, which pins it at 1.0 and leaves
        # the gate permanently open regardless of how bad the prefetches are.
        useful_delta = self._useful_total - self._useful_accounted
        self._useful_accounted = self._useful_total
        self.monitor.record(useful_delta, self._issued_since)
        self._issued_since = 0
        margin = self.monitor.margin(self.break_even)
        probing = self._probes_left > 0
        if probing:
            self._probes_left -= 1
        if self.monitor.exploring(self.exploration) or probing:
            depth = self.min_depth
        elif self.mode == "guard":
            depth = self.min_depth if margin > 0 else 0
        elif self.mode == "depth_adaptive":
            depth = depth_for_margin(margin)
        else:
            assert self.correlator is not None
            # The stride is tracked whether or not the gate is open, so
            # `observed_stride` reports what the detector currently believes
            # rather than only what happened to be acted upon.
            self.stride = self.correlator.confirmed_stride()
            depth = (self.min_depth
                     if margin > 0 and self.stride else 0)
        if depth == 0:
            self._closed_streak += 1
            if self._closed_streak >= self.REPROBE_AFTER:
                # The workload may have changed. Spend one interval finding out.
                self._probes_left = 1
                self._closed_streak = 0
        else:
            self._closed_streak = 0
        if depth != self._previous_depth:
            self.switches += 1
        self._previous_depth = depth
        self.active_depth = depth
        return depth

    def candidates(self, request: Request) -> list[int]:
        """Prefetch candidates for one request under the current depth."""
        use_stride = (self.mode == "correlate" and self.stride is not None)
        return read_ahead(request, self.active_depth, self.stride, use_stride)

    @property
    def margin(self) -> float:
        return self.monitor.margin(self.break_even)

    @property
    def precision(self) -> float:
        return self.monitor.precision

    @property
    def has_data(self) -> bool:
        return self.monitor.has_data


def depth_for_margin(margin: float, min_depth: int = 1, max_depth: int = 8) -> int:
    """Read-ahead depth from the precision margin above break-even.

    A small positive margin gets a shallow prefetch, a large one gets a deep
    one, and a non-positive margin stops prefetching entirely. Clamped, so a
    pathologically confident early estimate cannot issue a huge burst.
    """
    if margin <= 0:
        return 0
    if margin >= 0.30:
        return max_depth
    if margin >= 0.15:
        return max(2, max_depth // 2)
    return min_depth

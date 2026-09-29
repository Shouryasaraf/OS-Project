"""Small, address-scale independent features for fixed I/O windows."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, pstdev

from .trace import Request

FEATURE_NAMES = (
    "contiguous_ratio", "dominant_stride_ratio", "random_jump_ratio",
    "short_run_ratio", "unique_ratio", "gap_cv", "fast_gap_ratio", "read_ratio",
    # Contextual features. These describe the window *in relation to the
    # stream around it*, which the eight above cannot: workloads are
    # persistent, so a window that continues the previous one behaves
    # differently from an identical window that starts a new phase. The
    # optional first 8 above are also the complete set when no context is
    # supplied, so existing callers keep working.
    "reuse_ratio", "reuse_distance_log",
    "contiguous_delta", "stride_persist",
)

#: Number of features available without any stream context.
STREAM_FEATURES = 8

#: A "distant jump" is measured relative to the request size, not in absolute
#: blocks. The previous fixed 64-block threshold made a 128-block request look
#: sequential and a 4 KiB stride look random, so the same access pattern
#: classified differently on devices with different block sizes.
DISTANT_JUMP_SCALE = 8
#: Symmetric counterpart of the threshold above.
SHORT_RUN_SCALE = 0.5
#: Normalised intra-window speed contrast; fixed 0.15 ms assumed a device
#: arrival rate and made the feature device- and load-dependent.
FAST_GAP_FRACTION = 0.25
#: Shared with ``simulator.route_window``: contiguity at or above this counts
#: as genuinely sequential. Calibrated against the measured real-trace
#: distribution, where ``contiguous_ratio`` has median 0.032 and p90 0.097.
MIN_CONTIGUITY = 0.10


def extract(window: list[Request]) -> tuple[float, ...]:
    if len(window) < 8:
        raise ValueError("feature window must contain at least 8 requests")
    deltas = [b.lba - a.lba for a, b in zip(window, window[1:])]
    sizes = [max(1, a.size_blocks) for a in window[:-1]]
    contiguous = [d == a.size_blocks for d, a in zip(deltas, window)]
    noncontiguous = [d for d, is_contiguous in zip(deltas, contiguous)
                     if not is_contiguous and d != 0]
    dominant = Counter(noncontiguous).most_common(1)[0][1] if noncontiguous else 0
    gaps = [max(0.0, b.timestamp_ms - a.timestamp_ms)
            for a, b in zip(window, window[1:])]
    average_gap = mean(gaps)
    # Scale-free jump / short-run classification, per adjacent pair.
    far = 0
    short = 0
    for delta, size in zip(deltas, sizes):
        if abs(delta) > DISTANT_JUMP_SCALE * size:
            far += 1
        elif abs(delta) <= SHORT_RUN_SCALE * size:
            short += 1
    # Speed contrast relative to this window's own mean arrival interval.
    fast_threshold = FAST_GAP_FRACTION * average_gap
    return (
        sum(contiguous) / len(deltas),
        dominant / len(deltas),
        far / len(deltas),
        short / len(deltas),
        len({r.lba for r in window}) / len(window),
        min(pstdev(gaps) / average_gap, 10.0) if average_gap else 0.0,
        sum(g < fast_threshold for g in gaps) / len(gaps) if average_gap else 0.0,
        sum(r.operation == "R" for r in window) / len(window),
    )


class WindowContext:
    """Running stream state for the contextual features.

    Maintains a block-age table across windows so the reuse features know how
    recently each block was last touched, and remembers the previous window's
    spatial features. This is the information a real system would get from
    block-age tracking in its cache; it is genuinely available at decision
    time, so using it is not lookahead.
    """

    __slots__ = ("last_seen", "index", "previous", "previous_stride")

    def __init__(self):
        self.last_seen: dict[int, int] = {}
        self.index = 0
        self.previous: tuple[float, ...] | None = None
        self.previous_stride: int | None = None

    def observe(self, window: list[Request], stride: int | None) -> tuple:
        """Consume one window and return its full contextual feature vector."""
        base = extract(window)
        ages: list[int] = []
        for request in window:
            newest = 0
            for block in range(request.lba, request.lba + request.size_blocks):
                previous = self.last_seen.get(block)
                if previous is not None:
                    age = self.index - previous
                    if age > newest:
                        newest = age
                self.last_seen[block] = self.index
            ages.append(newest)
            self.index += 1

        revisited = [age for age in ages if age > 0]
        reuse_ratio = len(revisited) / len(ages) if ages else 0.0
        # Mean log2 reuse distance, a scale-free way to say "how far back did
        # the re-visit reach". log2 because reuse distances span 1 to ~1e6.
        if revisited:
            reuse_distance = sum(math.log2(age) for age in revisited) / len(revisited)
            reuse_distance /= 16.0
        else:
            reuse_distance = 0.0

        contiguous_delta = (base[0] - self.previous[0]
                           if self.previous is not None else 0.0)
        stride_persist = 0.0
        if self.previous is not None:
            same_stride = (stride is not None
                           and self.previous_stride is not None
                           and stride == self.previous_stride)
            stable = (base[0] >= MIN_CONTIGUITY
                      and self.previous[0] >= MIN_CONTIGUITY)
            stride_persist = 1.0 if (same_stride or stable) else 0.0

        self.previous = base
        self.previous_stride = stride
        return base + (reuse_ratio, min(reuse_distance, 4.0),
                       contiguous_delta, stride_persist)

    def result(self, window: list[Request], stride: int | None) -> tuple:
        """Contextual vector without advancing the stream state."""
        saved = (dict(self.last_seen), self.index, self.previous,
                 self.previous_stride)
        try:
            return self.observe(window, stride)
        finally:
            self.last_seen, self.index, self.previous, self.previous_stride = saved


def dominant_stride(window: list[Request]) -> int | None:
    """Most common forward stride in the window, or None.

    The candidate range is any forward motion past the end of the current
    request. The previous form used ``d > size_blocks and d <= 1024``, an
    absolute ceiling that on real traces (median 8-block requests, addresses
    up to 1e8) returned ``None`` on 100% of windows measured, so the
    window-level stride detector was structurally blind on real data.

    There is no magnitude cap now: a stride is defined by *repetition*, not
    size. The repetition requirement is a fraction of the window rather than
    an absolute count, so it behaves the same at any window size.
    """
    deltas = [b.lba - a.lba for a, b in zip(window, window[1:])]
    candidates = []
    for delta, a in zip(deltas, window):
        size = max(1, a.size_blocks)
        # Forward motion past the end of the current request. There is
        # deliberately **no upper magnitude cap**: repetition, not magnitude,
        # is what distinguishes a stride from a jump. A 4096-block delta seen
        # 15 times is a stride; the same delta seen once is a seek. The old
        # absolute `d <= 1024` cap conflated the two and returned None on 100%
        # of real windows, while a relative cap of 64x request sizes still
        # rejected legitimate large strides.
        if delta > size:
            candidates.append(delta)
    if not candidates:
        return None
    stride, count = Counter(candidates).most_common(1)[0]
    return stride if count >= max(2, len(deltas) // 4) else None

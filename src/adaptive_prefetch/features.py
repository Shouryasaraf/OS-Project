"""Small, address-scale independent features for fixed I/O windows."""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

from .trace import Request

FEATURE_NAMES = (
    "contiguous_ratio", "dominant_stride_ratio", "random_jump_ratio",
    "short_run_ratio", "unique_ratio", "gap_cv", "fast_gap_ratio", "read_ratio",
)

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


def dominant_stride(window: list[Request]) -> int | None:
    deltas = [b.lba - a.lba for a, b in zip(window, window[1:])]
    candidates = [d for d, a in zip(deltas, window)
                  if d > a.size_blocks and d <= 1024]
    if not candidates:
        return None
    stride, count = Counter(candidates).most_common(1)[0]
    return stride if count >= max(2, len(deltas) // 4) else None

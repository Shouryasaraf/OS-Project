"""Small, address-scale independent features for fixed I/O windows."""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

from .trace import Request

FEATURE_NAMES = (
    "contiguous_ratio", "dominant_stride_ratio", "random_jump_ratio",
    "short_run_ratio", "unique_ratio", "gap_cv", "fast_gap_ratio", "read_ratio",
)


def extract(window: list[Request]) -> tuple[float, ...]:
    if len(window) < 8:
        raise ValueError("feature window must contain at least 8 requests")
    deltas = [b.lba - a.lba for a, b in zip(window, window[1:])]
    contiguous = [d == a.size_blocks for d, a in zip(deltas, window)]
    noncontiguous = [d for d, is_contiguous in zip(deltas, contiguous)
                     if not is_contiguous and d != 0]
    dominant = Counter(noncontiguous).most_common(1)[0][1] if noncontiguous else 0
    gaps = [max(0.0, b.timestamp_ms - a.timestamp_ms)
            for a, b in zip(window, window[1:])]
    average_gap = mean(gaps)
    return (
        sum(contiguous) / len(deltas),
        dominant / len(deltas),
        sum(abs(d) > 64 for d in deltas) / len(deltas),
        sum(abs(d) <= 16 for d in deltas) / len(deltas),
        len({r.lba for r in window}) / len(window),
        min(pstdev(gaps) / average_gap, 10.0) if average_gap else 0.0,
        sum(g < 0.15 for g in gaps) / len(gaps),
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

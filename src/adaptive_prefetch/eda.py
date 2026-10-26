"""Exploratory data analysis and domain-shift measurement for I/O traces.

Everything here is read-only and stdlib-only. The point of this module is that
the project's headline claims ("100% classification accuracy", "adaptive beats
the baselines") are only meaningful relative to *where the data actually is*,
so this measures that directly instead of asserting it.

Typical use::

    from adaptive_prefetch.eda import profile_trace, domain_shift
    report = profile_trace(requests, "my-trace")
    shift = domain_shift(synthetic_windows, real_windows)
"""

from __future__ import annotations

import heapq
from collections import Counter
from statistics import mean, pstdev

from .features import FEATURE_NAMES, STREAM_FEATURES, extract
from .trace import CLASSES, Request


# --------------------------------------------------------------------------
# Trace-level structure
# --------------------------------------------------------------------------

def request_profile(requests: list[Request]) -> dict:
    """Basic request-level statistics for a loaded trace."""
    if not requests:
        return {"requests": 0}
    reads = [r for r in requests if r.operation == "R"]
    sizes = [r.size_blocks for r in requests]
    lbas = [r.lba for r in requests]
    deltas = [b.lba - a.lba for a, b in zip(requests, requests[1:])]
    buckets = Counter(
        "zero" if d == 0 else "neg" if d < 0 else
        "1-4" if d <= 4 else "5-64" if d <= 64 else ">64"
        for d in deltas
    )
    return {
        "requests": len(requests),
        "reads": len(reads),
        "read_fraction": len(reads) / len(requests),
        "streams": len({r.stream_id for r in requests}),
        "span_ms": requests[-1].timestamp_ms - requests[0].timestamp_ms,
        "lba_min": min(lbas),
        "lba_max": max(lbas),
        "lba_span": max(lbas) - min(lbas),
        "size_mean": mean(sizes),
        "size_max": max(sizes),
        "delta_buckets": {k: buckets.get(k, 0) for k in
                          ("zero", "1-4", "5-64", ">64", "neg")},
        "distinct_deltas": len(set(deltas)),
    }


def _block_accesses(requests: list[Request]) -> list[tuple[int, bool]]:
    """The 512-B block access stream, ``(block, is_read)``, in request order.

    Writes are included because the simulator's cache allocates on write
    (``LRUCache.write``): a block written and then read is a hit there.
    """
    return [(block, request.operation == "R") for request in requests
            for block in range(request.lba, request.lba + request.size_blocks)]


def lru_stack_distances(requests: list[Request]) -> tuple[Counter, int]:
    """Mattson LRU stack distance of every read block access.

    Returns ``(histogram, read_blocks)``. ``histogram[d]`` counts reads whose
    block was the ``d``-th most recently used block (1 = most recent) when it
    was accessed; first touches have no distance and are absent. Writes take
    part in recency exactly as in ``LRUCache``. By the inclusion property a
    read hits an LRU cache of ``c`` blocks iff its distance is at most ``c``,
    so one pass gives the demand hit ratio at every capacity. A Fenwick tree
    over access times makes this O(n log n).
    """
    accesses = _block_accesses(requests)
    size = len(accesses)
    tree = [0] * (size + 1)
    last: dict[int, int] = {}
    histogram: Counter = Counter()
    reads = 0
    live = 0
    for now, (block, is_read) in enumerate(accesses):
        reads += is_read
        previous = last.get(block)
        if previous is not None:
            # Markers sit at each block's most recent access time, so the
            # markers after `previous` are the distinct blocks touched since.
            index, before = previous + 1, 0
            while index > 0:
                before += tree[index]
                index -= index & -index
            if is_read:
                histogram[live - before + 1] += 1
            index = previous + 1
            while index <= size:
                tree[index] -= 1
                index += index & -index
        else:
            live += 1
        index = now + 1
        while index <= size:
            tree[index] += 1
            index += index & -index
        last[block] = now
    return histogram, reads


def lru_demand_curve(requests: list[Request],
                     capacities: list[int] | tuple[int, ...]) -> dict[int, float]:
    """LRU demand hit ratio (no prefetch) at each capacity, in one pass.

    At a given capacity this is exactly the hit ratio of the ``none`` mode,
    because ``none`` is the same LRU cache with nothing prefetched.
    """
    histogram, reads = lru_stack_distances(requests)
    result = {}
    for capacity in capacities:
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        hits = sum(count for distance, count in histogram.items()
                   if distance <= capacity)
        result[capacity] = hits / reads if reads else 0.0
    return result


def lru_demand_hit_ratio(requests: list[Request], capacity: int) -> float:
    """LRU demand hit ratio at one capacity; see :func:`lru_demand_curve`."""
    return lru_demand_curve(requests, (capacity,))[capacity]


def belady_demand_hit_ratio(requests: list[Request], capacity: int) -> float:
    """Belady MIN demand hit ratio: the best any demand-only cache can do.

    Offline optimal replacement over the block stream: on overflow, evict the
    resident block (the incoming one included, i.e. bypass is allowed) whose
    next *read* is farthest away. A block whose next access is a write is
    dead, because the write re-allocates it whatever happens. Built from the
    reconstructed block stream with a two-pass next-use array and a max-heap.

    This uses future knowledge and is a **reference, not a policy**. It bounds
    demand caching only: a prefetcher fetches blocks before their first read,
    so ``deep`` can and does exceed it.
    """
    if capacity < 1:
        raise ValueError("cache capacity must be positive")
    accesses = _block_accesses(requests)
    dead = len(accesses) + 1
    next_use = [dead] * len(accesses)
    upcoming: dict[int, tuple[int, bool]] = {}
    for now in range(len(accesses) - 1, -1, -1):
        block, is_read = accesses[now]
        following = upcoming.get(block)
        if following is not None and following[1]:
            next_use[now] = following[0]
        upcoming[block] = (now, is_read)

    resident: dict[int, int] = {}
    heap: list[tuple[int, int]] = []
    hits = reads = 0
    for now, (block, is_read) in enumerate(accesses):
        if is_read:
            reads += 1
            hits += block in resident
        resident[block] = next_use[now]
        heapq.heappush(heap, (-next_use[now], block))
        if len(resident) > capacity:
            # Lazy deletion: an entry is current only while it still matches
            # the block's recorded next use.
            while True:
                negative, victim = heapq.heappop(heap)
                if resident.get(victim) == -negative:
                    del resident[victim]
                    break
    return hits / reads if reads else 0.0


def locality_profile(requests: list[Request], capacity: int = 128) -> dict:
    """Re-access statistics and demand-cache references at ``capacity``.

    ``read_reuse_within_capacity`` (also kept under its old key
    ``cacheable_ceiling``) is the fraction of read blocks re-read within
    ``capacity`` *read accesses* of their previous read. It is a reuse
    statistic, **not a bound**: it ignores writes, counts accesses rather than
    distinct blocks, and says nothing about prefetching, so the ``none`` mode
    can exceed it. The demand-cache references are
    ``lru_demand_hit_ratio`` (Mattson; equals the ``none`` mode) and
    ``belady_demand_hit_ratio`` (offline-optimal replacement, no prefetch).
    """
    reads = [r for r in requests if r.operation == "R"]
    read_blocks = [b for r in reads for b in range(r.lba, r.lba + r.size_blocks)]
    if not read_blocks:
        return {"read_blocks": 0, "locality_ceiling": 0.0,
                "cacheable_ceiling": 0.0, "read_reuse_within_capacity": 0.0,
                "lru_demand_hit_ratio": 0.0, "belady_demand_hit_ratio": 0.0}
    total = Counter(read_blocks)
    repeats = sum(count - 1 for count in total.values())
    last_seen: dict[int, int] = {}
    cacheable = 0
    for index, block in enumerate(read_blocks):
        previous = last_seen.get(block)
        if previous is not None and index - previous <= capacity:
            cacheable += 1
        last_seen[block] = index
    return {
        "read_blocks": len(read_blocks),
        "distinct_read_blocks": len(total),
        "repeats": repeats,
        "locality_ceiling": repeats / len(read_blocks),
        "cacheable_ceiling": cacheable / len(read_blocks),
        "read_reuse_within_capacity": cacheable / len(read_blocks),
        "lru_demand_hit_ratio": lru_demand_hit_ratio(requests, capacity),
        "belady_demand_hit_ratio": belady_demand_hit_ratio(requests, capacity),
    }


def window_features(requests: list[Request], window_size: int = 32) -> list[tuple]:
    """Feature vectors for every complete window; the unit of classification."""
    if window_size < 8:
        raise ValueError("window_size must be at least 8")
    return [extract(requests[i:i + window_size])
            for i in range(0, len(requests) - window_size + 1, window_size)]


# --------------------------------------------------------------------------
# Cluster structure
# --------------------------------------------------------------------------

def kmeans(points: list[tuple], k: int, restarts: int = 8,
          iterations: int = 60) -> tuple[list[int], float]:
    """Plain k-means in raw feature space. Returns (labels, within-cluster SSE)."""
    if k < 1 or not points:
        raise ValueError("k must be >= 1 and points must be non-empty")
    if k == 1 or len(points) <= k:
        dimension = len(points[0])
        centroid = [mean([p[i] for p in points]) for i in range(dimension)]
        sse = sum(sum((p[i] - centroid[i]) ** 2 for i in range(dimension))
                  for p in points)
        return [0] * len(points), sse
    dimension = len(points[0])
    best_labels, best_sse = None, None
    # Deterministic seeding: pick evenly spaced starts, then refine.
    for restart in range(restarts):
        centres = [list(points[min(len(points) - 1,
                                   (restart * i * len(points)) // max(k, 1))])
                   for i in range(k)]
        labels = [0] * len(points)
        for _ in range(iterations):
            changed = False
            for index, point in enumerate(points):
                best = min(range(k), key=lambda c: sum(
                    (point[i] - centres[c][i]) ** 2 for i in range(dimension)))
                if labels[index] != best:
                    labels[index] = best
                    changed = True
            for c in range(k):
                members = [p for p, lab in zip(points, labels) if lab == c]
                if members:
                    centres[c] = [mean([m[i] for m in members])
                                  for i in range(dimension)]
            if not changed:
                break
        sse = 0.0
        for point, label in zip(points, labels):
            sse += sum((point[i] - centres[label][i]) ** 2
                       for i in range(dimension))
        if best_sse is None or sse < best_sse:
            best_sse, best_labels = sse, list(labels)
    return best_labels, best_sse  # type: ignore[return-value]


def elbow(points: list[tuple], k_max: int = 6) -> list[tuple[int, float]]:
    """SSE for k = 1..k_max, for judging whether k clusters exist at all."""
    return [(k, kmeans(points, k)[1]) for k in range(1, k_max + 1)]


# --------------------------------------------------------------------------
# Domain shift
# --------------------------------------------------------------------------

def domain_shift(reference: list[tuple], target: list[tuple]) -> dict:
    """Standardised shift between two feature populations.

    Reports, per feature, how many reference standard deviations the target
    mean sits away, and how many target points fall inside the reference
    range. A shift of many sigma, or near-zero coverage, means a model fitted
    on the reference is extrapolating rather than interpolating.
    """
    if not reference or not target:
        raise ValueError("both populations must be non-empty")
    result = {"features": {}, "mean_abs_z": 0.0, "max_abs_z": 0.0,
              "coverage": 0.0, "nearest_centroid_ratio": None}
    total_z, worst_z, covered, total = 0.0, 0.0, 0, 0
    # Domain shift is reported over the stream features only; the contextual
    # block depends on replay position and has no meaningful reference spread.
    for i, name in enumerate(FEATURE_NAMES[:STREAM_FEATURES]):
        ref = [p[i] for p in reference]
        tgt = [p[i] for p in target]
        sd = pstdev(ref) or 1.0
        lo, hi = min(ref), max(ref)
        z = (mean(tgt) - mean(ref)) / sd
        inside = sum(1 for value in tgt if lo <= value <= hi)
        result["features"][name] = {
            "ref_mean": mean(ref), "ref_sd": sd,
            "ref_min": lo, "ref_max": hi,
            "target_mean": mean(tgt), "target_min": min(tgt),
            "target_max": max(tgt), "shift_sd": z, "coverage": inside / len(tgt),
        }
        total_z += abs(z)
        worst_z = max(worst_z, abs(z))
        covered += inside
        total += len(tgt)
    result["mean_abs_z"] = total_z / STREAM_FEATURES
    result["max_abs_z"] = worst_z
    result["coverage"] = covered / total
    return result


def centroid_distances(reference: list[tuple], target: list[tuple],
                        labelled: list[tuple] | None = None) -> dict:
    """How far target windows sit from the reference class centroids.

    Compare this against how far a reference window sits from its own
    centroid. A ratio far above 1.0 means the target is nowhere near the
    region the classifier was fitted on.
    """
    if not reference or not target:
        raise ValueError("both populations must be non-empty")
    dimension = len(reference[0])
    sample = reference if labelled is None else labelled
    centres = [[mean([p[i] for p in sample]) for i in range(dimension)]]

    def nearest(point: tuple) -> float:
        return min(sum((point[i] - c[i]) ** 2 for i in range(dimension))
                   for c in centres) ** 0.5

    ref_distance = mean([nearest(p) for p in reference])
    target_distance = mean([nearest(p) for p in target])
    return {
        "reference_to_centroid": ref_distance,
        "target_to_centroid": target_distance,
        "ratio": (target_distance / ref_distance) if ref_distance else None,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _histogram(values: list[float], bins: int = 10,
               low: float = 0.0, high: float = 1.0) -> str:
    if not values:
        return "(no data)"
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int((value - low) / width)))
        counts[index] += 1
    peak = max(counts) or 1
    lines = []
    for index, count in enumerate(counts):
        bar = "#" * int(30 * count / peak)
        lines.append(f"  [{low + index * width:6.3f}, {low + (index + 1) * width:6.3f}) "
                     f"{count:5d} {bar}")
    return "\n".join(lines)


def profile_trace(requests: list[Request], name: str,
                  window_size: int = 32, capacity: int = 128) -> str:
    """Human-readable EDA report for one trace."""
    out: list[str] = [f"# Trace profile: {name}", ""]
    if not requests:
        return "\n".join(out + ["(empty trace)", ""])

    profile = request_profile(requests)
    out.append("## Request stream")
    out.append(f"- requests: {profile['requests']:,} "
               f"({profile['reads']:,} read, "
               f"{100 * profile['read_fraction']:.1f}%)")
    out.append(f"- streams: {profile['streams']}   "
               f"span: {profile['span_ms'] / 1000:.2f} s")
    out.append(f"- LBA range: {profile['lba_min']:,} .. {profile['lba_max']:,} "
               f"(span {profile['lba_span']:,} blocks)")
    out.append(f"- request size: mean {profile['size_mean']:.1f}, "
               f"max {profile['size_max']} blocks")
    out.append(f"- distinct consecutive deltas: {profile['distinct_deltas']:,}")
    out.append("- delta distribution:")
    total = max(1, sum(profile["delta_buckets"].values()))
    for bucket, count in profile["delta_buckets"].items():
        out.append(f"    {bucket:>6}: {count:7d}  ({100 * count / total:5.1f}%)")

    locality = locality_profile(requests, capacity)
    out.append("")
    out.append(f"## Locality (cache capacity {capacity} blocks)")
    if locality["read_blocks"]:
        out.append(f"- read blocks: {locality['read_blocks']:,} "
                   f"({locality['distinct_read_blocks']:,} distinct)")
        out.append(f"- read blocks that repeat an earlier read: "
                   f"{100 * locality['locality_ceiling']:.2f}%")
        out.append(f"- read-reuse within {capacity} reads (not a bound): "
                   f"{100 * locality['read_reuse_within_capacity']:.2f}%")
        out.append(f"- **LRU demand hit ratio (Mattson, no prefetch): "
                   f"{100 * locality['lru_demand_hit_ratio']:.2f}%**  "
                   f"<- equals the `none` mode")
        out.append(f"- **Belady MIN demand hit ratio (no prefetch): "
                   f"{100 * locality['belady_demand_hit_ratio']:.2f}%**  "
                   f"<- best possible replacement; prefetching can exceed it")
    else:
        out.append("- no read requests: every cache metric is 0/0 for this trace")

    vectors = window_features(requests, window_size)
    out.append("")
    out.append(f"## Windows ({window_size} requests each)")
    if not vectors:
        out.append("- no complete windows")
        return "\n".join(out + [""])
    out.append(f"- complete windows: {len(vectors)}")
    out.append("- per-feature mean / sd / min / max:")
    for i, feature in enumerate(FEATURE_NAMES[:STREAM_FEATURES]):
        column = [v[i] for v in vectors]
        out.append(f"    {feature:22} mean {mean(column):7.4f}  "
                   f"sd {pstdev(column):7.4f}  "
                   f"min {min(column):7.4f}  max {max(column):7.4f}")

    out.append("")
    out.append("- contiguous_ratio histogram:")
    out.append(_histogram([v[0] for v in vectors]))
    out.append("- random_jump_ratio histogram:")
    out.append(_histogram([v[2] for v in vectors]))

    if len(vectors) >= 4:
        out.append("")
        out.append("### Cluster structure (k-means, raw feature space)")
        for k, sse in elbow(vectors, min(6, max(2, len(vectors) // 2))):
            ratio = f"  (x{sse / 1.0:.3f})" if k > 1 else ""
            out.append(f"    k={k}: SSE = {sse:10.4f}{ratio}")
        out.append("    -> a real 4-class structure shows a clear elbow at k=4;")
        out.append("       a monotone decline means no cluster structure exists.")
    return "\n".join(out + [""])


def report_shift(reference: list[tuple], target: list[tuple],
                 reference_name: str, target_name: str) -> str:
    """Human-readable domain-shift report between two window populations."""
    shift = domain_shift(reference, target)
    out = [f"# Domain shift: {reference_name} (reference) -> {target_name}", ""]
    out.append(f"- mean |shift| across features: **{shift['mean_abs_z']:.2f} "
               f"reference sd**")
    out.append(f"- worst feature: **{shift['max_abs_z']:.2f} sd**")
    out.append(f"- target windows inside the reference per-feature range: "
               f"**{100 * shift['coverage']:.1f}%**")
    out.append("")
    out.append("| feature | ref mean | ref sd | target mean | shift (sd) | coverage |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, stats in shift["features"].items():
        out.append(f"| {name} | {stats['ref_mean']:.4f} | {stats['ref_sd']:.4f} "
                   f"| {stats['target_mean']:.4f} | {stats['shift_sd']:+.2f} "
                   f"| {100 * stats['coverage']:.0f}% |")
    if shift["mean_abs_z"] > 2 or shift["coverage"] < 0.5:
        out.append("")
        out.append("-> **The target sits well outside the reference region.** A "
                   "model fitted on the reference is extrapolating; accuracy "
                   "measured on the reference says nothing about the target.")
    return "\n".join(out + [""])


def label_support(labelled: list[tuple]) -> dict:
    """Count how many distinct classes a labelled population actually covers."""
    counts = Counter(label for _, label in labelled)
    return {
        "counts": {label: counts.get(label, 0) for label in CLASSES},
        "present": [label for label in CLASSES if counts.get(label, 0)],
        "missing": [label for label in CLASSES if not counts.get(label, 0)],
    }

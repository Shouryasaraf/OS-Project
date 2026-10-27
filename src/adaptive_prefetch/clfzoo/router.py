"""Part 2: every tuned classifier as the ``adaptive`` router on real traces.

The ``adaptive`` mode is used unchanged: ``simulator.replay_stream(mode=
"adaptive", model=...)`` with E1's multicorpus loading, slicing and cost
model (``multicorpus.run_job``'s exact call). What makes 15 classifiers x 19
traces x 2 capacities affordable is one observation about that loop:

* the vector a window is classified on (``WindowContext.observe(history,
  dominant_stride(history))``) depends only on the request stream, never on
  cache state or on earlier predictions, so it is computed **once per
  trace** (:func:`window_features`), and every classifier is batch-scored
  on the same vectors;
* the replay depends on the classifier only through the sequence of class
  labels it returns (``replay_stream`` reads no confidence without
  smoothing or pseudo-labels), so :class:`Replayed` hands the precomputed
  labels back in order, and two classifiers with the same label sequence on
  a trace have the same replay (computed once, :func:`label_key`).

``tests/test_clfzoo.py`` proves both: a replay driven by :class:`Replayed`
equals the per-call replay with the shipped GNB on every counter. This
module is stdlib-only.
"""

from __future__ import annotations

import hashlib

from ..baselines import DEFAULT_LINUX_RA_MAX_BLOCKS
from ..features import DEFAULT_AGE_TABLE_BOUND, WindowContext, dominant_stride
from ..simulator import LatencyModel, replay_stream
from ..trace import CLASSES, DEFAULT_MAX_MERGE_SECTORS, iter_oracle_chunks

WINDOW_SIZE = 32
CHUNK_REQUESTS = 250_000
#: E1's ``assumed`` cost model (multicorpus.LATENCIES["assumed"]).
ASSUMED = LatencyModel(5.0, 100.0, 50.0)
#: The counters a router row carries, all exact (no timings).
COUNTERS = ("read_blocks", "read_hits", "prefetches", "useful_prefetches",
            "policy_switches", "demand_ios", "prefetch_ios",
            "prefetch_ios_mergeable", "demand_latency_us", "prefetch_cost_us",
            "inference_calls")


def slice_chunks(path, skip_requests: int, max_requests: int | None):
    """The canonical request slice, exactly as ``multicorpus.run_job`` reads it."""
    return iter_oracle_chunks(path, CHUNK_REQUESTS, merge="auto",
                              max_merge_sectors=DEFAULT_MAX_MERGE_SECTORS,
                              skip_requests=skip_requests, max_requests=max_requests)


def window_features(chunks, window_size: int = WINDOW_SIZE,
                    age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND):
    """Every complete window's contextual vector, in stream order.

    Groups requests into windows exactly as ``replay_stream`` does (across
    chunk boundaries, a trailing partial window dropped) and builds each
    vector through the same ``WindowContext``. Returns
    ``(vectors, requests, age_table_evictions)``.
    """
    context = WindowContext(age_table_bound)
    vectors: list[tuple[float, ...]] = []
    history: list = []
    requests = 0
    for chunk in chunks:
        for request in chunk:
            requests += 1
            history.append(request)
            if len(history) == window_size:
                vectors.append(context.observe(history, dominant_stride(history)))
                history = []
    return vectors, requests, context.evictions


class Replayed:
    """A "classifier" that returns precomputed labels in call order.

    ``replay_stream`` calls ``predict`` once per complete window, in order,
    so feeding it the labels a classifier gave the same windows reproduces
    that classifier's replay. It refuses to run past or short of its labels:
    :meth:`check_consumed` after the replay.
    """

    name = "replayed"

    def __init__(self, labels):
        self.labels = list(labels)
        self.calls = 0
        unknown = set(self.labels) - set(CLASSES)
        if unknown:
            raise ValueError(f"labels outside CLASSES: {sorted(unknown)}")

    def predict(self, values):
        if self.calls >= len(self.labels):
            raise RuntimeError("more windows than precomputed labels")
        label = self.labels[self.calls]
        self.calls += 1
        return label, 1.0

    def check_consumed(self) -> None:
        if self.calls != len(self.labels):
            raise RuntimeError(f"replay used {self.calls} of {len(self.labels)} labels")


def replay_labels(chunks, capacity: int, labels, *, latency: LatencyModel = ASSUMED,
                  contextual: bool = True,
                  age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND):
    """``multicorpus.run_job``'s ``adaptive`` replay with precomputed labels."""
    model = Replayed(labels)
    metrics = replay_stream(chunks, capacity=capacity, window_size=WINDOW_SIZE,
                            mode="adaptive", model=model, latency_model=latency,
                            read_ahead_depth=0,
                            linux_ra_max_blocks=DEFAULT_LINUX_RA_MAX_BLOCKS,
                            contextual=contextual, age_table_bound=age_table_bound)
    model.check_consumed()
    return metrics


def counters(metrics) -> dict:
    out = {name: getattr(metrics, name) for name in COUNTERS}
    out.update({"hit_ratio": metrics.hit_ratio, "device_ios": metrics.device_ios,
                "blocks_moved": metrics.blocks_moved,
                "precision": metrics.prefetch_precision,
                "unused": metrics.unused_prefetches})
    return out


def label_key(labels) -> str:
    """Identity of a label sequence: equal keys give equal replays."""
    index = {label: str(i) for i, label in enumerate(CLASSES)}
    return hashlib.sha256("".join(index[l] for l in labels).encode()).hexdigest()[:16]


def agreement(labels, reference) -> float:
    """Fraction of windows on which two routers chose the same class."""
    if len(labels) != len(reference):
        raise ValueError("label sequences differ in length")
    if not labels:
        return 1.0
    return sum(a == b for a, b in zip(labels, reference)) / len(labels)


def class_shares(labels) -> dict:
    n = len(labels) or 1
    return {label: sum(l == label for l in labels) / n for label in CLASSES}


def committed_row_matches(row: dict, metrics, requests: int) -> list[str]:
    """Fields where a replay differs from a committed ``disk/sweep.jsonl`` row."""
    mine = {"requests": requests, "read_blocks": metrics.read_blocks,
            "hit_ratio": metrics.hit_ratio, "precision": metrics.prefetch_precision,
            "total_prefetches": metrics.prefetches,
            "useful_prefetches": metrics.useful_prefetches,
            "unused": metrics.unused_prefetches,
            "mean_us": metrics.mean_access_latency_us,
            "policy_switches": metrics.policy_switches,
            "demand_ios": metrics.demand_ios, "prefetch_ios": metrics.prefetch_ios,
            "prefetch_ios_mergeable": metrics.prefetch_ios_mergeable,
            "device_ios": metrics.device_ios, "blocks_moved": metrics.blocks_moved}
    return [name for name, value in mine.items() if row.get(name) != value]


__all__ = ["window_features", "Replayed", "replay_labels", "counters", "label_key",
           "agreement", "class_shares", "committed_row_matches", "slice_chunks"]

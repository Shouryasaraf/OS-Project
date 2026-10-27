"""Hindsight-labelled window router over an expanded action space (track HSR).

Stdlib only. The trained models live in :mod:`hindsight_models` (scikit-learn,
imported lazily); the experiment driver is :mod:`hindsight_experiment`.

Why this exists. The shipped ``adaptive`` mode routes each 32-request window
by a synthetic-trained classifier's *workload class*, and every class maps to
a candidate set of at most two blocks (``simulator.candidates``). Even a
perfect classifier is therefore capped near ``sequential`` (D11 measured
about +0.4 points of headroom), and E1 found ``adaptive`` worse than
``sequential``. This module changes the two things that cap it:

* **the action space** -- a window chooses among ``ACTIONS``, which include
  read-ahead depth (``deep@8/32/256``) as well as ``none`` and the two-block
  ``sequential`` policy;
* **the labels** -- instead of synthetic workload classes, each window of a
  *real* trace is labelled with the action that would have scored best on the
  next window, measured by replaying every fixed action over the training
  slice (:func:`record_windows`, :func:`label_windows`).

The labels are **trajectory labels**: window ``k+1``'s outcome under action
``a`` is read from a replay that ran ``a`` on every window, so the cache
state it starts from is the always-``a`` state, not the state the router's own
earlier choices would have produced. That is an approximation of the true
counterfactual (which would need one branch per window and action), and it is
stated wherever the labels are used.

Routing is causal and matches ``adaptive`` exactly: the 12 ``FEATURE_NAMES``
of window ``k`` (``features.WindowContext``) choose the action applied to
window ``k+1``; the first window never prefetches (the replay loop's gate).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .features import DEFAULT_AGE_TABLE_BOUND, WindowContext, dominant_stride
from .guard import read_ahead
from .simulator import (DEFAULT_CAPACITY, LatencyModel, LRUCache, Metrics,
                        _demand_read, _issue_prefetches, candidates)

#: The router's actions, in cost order: a tie between actions goes to the
#: earlier (cheaper) one. ``sequential`` is the shipped two-block policy.
ACTIONS = ("none", "sequential", "deep@8", "deep@32", "deep@256")
#: Per-window objectives a label can maximise (pre-registered):
#: ``hit``  hits in the window (U(c) at c = 0, i.e. hit ratio);
#: ``u0.5`` hits - 0.5 * prefetches issued in the window (U(c) at c = 0.5);
#: ``dev``  minus the window's device I/Os (device-work aware).
OBJECTIVES = ("hit", "u0.5", "dev")
UTIL_C = 0.5
WINDOW_SIZE = 32
#: Counters snapshotted at the end of every window, cumulative.
COUNTERS = ("read_blocks", "read_hits", "prefetches", "useful_prefetches",
            "device_ios", "blocks_moved")


def action_depth(action: str) -> int:
    """Blocks an action prefetches per read request (``sequential`` is 2)."""
    if action == "none":
        return 0
    if action == "sequential":
        return 2
    if action.startswith("deep@"):
        return int(action.split("@", 1)[1])
    raise ValueError(f"unknown hindsight action {action!r}; known: {ACTIONS}")


def action_candidates(request, action: str) -> list[int]:
    """The prefetch candidates ``action`` proposes after ``request``.

    ``sequential`` is exactly ``simulator.candidates(..., "sequential")`` and
    ``deep@N`` exactly ``guard.read_ahead(request, N)``, so a fixed router
    reproduces the ``sequential`` and ``deep`` modes bit for bit.
    """
    if action == "none":
        return []
    if action == "sequential":
        return candidates(request, "sequential", None)
    return read_ahead(request, action_depth(action))


class ScheduledRouter:
    """Follow a precomputed per-window action schedule.

    ``schedule[k]`` is the decision taken when window ``k`` completes; it
    applies to window ``k + 1``. Window 0 runs ``initial`` (it never
    prefetches anyway: the replay loop gates the first window). A schedule is
    either computed from the same causal window features a
    :class:`WindowRouter` would compute in the loop (batch inference; the
    equivalence is pinned by a test), or it is an **oracle** schedule built
    from the replayed slice's own hindsight labels -- a reference ceiling,
    never a policy.
    """

    def __init__(self, schedule, window_size: int = WINDOW_SIZE,
                 initial: str = "none"):
        if window_size < 8:
            raise ValueError("window_size must be at least 8")
        self.schedule = list(schedule)
        for action in set(self.schedule) | {initial}:
            action_depth(action)
        self.window_size = window_size
        self.action = initial
        self.seen = 0
        self.windows = 0
        self.switches = 0
        self.counts = {action: 0 for action in ACTIONS}

    def _decide(self, window_index: int, window) -> str:
        if window_index >= len(self.schedule):
            raise ValueError(f"schedule has {len(self.schedule)} decisions but "
                             f"window {window_index} completed")
        return self.schedule[window_index]

    def _observe(self, request) -> None:
        """Hook for subclasses that need the requests of a window."""

    def next_candidates(self, request) -> list[int]:
        proposed = action_candidates(request, self.action)
        self.counts[self.action] += 1
        self._observe(request)
        self.seen += 1
        if self.seen % self.window_size == 0:
            chosen = self._decide(self.windows, None)
            self.windows += 1
            if chosen != self.action:
                self.switches += 1
            self.action = chosen
        return proposed


class FixedRouter(ScheduledRouter):
    """Always ``action``: a fixed policy replayed through the ``routed`` mode."""

    def __init__(self, action: str, window_size: int = WINDOW_SIZE):
        super().__init__([], window_size, initial=action)
        self.fixed = action

    def _decide(self, window_index: int, window) -> str:
        return self.fixed


class WindowRouter(ScheduledRouter):
    """Choose each window's action in the loop from its 12 features.

    ``decide(features) -> action`` sees the full contextual feature vector of
    the window that just completed (``WindowContext.observe`` with the
    window's ``dominant_stride``, exactly as ``adaptive`` computes it).
    """

    def __init__(self, decide, window_size: int = WINDOW_SIZE,
                 age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND,
                 initial: str = "none"):
        super().__init__([], window_size, initial)
        self.decide = decide
        self.context = WindowContext(age_table_bound)
        self.history: list = []

    def _observe(self, request) -> None:
        self.history.append(request)

    def _decide(self, window_index: int, window) -> str:
        history, self.history = self.history, []
        values = self.context.observe(history, dominant_stride(history))
        chosen = self.decide(values)
        action_depth(chosen)
        return chosen


def contiguity_rule(values) -> str:
    """A fixed, untrained rule: deeper read-ahead for more contiguous windows.

    ``contiguous_ratio`` in [0, 1] picks ``ACTIONS[min(4, int(5 * ratio))]``.
    It exercises every action through the in-loop router path in tests and
    smoke runs; it is not one of the study's policies.
    """
    return ACTIONS[min(len(ACTIONS) - 1, int(len(ACTIONS) * values[0]))]


def feature_router(window_size: int = WINDOW_SIZE) -> "WindowRouter":
    """A fresh :class:`WindowRouter` driven by :func:`contiguity_rule`."""
    return WindowRouter(contiguity_rule, window_size)


def window_features(chunks, window_size: int = WINDOW_SIZE,
                    age_table_bound: int | None = DEFAULT_AGE_TABLE_BOUND) -> list[tuple]:
    """The 12-feature vector of every complete window, in stream order.

    Identical to what a :class:`WindowRouter` computes in the loop: the
    features depend on the request stream only, never on the cache, so they
    are the same under every action.
    """
    context = WindowContext(age_table_bound)
    out: list[tuple] = []
    history: list = []
    for chunk in chunks:
        for request in chunk:
            history.append(request)
            if len(history) == window_size:
                out.append(context.observe(history, dominant_stride(history)))
                history = []
    return out


def record_windows(chunks, action: str, capacity: int = DEFAULT_CAPACITY,
                   window_size: int = WINDOW_SIZE,
                   latency_model: LatencyModel | None = None):
    """Replay one fixed action and snapshot the counters after every window.

    The loop is ``simulator.replay_stream`` for ``mode="routed"`` with a
    :class:`FixedRouter`, built from the same shared helpers
    (``_demand_read``, ``_issue_prefetches``) and the same first-window gate;
    a test pins that its final ``Metrics`` equal the real loop's on every
    counter. Returns ``(metrics, ends)`` where ``ends[name]`` lists the
    cumulative value of each ``COUNTERS`` entry at the end of every complete
    window (a trailing partial window is not recorded).
    """
    action_depth(action)
    latency = latency_model or LatencyModel()
    metrics = Metrics()
    cache = LRUCache(capacity, metrics)
    ends: dict[str, list[int]] = {name: [] for name in COUNTERS}
    seen = 0
    for chunk in chunks:
        for request in chunk:
            tail_missed = _demand_read(cache, metrics, latency, request)
            proposed = action_candidates(request, action)
            if seen >= window_size:
                _issue_prefetches(cache, metrics, latency, None, proposed,
                                  request.lba + request.size_blocks, tail_missed)
            seen += 1
            if seen % window_size == 0:
                for name in COUNTERS:
                    ends[name].append(getattr(metrics, name))
    return metrics, ends


def window_deltas(ends: dict[str, list[int]]) -> dict[str, list[int]]:
    """Per-window counts from cumulative end-of-window snapshots."""
    out = {}
    for name, values in ends.items():
        previous = 0
        deltas = []
        for value in values:
            deltas.append(value - previous)
            previous = value
        out[name] = deltas
    return out


def window_utility(deltas: dict[str, list[int]], objective: str) -> list[float]:
    """The per-window objective (higher is better) from :func:`window_deltas`."""
    if objective == "hit":
        return [float(h) for h in deltas["read_hits"]]
    if objective == "u0.5":
        return [h - UTIL_C * p for h, p in zip(deltas["read_hits"], deltas["prefetches"])]
    if objective == "dev":
        return [-float(d) for d in deltas["device_ios"]]
    raise ValueError(f"unknown objective {objective!r}; known: {OBJECTIVES}")


def outcome_matrix(ends_by_action: dict[str, dict], objective: str) -> list[tuple]:
    """``matrix[w][i]``: objective of ``ACTIONS[i]`` on window ``w``."""
    missing = [a for a in ACTIONS if a not in ends_by_action]
    if missing:
        raise ValueError(f"no window record for actions {missing}")
    columns = [window_utility(window_deltas(ends_by_action[a]), objective)
               for a in ACTIONS]
    lengths = {len(c) for c in columns}
    if len(lengths) != 1:
        raise ValueError(f"window records disagree on the window count: {lengths}")
    return list(zip(*columns))


def best_action_index(values) -> int:
    """Index of the best value; a tie goes to the cheaper (earlier) action."""
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def label_windows(matrix: list[tuple]) -> list[str]:
    """Hindsight label for each decision point.

    The label of window ``k`` (whose features decide window ``k + 1``) is the
    action that scored best on window ``k + 1``. The last complete window has
    no next window and gets no label, so the result has ``len(matrix) - 1``
    entries.
    """
    return [ACTIONS[best_action_index(matrix[k + 1])] for k in range(len(matrix) - 1)]


def oracle_schedule(matrix: list[tuple]) -> list[str]:
    """Hindsight labels of the replayed slice itself, padded to one decision
    per complete window (the last decision targets no recorded window and is
    ``none``). A reference ceiling under the trajectory approximation, never
    a policy."""
    return label_windows(matrix) + ["none"]


def proxy_totals(matrix_rows: list[tuple], chosen: list[int]) -> float:
    """Sum of the objective a per-window choice would score under the
    trajectory approximation. ``chosen[k]`` is the action index applied to
    window ``k + 1``; window 0 counts as ``none``."""
    total = matrix_rows[0][0] if matrix_rows else 0.0
    for k in range(len(matrix_rows) - 1):
        total += matrix_rows[k + 1][chosen[k]]
    return total


def headroom_captured(matrix_rows: list[tuple], chosen: list[int]) -> float | None:
    """Share of the hindsight headroom over ``none`` a choice captures.

    ``(S(chosen) - S(none)) / (S(oracle) - S(none))`` with ``S`` the
    :func:`proxy_totals` sum. ``None`` when the trace has no headroom (every
    action ties with ``none`` on every window).
    """
    none = proxy_totals(matrix_rows, [0] * (len(matrix_rows) - 1))
    oracle = proxy_totals(matrix_rows, [best_action_index(matrix_rows[k + 1])
                                        for k in range(len(matrix_rows) - 1)])
    if oracle - none <= 0:
        return None
    return (proxy_totals(matrix_rows, chosen) - none) / (oracle - none)


def slice_objective(row, objective: str, none_row=None) -> float:
    """One slice's objective score, higher is better.

    ``hit``: hit ratio. ``u0.5``: (hits - 0.5 * prefetches) per read block.
    ``dev``: minus device I/Os relative to ``none`` on the same slice.
    ``row`` is a mapping with ``read_blocks``, ``read_hits`` (or
    ``hit_ratio``), ``total_prefetches`` and ``device_ios``.
    """
    read_blocks = row["read_blocks"]
    hits = row.get("read_hits")
    if hits is None:
        hits = row["hit_ratio"] * read_blocks
    if objective == "hit":
        return hits / read_blocks if read_blocks else 0.0
    if objective == "u0.5":
        return (hits - UTIL_C * row["total_prefetches"]) / read_blocks if read_blocks else 0.0
    if objective == "dev":
        if none_row is None or not none_row["device_ios"]:
            raise ValueError("dev objective needs the none row's device_ios")
        return -row["device_ios"] / none_row["device_ios"]
    raise ValueError(f"unknown objective {objective!r}; known: {OBJECTIVES}")


# --------------------------------------------------------- training records

def training_record(traces, skip_requests: int, max_requests: int, **extra) -> dict:
    """The training range of a router, in ``depth_model.scoring_conflict`` form."""
    return {"traces": [{"trace": t, "train": [skip_requests, skip_requests + max_requests]}
                       for t in traces], **extra}


def scoring_conflict(record: dict | None, trace: str, start: int,
                     stop: int | None) -> str | None:
    """Why requests ``[start, stop)`` of ``trace`` may not score a router.

    A router is refused on any slice that overlaps a slice it was trained or
    selected on (the same rule as ``learned_depth``).
    """
    from .depth_model import scoring_conflict as _conflict
    return _conflict(record, trace, start, stop)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_record(path: str | Path, record: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

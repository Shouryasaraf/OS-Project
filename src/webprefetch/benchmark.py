"""Validation-only policy search followed by a frozen held-out replay.

Policies are selected on validation (split 2) **separately at every cache
capacity** and frozen for the test replay (split 3). The pivot selected only
at capacity 8 and reused that choice at 2-32 (P1); that setting is kept as a
separately labelled ``capacity_transfer`` comparison, never as the headline.

Every comparison is reported three ways, each with a paired session
bootstrap CI: against the best fixed depth chosen for the same capacity,
against no prefetching (``fixed0``), and against the history-only
transition top-1 baseline. Utilities come from
:mod:`prefetch_common.utility`; the unified per-policy file is
``policy_metrics.csv`` and its columns are fixed by
:mod:`prefetch_common.schemas`.

Replays run in parallel (P9, :func:`replay_split`): each split is scored
once into a shared ranking table, stateless policies replay in session
shards, and each feedback-gate policy, whose state is shared across
sessions, replays the whole split in global time order in a task of its
own per capacity. Counts equal the serial replay's exactly. The pivot's
100k-view test prefix replay runs only with ``subset_check=True``.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prefetch_common import schemas
from prefetch_common.stats import CI
from prefetch_common.utility import c_star, row_U, utility_columns

from .cachecore import Cache, Counters, Policy
from .diagnostics import CALIBRATION_COLUMNS, calibration_rows, pair_kinds, think_time
from .model import FastScorer, load_artifacts
from .parallel import replay_parallel
from .scorers import DEFAULT_SCORERS, RANK_ROW, ScorerSpec, build_scorer, rank_table, session_shards
from .simulator import break_even_waste, check_replay, complete_views, finalize, replay, replay_partial

SCENARIO_COSTS = (0.0, 0.01, 0.05, 0.25)
FIXED_DEPTHS = (0, 1, 2, 4, 8)
ADAPTIVE_PENALTIES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
GATE_THRESHOLDS = (0.01, 0.02, 0.05, 0.1)
#: The feedback gate's break-even c and its exploration rate are both
#: selected on validation, like every other family's parameter.
FEEDBACK_COSTS = (0.005, 0.01, 0.02, 0.05, 0.25)
FEEDBACK_EPSILONS = (0.0005, 0.002, 0.005, 0.02)
CAPACITIES = (2, 4, 8, 16, 32)
#: The capacity the pivot selected at; ``capacity_transfer`` reuses it.
TRANSFER_CAPACITY = 8
FAMILIES = ("fixed", "adaptive", "adaptive_ca", "gate", "feedback")
CONTROLLERS = FAMILIES[1:]
SELECTIONS = ("per_capacity", "capacity_transfer")
SPLITS = {"validation": 2, "test": 3}
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 42
TRANSITION_TOP1 = "transition_top1"
#: A capacity no session reaches: the "infinite cache" of the scope lever.
UNBOUNDED_CAPACITY = 2 ** 31 - 1


def cost_key(cost: float) -> str:
    """The key a cost has in every selection map (``"0.0"``, ``"0.01"``)."""
    return str(float(cost))


def policy_grid(feedback: bool = True) -> list[Policy]:
    """Every candidate policy, in a fixed order."""
    grid = ([Policy(f"fixed{k}", "fixed", k) for k in FIXED_DEPTHS]
            + [Policy(f"adaptive_{p:g}", "adaptive", p) for p in ADAPTIVE_PENALTIES]
            + [Policy(f"adaptive_ca_{p:g}", "adaptive_ca", p) for p in ADAPTIVE_PENALTIES]
            + [Policy(f"gate_{t:g}", "gate", t) for t in GATE_THRESHOLDS])
    if feedback:
        grid += [Policy(f"feedback_{c:g}_eps{e:g}", "feedback", c, epsilon=e)
                 for c in FEEDBACK_COSTS for e in FEEDBACK_EPSILONS]
    return grid


def _utility(row: dict, cost: float) -> float:
    return row_U(row, cost)


# ---------------------------------------------------------------- replay

def _session_arrays(records: dict) -> tuple[np.ndarray, ...]:
    """``{session: [hits, issued, views, useful]}`` -> ``(keys, views, hits, issued, useful)``."""
    keys = np.fromiter(sorted(records), dtype=np.int64, count=len(records))
    values = np.asarray([records[int(k)] for k in keys], dtype=np.int32).reshape(-1, 4)
    return keys, values[:, 2], values[:, 0], values[:, 1], values[:, 3]


def _replay_task(task: tuple) -> tuple:
    """One shard of sessions (or, for a feedback policy, every row) at some capacities."""
    keys, positions, caps, session_caps, count_stats = task
    shared = _SHARED_REPLAY
    frame = shared["frame"]
    if keys is not None:
        frame = frame.loc[frame[shared["key"]].isin(keys)]
    policies = [shared["policies"][n] for n in positions]
    stateful = any(p.stateful for p in policies)
    names = [p.name for p in policies] if session_caps else ()
    part = replay_partial(frame, None, shared["table"], policies, caps,
                          order="global" if stateful else shared["order"],
                          cache_scope=shared["cache_scope"], sessions_for=names,
                          count_stats=count_stats)
    arrays = {(cap, name): _session_arrays(part["sessions"][cap][name])
              for cap in session_caps for name in names}
    return part["totals"], part["stats"] if count_stats else None, arrays


_SHARED_REPLAY: dict = {}


def _install_replay(shared: dict) -> None:
    _SHARED_REPLAY.clear()
    _SHARED_REPLAY.update(shared)


def _pool(tasks: list, shared: dict, workers: int) -> list:
    if workers <= 1 or len(tasks) <= 1:
        _install_replay(shared)
        try:
            return [_replay_task(task) for task in tasks]
        finally:
            _SHARED_REPLAY.clear()
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
    context = multiprocessing.get_context(method)
    processes = min(workers, len(tasks), os.cpu_count() or 1)
    with context.Pool(processes, initializer=_install_replay, initargs=(shared,)) as pool:
        return pool.map(_replay_task, tasks, chunksize=1)


class SplitReplay:
    """One split replayed under every policy and capacity.

    ``rows[cap][policy]`` are the aggregate rows, ``extra`` the scorer's
    ranking statistics, ``arrays[(cap, policy)]`` the per-session ``(keys,
    views, hits, issued, useful)`` for the capacities asked for, and
    ``frame``/``table`` the replayed rows and their rankings.
    """

    def __init__(self, rows, extra, arrays, frame, table):
        self.rows, self.extra, self.arrays = rows, extra, arrays
        self.frame, self.table = frame, table


def replay_split(frame: pd.DataFrame, spec: ScorerSpec, policies: list[Policy],
                 capacities: tuple[int, ...], *, workers: int = 1, max_views: int = 0,
                 session_caps: tuple[int, ...] = (), cache_scope: str = "session") -> SplitReplay:
    """Replay ``frame`` with every policy at every capacity, in one pool.

    The rows are scored once (:func:`~webprefetch.scorers.rank_table`) and
    every replay reads that table, which returns exactly the scorer's
    rankings. Stateless policies replay in session shards, as
    :func:`~webprefetch.parallel.replay_parallel` does; each feedback policy
    shares its gate across sessions, so it replays every row in global time
    order, one task per capacity (one gate per policy and capacity, so the
    split is exact). The integer counts equal the serial replay's.
    """
    capacities = tuple(capacities)
    stateless = [n for n, p in enumerate(policies) if not p.stateful]
    stateful = [n for n, p in enumerate(policies) if p.stateful]
    check_replay(frame, spec.index, [policies[n] for n in stateless], spec.order, cache_scope)
    if stateful:
        check_replay(frame, spec.index, [policies[n] for n in stateful], "global", cache_scope)
        if cache_scope != "session":
            raise ValueError("feedback policies are replayed with session-scoped caches only")
    selected = complete_views(frame, max_views).copy()
    selected[RANK_ROW] = np.arange(len(selected))
    table = rank_table(selected, spec, workers)
    key = "visitorid" if cache_scope == "visitor" else "session_id"
    session_caps = tuple(cap for cap in capacities if cap in session_caps)
    # Longest first: each feedback task covers every row.
    tasks = [(None, [n], (cap,), tuple(c for c in session_caps if c == cap), False)
             for n in stateful for cap in capacities]
    if stateless:
        tasks += [(chunk, stateless, capacities, session_caps, True)
                  for chunk in session_shards(selected, workers, key)]
    else:
        first = tasks[0]
        tasks[0] = first[:4] + (True,)
    shared = {"frame": selected, "table": table, "policies": policies, "order": spec.order,
              "cache_scope": cache_scope, "key": key}
    parts = _pool(tasks, shared, workers)
    totals = {cap: {} for cap in capacities}
    stats = None
    pieces: dict = {}
    for part_totals, part_stats, arrays in parts:
        for cap, by_name in part_totals.items():
            for name, counters in by_name.items():
                totals[cap].setdefault(name, Counters()).merge(counters)
        if part_stats is not None:
            if stats is None:
                stats = {k: (v if k != "calibration_bins" else [list(b) for b in v])
                         for k, v in part_stats.items()}
            else:
                for k, v in part_stats.items():
                    if k == "calibration_bins":
                        for mine, theirs in zip(stats[k], v):
                            for i in range(3):
                                mine[i] += theirs[i]
                    else:
                        stats[k] += v
        for label, values in arrays.items():
            pieces.setdefault(label, []).append(values)
    merged = {}
    for label, chunks in pieces.items():
        columns = [np.concatenate([c[i] for c in chunks]) for i in range(5)]
        order = np.argsort(columns[0], kind="mergesort")
        merged[label] = tuple(column[order] for column in columns)
    final = finalize({"totals": totals, "sessions": {cap: {} for cap in capacities},
                      "session_visitors": {}, "stats": stats}, policies, capacities)
    names = [p.name for p in policies]
    rows = {cap: dict(zip(names, final[cap][0])) for cap in capacities}
    extra = {k: v for k, v in final[capacities[0]][1].items() if k != "sessions"}
    return SplitReplay(rows, extra, merged, selected, table)


def replay_policies(frame: pd.DataFrame, spec: ScorerSpec, policies: list[Policy],
                    capacities: tuple[int, ...], *, workers: int = 1,
                    max_views: int = 0) -> tuple[dict, dict]:
    """``({cap: {policy: row}}, extra)`` without per-session records."""
    result = replay_split(frame, spec, policies, capacities, workers=workers, max_views=max_views)
    return result.rows, result.extra


def session_records(arrays: tuple[np.ndarray, ...]) -> dict:
    """Per-session arrays back to ``{session: [hits, issued, views, useful]}``."""
    keys, views, hits, issued, useful = arrays
    return {int(k): [int(h), int(i), int(v), int(u)]
            for k, v, h, i, u in zip(keys, views, hits, issued, useful)}


class PairedSessions:
    """Paired session bootstrap over many replays of one session set.

    Each column (a replay of one policy, keyed by any hashable label such as
    ``(scorer, cap, policy)``) is reduced on arrival to its totals and its
    per-rep resampled sums, so hundreds of comparisons cost one pass and the
    per-session arrays are not kept. Every replay of the same frame selection
    covers the same sessions; a column with another session set raises.

    The draws are exactly :func:`prefetch_common.stats.session_bootstrap`'s
    (numpy backend): ``default_rng(seed)``, one ``integers(0, n, n)`` per
    rep, the ratio ``sum(delta) / sum(views)`` and a percentile interval.
    Hit and issued sums are integers, so a cost-0 interval is reproduced
    bit for bit and other costs agree to float rounding.
    """

    def __init__(self, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED,
                 flush_every: int = 32, batch: int = 50):
        if reps < 1:
            raise ValueError("reps must be positive")
        self.reps, self.seed = reps, seed
        self.flush_every, self.batch = flush_every, batch
        self.keys: np.ndarray | None = None
        self.views: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        self.view_sums: np.ndarray | None = None
        self.totals: dict = {}
        self.sums: dict = {}
        self._pending: dict = {}

    @property
    def sessions(self) -> int:
        return 0 if self.keys is None else len(self.keys)

    def __contains__(self, label) -> bool:
        return label in self.totals

    def _draw(self) -> None:
        n = len(self.keys)
        rng = np.random.default_rng(self.seed)
        draws = [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(self.reps)]
        dtype = np.uint8 if max(int(d.max()) for d in draws) < 256 else np.uint16
        self.counts = np.asarray(draws, dtype=dtype)
        self.view_sums = self._resample(self.views[:, None])[:, 0]

    def _resample(self, matrix: np.ndarray) -> np.ndarray:
        out = np.empty((self.reps, matrix.shape[1]))
        matrix = matrix.astype(np.float64)
        for start in range(0, self.reps, self.batch):
            out[start:start + self.batch] = self.counts[start:start + self.batch].astype(np.float64) @ matrix
        return out

    def add(self, label, arrays: tuple[np.ndarray, ...]) -> None:
        keys, views, hits, issued = arrays[:4]
        if self.keys is None:
            if not len(keys):
                raise ValueError("bootstrap of no sessions")
            self.keys, self.views = keys, views
            self._draw()
        elif not (np.array_equal(keys, self.keys) and np.array_equal(views, self.views)):
            raise ValueError(f"{label} replayed a different session set")
        self.totals[label] = (int(hits.sum()), int(issued.sum()))
        self._pending[label] = (hits, issued)
        if len(self._pending) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        labels = list(self._pending)
        matrix = np.column_stack([column for label in labels for column in self._pending[label]])
        sums = self._resample(matrix)
        for i, label in enumerate(labels):
            self.sums[label] = (sums[:, 2 * i], sums[:, 2 * i + 1])
        self._pending.clear()

    def ci(self, policy, baseline, cost: float, level: float = 0.95) -> CI:
        self._flush()
        (h_a, i_a), (h_b, i_b) = self.totals[policy], self.totals[baseline]
        (sh_a, si_a), (sh_b, si_b) = self.sums[policy], self.sums[baseline]
        draws = ((sh_a - sh_b) - cost * (si_a - si_b)) / self.view_sums
        tail = round((1.0 - level) / 2.0, 12)
        low, high = (float(x) for x in np.quantile(draws, [tail, 1.0 - tail]))
        estimate = ((h_a - h_b) - cost * (i_a - i_b)) / float(self.views.sum())
        return CI(float(estimate), low, high, self.sessions)


# ------------------------------------------------------------- selection

def select_policies(rows_by_cap: dict[int, dict[str, dict]], policies: list[Policy],
                    costs=SCENARIO_COSTS) -> dict[int, dict[str, dict[str, str]]]:
    """``selection[cap][cost_key][family]``: the validation winner per capacity.

    Within each family the highest ``U(cost)`` wins; ties go to fewer issued
    prefetches, then to the name, exactly as the pivot's cap-8 rule did.
    """
    kinds = {p.name: p.kind for p in policies}
    selection: dict[int, dict] = {}
    for cap, rows in rows_by_cap.items():
        selection[cap] = {}
        for cost in costs:
            best = {}
            for family in FAMILIES:
                names = [name for name in rows if kinds.get(name) == family]
                if names:
                    best[family] = max(names, key=lambda n: (_utility(rows[n], cost),
                                                             -rows[n]["issued"], n))
            selection[cap][cost_key(cost)] = best
    return selection


def transfer_selection(selection: dict, capacities=CAPACITIES,
                       source: int = TRANSFER_CAPACITY) -> dict:
    """The pivot's setting: the ``source``-capacity choice at every capacity."""
    return {cap: selection[source] for cap in capacities}


def needed_policies(selection: dict, transfer: dict) -> dict[int, list[str]]:
    """The policies the comparisons at each capacity need per-session records for."""
    out = {}
    for cap in selection:
        names = {"fixed0"}
        for chosen in (selection[cap], transfer[cap]):
            for families in chosen.values():
                names.update(families.values())
        out[cap] = sorted(names)
    return out


def verdict(family: str, vs_best: CI, vs_none: CI) -> str:
    """The report label (P11): never call a controller a winner when it
    loses to not prefetching at all."""
    if vs_best.low > 0:
        text = f"{family} wins"
    elif vs_best.high < 0:
        text = f"{family} loses"
    else:
        text = f"{family} within noise"
    if vs_none.high < 0:
        text += "; loses to no prefetch"
    return text


def comparison_rows(scorer: str, test_rows: dict, transition_rows: dict, selection: dict,
                    boot: PairedSessions, costs=SCENARIO_COSTS) -> list[dict]:
    """``comparisons.csv`` rows for one scorer (schema ``web/comparisons.csv``)."""
    out = []
    transfer = transfer_selection(selection, tuple(selection))
    for kind, chosen in zip(SELECTIONS, (selection, transfer)):
        for cap in sorted(chosen):
            rows = test_rows[cap]
            trans = transition_rows.get(cap)
            for cost in costs:
                families = chosen[cap][cost_key(cost)]
                fixed = families["fixed"]
                for family in CONTROLLERS:
                    if family not in families:
                        continue
                    name = families[family]
                    a, f, z = (scorer, cap, name), (scorer, cap, fixed), (scorer, cap, "fixed0")
                    t = (TRANSITION_TOP1, cap, "fixed1")
                    vs_best = boot.ci(a, f, cost)
                    vs_none = boot.ci(a, z, cost)
                    vs_trans = boot.ci(a, t, cost) if t in boot else None
                    out.append({
                        "split": "test", "selection": kind, "scorer": scorer, "capacity": cap,
                        "cost": cost, "adaptive_policy": name, "best_fixed_policy": fixed,
                        "adaptive_utility": _utility(rows[name], cost),
                        "best_fixed_utility": _utility(rows[fixed], cost),
                        "fixed0_utility": _utility(rows["fixed0"], cost),
                        "transition_top1_utility": _utility(trans, cost) if trans else None,
                        "vs_best_fixed_same_cap": vs_best.estimate,
                        "vs_best_fixed_same_cap_lo": vs_best.low,
                        "vs_best_fixed_same_cap_hi": vs_best.high,
                        "vs_fixed0": vs_none.estimate, "vs_fixed0_lo": vs_none.low,
                        "vs_fixed0_hi": vs_none.high,
                        "vs_transition_top1": vs_trans.estimate if vs_trans else None,
                        "vs_transition_top1_lo": vs_trans.low if vs_trans else None,
                        "vs_transition_top1_hi": vs_trans.high if vs_trans else None,
                        "adaptive_hit_ratio": rows[name]["hit_ratio"],
                        "best_fixed_hit_ratio": rows[fixed]["hit_ratio"],
                        "adaptive_issued_per_view": rows[name]["issued_per_view"],
                        "best_fixed_issued_per_view": rows[fixed]["issued_per_view"],
                        "label": verdict(family, vs_best, vs_none),
                        "reps": boot.reps, "seed": boot.seed})
    return out


def policy_metric_rows(split: str, scorer: str, rows_by_cap: dict,
                       cache_scope: str = "session") -> list[dict]:
    """``policy_metrics.csv`` rows; ``c_star`` is against the same replay's fixed0."""
    out = []
    for cap, rows in rows_by_cap.items():
        none = rows.get("fixed0")
        for name, row in rows.items():
            star = c_star(row, none) if none is not None and name != "fixed0" else None
            out.append({"split": split, "scorer": scorer, "cache_scope": cache_scope,
                        "capacity": cap, "policy": name, "views": row["views"],
                        "hit_ratio": row["hit_ratio"],
                        "prefetch_hits_per_view": row["prefetch_hits_per_view"],
                        "issued_per_view": row["issued_per_view"], "precision": row["precision"],
                        "wasted_per_view": row["wasted_per_view"],
                        "demand_evicted_by_prefetch_per_view":
                            row["demand_evicted_by_prefetch_per_view"],
                        "c_star": star, **utility_columns(row)})
    return out


# ----------------------------------------------------------- legacy views

def _reliability_svg(rows: list[dict]) -> str:
    """Dependency-free candidate-level reliability curve."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 450" role="img" aria-label="Candidate probability reliability curve">',
             '<rect width="620" height="450" fill="#fff"/>',
             '<text x="55" y="28" font-size="19" font-family="sans-serif">Candidate-level reliability</text>',
             '<path d="M65 370 L380 370 M65 370 L65 55" stroke="#333" fill="none" stroke-width="2"/>',
             '<path d="M65 370 L380 55" stroke="#aaa" fill="none" stroke-dasharray="5 5"/>',
             '<text x="145" y="420" font-size="14" font-family="sans-serif">Mean predicted probability</text>',
             '<text transform="translate(20 300) rotate(-90)" font-size="14" font-family="sans-serif">Observed frequency</text>']
    # Plot only bins with observations; markers scale with bin population.
    for split, color in (("validation", "#2463a5"), ("test", "#d06a2c")):
        selected = [row for row in rows if row["split"] == split and row["count"]]
        points = []
        for row in selected:
            x = 65 + 315 * row["mean_probability"]
            y = 370 - 315 * row["observed_frequency"]
            points.append(f"{x:.2f},{y:.2f}")
            radius = min(7, max(2, row["count"] ** .25 / 4))
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}"/>')
        if points:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        skill = selected[0]["brier_skill_score"] if selected else None
        ece = selected[0]["ece"] if selected else 0
        y = 80 if split == "validation" else 105
        label = f"{split} BSS {skill:.4f}, ECE {ece:.5f}" if skill is not None else f"{split} ECE {ece:.5f}"
        parts.append(f'<text x="400" y="{y}" fill="{color}" font-size="13" font-family="sans-serif">{label}</text>')
    parts += ['<text x="62" y="390" font-size="12">0</text>',
              '<text x="375" y="390" font-size="12">1</text>',
              '<text x="48" y="60" font-size="12">1</text>', '</svg>']
    return "\n".join(parts)


def _real_session_demo(test: pd.DataFrame, index, model, policy: Policy,
                       capacity: int = TRANSFER_CAPACITY) -> dict:
    """Show the first held-out session with at least five views, without picking a success.

    ``test`` must already hold complete sessions only (P8): a raw prefix can
    cut the chosen session short.
    """
    lengths = test.groupby("session_id", sort=False).size()
    eligible_ids = lengths.index[lengths.ge(5)]
    if len(eligible_ids) == 0:
        return {"selection_rule": "first complete test session with at least five views",
                "session_id": None, "policy": policy.name, "capacity": capacity, "steps": [],
                "observed_hits": 0, "prefetches_issued": 0,
                "note": "the replayed test prefix holds no session with five views"}
    session_id = int(eligible_ids[0])
    rows = test.loc[test.session_id.eq(session_id)]
    scorer = FastScorer(model)
    cache = Cache(capacity, Counters())
    steps = []
    for row in rows.itertuples(index=False):
        hit = cache.demand(int(row.itemid))
        ranked = scorer.score(row, index)
        depth = policy.depth(ranked)
        prior_issued = cache.counters.issued
        chosen = [item for item, _ in ranked[:depth]]
        for item in chosen:
            cache.prefetch(item)
        steps.append({
            "timestamp": int(row.timestamp), "viewed_item": int(row.itemid),
            "cache_hit": hit, "selected_depth": depth,
            "issued_now": cache.counters.issued - prior_issued,
            "prefetched_candidates": chosen,
            "top_scored_candidates": [[item, round(prob, 4)] for item, prob in ranked[:5]],
            "actual_next_item_after_decision": int(row.next_item) if bool(row.eligible) else None,
        })
    return {"selection_rule": "first complete test session with at least five views",
            "session_id": session_id, "policy": policy.name, "capacity": capacity,
            "steps": steps, "observed_hits": cache.counters.hits,
            "prefetches_issued": cache.counters.issued}


def _legacy_comparisons(scorer: str, rows: dict, selection: dict, boot: PairedSessions,
                        coverage: float, kind: str) -> list[dict]:
    """The pivot's comparisons.csv layout (the demo and report read it)."""
    out = []
    for cap in sorted(selection):
        by_name = rows[cap]
        for cost in SCENARIO_COSTS:
            names = selection[cap][cost_key(cost)]
            fixed, adaptive, gate = by_name[names["fixed"]], by_name[names["adaptive"]], by_name[names["gate"]]
            vs_best = boot.ci((scorer, cap, names["adaptive"]), (scorer, cap, names["fixed"]), cost)
            vs_none = boot.ci((scorer, cap, names["adaptive"]), (scorer, cap, "fixed0"), cost)
            out.append({
                "selection": kind, "capacity": cap, "cost_per_prefetch": cost,
                "fixed_policy": fixed["policy"], "adaptive_policy": adaptive["policy"],
                "gate_policy": gate["policy"],
                "fixed_utility": _utility(fixed, cost), "adaptive_utility": _utility(adaptive, cost),
                "gate_utility": _utility(gate, cost),
                "adaptive_minus_fixed_utility": vs_best.estimate,
                "paired_95pct_low": vs_best.low, "paired_95pct_high": vs_best.high,
                "fixed_hit_ratio": fixed["hit_ratio"], "adaptive_hit_ratio": adaptive["hit_ratio"],
                "no_prefetch_hit_ratio": by_name["fixed0"]["hit_ratio"],
                "adaptive_minus_no_prefetch_utility": vs_none.estimate,
                "no_prefetch_95pct_low": vs_none.low, "no_prefetch_95pct_high": vs_none.high,
                "fixed_issued_per_view": fixed["issued_per_view"],
                "adaptive_issued_per_view": adaptive["issued_per_view"],
                "candidate_coverage_at_8": coverage,
                "label": verdict("adaptive", vs_best, vs_none)})
    return out


def _cohorts(test: pd.DataFrame, index) -> list[dict]:
    counts = {}
    for row in test.itertuples(index=False):
        if not bool(row.eligible):
            continue
        month = pd.Timestamp(int(row.timestamp), unit="ms", tz="UTC").strftime("%Y-%m")
        source_status = "seen_source" if int(row.itemid) in index.global_count else "cold_source"
        target_status = "seen_target" if int(row.next_item) in index.global_count else "cold_target"
        covered = int(any(item == row.next_item for item, _ in index.candidates(int(row.itemid))[:8]))
        for dimension, cohort in (("month", month), ("source", source_status), ("target", target_status)):
            record = counts.setdefault((dimension, cohort), [0, 0])
            record[0] += 1
            record[1] += covered
    return [{"dimension": d, "cohort": c, "eligible_pairs": v[0],
             "candidate_coverage_at_8": v[1] / v[0]} for (d, c), v in sorted(counts.items())]


def _strip(rows) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "depth_counts"} for row in rows]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(columns) if columns else None).to_csv(path, index=False)


# -------------------------------------------------------------- benchmark

def run_benchmark(prepared: str | Path, model_dir: str | Path,
                  output_dir: str | Path, validation_views: int = 0,
                  test_views: int = 0, *, results_dir: str | Path | None = None,
                  scorers=DEFAULT_SCORERS, workers: int = 1, feedback: bool = True,
                  subset_check: bool = False, reps: int = BOOTSTRAP_REPS,
                  seq_path: str | Path | None = None, log=print) -> dict:
    """Select on validation per capacity, replay test frozen, write every table.

    ``output_dir`` receives the pivot-layout files the demo and report read;
    ``results_dir`` (default ``output_dir/results``) receives the schema
    files ``policy_metrics.csv`` and ``comparisons.csv``, plus
    ``calibration.csv``, ``dataset_profile.json`` and ``think_time.json``.
    Zero view limits mean the full split; a limit keeps complete sessions only.
    """
    started = time.monotonic()
    if validation_views < 0 or test_views < 0:
        raise ValueError("view limits must be non-negative; zero means full split")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir) if results_dir is not None else output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    views = pd.read_csv(prepared)
    artifacts = load_artifacts(model_dir)
    index, model = artifacts
    frames = {label: views.loc[views.split.eq(split)] for label, split in SPLITS.items()}
    limits = {"validation": validation_views, "test": test_views}
    if len(frames["validation"]) < validation_views or len(frames["test"]) < test_views:
        raise ValueError("requested replay size exceeds a chronological split")
    selected = {label: complete_views(frames[label], limits[label]) for label in SPLITS}
    grid = policy_grid(feedback)
    by_name = {p.name: p for p in grid}

    statuses, specs = {}, {}
    rows = {label: {} for label in SPLITS}
    extras = {label: {} for label in SPLITS}
    selection, calibration = {}, []
    validation_arrays = {}
    boot = PairedSessions(reps)
    for name in scorers:
        specs_here = {label: build_scorer(name, views, split, model_dir, artifacts=artifacts,
                                          seq_path=seq_path) for label, split in SPLITS.items()}
        failed = next((spec for spec in specs_here.values() if not spec.ok), None)
        spec = failed or specs_here["test"]
        statuses[name] = {"status": spec.status, "order": spec.order, "train_range": spec.train_range}
        if failed is not None:
            log(f"[benchmark] {name}: {failed.status}")
            continue
        for label in SPLITS:
            specs[(name, label)] = specs_here[label]
            log(f"[benchmark] {name} {label}: {len(grid)} policies x {len(CAPACITIES)} capacities")
            # Validation keeps the cap-8 session records (break-even, pivot
            # layout); test keeps them at every capacity for the comparisons.
            result = replay_split(frames[label], specs_here[label], grid, CAPACITIES,
                                  workers=workers, max_views=limits[label],
                                  session_caps=CAPACITIES if label == "test" else (TRANSFER_CAPACITY,))
            rows[label][name], extras[label][name] = result.rows, result.extra
            calibration += calibration_rows(result.table, result.frame, name, label)
            if label == "validation":
                selection[name] = select_policies(result.rows, grid)
                cap8 = selection[name][TRANSFER_CAPACITY]
                keep = {cap8[cost_key(c)][f] for c in SCENARIO_COSTS for f in ("fixed", "adaptive")}
                validation_arrays[name] = {n: result.arrays[(TRANSFER_CAPACITY, n)] for n in keep}
            else:
                needed = needed_policies(selection[name], transfer_selection(selection[name]))
                for cap, names in needed.items():
                    for policy in names:
                        boot.add((name, cap, policy), result.arrays[(cap, policy)])
            del result
    ok = [name for name in scorers if statuses[name]["status"] == "ok"]
    if not ok:
        raise ValueError(f"no scorer could run: {statuses}")
    primary = "logistic" if "logistic" in ok else ok[0]

    log("[benchmark] transition top-1 baseline")
    transition_spec = (specs.get(("trans_s0", "test"))
                       or build_scorer("trans_s0", views, 3, model_dir, artifacts=artifacts))
    transition = replay_split(frames["test"], transition_spec, [by_name["fixed0"], by_name["fixed1"]],
                              CAPACITIES, workers=workers, max_views=test_views,
                              session_caps=CAPACITIES)
    transition_rows = transition.rows
    for cap in CAPACITIES:
        boot.add((TRANSITION_TOP1, cap, "fixed1"), transition.arrays[(cap, "fixed1")])
    del transition
    log(f"[benchmark] bootstrap: {reps} reps over {boot.sessions} test sessions")
    comparisons = []
    for name in ok:
        comparisons += comparison_rows(name, rows["test"][name],
                                       {cap: r["fixed1"] for cap, r in transition_rows.items()},
                                       selection[name], boot)

    metric_rows = []
    for name in ok:
        for label in SPLITS:
            metric_rows += policy_metric_rows(label, name, rows[label][name])
    if "trans_s0" not in ok:
        metric_rows += policy_metric_rows("test", "trans_s0", transition_rows)
    schemas.write_rows("web/policy_metrics.csv", metric_rows, results_dir / "policy_metrics.csv")
    schemas.write_rows("web/comparisons.csv", comparisons, results_dir / "comparisons.csv")
    _write_csv(results_dir / "calibration.csv", calibration, CALIBRATION_COLUMNS)
    profile_path = Path(prepared).parent / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    _write_json(results_dir / "dataset_profile.json", {
        **profile, "replayed_views": {label: int(len(selected[label])) for label in SPLITS},
        "pair_kinds": {label: pair_kinds(selected[label], SPLITS[label]) for label in SPLITS}})
    _write_json(results_dir / "think_time.json",
                {label: think_time(selected[label], SPLITS[label]) for label in SPLITS})

    # The pivot-layout files, for the primary scorer.
    primary_rows = {label: rows[label][primary] for label in SPLITS}
    primary_selection = selection[primary]
    coverage = extras["test"][primary]["oracle_candidate_coverage_at_8"]
    legacy = _legacy_comparisons(primary, primary_rows["test"], primary_selection, boot,
                                 coverage, "per_capacity")
    transfer_rows = _legacy_comparisons(primary, primary_rows["test"],
                                        transfer_selection(primary_selection), boot,
                                        coverage, "capacity_transfer")
    _write_csv(output_dir / "validation_metrics.csv",
               _strip(r for cap in CAPACITIES for r in primary_rows["validation"][cap].values()))
    _write_csv(output_dir / "test_metrics.csv",
               _strip(r for cap in CAPACITIES for r in primary_rows["test"][cap].values()))
    _write_csv(output_dir / "comparisons.csv", legacy)
    _write_csv(output_dir / "capacity_transfer.csv", transfer_rows)
    _write_csv(output_dir / "calibration.csv", calibration, CALIBRATION_COLUMNS)
    _write_csv(output_dir / "transition_baseline.csv",
               _strip([dict(transition_rows[TRANSFER_CAPACITY]["fixed1"], policy=TRANSITION_TOP1)]))
    primary_calibration = [row for row in calibration if row["scorer"] == primary]
    (output_dir / "reliability.svg").write_text(_reliability_svg(primary_calibration), encoding="utf-8")

    # Break-even waste on validation at the transfer capacity (pivot layout).
    cap8 = primary_selection[TRANSFER_CAPACITY]
    sessions = {n: session_records(a) for n, a in validation_arrays[primary].items()}
    validation_replayed = extras["validation"][primary]["replayed_views"]
    break_even, curve = [], []
    for cost in SCENARIO_COSTS:
        names = cap8[cost_key(cost)]
        break_even.append({"scenario_cost": cost, "adaptive_policy": names["adaptive"],
                           "fixed_policy": names["fixed"],
                           **break_even_waste(sessions, names["adaptive"], names["fixed"])})
        a, b = sessions[names["adaptive"]], sessions[names["fixed"]]
        dh = sum(a[k][0] - b[k][0] for k in a)
        dw = sum((a[k][1] - a[k][3]) - (b[k][1] - b[k][3]) for k in a)
        for point in range(0, 301, 5):
            curve.append({"scenario_cost": cost, "waste_cost": point / 1000,
                          "validation_utility_delta_per_view":
                              (dh - (point / 1000) * dw) / validation_replayed})
    del validation_arrays, sessions
    _write_csv(output_dir / "break_even.csv", break_even)
    _write_csv(output_dir / "break_even_curve.csv", curve)

    subset = []
    if subset_check:
        # The pivot's historical 100k-view prefix, a diagnostic only (P9).
        budget = min(100000, test_views) if test_views else 100000
        names = sorted({cap8[cost_key(c)][f] for c in SCENARIO_COSTS for f in ("fixed", "adaptive")})
        prefix = replay_parallel(frames["test"], index, FastScorer(model), [by_name[n] for n in names],
                                 (TRANSFER_CAPACITY,), workers, max_views=budget,
                                 with_sessions=False)[TRANSFER_CAPACITY]
        prefix_rows = {r["policy"]: r for r in prefix[0]}
        for cost in SCENARIO_COSTS:
            chosen = cap8[cost_key(cost)]
            full = next(r for r in legacy if r["capacity"] == TRANSFER_CAPACITY
                        and r["cost_per_prefetch"] == cost)
            subset.append({"cost": cost, "subset_views": prefix[1]["replayed_views"],
                           "full_views": extras["test"][primary]["replayed_views"],
                           "subset_delta": _utility(prefix_rows[chosen["adaptive"]], cost)
                           - _utility(prefix_rows[chosen["fixed"]], cost),
                           "full_delta": full["adaptive_minus_fixed_utility"],
                           "subset_coverage": prefix[1]["oracle_candidate_coverage_at_8"],
                           "full_coverage": coverage})
        _write_csv(output_dir / "subset_vs_full.csv", subset)

    _write_csv(output_dir / "cohorts.csv", _cohorts(selected["test"], index))
    from .diagnostics import evaluate_models
    model_diagnostics = evaluate_models(views, index, model, model_dir, output_dir)
    demo_policy = by_name[primary_selection[TRANSFER_CAPACITY][cost_key(0.01)]["adaptive"]]
    demo = _real_session_demo(selected["test"], index, model, demo_policy)
    _write_json(output_dir / "demo.json", demo)

    calibration_summary = {}
    for row in calibration:
        calibration_summary.setdefault(row["scorer"], {})[row["split"]] = {
            key: row[key] for key in ("brier", "brier_reference", "brier_skill_score", "ece")}
    serial = {name: {str(cap): sel for cap, sel in selection[name].items()} for name in ok}
    summary = {
        "validation_views": extras["validation"][primary]["replayed_views"],
        "test_views": extras["test"][primary]["replayed_views"],
        "partial_run": bool(validation_views or test_views),
        "limits": {"validation_views": validation_views, "test_views": test_views},
        "selection_rule": "per capacity on validation (P1); capacity_transfer = the cap-8 choice reused",
        "validation_capacity": "per capacity", "test_capacities": list(CAPACITIES),
        "cost_scenarios": list(SCENARIO_COSTS),
        "cost_note": "Object cost is an assumed sensitivity parameter, not observed bytes or latency.",
        "primary_scorer": primary, "scorers": statuses,
        "validation_candidate_coverage_at_8": extras["validation"][primary]["oracle_candidate_coverage_at_8"],
        "validation_ranking_hit_at_1": extras["validation"][primary]["ranking_hit_at_1"],
        "validation_ranking_mrr_at_8": extras["validation"][primary]["ranking_mrr_at_8"],
        "test_candidate_coverage_at_8": {str(cap): coverage for cap in CAPACITIES},
        "ranking": {label: {name: {k: v for k, v in extras[label][name].items()
                                   if k not in ("calibration_bins", "session_visitors")}
                            for name in ok} for label in SPLITS},
        "validation_ece": calibration_summary[primary]["validation"]["ece"],
        "test_ece": calibration_summary[primary]["test"]["ece"],
        "calibration": calibration_summary,
        "calibration_rows": [row for row in primary_calibration if row["reported"]],
        "subset_vs_full": subset, "break_even_waste": break_even,
        "transition_top1": _strip([transition_rows[TRANSFER_CAPACITY]["fixed1"]])[0],
        "model_diagnostics": model_diagnostics,
        # The cap-8 selection under its pivot name, so older readers still work.
        "validation_selection": serial[primary][str(TRANSFER_CAPACITY)],
        "validation_selection_per_capacity": serial,
        "comparisons": legacy, "capacity_transfer": transfer_rows,
        "bootstrap": {"reps": reps, "seed": BOOTSTRAP_SEED, "sessions": boot.sessions},
        "workers": workers,
        "wall_s": round(time.monotonic() - started, 1),
    }
    _write_json(output_dir / "benchmark.json", summary)
    log(f"[benchmark] done in {summary['wall_s']} s")
    return summary


def run_sensitivity(raw: str | Path, processed: str | Path,
                    model_dir: str | Path, output_dir: str | Path) -> None:
    """Isolate segmentation sensitivity with a frozen trained model/policies.

    Uses the historical view budgets for this expensive supplemental analysis;
    full-period results remain in benchmark.json and comparisons.csv.
    """
    from .data import prepare
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "benchmark.json").read_text())
    index, model = load_artifacts(model_dir)
    rows = []
    for gap in (15, 30, 60):
        if gap == 30:
            views = pd.read_csv(Path(processed) / "views.csv.gz")
        else:
            directory = Path(processed) / f"gap{gap}"
            prepare(raw, directory, gap)
            views = pd.read_csv(directory / "views.csv.gz")
        test = views.loc[views.split.eq(3)]
        names = summary["validation_selection"]["0.01"]
        policies = [Policy(names["fixed"], "fixed", int(names["fixed"].replace("fixed", ""))),
                    Policy(names["adaptive"], "adaptive", float(names["adaptive"].split("_")[1]))]
        result, extra, _ = replay(test, index, model, policies, 8, 100000)
        mapping = {row["policy"]: row for row in result}
        fixed, adaptive = mapping[names["fixed"]], mapping[names["adaptive"]]
        rows.append({"session_gap_minutes":gap,"frozen_from_gap_minutes":30,
                     "approximate_view_budget":100000,"replayed_views":extra["replayed_views"],
                     "eligible_pairs":extra["oracle_eligible_pairs"],
                     "candidate_coverage_at_8":extra["oracle_candidate_coverage_at_8"],
                     "fixed_hit_ratio":fixed["hit_ratio"],"adaptive_hit_ratio":adaptive["hit_ratio"],
                     "adaptive_minus_fixed_utility_at_0.01":_utility(adaptive,.01)-_utility(fixed,.01)})
    pd.DataFrame(rows).to_csv(output_dir / "session_gap_sensitivity.csv", index=False)


# ------------------------------------------------------ further analyses

def run_freshness(prepared: str | Path, results_dir: str | Path) -> list[dict]:
    """``freshness.csv``: coverage@k by index freshness, validation and test."""
    from .diagnostics import freshness_rows
    views = pd.read_csv(prepared)
    rows = [row for label, split in SPLITS.items() for row in freshness_rows(views, split, label)]
    schemas.write_rows("web/freshness.csv", rows, Path(results_dir) / "freshness.csv")
    return rows


#: The cost the controller-type levers are compared at (the pivot's headline).
DECOMPOSITION_COST = 0.01
#: Every lever is reported as a hit-ratio delta and as a utility delta.
DECOMPOSITION_METRICS = (("hit_ratio", 0.0), ("utility_per_view", DECOMPOSITION_COST))


def decomposition_levers(selection: dict, available: set[str]) -> list[dict]:
    """What each "what moves the needle" row compares.

    ``selection`` is the benchmark's ``validation_selection_per_capacity``;
    every controller is the one validation chose for that capacity and cost,
    never a test-picked one. Levers whose scorer is unavailable are left out.
    """
    logistic = selection["logistic"]
    key = cost_key(DECOMPOSITION_COST)
    levers = [
        {"lever": "ranker", "scorer": "trans_s0", "policy": "fixed1", "baseline_scorer": "logistic",
         "baseline": "fixed1", "capacity": 8,
         "note": "history-only transition top-1 minus the trained logistic ranker, both depth 1"},
        {"lever": "freshness", "scorer": "trans_online", "policy": "fixed4",
         "baseline_scorer": "logistic", "baseline": "fixed4", "capacity": 8,
         "note": "causal online transition counts minus the split-0 logistic ranker, depth 4"},
    ]
    deep = logistic["8"][cost_key(0.0)]["fixed"]
    levers.append({"lever": "depth", "scorer": "logistic", "policy": deep, "baseline_scorer": "logistic",
                   "baseline": "fixed1", "capacity": 8,
                   "note": f"validation's best fixed depth at cost 0 ({deep}) minus depth 1"})
    for cap in (8, 32):
        chosen = logistic[str(cap)][key]
        for family in ("adaptive", "adaptive_ca"):
            if family in chosen:
                levers.append({"lever": "controller", "scorer": "logistic", "policy": chosen[family],
                               "baseline_scorer": "logistic", "baseline": chosen["fixed"],
                               "capacity": cap,
                               "note": f"validated {family} minus the validated best fixed depth "
                                       f"at this capacity, cost {DECOMPOSITION_COST:g}"})
    chosen = logistic["8"][key]
    if "feedback" in chosen:
        levers.append({"lever": "feedback", "scorer": "logistic", "policy": chosen["feedback"],
                       "baseline_scorer": "logistic", "baseline": chosen["adaptive"], "capacity": 8,
                       "note": f"validated feedback gate minus validated adaptive, cost {DECOMPOSITION_COST:g}"})
    levers.append({"lever": "cache_scope", "scorer": "logistic", "policy": "fixed0",
                   "baseline_scorer": "logistic", "baseline": "fixed0", "capacity": None,
                   "cache_scope": "visitor", "baseline_scope": "session",
                   "note": "no prefetching, infinite cache: one cache per visitor minus one per session"})
    levers.append({"lever": "sequence_model", "scorer": "seq", "policy": "fixed4",
                   "baseline_scorer": "trans_online", "baseline": "fixed4", "capacity": 8,
                   "note": "GPU sequence model minus causal online transition counts, depth 4"})
    return [lever for lever in levers
            if lever["scorer"] in available and lever["baseline_scorer"] in available]


def run_decomposition(prepared: str | Path, model_dir: str | Path, output_dir: str | Path,
                      results_dir: str | Path, *, test_views: int | None = None,
                      workers: int = 1, reps: int = BOOTSTRAP_REPS,
                      seq_path: str | Path | None = None, log=print) -> list[dict]:
    """``decomposition.csv`` (T5): each lever as a paired session-bootstrap delta on test.

    Reads the validation selection from ``output_dir/benchmark.json`` and
    replays the same test sessions the benchmark did (its view limit, unless
    ``test_views`` overrides it).
    """
    summary = json.loads((Path(output_dir) / "benchmark.json").read_text(encoding="utf-8"))
    selection = summary["validation_selection_per_capacity"]
    if "logistic" not in selection:
        raise ValueError("the decomposition needs a benchmark run that includes the logistic scorer")
    if test_views is None:
        test_views = summary.get("limits", {}).get("test_views", 0)
    views = pd.read_csv(prepared)
    artifacts = load_artifacts(model_dir)
    test = views.loc[views.split.eq(3)]
    specs, skipped = {}, {}
    for name in ("logistic", "trans_s0", "trans_online", "seq"):
        spec = build_scorer(name, views, 3, model_dir, artifacts=artifacts, seq_path=seq_path)
        (specs if spec.ok else skipped)[name] = spec if spec.ok else spec.status
    for name, status in skipped.items():
        log(f"[decompose] {name}: {status}; its lever is left out")
    levers = decomposition_levers(selection, set(specs))
    by_name = {p.name: p for p in policy_grid(True)}
    plans: dict[tuple, dict] = {}
    for lever in levers:
        cap = lever["capacity"] if lever["capacity"] is not None else UNBOUNDED_CAPACITY
        for scorer, name, scope in ((lever["scorer"], lever["policy"], lever.get("cache_scope", "session")),
                                    (lever["baseline_scorer"], lever["baseline"],
                                     lever.get("baseline_scope", "session"))):
            plans.setdefault((scorer, scope), {}).setdefault(cap, {})[name] = by_name[name]
    paired = PairedSessions(reps)
    for (scorer, scope), plan in plans.items():
        log(f"[decompose] replaying {scorer} ({scope} caches)")
        caps = tuple(sorted(plan))
        policies = list({p.name: p for cap in caps for p in plan[cap].values()}.values())
        result = replay_split(test, specs[scorer], policies, caps, workers=workers,
                              max_views=test_views, session_caps=caps, cache_scope=scope)
        for cap in caps:
            for name in plan[cap]:
                paired.add((scorer, scope, cap, name), result.arrays[(cap, name)])
        del result
    rows = []
    for lever in levers:
        cap = lever["capacity"] if lever["capacity"] is not None else UNBOUNDED_CAPACITY
        scope = lever.get("cache_scope", "session")
        a = (lever["scorer"], scope, cap, lever["policy"])
        b = (lever["baseline_scorer"], lever.get("baseline_scope", "session"), cap, lever["baseline"])
        # No-prefetch rows issue nothing, so their utility is their hit ratio.
        metrics = DECOMPOSITION_METRICS[:1] if lever["lever"] == "cache_scope" else DECOMPOSITION_METRICS
        for metric, cost in metrics:
            ci = paired.ci(a, b, cost)
            rows.append({"lever": lever["lever"], "policy": lever["policy"], "baseline": lever["baseline"],
                         "scorer": lever["scorer"], "baseline_scorer": lever["baseline_scorer"],
                         "split": "test", "capacity": lever["capacity"], "cache_scope": scope,
                         "metric": metric, "cost": cost, "estimate": ci.estimate,
                         "ci_lo": ci.low, "ci_hi": ci.high, "sessions": ci.n,
                         "reps": paired.reps, "seed": paired.seed, "note": lever["note"]})
    schemas.write_rows("web/decomposition.csv", rows, Path(results_dir) / "decomposition.csv")
    return rows


def run_learning_curve(prepared: str | Path, model_dir: str | Path, results_dir: str | Path, *,
                       decisions=None, validation_views: int = 0, workers: int = 1) -> list[dict]:
    """``learning_curve.csv`` (P13), validation only."""
    from .diagnostics import LEARNING_CURVE_COLUMNS, LEARNING_CURVE_DECISIONS, learning_curve_rows
    views = pd.read_csv(prepared)
    index, _ = load_artifacts(model_dir)
    rows = learning_curve_rows(views, index, decisions or LEARNING_CURVE_DECISIONS,
                               validation_views=validation_views, workers=workers)
    _write_csv(Path(results_dir) / "learning_curve.csv", rows, LEARNING_CURVE_COLUMNS)
    return rows

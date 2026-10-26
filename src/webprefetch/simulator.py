"""One causal request stream, equal cache rules for every prefetch policy."""

from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
import pandas as pd

# Policy, Counters and Cache live in the numpy-free cachecore; they are
# re-exported here so existing imports keep working.
from .cachecore import (DEPTHS, UTILITY_PREFETCH_COST, Cache, Counters,  # noqa: F401
                        FeedbackGate, GateTracker, Policy)
from .data import SESSION_GAP_MS
from .model import CandidateIndex, FastScorer

ORDERS = ("session", "global")
CACHE_SCOPES = ("session", "visitor")


def complete_views(views: pd.DataFrame, max_views: int) -> pd.DataFrame:
    """Return complete sessions within an approximate chronological view budget."""
    if max_views <= 0 or max_views >= len(views):
        return views
    positions = pd.DataFrame({"session_id": views.session_id.to_numpy(),
                              "position": np.arange(len(views))})
    last = positions.groupby("session_id", sort=False).position.max()
    complete = last.index[last.lt(max_views)]
    selected = views.iloc[:max_views]
    return selected.loc[selected.session_id.isin(complete)]


class TransitionScorer:
    """History-only source transition order; no trained ranker."""
    def score(self, row, index: CandidateIndex) -> list[tuple[int, float]]:
        candidates = index.candidates(int(row.itemid))
        support = index.source_total.get(int(row.itemid), 0)
        return [(item, count / support if support else 0.0) for item, count in candidates]


def check_replay(views: pd.DataFrame, index, policies: list[Policy],
                 order: str, cache_scope: str) -> None:
    """The one parameter validator shared by the serial and parallel replays."""
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    if cache_scope not in CACHE_SCOPES:
        raise ValueError(f"cache_scope must be one of {CACHE_SCOPES}")
    if order != "global":
        # Session order visits a later session's early views after an earlier
        # session's late ones, so shared state would see the future.
        stateful = [policy.name for policy in policies if policy.stateful]
        if stateful:
            raise ValueError(f"{stateful} share state across sessions and need order='global'")
        if getattr(index, "online", False):
            raise ValueError("an online index needs order='global'")
    if cache_scope == "visitor" and "visitorid" not in views:
        raise ValueError("cache_scope='visitor' needs a visitorid column")


def _ordered(selected: pd.DataFrame, order: str) -> pd.DataFrame:
    if order == "session":
        # Sessions in first-appearance order, each in its own row order:
        # exactly the iteration order of groupby("session_id", sort=False).
        codes, _ = pd.factorize(selected.session_id)
        return selected.iloc[np.argsort(codes, kind="mergesort")]
    keys = ["timestamp", "source_row"] if "source_row" in selected else ["timestamp"]
    return selected.sort_values(keys, kind="mergesort")


def replay_partial(selected: pd.DataFrame, index, scorer, policies: list[Policy],
                   capacities: tuple[int, ...], *, order: str = "session",
                   cache_scope: str = "session", sessions_for=(), demo_limit: int = 0,
                   session_gap_ms: int = SESSION_GAP_MS, count_stats: bool = True) -> dict:
    """Replay ``selected`` and return raw, additive totals (see :func:`finalize`).

    Every field is a sum or a disjoint per-session map, so totals from shards
    of distinct sessions (or visitors) add up to the serial replay exactly.
    ``count_stats=False`` skips the scorer-level ranking statistics, for a
    shard that re-scores rows another shard already counted.
    """
    frame = _ordered(selected, order)
    if getattr(index, "online", False):
        index = index.fresh()
    advance = getattr(index, "advance", None)
    sessions_for = set(sessions_for)
    totals = {cap: {p.name: Counters() for p in policies} for cap in capacities}
    gates = {(cap, p.name): FeedbackGate.for_policy(p)
             for cap in capacities for p in policies if p.stateful}
    sessions = {cap: {name: {} for name in sessions_for} for cap in capacities}
    session_visitors: dict[int, int] = {}
    keep_visitors = bool(sessions_for) and "visitorid" in frame
    calibration = [[0, 0.0, 0] for _ in range(10)]
    covered = eligible = top1 = 0
    novel_eligible = novel_top1 = novel_covered = 0
    reciprocal = 0.0
    demo: list[dict] = []
    caches: dict[int, list] = {}
    history: dict[int, set] = {}
    expiries: list = []
    sequence = 0
    session_ids = frame.session_id.to_numpy()
    scope_ids = frame.visitorid.to_numpy() if cache_scope == "visitor" else session_ids
    last_session = ~pd.Series(session_ids).duplicated(keep="last").to_numpy()
    last_scope = ~pd.Series(scope_ids).duplicated(keep="last").to_numpy()
    for position, row in enumerate(frame.itertuples(index=False)):
        if order == "global":
            now = int(row.timestamp)
            while expiries and expiries[0][0] <= now:
                _, _, expired, trackers = heapq.heappop(expiries)
                for tracker in trackers:
                    tracker.retire(expired)
            if advance is not None:
                advance(now)
        ranked = scorer.score(row, index)
        session_id = int(session_ids[position])
        item = int(row.itemid)
        if count_stats:
            is_eligible = bool(row.eligible)
            for candidate, probability in ranked:
                bucket = min(9, int(probability * 10))
                calibration[bucket][0] += 1
                calibration[bucket][1] += probability
                calibration[bucket][2] += int(is_eligible and candidate == row.next_item)
            seen = history.setdefault(session_id, set())
            seen.add(item)
            if is_eligible:
                eligible += 1
                actual_rank = next((n for n, (candidate, _) in enumerate(ranked[:8], 1)
                                    if candidate == row.next_item), None)
                if actual_rank is not None:
                    covered += 1
                    top1 += int(actual_rank == 1)
                    reciprocal += 1 / actual_rank
                # Novel: the next item is not among this session's views so
                # far, so a session cache could not already hold it.
                if int(row.next_item) not in seen:
                    novel_eligible += 1
                    if actual_rank is not None:
                        novel_covered += 1
                        novel_top1 += int(actual_rank == 1)
                if len(demo) < demo_limit:
                    demo.append({"timestamp": int(row.timestamp), "session_id": session_id,
                                 "current_item": item, "actual_next_item": int(row.next_item),
                                 "top_candidates": [[c, round(p, 4)] for c, p in ranked[:8]]})
        if keep_visitors:
            session_visitors[session_id] = int(row.visitorid)
        scope = int(scope_ids[position])
        states = caches.get(scope)
        if states is None:
            states = []
            for cap in capacities:
                for policy in policies:
                    tracker = GateTracker(gates[(cap, policy.name)]) if policy.stateful else None
                    cache = Cache(cap, totals[cap][policy.name],
                                  on_evict=tracker.on_evict if tracker is not None else None)
                    states.append((cap, policy, cache, tracker))
            caches[scope] = states
        chosen: dict[str, tuple[int, list]] = {}
        for cap, policy, cache, tracker in states:
            counters = cache.counters
            before = (counters.hits, counters.issued, counters.useful)
            if tracker is not None:
                tracker.demand(cache, item)
                depth = tracker.prefetch(cache, ranked, session_id)
            else:
                cache.demand(item)
                if policy.cache_aware:
                    depth, nominal = policy.choose(ranked, cache.blocks)
                else:
                    # Residency-blind decisions are the same for every capacity.
                    if policy.name not in chosen:
                        chosen[policy.name] = policy.choose(ranked, ())
                    depth, nominal = chosen[policy.name]
                for candidate, _ in nominal:
                    cache.prefetch(candidate)
            counters.depth_counts[depth] += 1
            if policy.name in sessions_for:
                record = sessions[cap][policy.name].setdefault(session_id, [0, 0, 0, 0])
                record[0] += counters.hits - before[0]
                record[1] += counters.issued - before[1]
                record[2] += 1
                record[3] += counters.useful - before[2]
        if last_session[position]:
            history.pop(session_id, None)
            trackers = [state[3] for state in states if state[3] is not None]
            if trackers:
                # Unused prefetches are credited once the session has been
                # idle for the session gap, which is when a live system
                # would know it had ended.
                heapq.heappush(expiries, (int(row.timestamp) + session_gap_ms, sequence,
                                          session_id, trackers))
                sequence += 1
        if last_scope[position]:
            del caches[scope]
    return {"totals": totals, "sessions": sessions, "session_visitors": session_visitors,
            "stats": {"covered": covered, "eligible": eligible, "top1": top1,
                      "reciprocal": reciprocal, "novel_eligible": novel_eligible,
                      "novel_top1": novel_top1, "novel_covered": novel_covered,
                      "replayed_views": len(frame), "calibration_bins": calibration},
            "demo": demo}


def finalize(partial: dict, policies: list[Policy],
             capacities: tuple[int, ...]) -> dict[int, tuple[list[dict], dict]]:
    """Turn raw totals into the per-capacity (rows, extra) pairs callers read."""
    stats = partial["stats"]
    eligible, novel = stats["eligible"], stats["novel_eligible"]
    extra = {"oracle_candidate_coverage_at_8": stats["covered"] / eligible if eligible else 0.0,
             "oracle_eligible_pairs": eligible,
             "ranking_hit_at_1": stats["top1"] / eligible if eligible else 0.0,
             "ranking_mrr_at_8": stats["reciprocal"] / eligible if eligible else 0.0,
             "novel_eligible_pairs": novel,
             "novel_hit_at_1": stats["novel_top1"] / novel if novel else 0.0,
             "novel_coverage_at_8": stats["novel_covered"] / novel if novel else 0.0,
             "replayed_views": stats["replayed_views"],
             "calibration_bins": stats["calibration_bins"]}
    if partial["session_visitors"]:
        extra["session_visitors"] = partial["session_visitors"]
    out = {}
    for cap in capacities:
        sessions = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
        for name, records in partial["sessions"][cap].items():
            sessions[name].update(records)
        rows = [partial["totals"][cap][p.name].row(p.name, cap) for p in policies]
        out[cap] = (rows, {**extra, "sessions": sessions})
    return out


def replay(views: pd.DataFrame, index: CandidateIndex, model,
           policies: list[Policy], capacity: int, max_views: int = 50000,
           with_sessions: bool = False, scorer=None, *, order: str = "session",
           cache_scope: str = "session",
           session_gap_ms: int = SESSION_GAP_MS) -> tuple[list[dict], dict, list[dict]]:
    """Replay complete sessions through one cache size.

    By default each visitor session has its own browser cache (cross-visitor
    sharing would model a proxy, not client-side page prefetching), and
    sessions are replayed one after another. ``cache_scope="visitor"`` keeps
    one cache across a visitor's sessions. ``order="global"`` interleaves
    sessions by timestamp; it is required by the feedback gate and the online
    index, whose state is shared across sessions.
    """
    if max_views < 0:
        raise ValueError("max_views must be non-negative")
    check_replay(views, index, policies, order, cache_scope)
    selected = complete_views(views, max_views)
    scorer = scorer if scorer is not None else FastScorer(model)
    names = [p.name for p in policies] if with_sessions else ()
    partial = replay_partial(selected, index, scorer, policies, (capacity,), order=order,
                             cache_scope=cache_scope, sessions_for=names, demo_limit=8,
                             session_gap_ms=session_gap_ms)
    rows, extra = finalize(partial, policies, (capacity,))[capacity]
    if not with_sessions:
        extra["sessions"] = None
    return rows, extra, partial["demo"]


def replay_capacities(views: pd.DataFrame, index: CandidateIndex, model,
                      policies: list[Policy], capacities: tuple[int, ...],
                      max_views: int = 0, scorer=None, *, order: str = "session",
                      cache_scope: str = "session", with_sessions: bool = True,
                      session_gap_ms: int = SESSION_GAP_MS) -> dict[int, tuple[list[dict], dict]]:
    """Score each request once, applying identical decisions to several caches."""
    check_replay(views, index, policies, order, cache_scope)
    selected = complete_views(views, max_views)
    scorer = scorer if scorer is not None else FastScorer(model)
    names = [p.name for p in policies] if with_sessions else ()
    partial = replay_partial(selected, index, scorer, policies, tuple(capacities), order=order,
                             cache_scope=cache_scope, sessions_for=names,
                             session_gap_ms=session_gap_ms)
    return finalize(partial, policies, tuple(capacities))


def expected_calibration_error(bins: list[list[float]]) -> float:
    total = sum(row[0] for row in bins)
    return (sum(abs(row[1] - row[2]) for row in bins) / total) if total else 0.0


def break_even_waste(sessions: dict, adaptive: str, baseline: str,
                     repetitions: int = 500, seed: int = 42) -> dict:
    """Waste cost at which ``adaptive`` and ``baseline`` tie, with a paired CI.

    ``direction`` reads: ``wins_below`` (adaptive gains hits but wastes more,
    so it wins below the threshold), ``wins_above`` (it loses hits but wastes
    less), ``dominates`` (no fewer hits and less waste: it wins at every
    non-negative cost, including equal waste with more hits), ``dominated``
    (the mirror case) or ``constant`` (equal hits and equal waste). Bootstrap
    draws whose waste difference is 0 have no threshold; ``dropped_draws``
    counts them instead of hiding them.
    """
    left, right = sessions[adaptive], sessions[baseline]
    keys = sorted(set(left) | set(right))
    hits = np.asarray([left[k][0] - right[k][0] for k in keys], dtype=float)
    waste = np.asarray([(left[k][1] - left[k][3]) - (right[k][1] - right[k][3])
                        for k in keys], dtype=float)
    delta_h, delta_w = hits.sum(), waste.sum()
    threshold = float(delta_h / delta_w) if delta_w else None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        ix = rng.integers(0, len(keys), len(keys))
        denominator = waste[ix].sum()
        if denominator:
            draws.append(float(hits[ix].sum() / denominator))
    interval = [float(x) for x in np.quantile(draws, [0.025, 0.975])] if draws else [None, None]
    if delta_w > 0:
        direction = "wins_below" if delta_h > 0 else "dominated"
    elif delta_w < 0:
        direction = "dominates" if delta_h >= 0 else "wins_above"
    elif delta_h:
        direction = "dominates" if delta_h > 0 else "dominated"
    else:
        direction = "constant"
    return {"delta_hits": int(delta_h), "delta_waste": int(delta_w),
            "threshold": threshold, "paired_95pct_low": interval[0],
            "paired_95pct_high": interval[1], "direction": direction,
            "dropped_draws": repetitions - len(draws)}


def paired_bootstrap(sessions: dict, adaptive: str, baseline: str,
                     repetitions: int = 500, seed: int = 42,
                     cost: float = UTILITY_PREFETCH_COST) -> tuple[float, float]:
    left, right = sessions[adaptive], sessions[baseline]
    keys = sorted(set(left) | set(right))
    delta = np.asarray([(left.get(k, [0, 0, 0])[0] - right.get(k, [0, 0, 0])[0])
                        - cost *
                        (left.get(k, [0, 0, 0])[1] - right.get(k, [0, 0, 0])[1])
                        for k in keys], dtype=np.float64)
    denominators = np.asarray([left.get(k, right.get(k))[2] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for i in range(repetitions):
        chosen = rng.integers(0, len(keys), len(keys))
        samples[i] = delta[chosen].sum() / denominators[chosen].sum()
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))

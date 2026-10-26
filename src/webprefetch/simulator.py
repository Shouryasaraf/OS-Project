"""One causal request stream, equal cache rules for every prefetch policy."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model import CandidateIndex, FastScorer

DEPTHS = (0, 1, 2, 4, 8)
UTILITY_PREFETCH_COST = 0.25  # Sensitivity parameter, not a measured network cost.


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    parameter: float = 0.0

    def depth(self, ranked: list[tuple[int, float]]) -> int:
        if self.kind == "fixed":
            return int(self.parameter)
        if not ranked:
            return 0
        if self.kind == "gate":
            return 2 if ranked[0][1] >= self.parameter else 0
        if self.kind == "adaptive":
            probabilities = [prob for _, prob in ranked[:8]]
            prefix = []
            running = 0.0
            for probability in probabilities:
                running += probability
                prefix.append(running)
            values = [0.0] + [min(1.0, prefix[min(k, len(prefix)) - 1])
                              - self.parameter * min(k, len(prefix))
                              for k in DEPTHS[1:]]
            return DEPTHS[max(range(len(values)), key=values.__getitem__)]
        raise ValueError(f"unknown policy kind {self.kind}")


@dataclass
class Counters:
    views: int = 0
    hits: int = 0
    prefetch_hits: int = 0
    issued: int = 0
    useful: int = 0
    evicted_unused: int = 0
    demand_evicted_by_prefetch: int = 0
    depth_counts: Counter = field(default_factory=Counter)

    def row(self, name: str, capacity: int) -> dict:
        return {
            "policy": name, "capacity": capacity, "views": self.views,
            "hits": self.hits, "hit_ratio": self.hits / self.views if self.views else 0.0,
            "prefetch_hits": self.prefetch_hits, "issued": self.issued,
            "useful": self.useful,
            "precision": self.useful / self.issued if self.issued else 0.0,
            "unused": self.issued - self.useful,
            "evicted_unused": self.evicted_unused,
            "demand_evicted_by_prefetch": self.demand_evicted_by_prefetch,
            "issued_per_view": self.issued / self.views if self.views else 0.0,
            "utility_per_view": (self.hits - UTILITY_PREFETCH_COST * self.issued) / self.views
            if self.views else 0.0,
            "depth_counts": dict(self.depth_counts),
        }


class Cache:
    def __init__(self, capacity: int, counters: Counters):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.counters = counters
        self.blocks: OrderedDict[int, bool] = OrderedDict()

    def _insert(self, item: int, prefetched: bool) -> None:
        if len(self.blocks) >= self.capacity:
            # Speculative objects are the first victims for every policy.
            victim = next((key for key, speculative in self.blocks.items() if speculative), None)
            if victim is None:
                victim, was_prefetched = self.blocks.popitem(last=False)
            else:
                was_prefetched = self.blocks.pop(victim)
            if was_prefetched:
                self.counters.evicted_unused += 1
            elif prefetched:
                self.counters.demand_evicted_by_prefetch += 1
        self.blocks[item] = prefetched

    def demand(self, item: int) -> bool:
        self.counters.views += 1
        if item in self.blocks:
            self.counters.hits += 1
            if self.blocks[item]:
                self.counters.useful += 1
                self.counters.prefetch_hits += 1
            self.blocks[item] = False
            self.blocks.move_to_end(item)
            return True
        self._insert(item, False)
        return False

    def prefetch(self, item: int) -> None:
        if item in self.blocks:
            return
        self._insert(item, True)
        self.counters.issued += 1


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


def replay(views: pd.DataFrame, index: CandidateIndex, model,
           policies: list[Policy], capacity: int, max_views: int = 50000,
           with_sessions: bool = False, scorer=None) -> tuple[list[dict], dict, list[dict]]:
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    if max_views < 0:
        raise ValueError("max_views must be non-negative")
    selected = complete_views(views, max_views)
    scorer = scorer if scorer is not None else FastScorer(model)
    totals = {p.name: Counters() for p in policies}
    sessions = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])) if with_sessions else None
    calibration = [[0, 0.0, 0] for _ in range(10)]
    oracle_covered = 0
    oracle_eligible = 0
    oracle_top1 = 0
    oracle_reciprocal_rank = 0.0
    demo: list[dict] = []
    # Each visitor session has its own browser cache. Cross-visitor cache sharing
    # would model a proxy cache, not client-side page prefetching.
    for session_id, session_views in selected.groupby("session_id", sort=False):
        states = {p.name: (Cache(capacity, totals[p.name]), p) for p in policies}
        for row in session_views.itertuples(index=False):
            ranked = scorer.score(row, index)
            for item, probability in ranked:
                bucket = min(9, int(probability * 10))
                calibration[bucket][0] += 1
                calibration[bucket][1] += probability
                calibration[bucket][2] += int(bool(row.eligible) and item == row.next_item)
            if bool(row.eligible):
                oracle_eligible += 1
                actual_rank = next((n for n, (item, _) in enumerate(ranked[:8], 1)
                                    if item == row.next_item), None)
                if actual_rank is not None:
                    oracle_covered += 1
                    oracle_top1 += int(actual_rank == 1)
                    oracle_reciprocal_rank += 1 / actual_rank
                if len(demo) < 8:
                    demo.append({"timestamp": int(row.timestamp), "session_id": int(row.session_id),
                                 "current_item": int(row.itemid), "actual_next_item": int(row.next_item),
                                 "top_candidates": [[item, round(prob, 4)] for item, prob in ranked[:8]]})
            for name, (cache, policy) in states.items():
                prior_hits = cache.counters.hits
                prior_issued = cache.counters.issued
                prior_useful = cache.counters.useful
                cache.demand(int(row.itemid))
                depth = policy.depth(ranked)
                cache.counters.depth_counts[depth] += 1
                for item, _ in ranked[:depth]:
                    cache.prefetch(item)
                if sessions is not None:
                    record = sessions[name][int(session_id)]
                    record[0] += cache.counters.hits - prior_hits
                    record[1] += cache.counters.issued - prior_issued
                    record[2] += 1
                    record[3] += cache.counters.useful - prior_useful
    rows = [totals[p.name].row(p.name, capacity) for p in policies]
    extra = {"oracle_candidate_coverage_at_8": oracle_covered / oracle_eligible
             if oracle_eligible else 0.0, "oracle_eligible_pairs": oracle_eligible,
             "ranking_hit_at_1": oracle_top1 / oracle_eligible if oracle_eligible else 0.0,
             "ranking_mrr_at_8": oracle_reciprocal_rank / oracle_eligible if oracle_eligible else 0.0,
             "sessions": sessions, "replayed_views": len(selected), "calibration_bins": calibration}
    return rows, extra, demo


def replay_capacities(views: pd.DataFrame, index: CandidateIndex, model,
                      policies: list[Policy], capacities: tuple[int, ...],
                      max_views: int = 0) -> dict[int, tuple[list[dict], dict]]:
    """Score each request once, applying identical decisions to several caches."""
    selected = complete_views(views, max_views)
    scorer = FastScorer(model)
    totals = {cap: {p.name: Counters() for p in policies} for cap in capacities}
    sessions = {cap: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])) for cap in capacities}
    covered = eligible = top1 = 0
    reciprocal = 0.0
    calibration = [[0, 0.0, 0] for _ in range(10)]
    for session_id, session_views in selected.groupby("session_id", sort=False):
        states = {cap: {p.name: Cache(cap, totals[cap][p.name]) for p in policies}
                  for cap in capacities}
        for row in session_views.itertuples(index=False):
            ranked = scorer.score(row, index)
            for item, probability in ranked:
                bucket = min(9, int(probability * 10))
                calibration[bucket][0] += 1
                calibration[bucket][1] += probability
                calibration[bucket][2] += int(bool(row.eligible) and item == row.next_item)
            if bool(row.eligible):
                eligible += 1
                actual_rank = next((n for n, (item, _) in enumerate(ranked[:8], 1)
                                    if item == row.next_item), None)
                if actual_rank is not None:
                    covered += 1
                    top1 += int(actual_rank == 1)
                    reciprocal += 1 / actual_rank
            depths = {p.name: p.depth(ranked) for p in policies}
            for cap in capacities:
                for p in policies:
                    cache = states[cap][p.name]
                    before = (cache.counters.hits, cache.counters.issued, cache.counters.useful)
                    cache.demand(int(row.itemid))
                    depth = depths[p.name]
                    cache.counters.depth_counts[depth] += 1
                    for item, _ in ranked[:depth]:
                        cache.prefetch(item)
                    record = sessions[cap][p.name][int(session_id)]
                    record[0] += cache.counters.hits - before[0]
                    record[1] += cache.counters.issued - before[1]
                    record[2] += 1
                    record[3] += cache.counters.useful - before[2]
    extra = {"oracle_candidate_coverage_at_8": covered / eligible if eligible else 0.0,
             "oracle_eligible_pairs": eligible, "ranking_hit_at_1": top1 / eligible if eligible else 0.0,
             "ranking_mrr_at_8": reciprocal / eligible if eligible else 0.0,
             "replayed_views": len(selected), "calibration_bins": calibration}
    return {cap: ([totals[cap][p.name].row(p.name, cap) for p in policies],
                  {**extra, "sessions": sessions[cap]}) for cap in capacities}


def expected_calibration_error(bins: list[list[float]]) -> float:
    total = sum(row[0] for row in bins)
    return (sum(abs(row[1] - row[2]) for row in bins) / total) if total else 0.0


def break_even_waste(sessions: dict, adaptive: str, baseline: str,
                     repetitions: int = 500, seed: int = 42) -> dict:
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
    return {"delta_hits": int(delta_h), "delta_waste": int(delta_w),
            "threshold": threshold, "paired_95pct_low": interval[0],
            "paired_95pct_high": interval[1],
            "direction": "wins_below" if delta_w > 0 else "wins_above" if delta_w < 0 else "constant"}


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

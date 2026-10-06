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
            prefix = np.cumsum([prob for _, prob in ranked[:8]])
            values = [0.0] + [float(min(1.0, prefix[min(k, len(prefix)) - 1])
                                     - self.parameter * k) if prefix.size else -self.parameter * k
                              for k in DEPTHS[1:]]
            return DEPTHS[int(np.argmax(values))]
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


def replay(views: pd.DataFrame, index: CandidateIndex, model,
           policies: list[Policy], capacity: int, max_views: int = 50000,
           with_sessions: bool = False) -> tuple[list[dict], dict, list[dict]]:
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("policy names must be unique")
    if max_views < 0:
        raise ValueError("max_views must be non-negative")
    selected = views if max_views == 0 else views.iloc[:max_views]
    scorer = FastScorer(model)
    totals = {p.name: Counters() for p in policies}
    sessions = defaultdict(lambda: defaultdict(lambda: [0, 0, 0])) if with_sessions else None
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
    rows = [totals[p.name].row(p.name, capacity) for p in policies]
    extra = {"oracle_candidate_coverage_at_8": oracle_covered / oracle_eligible
             if oracle_eligible else 0.0, "oracle_eligible_pairs": oracle_eligible,
             "ranking_hit_at_1": oracle_top1 / oracle_eligible if oracle_eligible else 0.0,
             "ranking_mrr_at_8": oracle_reciprocal_rank / oracle_eligible if oracle_eligible else 0.0,
             "sessions": sessions}
    return rows, extra, demo


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

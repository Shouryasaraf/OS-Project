"""The one browser-cache model: policies, counters and the speculative-first LRU.

Stdlib only, so the replay simulator and any live client can wrap this same
class instead of keeping hand copies that can drift apart. (The demo server
still has its own cache loop; the replay simulator is the only user so far.)
"""

from __future__ import annotations

import random
from bisect import bisect_right
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Callable

DEPTHS = (0, 1, 2, 4, 8)
UTILITY_PREFETCH_COST = 0.25  # Sensitivity parameter, not a measured network cost.
KINDS = ("fixed", "gate", "adaptive", "adaptive_ca", "feedback")
# Score-bin edges of the feedback gate's buckets; the last bin is [0.4, 1].
SCORE_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0)
FEEDBACK_RANKS = 8


@dataclass(frozen=True)
class Policy:
    """A prefetch rule. ``parameter`` is the depth, threshold, penalty or cost.

    ``adaptive`` charges its penalty on every nominal slot, including slots
    whose item is already resident (kept so the committed pivot rows still
    reproduce). ``adaptive_ca`` drops resident items first and then applies
    the same prefix-sum objective, so it is charged only for real fetches.
    ``feedback`` is stateful and is driven by :class:`FeedbackGate`;
    ``epsilon`` and ``pollution`` apply to it alone.
    """
    name: str
    kind: str
    parameter: float = 0.0
    epsilon: float = 0.0
    pollution: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown policy kind {self.kind}")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be within [0, 1]")
        if self.pollution < 0:
            raise ValueError("pollution weight must be non-negative")

    @property
    def cache_aware(self) -> bool:
        return self.kind in ("adaptive_ca", "feedback")

    @property
    def stateful(self) -> bool:
        """True when decisions depend on outcomes from earlier sessions."""
        return self.kind == "feedback"

    def depth(self, ranked: list[tuple[int, float]]) -> int:
        if self.kind == "fixed":
            return int(self.parameter)
        if self.kind == "feedback":
            raise ValueError("feedback decides per candidate; drive it with FeedbackGate")
        if not ranked:
            return 0
        if self.kind == "gate":
            return 2 if ranked[0][1] >= self.parameter else 0
        # adaptive and adaptive_ca share the objective; they differ in what
        # ``ranked`` holds (adaptive_ca receives only non-resident items).
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

    def choose(self, ranked: list[tuple[int, float]], resident) -> tuple[int, list[tuple[int, float]]]:
        """Return (depth, nominal candidates) for a stateless policy.

        ``resident`` is the cache's membership view after the current demand.
        """
        if self.cache_aware:
            ranked = [pair for pair in ranked if pair[0] not in resident]
        depth = self.depth(ranked)
        return depth, ranked[:depth]


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

    def merge(self, other: "Counters") -> "Counters":
        """Add another shard's counts in place (all fields are additive)."""
        for name in ("views", "hits", "prefetch_hits", "issued", "useful",
                     "evicted_unused", "demand_evicted_by_prefetch"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.depth_counts.update(other.depth_counts)
        return self

    def row(self, name: str, capacity: int) -> dict:
        views = self.views
        unused = self.issued - self.useful
        return {
            "policy": name, "capacity": capacity, "views": views,
            "hits": self.hits, "hit_ratio": self.hits / views if views else 0.0,
            "prefetch_hits": self.prefetch_hits, "issued": self.issued,
            "useful": self.useful,
            "precision": self.useful / self.issued if self.issued else 0.0,
            "unused": unused,
            "evicted_unused": self.evicted_unused,
            "demand_evicted_by_prefetch": self.demand_evicted_by_prefetch,
            "issued_per_view": self.issued / views if views else 0.0,
            "utility_per_view": (self.hits - UTILITY_PREFETCH_COST * self.issued) / views
            if views else 0.0,
            "prefetch_hits_per_view": self.prefetch_hits / views if views else 0.0,
            "wasted_per_view": unused / views if views else 0.0,
            "demand_evicted_by_prefetch_per_view":
                self.demand_evicted_by_prefetch / views if views else 0.0,
            # (hit_ratio - hit_ratio of fixed0) / issued_per_view needs the
            # no-prefetch row of the same replay, so the report fills it in.
            "c_star": None,
            "depth_counts": dict(self.depth_counts),
        }


class Cache:
    """Per-scope LRU whose speculative (unused prefetched) objects are evicted first.

    ``on_evict(item, speculative)`` is called after every eviction; it lets a
    feedback gate or a live client observe outcomes without copying this class.
    """

    def __init__(self, capacity: int, counters: Counters,
                 on_evict: Callable[[int, bool], None] | None = None):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.counters = counters
        self.on_evict = on_evict
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
            if self.on_evict is not None:
                self.on_evict(victim, was_prefetched)
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

    def prefetch(self, item: int) -> bool:
        """Fetch ``item`` speculatively; a resident item is a no-op (False)."""
        if item in self.blocks:
            return False
        self._insert(item, True)
        self.counters.issued += 1
        return True


def score_bin(probability: float) -> int:
    return min(len(SCORE_EDGES) - 2, max(0, bisect_right(SCORE_EDGES, probability) - 1))


class FeedbackGate:
    """Issue a candidate only while its bucket's *observed* precision beats the cost.

    Ported from the disk study's precision gate: the decision uses outcomes,
    not the scorer's confidence. A bucket is (rank, score bin). It opens when
    the Beta(1,1) posterior mean of its precision, minus ``pollution`` times
    its smoothed rate of displacing demand objects, exceeds ``cost``. With
    probability ``epsilon`` a gated candidate is issued anyway, so a closed
    bucket keeps gathering evidence. One gate serves every session of a
    (policy, capacity) pair, which is why it needs the global time order.
    """

    def __init__(self, cost: float, epsilon: float = 0.0, pollution: float = 0.0,
                 seed: int = 42, ranks: int = FEEDBACK_RANKS):
        self.cost = cost
        self.epsilon = epsilon
        self.pollution = pollution
        self.ranks = ranks
        self.rng = random.Random(seed)
        self.issued: Counter = Counter()
        self.used: Counter = Counter()
        self.displaced: Counter = Counter()

    @classmethod
    def for_policy(cls, policy: Policy, seed: int = 42) -> "FeedbackGate":
        if policy.kind != "feedback":
            raise ValueError(f"{policy.name} is not a feedback policy")
        return cls(policy.parameter, policy.epsilon, policy.pollution, seed)

    def bucket(self, rank: int, probability: float) -> tuple[int, int]:
        return rank, score_bin(probability)

    def precision(self, bucket) -> float:
        return (self.used[bucket] + 1) / (self.issued[bucket] + 2)

    def is_open(self, bucket) -> bool:
        penalty = self.pollution * self.displaced[bucket] / (self.issued[bucket] + 2)
        return self.precision(bucket) - penalty > self.cost

    def credit(self, bucket, used: bool) -> None:
        self.issued[bucket] += 1
        self.used[bucket] += int(used)

    def explore(self) -> list[bool]:
        # Always draw a full row, so the random stream does not depend on how
        # many candidates a view happens to have.
        if self.epsilon <= 0:
            return [False] * self.ranks
        return [self.rng.random() < self.epsilon for _ in range(self.ranks)]


class GateTracker:
    """Connects one cache to a shared :class:`FeedbackGate`.

    It remembers which bucket issued each speculative object and credits that
    bucket when the outcome is observed: used on a demand hit, unused on
    eviction or when the issuing session expires (:meth:`retire`).
    """

    def __init__(self, gate: FeedbackGate):
        self.gate = gate
        self.tags: dict[int, tuple[tuple[int, int], int]] = {}
        self._issuing = None

    def on_evict(self, item: int, speculative: bool) -> None:
        if speculative:
            tag = self.tags.pop(item, None)
            if tag is not None:
                self.gate.credit(tag[0], False)
        elif self._issuing is not None:
            self.gate.displaced[self._issuing] += 1

    def demand(self, cache: Cache, item: int) -> bool:
        if cache.blocks.get(item):
            tag = self.tags.pop(item, None)
            if tag is not None:
                self.gate.credit(tag[0], True)
        return cache.demand(item)

    def prefetch(self, cache: Cache, ranked: list[tuple[int, float]], session: int) -> int:
        """Issue every open (or explored) non-resident candidate; return the count."""
        explore = self.gate.explore()
        issued = 0
        for rank, (item, probability) in enumerate(ranked[:self.gate.ranks]):
            if item in cache.blocks:
                continue
            bucket = self.gate.bucket(rank, probability)
            if self.gate.is_open(bucket) or explore[rank]:
                self._issuing = bucket
                try:
                    cache.prefetch(item)
                finally:
                    self._issuing = None
                self.tags[item] = (bucket, session)
                issued += 1
        return issued

    def retire(self, session: int) -> None:
        """The session has expired: its still-unused prefetches count as unused."""
        for item in [item for item, (_, owner) in self.tags.items() if owner == session]:
            self.gate.credit(self.tags.pop(item)[0], False)

"""Causal per-request prefetch baselines.

Predictors observe the current request and propose blocks for *later* reads.
The simulator applies the same cache, warm-up, and cost rules to every mode.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Protocol

from .trace import Request


class CandidatePredictor(Protocol):
    def next_candidates(self, request: Request) -> list[int]:
        """Update state from this request and return future block candidates."""


class StridePrefetcher:
    def __init__(self, degree: int = 2, min_repetitions: int = 2):
        if degree < 1 or min_repetitions < 2:
            raise ValueError("degree >= 1 and min_repetitions >= 2 are required")
        self.degree = degree
        self.min_repetitions = min_repetitions
        self.last_lba: int | None = None
        self.last_delta: int | None = None
        self.repetitions = 0

    def next_candidates(self, request: Request) -> list[int]:
        if request.operation != "R":
            return []
        if self.last_lba is None:
            self.last_lba = request.lba
            return []
        delta = request.lba - self.last_lba
        self.repetitions = self.repetitions + 1 if delta == self.last_delta else 1
        self.last_lba = request.lba
        self.last_delta = delta
        if delta <= 0 or self.repetitions < self.min_repetitions:
            return []
        end = request.lba + request.size_blocks
        return [request.lba + k * delta for k in range(1, self.degree + 1)
                if request.lba + k * delta >= end]


class MarkovPrefetcher:
    """First- or second-order Markov transitions between address deltas."""

    def __init__(self, order: int = 1, top_k: int = 2,
                 confidence_threshold: float = 0.3, max_delta: int = 4096):
        if order not in {1, 2} or top_k < 1 or not 0 <= confidence_threshold <= 1:
            raise ValueError("order must be 1 or 2; top_k >= 1; threshold in [0,1]")
        self.order = order
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.max_delta = max_delta
        self.last_lba: int | None = None
        self.deltas: list[int] = []
        self.transitions: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)

    def next_candidates(self, request: Request) -> list[int]:
        if request.operation != "R":
            return []
        if self.last_lba is None:
            self.last_lba = request.lba
            return []
        delta = request.lba - self.last_lba
        self.last_lba = request.lba
        if len(self.deltas) >= self.order and abs(delta) <= self.max_delta:
            self.transitions[tuple(self.deltas[-self.order:])][delta] += 1
        self.deltas.append(delta)
        if len(self.deltas) < self.order:
            return []
        counts = self.transitions.get(tuple(self.deltas[-self.order:]))
        if not counts:
            return []
        total = sum(counts.values())
        predicted = counts.most_common(self.top_k)
        if predicted[0][1] / total < self.confidence_threshold:
            return []
        end = request.lba + request.size_blocks
        return [request.lba + next_delta for next_delta, _ in predicted
                if request.lba + next_delta >= end]

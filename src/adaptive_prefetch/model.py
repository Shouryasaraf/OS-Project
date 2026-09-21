"""Incrementally updated Gaussian Naive Bayes classifier."""

from __future__ import annotations

import math

from .trace import CLASSES


class OnlineGaussianNB:
    """Running means/variances allow one-example-at-a-time updates.

    This is supervised learning: every update requires a trustworthy label.
    """

    def __init__(self, n_features: int, variance_floor: float = 0.0025):
        self.n_features = n_features
        self.variance_floor = variance_floor
        self.count = {label: 0 for label in CLASSES}
        self.mean = {label: [0.0] * n_features for label in CLASSES}
        self.m2 = {label: [0.0] * n_features for label in CLASSES}

    def update(self, values: tuple[float, ...], label: str) -> None:
        if label not in CLASSES or len(values) != self.n_features:
            raise ValueError("unknown class or incorrect feature count")
        self.count[label] += 1
        n = self.count[label]
        for i, value in enumerate(values):
            difference = value - self.mean[label][i]
            self.mean[label][i] += difference / n
            self.m2[label][i] += difference * (value - self.mean[label][i])

    def predict(self, values: tuple[float, ...]) -> tuple[str, float]:
        if len(values) != self.n_features:
            raise ValueError("incorrect feature count")
        total = sum(self.count.values())
        if not total:
            raise ValueError("train the classifier before predicting")
        scores: dict[str, float] = {}
        for label in CLASSES:
            n = self.count[label]
            if not n:
                continue
            score = math.log(n / total)
            for i, value in enumerate(values):
                variance = max(self.m2[label][i] / max(n - 1, 1), self.variance_floor)
                distance = value - self.mean[label][i]
                score += -0.5 * (math.log(2 * math.pi * variance)
                                 + distance * distance / variance)
            scores[label] = score
        winner = max(scores, key=scores.get)
        maximum = scores[winner]
        confidence = 1 / sum(math.exp(score - maximum) for score in scores.values())
        return winner, confidence

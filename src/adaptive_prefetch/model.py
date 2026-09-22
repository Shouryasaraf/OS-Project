from __future__ import annotations

import math
from .trace import CLASSES


class OnlineGaussianNB:
    """Incrementally updated Gaussian Naive Bayes classifier with exponential decay

    Supports concept drift adaptation by decaying historical statistics.
    """

    def __init__(
        self,
        n_features: int,
        variance_floor: float = 0.0025,
        decay_factor: float = 0.995,
    ):
        self.n_features = n_features
        self.variance_floor = variance_floor
        self.decay_factor = decay_factor  # Gamma: values < 1.0 discount old I/O history
        self.count = {label: 0.0 for label in CLASSES}
        self.mean = {label: [0.0] * n_features for label in CLASSES}
        self.m2 = {label: [0.0] * n_features for label in CLASSES}

    def update(self, values: tuple[float, ...], label: str) -> None:
        if label not in CLASSES or len(values) != self.n_features:
            raise ValueError("unknown class or incorrect feature count")

        # Apply exponential decay to all class counts and M2 variances
        for c in CLASSES:
            self.count[c] *= self.decay_factor
            for i in range(self.n_features):
                self.m2[c][i] *= self.decay_factor

        self.count[label] += 1.0
        n = self.count[label]

        # Incremental mean and M2 update with decayed weights
        for i, value in enumerate(values):
            difference = value - self.mean[label][i]
            self.mean[label][i] += difference / n
            self.m2[label][i] += difference * (value - self.mean[label][i])

    def predict(self, values: tuple[float, ...]) -> tuple[str, float]:
        if len(values) != self.n_features:
            raise ValueError("incorrect feature count")

        total = sum(self.count.values())
        if total == 0:
            raise ValueError("train the classifier before predicting")

        scores: dict[str, float] = {}
        for label in CLASSES:
            n = self.count[label]
            if n <= 1e-5:
                continue

            score = math.log(n / total)
            for i, value in enumerate(values):
                # Variance calculation adjusted for effective sample weight n
                variance = max(
                    self.m2[label][i] / max(n - 1.0, 1.0), self.variance_floor
                )
                distance = value - self.mean[label][i]
                score += -0.5 * (
                    math.log(2 * math.pi * variance)
                    + (distance * distance) / variance
                )
            scores[label] = score

        if not scores:
            return CLASSES[0], 0.0

        winner = max(scores, key=scores.get)
        maximum = scores[winner]
        confidence = 1.0 / sum(
            math.exp(score - maximum) for score in scores.values()
        )
        return winner, confidence
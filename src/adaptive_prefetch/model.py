"""Incremental window classifiers.

Both classifiers share one contract so they are interchangeable everywhere:

    update(values: tuple[float, ...], label: str) -> None
    predict(values: tuple[float, ...]) -> tuple[str, float]   # (class, confidence)

``OnlineGaussianNB`` is the original diagonal-covariance model.
``OnlineQDA`` keeps a full per-class scatter matrix, which lifts the accuracy
ceiling on workloads whose classes overlap; see ``docs/CLASSIFIER_EVAL.md``.
"""

from __future__ import annotations

import math

from .trace import CLASSES

K = len(CLASSES)
_CLASS_INDEX = {label: index for index, label in enumerate(CLASSES)}


class OnlineGaussianNB:
    """Incrementally updated Gaussian Naive Bayes classifier with exponential decay

    Supports concept drift adaptation by decaying historical statistics.
    """

    #: Registry key; must match the key used by ``artifacts`` persistence.
    name = "gnb"

    def __init__(
        self,
        n_features: int,
        variance_floor: float = 0.0025,
        decay_factor: float = 0.995,
    ):
        if not 0 < decay_factor <= 1:
            raise ValueError("decay_factor must be in (0, 1]")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        self.name = "gnb"
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
                # Variance is biased low under exponential decay: the divisor
                # is the saturated effective count 1/(1-gamma), not a true
                # sample count, so the decayed scatter under-estimates sigma^2.
                variance = max(
                    self.m2[label][i] * (1.0 / self.decay_factor) /
                    max(n - 1.0, 1.0),
                    self.variance_floor,
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


class OnlineQDA:
    """Online Quadratic Discriminant Analysis with shrinkage and decay.

    Same contract as :class:`OnlineGaussianNB`, but each class keeps a full
    ``d x d`` scatter matrix instead of ``d`` independent variances. This
    matters because ``contiguous_ratio``, ``dominant_stride_ratio`` and
    ``short_run_ratio`` are all functions of the same delta sequence and are
    strongly negatively correlated within a class -- exactly the structure a
    diagonal model cannot represent. Naive Bayes plateaus around 92.5% on
    overlapping classes and 64x more labelled data moves that by 0.16
    percentage points; this reaches ~98%.

    ``shrink`` pulls each covariance toward its own diagonal (a
    Ledoit-Wolf-style target), which keeps the Cholesky factorisation well
    conditioned when a class has few samples or a near-degenerate scatter
    matrix.
    """

    #: Registry key; must match the key used by ``artifacts`` persistence.
    name = "qda"

    def __init__(
        self,
        n_features: int,
        variance_floor: float = 1e-3,
        decay_factor: float = 0.995,
        shrink: float = 0.05,
    ):
        if not 0 < decay_factor <= 1:
            raise ValueError("decay_factor must be in (0, 1]")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        if not 0.0 <= shrink < 1.0:
            raise ValueError("shrink must be in [0, 1)")
        self.name = "qda"
        self.n_features = n_features
        self.variance_floor = variance_floor
        self.decay_factor = decay_factor
        self.shrink = shrink
        self._log_decay = math.log(decay_factor) if decay_factor > 0 else 0.0
        # Same attribute names and label-keyed shape as OnlineGaussianNB, so
        # callers can read ``model.count["sequential"]`` from either model.
        # The scatter matrices stay flat d*d lists: keeping them flat turns the
        # per-update decay into a list comprehension, which dominates update
        # cost in pure Python.
        self.count = {label: 0.0 for label in CLASSES}
        self.mean = {label: [0.0] * n_features for label in CLASSES}
        self.scatter = {label: [0.0] * (n_features * n_features) for label in CLASSES}
        self._cholesky: dict[str, list[list[float]] | None] = {label: None for label in CLASSES}
        self._logdet = {label: 0.0 for label in CLASSES}
        # Log of the pure decay applied since the last factorisation. Because
        # S -> g*S implies L -> sqrt(g)*L, logdet -> logdet + d*log(g) and the
        # Mahalanobis distance -> maha/g, an untouched class needs no refactor
        # and costs nothing. Without this a 1:1 predict/update stream refactors
        # all four classes on every call.
        self._pending = {label: 0.0 for label in CLASSES}
        self._dirty = {label: True for label in CLASSES}

    def update(self, values: tuple[float, ...], label: str) -> None:
        if label not in _CLASS_INDEX or len(values) != self.n_features:
            raise ValueError("unknown class or incorrect feature count")
        decay = self.decay_factor
        for name in CLASSES:
            self.count[name] *= decay
            self.scatter[name] = [v * decay for v in self.scatter[name]]
            self._pending[name] += self._log_decay

        self.count[label] += 1.0
        inverse = 1.0 / self.count[label]
        mean = self.mean[label]
        scatter = self.scatter[label]
        width = self.n_features
        for a in range(width):
            difference = values[a] - mean[a]
            mean[a] += difference * inverse
            base = a * width
            for b in range(width):
                scatter[base + b] += difference * (values[b] - mean[b])
        self._dirty[label] = True
        self._pending[label] = 0.0

    def _factor(self, label: str) -> None:
        """Cholesky factor of the shrunk covariance, plus its log-determinant."""
        d = self.n_features
        scatter = self.scatter[label]
        inverse = 1.0 / max(self.count[label] - 1.0, 1.0)
        full = 1.0 - self.shrink
        diagonal = [max(scatter[a * d + a] * inverse, self.variance_floor)
                    for a in range(d)]
        factor = [[0.0] * d for _ in range(d)]
        logdet = 0.0
        for a in range(d):
            row = factor[a]
            base = a * d
            for b in range(a + 1):
                total = full * scatter[base + b] * inverse
                if a == b:
                    total += self.shrink * diagonal[a]
                previous = factor[b]
                for c in range(b):
                    total -= row[c] * previous[c]
                if a == b:
                    root = math.sqrt(max(total, 1e-12))
                    row[a] = root
                    logdet += 2.0 * math.log(root)
                else:
                    row[b] = total / previous[b]
        self._cholesky[label] = factor
        self._logdet[label] = logdet
        self._dirty[label] = False

    def _mahalanobis(self, label: str, values: tuple[float, ...]) -> float:
        d = self.n_features
        factor = self._cholesky[label]
        mean = self.mean[label]
        total = 0.0
        solved = []
        for a in range(d):
            value = values[a] - mean[a]
            row = factor[a]
            for c in range(a):
                value -= row[c] * solved[c]
            component = value / row[a]
            solved.append(component)
            total += component * component
        return total

    def predict(self, values: tuple[float, ...]) -> tuple[str, float]:
        if len(values) != self.n_features:
            raise ValueError("incorrect feature count")
        total = sum(self.count.values())
        if total <= 0:
            raise ValueError("train the classifier before predicting")

        d = self.n_features
        scores: dict[str, float] = {}
        for label in CLASSES:
            # Fewer than two effective samples gives a degenerate covariance.
            if self.count[label] <= 1.5:
                continue
            if self._dirty[label]:
                self._factor(label)
            penalty = self._pending[label]
            distance = self._mahalanobis(label, values)
            if penalty:
                distance *= math.exp(-penalty)
            scores[label] = (
                -0.5 * (self._logdet[label] + d * penalty + distance)
                + math.log(self.count[label] / total)
            )

        if not scores:
            return CLASSES[0], 0.0
        winner = max(scores, key=scores.get)
        maximum = scores[winner]
        confidence = 1.0 / sum(
            math.exp(score - maximum) for score in scores.values()
        )
        return winner, confidence


#: Registry consumed by ``benchmark.make_model`` and the CLI. Keys are the
#: model ``name`` attributes, so an artifact saved by ``artifacts.save_model``
#: round-trips through this registry without a second lookup table.
CLASSIFIERS = {cls.name: cls for cls in (OnlineGaussianNB, OnlineQDA)}

#: Naive Bayes stays the default, which is the opposite of what the model
#: comparison initially suggested. Measured over disjoint seeds:
#:
#:   * On cleanly separable windows both are 100.00%, so the algorithm is not
#:     the bottleneck -- the *features* were (see ``features.py``).
#:   * On windows that straddle a phase change QDA wins decisively
#:     (86.3% vs 62.6% at a 75% blend), because the diagonal assumption
#:     cannot represent correlated features.
#:   * But this project's dominant risk is covariate shift from synthetic
#:     training windows to real traces. There GNB transfers perfectly across
#:     a 128x block-size change while QDA falls to ~79%, because a full
#:     covariance memorises joint structure that also moves.
#:
#: Parsimony wins on the risk that actually occurs, so GNB is the default and
#: QDA is opt-in via ``--classifier qda``.
DEFAULT_CLASSIFIER = "gnb"


def make_classifier(name: str, n_features: int, **kwargs):
    """Instantiate a registered classifier by name."""
    try:
        factory = CLASSIFIERS[name]
    except KeyError:
        raise ValueError(
            f"unknown classifier {name!r}; choose from "
            f"{', '.join(sorted(CLASSIFIERS))}"
        ) from None
    if factory is OnlineGaussianNB:
        allowed = {"variance_floor", "decay_factor"}
    else:
        allowed = {"variance_floor", "decay_factor", "shrink"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(
            f"{name} does not accept: {', '.join(sorted(unknown))}")
    return factory(n_features, **kwargs)

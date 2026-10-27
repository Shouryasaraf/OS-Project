"""Classifier families, random search and grouped CV for the hindsight router.

Optional: needs numpy and scikit-learn (the ``[web]`` extra), imported here
only, so the stdlib disk core never pays for it and system ``python3`` skips
the tests that use it.

Each family is a scikit-learn estimator with a pre-registered random-search
space (``SPACES``; Bergstra and Bengio 2012). A configuration is scored by
grouped cross-validation over traces (whole twin families held out
together), and the score is the share of hindsight headroom over ``none``
the predicted actions capture on the held-out traces' windows
(``hindsight.headroom_captured``), averaged over held-out traces. That is
the objective the router exists for, under the trajectory approximation;
label accuracy is reported beside it only as a diagnostic.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from . import hindsight

#: Families in report order. ``gnb`` is the shipped classifier family, here
#: trained on real-trace hindsight labels instead of synthetic classes.
FAMILIES = ("gnb", "logreg", "tree", "forest", "extra_trees", "hgb", "mlp", "knn")
#: Families whose ``fit`` takes ``sample_weight`` (the ``weighting`` option).
WEIGHTED = ("gnb", "logreg", "tree", "forest", "extra_trees", "hgb")
CONFIGS_PER_FAMILY = 8
SEARCH_SEED = 42
FOLDS = 5


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_params(family: str, rng: random.Random) -> dict:
    """One draw from ``family``'s pre-registered search space."""
    weighting = rng.choice(("none", "gap")) if family in WEIGHTED else "none"
    if family == "gnb":
        params = {"var_smoothing": _log_uniform(rng, 1e-12, 1e-1)}
    elif family == "logreg":
        params = {"C": _log_uniform(rng, 1e-3, 1e2),
                  "class_weight": rng.choice((None, "balanced"))}
    elif family == "tree":
        params = {"max_depth": rng.choice((3, 4, 6, 8, 12, 16, 20, None)),
                  "min_samples_leaf": rng.choice((1, 5, 20, 50, 100, 200)),
                  "criterion": rng.choice(("gini", "entropy")),
                  "class_weight": rng.choice((None, "balanced"))}
    elif family in ("forest", "extra_trees"):
        # Amended 2026-10-27 before the selection run: {100, 200} -> {50, 100}
        # (time; results/hindsight/PREREGISTERED.md, amendment A1).
        params = {"n_estimators": rng.choice((50, 100)),
                  "max_depth": rng.choice((8, 12, 16, None)),
                  "min_samples_leaf": rng.choice((1, 5, 20, 50)),
                  "max_features": rng.choice(("sqrt", 0.5, 1.0)),
                  "class_weight": rng.choice((None, "balanced"))}
    elif family == "hgb":
        params = {"learning_rate": _log_uniform(rng, 0.02, 0.3),
                  "max_leaf_nodes": rng.choice((15, 31, 63, 127)),
                  "max_depth": rng.choice((None, 4, 8)),
                  "l2_regularization": rng.choice((0.0, 0.1, 1.0)),
                  "max_iter": rng.choice((100, 200, 400)),
                  "class_weight": rng.choice((None, "balanced"))}
    elif family == "mlp":
        params = {"hidden_layer_sizes": rng.choice(((32,), (64,), (64, 32), (128, 64))),
                  "alpha": _log_uniform(rng, 1e-5, 1e-1),
                  "learning_rate_init": _log_uniform(rng, 1e-4, 1e-2)}
    elif family == "knn":
        params = {"n_neighbors": rng.choice((1, 3, 5, 11, 21, 51, 101)),
                  "weights": rng.choice(("uniform", "distance"))}
    else:
        raise ValueError(f"unknown family {family!r}; known: {FAMILIES}")
    params["weighting"] = weighting
    return params


def search_space(family: str, n: int = CONFIGS_PER_FAMILY,
                 seed: int = SEARCH_SEED) -> list[dict]:
    """The ``n`` random-search configurations of ``family`` (deterministic)."""
    rng = random.Random(f"{seed}:{family}")
    return [sample_params(family, rng) for _ in range(n)]


def build(family: str, params: dict):
    """A fresh unfitted estimator for ``family`` with ``params``."""
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    p = {k: v for k, v in params.items() if k != "weighting"}
    if family == "gnb":
        return GaussianNB(**p)
    if family == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **p))
    if family == "tree":
        return DecisionTreeClassifier(random_state=SEARCH_SEED, **p)
    if family == "forest":
        return RandomForestClassifier(random_state=SEARCH_SEED, n_jobs=1, **p)
    if family == "extra_trees":
        return ExtraTreesClassifier(random_state=SEARCH_SEED, n_jobs=1, **p)
    if family == "hgb":
        return HistGradientBoostingClassifier(random_state=SEARCH_SEED, **p)
    if family == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            max_iter=200, early_stopping=True, random_state=SEARCH_SEED, **p))
    if family == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**p))
    raise ValueError(f"unknown family {family!r}; known: {FAMILIES}")


class TraceData:
    """One trace slice's decision points: features, labels and outcomes.

    ``X[k]`` are window ``k``'s 12 features, ``matrix[k]`` the objective of
    every action on window ``k`` (``hindsight.outcome_matrix``), ``y[k]`` the
    index of the label of window ``k`` (best action on window ``k + 1``).
    """

    def __init__(self, trace: str, features, matrix):
        import numpy as np
        if len(features) != len(matrix):
            raise ValueError(f"{trace}: {len(features)} feature windows but "
                             f"{len(matrix)} outcome windows")
        self.trace = trace
        self.matrix = [tuple(row) for row in matrix]
        self.X = np.asarray(features[:-1], dtype=float)
        self.y = np.asarray([hindsight.best_action_index(self.matrix[k + 1])
                             for k in range(len(self.matrix) - 1)], dtype=int)
        values = np.asarray(self.matrix[1:], dtype=float)
        gap = values.max(axis=1) - values.min(axis=1)
        mean_gap = gap.mean()
        # ``gap`` weighting: a window counts by how much the choice matters
        # there, normalised so every trace carries the same total weight.
        self.gap_weight = gap / mean_gap if mean_gap > 0 else np.ones_like(gap)

    def subsample(self, n: int, seed: int = SEARCH_SEED):
        """Indices of up to ``n`` decision points, uniformly at random."""
        k = len(self.y)
        if k <= n:
            return list(range(k))
        rng = random.Random(f"{seed}:{self.trace}")
        return sorted(rng.sample(range(k), n))

    def score(self, predicted) -> float | None:
        return hindsight.headroom_captured(self.matrix, [int(a) for a in predicted])


def stack(datasets: list[TraceData], per_trace: int, weighting: str):
    """Training arrays from each trace's subsample."""
    import numpy as np
    xs, ys, ws = [], [], []
    for data in datasets:
        index = data.subsample(per_trace)
        xs.append(data.X[index])
        ys.append(data.y[index])
        ws.append(data.gap_weight[index] if weighting == "gap"
                  else np.ones(len(index)))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


class ConstantModel:
    """Predicts one action index; used when a training set has one label."""

    def __init__(self, value: int):
        self.value = int(value)

    def predict(self, X):
        import numpy as np
        return np.full(len(X), self.value, dtype=int)


def fit(family: str, params: dict, datasets: list[TraceData], per_trace: int):
    X, y, w = stack(datasets, per_trace, params.get("weighting", "none"))
    if family in ("mlp", "hgb"):
        # Their early stopping holds out a stratified split, which refuses a
        # class with a single sample; such a sample is dropped (only then).
        import numpy as np
        values, counts = np.unique(y, return_counts=True)
        rare = values[counts < 2]
        if len(rare) and len(values) > len(rare):
            keep = ~np.isin(y, rare)
            X, y, w = X[keep], y[keep], w[keep]
    if len(set(y.tolist())) == 1:
        return ConstantModel(y[0])
    if family == "knn" and params["n_neighbors"] > len(y):
        # A bounded (smoke) training set can be smaller than k.
        params = dict(params, n_neighbors=len(y))
    model = build(family, params)
    if family in WEIGHTED and params.get("weighting") == "gap":
        if family == "logreg":
            model.fit(X, y, logisticregression__sample_weight=w)
        else:
            model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)
    return model


def group_folds(traces: list[str], groups: dict[str, str], folds: int = FOLDS,
                seed: int = SEARCH_SEED) -> list[list[str]]:
    """Held-out trace lists, one per fold; a twin family stays in one fold."""
    labels = sorted({groups[t] for t in traces})
    random.Random(seed).shuffle(labels)
    out: list[list[str]] = [[] for _ in range(min(folds, len(labels)))]
    for index, label in enumerate(labels):
        out[index % len(out)].extend(sorted(t for t in traces if groups[t] == label))
    return out


def cross_validate(family: str, params: dict, datasets: list[TraceData],
                   groups: dict[str, str], per_trace: int) -> dict:
    """Grouped-CV score of one configuration (mean headroom captured)."""
    by_trace = {d.trace: d for d in datasets}
    scores, accuracy = [], []
    per_fold = []
    for held_out in group_folds(sorted(by_trace), groups):
        train = [by_trace[t] for t in sorted(by_trace) if t not in held_out]
        model = fit(family, params, train, per_trace)
        fold = []
        for trace in held_out:
            data = by_trace[trace]
            predicted = model.predict(data.X)
            accuracy.append(float((predicted == data.y).mean()))
            value = data.score(predicted)
            if value is not None:
                scores.append(value)
                fold.append(value)
        per_fold.append(sum(fold) / len(fold) if fold else None)
    return {"cv_score": sum(scores) / len(scores) if scores else None,
            "cv_accuracy": sum(accuracy) / len(accuracy) if accuracy else None,
            "cv_fold_scores": per_fold, "cv_traces_scored": len(scores)}


def predict_schedule(model, features) -> list[str]:
    """One action per complete window, from batch inference on the features
    the router would compute in the loop."""
    import numpy as np
    if not features:
        return []
    predicted = model.predict(np.asarray(features, dtype=float))
    return [hindsight.ACTIONS[int(a)] for a in predicted]


def save(model, path: str | Path, record: dict) -> Path:
    """Persist ``model`` with joblib and its training record beside it.

    The record names every trace and request range the router was trained
    or selected on and the artifact's SHA-256, so a scorer can refuse an
    overlapping slice (``hindsight.scoring_conflict``).
    """
    import joblib
    from prefetch_common.provenance import sha256_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    record = dict(record, artifact_sha256=sha256_file(path))
    hindsight.write_record(path.with_suffix(".training.json"), record)
    return path


def load(path: str | Path):
    """``(model, record)``; refuses an artifact whose hash does not match."""
    import joblib
    from prefetch_common.provenance import sha256_file
    path = Path(path)
    record = json.loads(path.with_suffix(".training.json").read_text(encoding="utf-8"))
    if record.get("artifact_sha256") != sha256_file(path):
        raise ValueError(f"{path}: artifact does not match its training record")
    return joblib.load(path), record

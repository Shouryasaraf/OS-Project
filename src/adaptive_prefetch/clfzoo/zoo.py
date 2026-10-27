"""The classifier families, their search spaces and a router wrapper.

Every family is fitted on the 12-feature contextual vectors of
:mod:`.data` and exposes the shipped classifier contract
``predict(values) -> (class, confidence)`` through :class:`Router`, so a
fitted model drops into ``simulator.replay(mode="adaptive", model=...)``
unchanged. scikit-learn (``Pedregosa et al. 2011``) is imported lazily: this
module is optional and the disk core stays stdlib-only.

Families and their canonical references (BibTeX keys in ``paper/refs.bib``
and ``paper/refs_CLF.bib``):

==============  ===============================================  =====================
family          estimator                                        reference
==============  ===============================================  =====================
gnb_online      shipped ``model.OnlineGaussianNB`` (as-is)       domingos1997optimality
qda_online      shipped ``model.OnlineQDA`` (as-is)              hastie2009elements
gnb             ``GaussianNB`` (var_smoothing)                   domingos1997optimality
lda             ``LinearDiscriminantAnalysis`` (shrinkage)       fisher1936use, ledoit2004well
qda             ``QuadraticDiscriminantAnalysis`` (reg_param)    hastie2009elements
logreg          multinomial ``LogisticRegression`` (C, scaling)  cox1958regression
knn             ``KNeighborsClassifier`` (k, weights, scaling)   cover1967nearest
svm_linear      ``SVC(kernel="linear")`` + Platt (C)             cortes1995support, platt1999probabilistic
svm_rbf         ``SVC(kernel="rbf")`` + Platt (C, gamma)         cortes1995support, platt1999probabilistic
cart            ``DecisionTreeClassifier`` (depth, leaf)         breiman1984classification
rf              ``RandomForestClassifier``                       breiman2001random
extra_trees     ``ExtraTreesClassifier``                         geurts2006extremely
hgb             ``HistGradientBoostingClassifier``               friedman2001greedy, ke2017lightgbm
mlp             ``MLPClassifier`` (Adam)                         rumelhart1986learning, kingma2015adam
sgd_online      ``SGDClassifier(log_loss)`` by ``partial_fit``   bottou2010large
==============  ===============================================  =====================

Hyperparameters are drawn by seeded random search (Bergstra and Bengio 2012)
from :data:`SPACES`; the first candidate of every family is its library
default. A streaming Hoeffding tree (Domingos and Hulten 2000) is **not**
implemented: a correct stdlib version was out of scope for the time budget,
and ``sgd_online`` is the online alternative.
"""

from __future__ import annotations

import math
import random
from time import perf_counter

from ..model import OnlineGaussianNB, OnlineQDA
from ..trace import CLASSES
from .data import N_FEATURES

#: Report order. The first two are the shipped baselines, never tuned.
FAMILIES = ("gnb_online", "qda_online", "gnb", "lda", "qda", "logreg", "knn",
            "svm_linear", "svm_rbf", "cart", "rf", "extra_trees", "hgb", "mlp",
            "sgd_online")
SHIPPED = ("gnb_online", "qda_online")
BASELINE_FAMILY = "gnb_online"

CITATIONS = {
    "gnb_online": ("domingos1997optimality",),
    "qda_online": ("hastie2009elements",),
    "gnb": ("domingos1997optimality", "zhang2004optimality"),
    "lda": ("fisher1936use", "ledoit2004well"),
    "qda": ("hastie2009elements",),
    "logreg": ("cox1958regression",),
    "knn": ("cover1967nearest",),
    "svm_linear": ("cortes1995support", "platt1999probabilistic"),
    "svm_rbf": ("cortes1995support", "platt1999probabilistic"),
    "cart": ("breiman1984classification",),
    "rf": ("breiman2001random",),
    "extra_trees": ("geurts2006extremely",),
    "hgb": ("friedman2001greedy", "ke2017lightgbm"),
    "mlp": ("rumelhart1986learning", "kingma2015adam"),
    "sgd_online": ("bottou2010large",),
}
COMMON_CITATIONS = ("pedregosa2011sklearn", "bergstra2012random")

#: Random-search candidates per tunable family (the default is candidate 0).
N_CANDIDATES = 16
SEARCH_SEED = 42
#: libsvm iteration cap (amendment 1 of PREREGISTERED.md): uncapped, a
#: linear SVC with a large C on unscaled vectors did not finish in minutes.
SVC_MAX_ITER = 200_000


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return float(10 ** rng.uniform(math.log10(low), math.log10(high)))


def _sample(family: str, rng: random.Random) -> dict:
    """One random-search draw for ``family`` (the space is this function)."""
    if family == "gnb":
        return {"var_smoothing": _log_uniform(rng, 1e-12, 1.0)}
    if family == "lda":
        return {"shrinkage": rng.choice([None, "auto", round(rng.uniform(0, 1), 4)])}
    if family == "qda":
        return {"reg_param": round(rng.uniform(0.0, 1.0), 4)}
    if family == "logreg":
        return {"C": _log_uniform(rng, 1e-3, 1e3), "scale": rng.choice([True, False])}
    if family == "knn":
        return {"n_neighbors": rng.randint(1, 50),
                "weights": rng.choice(["uniform", "distance"]),
                "scale": rng.choice([True, False])}
    if family == "svm_linear":
        return {"C": _log_uniform(rng, 1e-3, 1e3), "scale": rng.choice([True, False])}
    if family == "svm_rbf":
        return {"C": _log_uniform(rng, 1e-2, 1e3), "gamma": _log_uniform(rng, 1e-4, 1e1),
                "scale": rng.choice([True, False])}
    if family == "cart":
        return {"max_depth": rng.choice([2, 3, 4, 6, 8, 12, 16, None]),
                "min_samples_leaf": rng.randint(1, 20)}
    if family in ("rf", "extra_trees"):
        return {"n_estimators": rng.choice([100, 200, 400]),
                "max_depth": rng.choice([4, 8, 16, None]),
                "max_features": rng.choice(["sqrt", 0.5, 1.0]),
                "min_samples_leaf": rng.randint(1, 10)}
    if family == "hgb":
        return {"learning_rate": _log_uniform(rng, 0.01, 0.3),
                "max_iter": rng.choice([100, 200, 400]),
                "max_leaf_nodes": rng.choice([7, 15, 31, 63]),
                "l2_regularization": _log_uniform(rng, 1e-4, 10.0)}
    if family == "mlp":
        return {"hidden_layer_sizes": rng.choice([[32], [64], [128], [64, 32], [128, 64]]),
                "alpha": _log_uniform(rng, 1e-6, 1e-1),
                "learning_rate_init": _log_uniform(rng, 1e-4, 1e-2)}
    if family == "sgd_online":
        return {"alpha": _log_uniform(rng, 1e-6, 1e-1), "epochs": rng.choice([1, 5, 20])}
    raise ValueError(f"{family} has no search space")


DEFAULTS = {
    "gnb": {"var_smoothing": 1e-9},
    "lda": {"shrinkage": None},
    "qda": {"reg_param": 0.0},
    "logreg": {"C": 1.0, "scale": True},
    "knn": {"n_neighbors": 5, "weights": "uniform", "scale": True},
    "svm_linear": {"C": 1.0, "scale": True},
    "svm_rbf": {"C": 1.0, "gamma": "scale", "scale": True},
    "cart": {"max_depth": None, "min_samples_leaf": 1},
    "rf": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt",
           "min_samples_leaf": 1},
    "extra_trees": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt",
                    "min_samples_leaf": 1},
    "hgb": {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31,
            "l2_regularization": 0.0},
    "mlp": {"hidden_layer_sizes": [100], "alpha": 1e-4, "learning_rate_init": 1e-3},
    "sgd_online": {"alpha": 1e-4, "epochs": 5},
}


def candidates(family: str, n: int = N_CANDIDATES, seed: int = SEARCH_SEED) -> list[dict]:
    """The random-search candidates of ``family``: its default, then draws.

    Seeded per family, so adding a family never changes another's draws.
    The shipped online models have one candidate, the shipped configuration.
    """
    if family in SHIPPED:
        return [{}]
    rng = random.Random(f"{seed}:{family}")
    out = [dict(DEFAULTS[family])]
    while len(out) < n:
        draw = _sample(family, rng)
        if draw not in out:
            out.append(draw)
    return out


def _sklearn_estimator(family: str, params: dict, seed: int):
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    p = dict(params)
    scale = p.pop("scale", False)
    if family == "gnb":
        model = GaussianNB(**p)
    elif family == "lda":
        shrinkage = p["shrinkage"]
        model = (LinearDiscriminantAnalysis() if shrinkage is None else
                 LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage))
    elif family == "qda":
        model = QuadraticDiscriminantAnalysis(**p)
    elif family == "logreg":
        # lbfgs on >2 classes fits the multinomial (softmax) model.
        model = LogisticRegression(C=p["C"], max_iter=5000)
    elif family == "knn":
        model = KNeighborsClassifier(**p)
    elif family == "svm_linear":
        model = SVC(kernel="linear", C=p["C"], probability=True, random_state=seed,
                    max_iter=SVC_MAX_ITER)
    elif family == "svm_rbf":
        model = SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], probability=True,
                    random_state=seed, max_iter=SVC_MAX_ITER)
    elif family == "cart":
        model = DecisionTreeClassifier(random_state=seed, **p)
    elif family == "rf":
        model = RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    elif family == "extra_trees":
        model = ExtraTreesClassifier(random_state=seed, n_jobs=1, **p)
    elif family == "hgb":
        model = HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **p)
    elif family == "mlp":
        model = MLPClassifier(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]),
                              alpha=p["alpha"], learning_rate_init=p["learning_rate_init"],
                              solver="adam", max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"unknown sklearn family {family!r}")
    return make_pipeline(StandardScaler(), model) if scale else model


class Router:
    """A fitted classifier behind the shipped ``predict`` contract.

    ``predict(values) -> (class, confidence)`` with ``class`` in
    ``trace.CLASSES`` and ``confidence`` the top posterior, exactly the
    contract ``simulator.replay`` calls once per complete window.
    :meth:`proba_batch` gives the full posterior in ``CLASSES`` order.
    """

    def __init__(self, family: str, params: dict, estimator=None, online=None):
        self.family = family
        self.params = dict(params)
        self.name = family
        self._estimator = estimator
        self._online = online
        self._order = None
        if estimator is not None:
            classes = [str(c) for c in estimator.classes_]
            self._order = [classes.index(label) for label in CLASSES]

    # -- shipped contract
    def predict(self, values) -> tuple[str, float]:
        if self._online is not None and not isinstance(self._online, _SGDOnline):
            return self._online.predict(tuple(values))
        probs = self.proba_batch([values])[0]
        best = max(range(len(CLASSES)), key=lambda i: probs[i])
        return CLASSES[best], probs[best]

    # -- batch helpers (numpy)
    def proba_batch(self, rows):
        import numpy as np
        X = np.asarray(rows, dtype=float)
        if self._online is not None:
            if isinstance(self._online, _SGDOnline):
                return self._online.proba(X)
            return np.array([online_posterior(self._online, tuple(row)) for row in X])
        raw = self._estimator.predict_proba(X)
        return raw[:, self._order]

    def predict_batch(self, rows) -> list[str]:
        """Labels for many windows, identical to calling :meth:`predict` on each."""
        if self._online is not None and not isinstance(self._online, _SGDOnline):
            return [self._online.predict(tuple(row))[0] for row in rows]
        probs = self.proba_batch(rows)
        # argmax keeps the first maximum, as max() over range does above.
        return [CLASSES[int(i)] for i in probs.argmax(axis=1)]


class _SGDOnline:
    """Multinomial-by-OvR logistic SGD trained with ``partial_fit`` only."""

    def __init__(self, alpha: float, epochs: int, seed: int):
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        import numpy as np
        order = list(range(len(y)))
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            # One streaming pass per epoch in a seeded order: a single pass
            # in make_model's class-after-class order would only ever see
            # the last class at the end of training.
            rng.shuffle(order)
            for start in range(0, len(order), 32):
                batch = order[start:start + 32]
                self.model.partial_fit(X[batch], np.asarray(y)[batch],
                                       classes=np.array(CLASSES))
        classes = [str(c) for c in self.model.classes_]
        self.order = [classes.index(label) for label in CLASSES]
        return self

    def proba(self, X):
        return self.model.predict_proba(X)[:, self.order]


def online_posterior(model, values) -> list[float]:
    """Full posterior of a shipped online model, in ``CLASSES`` order.

    Recomputes the class scores the model's own ``predict`` uses (same
    formulae, same skipped classes) and normalises them; the argmax and the
    top probability equal ``model.predict(values)``.
    """
    if isinstance(model, OnlineGaussianNB):
        total = sum(model.count.values())
        scores = {}
        for label in CLASSES:
            n = model.count[label]
            if n <= 1e-5:
                continue
            score = math.log(n / total)
            for i, value in enumerate(values):
                variance = max(model.m2[label][i] * (1.0 / model.decay_factor) /
                               max(n - 1.0, 1.0), model.variance_floor)
                distance = value - model.mean[label][i]
                score += -0.5 * (math.log(2 * math.pi * variance)
                                 + (distance * distance) / variance)
            scores[label] = score
    elif isinstance(model, OnlineQDA):
        total = sum(model.count.values())
        d = model.n_features
        scores = {}
        for label in CLASSES:
            if model.count[label] <= 1.5:
                continue
            if model._dirty[label]:
                model._factor(label)
            penalty = model._pending[label]
            distance = model._mahalanobis(label, values)
            if penalty:
                distance *= math.exp(-penalty)
            scores[label] = (-0.5 * (model._logdet[label] + d * penalty + distance)
                             + math.log(model.count[label] / total))
    else:
        raise TypeError(f"not a shipped online model: {type(model).__name__}")
    if not scores:
        return [1.0 if label == CLASSES[0] else 0.0 for label in CLASSES]
    top = max(scores.values())
    weights = {label: math.exp(s - top) for label, s in scores.items()}
    norm = sum(weights.values())
    return [weights.get(label, 0.0) / norm for label in CLASSES]


def fit(family: str, params: dict, rows, labels, seed: int = 0) -> Router:
    """Fit ``family`` with ``params`` on vectors ``rows`` (in stream order).

    The shipped online models are updated one vector at a time in the given
    order, exactly as ``benchmark.make_model`` fits them.
    """
    if family == "gnb_online":
        model = OnlineGaussianNB(N_FEATURES)
        for values, label in zip(rows, labels):
            model.update(tuple(values), label)
        return Router(family, params, online=model)
    if family == "qda_online":
        model = OnlineQDA(N_FEATURES)
        for values, label in zip(rows, labels):
            model.update(tuple(values), label)
        return Router(family, params, online=model)
    import numpy as np
    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels)
    if family == "sgd_online":
        online = _SGDOnline(params["alpha"], params["epochs"], seed).fit(X, y)
        return Router(family, params, online=online)
    estimator = _sklearn_estimator(family, params, seed)
    estimator.fit(X, y)
    return Router(family, params, estimator=estimator)


def predict_cost_us(router: Router, rows, calls: int = 64) -> float:
    """Median wall time of one deployed ``predict`` call, in microseconds.

    One call per window is how ``adaptive`` uses a classifier. The host is
    shared, so this is a contended timing, never a cost claim on its own.
    """
    times = []
    for row in list(rows)[:calls]:
        start = perf_counter()
        router.predict(tuple(row))
        times.append((perf_counter() - start) * 1e6)
    times.sort()
    return times[len(times) // 2] if times else 0.0

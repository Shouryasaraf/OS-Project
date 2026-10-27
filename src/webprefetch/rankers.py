"""Alternative candidate rankers for the web study (track WRK).

The shipped ranker (:mod:`webprefetch.model`) is a logistic regression over
nine causal candidate features, trained on a 30,000-decision sample of split
1 and tuned over three values of C. This module asks whether another model
family, tuned properly, ranks the same candidates better, and whether that
moves the replayed cache metrics. Every family sees **exactly the same
candidates and the same nine features** (:func:`webprefetch.model.features`
on the split-0 :class:`~webprefetch.index.CandidateIndex`), so only the
scoring function changes.

Families (``FAMILIES``), each an sklearn estimator that outputs a
probability that a candidate is the next view:

=================  =====================================================
``logistic_tuned``  L1/L2 logistic regression, C, penalty and class weight searched
``hgb``             histogram gradient boosting (LightGBM-style)
``rf``              random forest
``et``              extremely randomised trees
``mlp``             multilayer perceptron trained with Adam
=================  =====================================================

Protocol (GRILL, ``results/web_rankers/PREREGISTERED.md``): split 1 trains
(``TRAIN_DECISIONS`` decisions, evenly spaced as in ``train_ranker``), a
``TUNE_DECISIONS`` sample of split 2 selects each family's hyperparameters by
random search (``SEARCH_SPACES``, ``TRIALS``, seed :data:`SEARCH_SEED`), the
whole of split 2 then selects the prefetch policy per capacity exactly as E4
does (:func:`webprefetch.benchmark.select_policies`), and split 3 is replayed
once with everything frozen. Rankings are computed once per split in a batch
(:func:`rank_split`) and replayed through :class:`ArrayScorer`, which returns
exactly what the per-request :class:`ModelScorer` would.

sklearn, numpy and pandas are imported here, so this module is optional: the
disk core and system ``python3`` never import it.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import FEATURE_NAMES, features

FAMILIES = ("logistic_tuned", "hgb", "rf", "et", "mlp")
#: The shipped ranker, the history-only top-1 and the causal online counts.
BASELINES = ("logistic", "trans_s0", "trans_online")
#: Split-1 decisions every family trains on (the shipped ranker used 30,000;
#: its learning curve was flat, the tree families' need not be).
TRAIN_DECISIONS = 100_000
#: Split-2 decisions the hyperparameter search scores on (the shipped ranker
#: chose C on 10,000, which hold only a few hundred positive candidates).
TUNE_DECISIONS = 100_000
SEARCH_SEED = 42
#: Random-search budget per family (Bergstra and Bengio 2012).
TRIALS = {"logistic_tuned": 16, "hgb": 24, "rf": 12, "et": 12, "mlp": 12}
#: The ranking cut-off of every ranking metric (the policies prefetch at most 8).
TOP_K = 8


def _loguniform(low: float, high: float):
    from scipy.stats import loguniform
    return loguniform(low, high)


def search_spaces() -> dict:
    """The pre-registered search space of every family (sklearn parameter names)."""
    return {
        "logistic_tuned": {"C": _loguniform(1e-3, 1e2), "l1_ratio": [0.0, 1.0],
                           "class_weight": [None, "balanced"]},
        "hgb": {"learning_rate": _loguniform(0.01, 0.3), "max_iter": [100, 200, 400],
                "max_leaf_nodes": [7, 15, 31, 63], "min_samples_leaf": [20, 50, 100, 200],
                "l2_regularization": _loguniform(1e-4, 10.0)},
        # Fully grown forests (no depth cap, one-sample leaves) took over ten
        # minutes per fit in the pilot, so depth and leaf size are bounded.
        "rf": {"n_estimators": [100, 200], "max_depth": [8, 12, 16, 20],
               "min_samples_leaf": [5, 20, 50, 100], "max_features": ["sqrt", 0.5],
               "max_samples": [0.25, 0.5]},
        "et": {"n_estimators": [100, 200], "max_depth": [8, 12, 16, 20],
               "min_samples_leaf": [5, 20, 50, 100], "max_features": ["sqrt", 0.5]},
        "mlp": {"hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                "alpha": _loguniform(1e-5, 1e-2), "learning_rate_init": _loguniform(1e-4, 1e-2),
                "batch_size": [256, 1024]},
    }


def make_estimator(family: str, params: dict, n_jobs: int = 1):
    """A fresh, seeded estimator of ``family`` with ``params``."""
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(params)
    if family == "logistic_tuned":
        # l1_ratio 1 is the L1 (lasso) penalty, 0 the L2 (ridge) one.
        solver = "liblinear" if params.get("l1_ratio") == 1.0 else "lbfgs"
        return make_pipeline(StandardScaler(), LogisticRegression(
            solver=solver, max_iter=1000, random_state=SEARCH_SEED, **params))
    if family == "hgb":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEARCH_SEED,
                                              **params)
    if family == "rf":
        return RandomForestClassifier(random_state=SEARCH_SEED, n_jobs=n_jobs, **params)
    if family == "et":
        return ExtraTreesClassifier(random_state=SEARCH_SEED, n_jobs=n_jobs, **params)
    if family == "mlp":
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        # early_stopping holds out 10% of the *training* rows (split 1).
        return make_pipeline(StandardScaler(), MLPClassifier(
            early_stopping=True, max_iter=60, n_iter_no_change=5, random_state=SEARCH_SEED,
            **params))
    raise ValueError(f"unknown ranker family {family!r}; expected one of {FAMILIES}")


def sample_trials(family: str, trials: int | None = None, seed: int = SEARCH_SEED) -> list[dict]:
    """The random-search configurations of ``family``, in trial order."""
    from sklearn.model_selection import ParameterSampler
    n = TRIALS[family] if trials is None else trials
    out = []
    for params in ParameterSampler(search_spaces()[family], n_iter=n, random_state=seed):
        out.append({k: (float(v) if isinstance(v, (float, np.floating)) else
                        int(v) if isinstance(v, (np.integer,)) else v)
                    for k, v in sorted(params.items())})
    return out


# -------------------------------------------------------------- design

@dataclass
class Design:
    """Every candidate of every decision row, flattened.

    ``x[m, 9]`` are the candidate features, ``items[m]`` the candidates and
    ``y[m]`` the label (eligible and the candidate is the next view);
    ``lengths[n]`` the candidate count of each of the ``n`` frame rows, in
    frame order. ``eligible[n]`` and ``covered[n]`` (the next view is a
    candidate at all) describe the rows.
    """
    x: np.ndarray
    y: np.ndarray
    items: np.ndarray
    lengths: np.ndarray
    eligible: np.ndarray

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.lengths)])


def sample_decisions(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """``train_ranker``'s evenly spaced sample (all rows when ``limit`` is 0)."""
    if limit and len(frame) > limit:
        return frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=np.int64)]
    return frame


_DESIGN: dict = {}


def _design_chunk(bounds: tuple[int, int]):
    frame, index = _DESIGN["frame"], _DESIGN["index"]
    x, y, items, lengths = [], [], [], []
    for row in frame.iloc[bounds[0]:bounds[1]].itertuples(index=False):
        candidates = index.candidates(int(row.itemid))
        lengths.append(len(candidates))
        positive = bool(row.eligible)
        for rank, (item, count) in enumerate(candidates):
            x.append(features(row, item, count, rank, index))
            y.append(int(positive and item == row.next_item))
            items.append(item)
    return (np.asarray(x, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)),
            np.asarray(y, dtype=np.int8), np.asarray(items, dtype=np.int64),
            np.asarray(lengths, dtype=np.int64))


def design(frame: pd.DataFrame, index, workers: int = 1) -> Design:
    """The candidate design matrix of ``frame``, built in contiguous chunks.

    The rows, their order and every value are exactly those
    ``train_ranker`` builds, so a model fit on ``design(sample).x`` sees the
    shipped ranker's training matrix.
    """
    n = len(frame)
    chunks = max(1, min(workers * 4, n // 2000 or 1))
    edges = np.linspace(0, n, chunks + 1, dtype=np.int64)
    tasks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    _DESIGN.update(frame=frame, index=index)
    try:
        if workers <= 1 or len(tasks) <= 1:
            parts = [_design_chunk(task) for task in tasks]
        else:
            context = multiprocessing.get_context(
                "fork" if "fork" in multiprocessing.get_all_start_methods() else None)
            with context.Pool(min(workers, len(tasks))) as pool:
                parts = pool.map(_design_chunk, tasks, chunksize=1)
    finally:
        _DESIGN.clear()
    if not parts:
        empty = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        return Design(empty, np.zeros(0, np.int8), np.zeros(0, np.int64),
                      np.zeros(0, np.int64), np.zeros(0, bool))
    return Design(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                  np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]),
                  frame.eligible.to_numpy(dtype=bool))


# ------------------------------------------------------------- ranking

def rank_order(probs: np.ndarray, items: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Flat indices that sort each row's candidates by (-probability, item).

    The tie rule is the shipped scorers' (``ranked.sort(key=(-p, item))``).
    """
    groups = np.repeat(np.arange(len(lengths)), lengths)
    return np.lexsort((items, -probs, groups))


def ranking_counts(probs: np.ndarray, d: Design, k: int = TOP_K) -> dict:
    """Per-row ranking outcome: ``rank[n]`` of the next view (0 = not in the top k)."""
    order = rank_order(probs, d.items, d.lengths)
    offsets = d.offsets
    within = np.arange(len(order)) - np.repeat(offsets[:-1], d.lengths)
    positive = d.y[order].astype(bool)
    rank = np.zeros(len(d.lengths), dtype=np.int64)
    rows = np.repeat(np.arange(len(d.lengths)), d.lengths)[positive]
    ranks = within[positive] + 1
    keep = ranks <= k
    rank[rows[keep]] = ranks[keep]
    return {"rank": rank, "order": order}


def ranking_summary(probs: np.ndarray, d: Design, k: int = TOP_K) -> dict:
    """hit@1, coverage@k (hit@k), MRR@k per eligible decision, and the log loss."""
    from sklearn.metrics import log_loss
    rank = ranking_counts(probs, d, k)["rank"]
    eligible = int(d.eligible.sum())
    hits = rank[d.eligible]
    clipped = np.clip(probs, 1e-15, 1 - 1e-15)
    return {"eligible": eligible,
            "hit_at_1": float((hits == 1).sum() / eligible) if eligible else 0.0,
            f"coverage_at_{k}": float((hits > 0).sum() / eligible) if eligible else 0.0,
            f"mrr_at_{k}": float(np.where(hits > 0, 1.0 / np.maximum(hits, 1), 0.0).sum()
                                 / eligible) if eligible else 0.0,
            "log_loss": float(log_loss(d.y, clipped, labels=[0, 1])) if len(d.y) else None}


def predict(model, x: np.ndarray, batch: int = 1_000_000) -> np.ndarray:
    """Positive-class probabilities as float64, in bounded batches."""
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch):
        out[start:start + batch] = model.predict_proba(x[start:start + batch])[:, 1]
    return out


class ArrayScorer:
    """A batch-scored ranking looked up by ``(session_id, position)``.

    It holds, for every row of one split, the candidates sorted by
    (-probability, item) as padded arrays, and returns exactly the list a
    :class:`ModelScorer` returns for that row. The index argument is ignored.
    """

    def __init__(self, keys: dict, items: np.ndarray, probs: np.ndarray, lengths: np.ndarray):
        self.keys, self.items, self.probs, self.lengths = keys, items, probs, lengths

    def score(self, row, index=None) -> list[tuple[int, float]]:
        r = self.keys.get((int(row.session_id), int(row.position)))
        if r is None:
            return []
        k = self.lengths[r]
        return list(zip(self.items[r, :k].tolist(), self.probs[r, :k].tolist()))


def array_scorer(frame: pd.DataFrame, d: Design, probs: np.ndarray) -> ArrayScorer:
    """The :class:`ArrayScorer` of ``frame`` from its design and probabilities."""
    order = rank_order(probs, d.items, d.lengths)
    n = len(d.lengths)
    width = max(1, int(d.lengths.max()) if n else 1)
    items = np.full((n, width), -1, dtype=np.int64)
    out = np.zeros((n, width), dtype=np.float64)
    within = np.arange(len(order)) - np.repeat(d.offsets[:-1], d.lengths)
    rows = np.repeat(np.arange(n), d.lengths)
    items[rows, within] = d.items[order]
    out[rows, within] = probs[order]
    keys = {(int(s), int(p)): r for r, (s, p) in
            enumerate(zip(frame.session_id.to_numpy(), frame.position.to_numpy()))}
    return ArrayScorer(keys, items, out, d.lengths.copy())


class ModelScorer:
    """Per-request scoring with any sklearn probability model (the online path)."""

    def __init__(self, model):
        self.model = model

    def score(self, row, index) -> list[tuple[int, float]]:
        candidates = index.candidates(int(row.itemid))
        if not candidates:
            return []
        matrix = np.asarray([features(row, item, count, rank, index)
                             for rank, (item, count) in enumerate(candidates)], dtype=np.float32)
        probabilities = self.model.predict_proba(matrix)[:, 1]
        ranked = [(item, float(p)) for (item, _), p in zip(candidates, probabilities)]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked


# -------------------------------------------------------------- search

_SEARCH: dict = {}


def _limit_threads(n: int) -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:  # pragma: no cover - sklearn ships threadpoolctl
        pass


def _trial(task: tuple) -> dict:
    family, number, params, threads = task
    _limit_threads(threads)
    train, tune = _SEARCH["train"], _SEARCH["tune"]
    started = time.perf_counter()
    model = make_estimator(family, params, n_jobs=threads).fit(train.x, train.y)
    fit_s = time.perf_counter() - started
    started = time.perf_counter()
    probs = predict(model, tune.x)
    score_s = time.perf_counter() - started
    summary = ranking_summary(probs, tune)
    return {"family": family, "trial": number, "params": params, "fit_s": fit_s,
            "score_s": score_s, "load_avg": os.getloadavg()[0], **summary}


def selection_key(row: dict) -> tuple:
    """Higher validation MRR@8 wins; then lower log loss; then the earlier trial."""
    return (-row[f"mrr_at_{TOP_K}"], row["log_loss"], row["trial"])


def random_search(train: Design, tune: Design, families=FAMILIES, trials: dict | None = None,
                  workers: int = 1, log=print) -> list[dict]:
    """Every trial of every family, scored on the tuning design.

    Trials run in a fork pool of ``workers`` single-threaded processes
    (forest trials with ``threads`` jobs each when there are fewer trials than
    workers), longest families first.
    """
    trials = dict(TRIALS if trials is None else trials)
    tasks = []
    for family in families:
        for number, params in enumerate(sample_trials(family, trials.get(family))):
            tasks.append((family, number, params, 1))
    # The forests take longest: start them first so the pool drains evenly.
    weight = {"rf": 0, "et": 1, "mlp": 2, "hgb": 3, "logistic_tuned": 4}
    tasks.sort(key=lambda t: (weight.get(t[0], 9), t[1]))
    _SEARCH.update(train=train, tune=tune)
    try:
        if workers <= 1:
            rows = [_trial(task) for task in tasks]
        else:
            context = multiprocessing.get_context(
                "fork" if "fork" in multiprocessing.get_all_start_methods() else None)
            rows = []
            with context.Pool(min(workers, len(tasks))) as pool:
                for row in pool.imap_unordered(_trial, tasks):
                    log(f"[rankers] {row['family']} trial {row['trial']}: "
                        f"MRR@8 {row[f'mrr_at_{TOP_K}']:.5f} log loss {row['log_loss']:.6f} "
                        f"fit {row['fit_s']:.1f} s")
                    rows.append(row)
    finally:
        _SEARCH.clear()
    rows.sort(key=lambda r: (families.index(r["family"]), r["trial"]))
    best = {}
    for row in rows:
        if row["family"] not in best or selection_key(row) < selection_key(best[row["family"]]):
            best[row["family"]] = row
    for row in rows:
        row["selected"] = best[row["family"]] is row
    return rows


def fit_selected(family: str, params: dict, train: Design, threads: int = 1):
    """Refit the selected configuration (deterministic: seeded) and time it."""
    _limit_threads(threads)
    started = time.perf_counter()
    model = make_estimator(family, params, n_jobs=threads).fit(train.x, train.y)
    return model, time.perf_counter() - started


def _refit(task: tuple) -> tuple:
    family, params, path, record = task
    model, fit_s = fit_selected(family, params, _SEARCH["train"], 1)
    summary = ranking_summary(predict(model, _SEARCH["tune"].x), _SEARCH["tune"])
    save_model(path, family, model, {**record, "refit_tune_summary": summary})
    return family, fit_s, summary, os.getloadavg()[0]


def refit_all(chosen: dict, train: Design, tune: Design, directory: Path, records: dict,
              workers: int = 1) -> dict:
    """Refit every family's selected trial single-threaded, as its trial ran,
    so the model is the one the search scored; each is saved to
    ``directory`` and loaded back. Returns ``{family: (model, fit_s, tune
    summary, load)}``."""
    import joblib
    tasks = [(family, chosen[family]["params"], directory, records[family]) for family in chosen]
    _SEARCH.update(train=train, tune=tune)
    try:
        if workers <= 1 or len(tasks) <= 1:
            parts = [_refit(task) for task in tasks]
        else:
            context = multiprocessing.get_context(
                "fork" if "fork" in multiprocessing.get_all_start_methods() else None)
            with context.Pool(min(workers, len(tasks))) as pool:
                parts = pool.map(_refit, tasks, chunksize=1)
    finally:
        _SEARCH.clear()
    return {family: (joblib.load(Path(directory) / f"{family}.joblib"), fit_s, summary, load)
            for family, fit_s, summary, load in parts}


def params_text(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def timing_per_request(model, frame: pd.DataFrame, index, requests: int = 1000) -> float:
    """Mean wall microseconds of one :class:`ModelScorer` call (features +
    predict + sort) over the first ``requests`` rows. Contended host: the
    caller records the load average with it."""
    scorer = ModelScorer(model)
    rows = list(frame.iloc[:requests].itertuples(index=False))
    if not rows:
        return 0.0
    scorer.score(rows[0], index)  # warm-up
    started = time.perf_counter()
    for row in rows:
        scorer.score(row, index)
    return (time.perf_counter() - started) / len(rows) * 1e6


def save_model(directory: str | Path, family: str, model, record: dict) -> Path:
    """``models/web/rankers/<family>.joblib`` plus its training record."""
    import joblib
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / f"{family}.joblib", compress=3)
    (directory / f"{family}.json").write_text(json.dumps(record, indent=2, default=str),
                                              encoding="utf-8")
    return directory / f"{family}.joblib"


# ---------------------------------------------------------- experiment

#: The capacities E4's headline comparisons (W1, W2) are made at.
CAPACITIES = (8, 32)
#: The cost of the headline utility comparisons (E4's W1-W6).
HEADLINE_COST = 0.01
#: E4's committed test row (results/web/policy_metrics.csv: test, logistic,
#: cap 8, fixed1) that the regenerated shipped ranker must reproduce exactly
#: on the full test split before any comparison is trusted.
E4_CHECK = {"views": 532_844, "capacity": 8, "policy": "fixed1", "hits": 65_944,
            "issued": 445_796}
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 42
#: Ranking metrics compared between rankers, with their denominators.
RANKING_METRICS = (("hit_at_1", "eligible"), (f"coverage_at_{TOP_K}", "eligible"),
                   (f"mrr_at_{TOP_K}", "eligible"), ("novel_hit_at_1", "novel_eligible"),
                   (f"novel_coverage_at_{TOP_K}", "novel_eligible"))
SPLIT_LABELS = {"validation": 2, "test": 3}


def three_way(ci) -> str:
    """The pre-registered verdict of a difference A - B."""
    if ci.low > 0:
        return "better"
    if ci.high < 0:
        return "worse"
    return "no difference detected"


def session_ranking(frame: pd.DataFrame, table, k: int = TOP_K) -> dict:
    """Per-session ranking counts of the rankings a replay used.

    ``table`` is the replay's :class:`~webprefetch.scorers.RankTable` over
    ``frame`` (whose ``rank_row`` column indexes it). Returns sorted session
    keys and, per session, the eligible next views, the top-1 hits, the
    top-``k`` hits, the reciprocal-rank sum within the top ``k`` and the same
    three counts over *novel* next views (not viewed earlier in the session),
    exactly the replay simulator's definitions.
    """
    ordered = frame.sort_values(["session_id", "position"], kind="mergesort")
    sessions = ordered.session_id.to_numpy()
    rows = ordered.rank_row.to_numpy()
    items, nexts = ordered.itemid.to_numpy(), ordered.next_item.to_numpy()
    eligible = ordered.eligible.to_numpy(dtype=bool)
    keys, first = np.unique(sessions, return_index=True)
    slot = np.searchsorted(keys, sessions)
    out = {name: np.zeros(len(keys), dtype=np.float64 if "mrr" in name else np.int64)
           for name in ("eligible", "hit_at_1", f"coverage_at_{k}", f"mrr_at_{k}",
                        "novel_eligible", "novel_hit_at_1", f"novel_coverage_at_{k}")}
    seen: set = set()
    current = None
    for n in range(len(rows)):
        if sessions[n] != current:
            current, seen = sessions[n], set()
        seen.add(int(items[n]))
        if not eligible[n]:
            continue
        s, r, target = slot[n], rows[n], int(nexts[n])
        length = min(k, int(table.lengths[r]))
        top = table.items[r, :length]
        found = np.flatnonzero(top == target)
        rank = int(found[0]) + 1 if len(found) else 0
        novel = target not in seen
        out["eligible"][s] += 1
        out["novel_eligible"][s] += novel
        if rank:
            out["hit_at_1"][s] += rank == 1
            out[f"coverage_at_{k}"][s] += 1
            out[f"mrr_at_{k}"][s] += 1.0 / rank
            if novel:
                out["novel_hit_at_1"][s] += rank == 1
                out[f"novel_coverage_at_{k}"][s] += 1
    out["keys"] = keys
    return out


class RatioBootstrap:
    """Paired session bootstrap of per-unit ratio metrics.

    The draws are :class:`webprefetch.benchmark.PairedSessions`' (and so
    ``prefetch_common.stats.session_bootstrap``'s numpy backend):
    ``default_rng(seed)``, one ``integers(0, n, n)`` per rep, the ratio
    ``sum(delta) / sum(denominator)`` and a 95% percentile interval. The
    numerators here may be fractional (reciprocal ranks), which
    ``PairedSessions`` does not accept.
    """

    def __init__(self, keys: np.ndarray, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED,
                 batch: int = 50):
        self.keys, self.reps, self.seed, self.batch = keys, reps, seed, batch
        n = len(keys)
        if not n:
            raise ValueError("bootstrap of no sessions")
        rng = np.random.default_rng(seed)
        counts = np.empty((reps, n), dtype=np.uint16)
        for rep in range(reps):
            counts[rep] = np.bincount(rng.integers(0, n, n), minlength=n)
        self.counts = counts.astype(np.uint8) if int(counts.max()) < 256 else counts

    def resample(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        out = np.empty((self.reps, matrix.shape[1]))
        for start in range(0, self.reps, self.batch):
            out[start:start + self.batch] = (self.counts[start:start + self.batch]
                                             .astype(np.float64) @ matrix)
        return out

    def ci(self, a: np.ndarray, b: np.ndarray, denominator: np.ndarray):
        from prefetch_common.stats import CI
        sums = self.resample(np.column_stack([a - b, denominator]))
        draws = sums[:, 0] / sums[:, 1]
        low, high = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
        estimate = float((a - b).sum() / denominator.sum())
        return CI(estimate, low, high, len(self.keys))


def _spec(name: str, scorer, train_range: str):
    from .scorers import ScorerSpec
    return ScorerSpec(name, "ok", scorer, None, "session", train_range)


def _load(_=None) -> float:
    return round(os.getloadavg()[0], 2)


def run_experiment(prepared: str | Path, model_dir: str | Path, results_dir: str | Path, *,
                   artifact_dir: str | Path | None = None, families=FAMILIES,
                   trials: dict | None = None, train_decisions: int = TRAIN_DECISIONS,
                   tune_decisions: int = TUNE_DECISIONS, validation_views: int = 0,
                   test_views: int = 0, capacities=CAPACITIES, workers: int = 1,
                   reps: int = BOOTSTRAP_REPS, timing_requests: int = 1000,
                   min_positives: int = 100, log=print) -> dict:
    """The whole track: search, select, replay, compare; write every results file.

    ``results_dir`` receives ``tuning.csv``, ``ranking_metrics.csv``,
    ``policy_metrics.csv`` and ``comparisons.csv`` (schemas
    ``web_rankers/*``) and ``summary.json``. Zero view limits mean full
    splits; a limit keeps complete sessions only (a bounded run, which the
    caller must point away from ``results/``).
    """
    from prefetch_common import schemas
    from . import benchmark as B
    from .model import FastScorer, load_artifacts
    from .scorers import build_scorer
    from .simulator import complete_views

    started = time.monotonic()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else Path(model_dir) / "rankers"
    families = tuple(families)
    views = pd.read_csv(prepared)
    artifacts = load_artifacts(model_dir)
    index, shipped = artifacts
    loads = {"start": _load(None)}

    log(f"[rankers] designs: {train_decisions:,} split-1 and {tune_decisions:,} split-2 decisions")
    train = design(sample_decisions(views.loc[views.split.eq(1)], train_decisions), index, workers)
    tune = design(sample_decisions(views.loc[views.split.eq(2)], tune_decisions), index, workers)
    # train_ranker's floor; the synthetic test fixture lowers it.
    if int(train.y.sum()) < min_positives or not tune.y.any():
        raise ValueError("too few positive candidates to train or tune the rankers")

    search_started = time.monotonic()
    tuning = random_search(train, tune, families, trials, workers, log)
    search_s = time.monotonic() - search_started
    reference = ranking_summary(predict(shipped, tune.x), tune)
    tuning_rows = [{"family": "logistic", "trial": -1, "params": {"C": float(
                        shipped.steps[1][1].C)}, "selected": True,
                    "train_decisions": 30000, "tune_decisions": tune_decisions,
                    "tune_eligible": reference["eligible"], "hit_at_1": reference["hit_at_1"],
                    f"coverage_at_{TOP_K}": reference[f"coverage_at_{TOP_K}"],
                    f"mrr_at_{TOP_K}": reference[f"mrr_at_{TOP_K}"],
                    "log_loss": reference["log_loss"], "fit_s": None, "score_s": None,
                    "threads": None, "load_avg": loads["start"],
                    "note": "shipped ranker (models/web/ranker.joblib), not searched; reference"}]
    for row in tuning:
        tuning_rows.append({"family": row["family"], "trial": row["trial"], "params": row["params"],
                            "selected": row["selected"], "train_decisions": train_decisions,
                            "tune_decisions": tune_decisions, "tune_eligible": row["eligible"],
                            "hit_at_1": row["hit_at_1"],
                            f"coverage_at_{TOP_K}": row[f"coverage_at_{TOP_K}"],
                            f"mrr_at_{TOP_K}": row[f"mrr_at_{TOP_K}"], "log_loss": row["log_loss"],
                            "fit_s": row["fit_s"], "score_s": row["score_s"], "threads": 1,
                            "load_avg": row["load_avg"], "note": None})
    schemas.write_rows("web_rankers/tuning.csv", tuning_rows, results_dir / "tuning.csv")

    models, costs = {}, {}
    threads = max(1, min(16, workers))
    chosen = {f: next(r for r in tuning if r["family"] == f and r["selected"]) for f in families}
    records = {f: {"family": f, "params": chosen[f]["params"], "train_range":
                   f"split 1, {train_decisions} evenly spaced decisions; hyperparameters: "
                   f"split 2, {tune_decisions} decisions",
                   "tune_mrr_at_8": chosen[f][f"mrr_at_{TOP_K}"]} for f in families}
    refits = refit_all(chosen, train, tune, artifact_dir, records, workers)
    for family in families:
        model, fit_s, summary, load = refits[family]
        models[family] = model
        # The refit is the searched trial (same seed, one thread): same score.
        same = summary[f"mrr_at_{TOP_K}"] == chosen[family][f"mrr_at_{TOP_K}"]
        costs[family] = {"train_s": fit_s, "train_threads": 1, "train_load": round(load, 2),
                         "params": chosen[family]["params"], "refit_matches_trial": same}
        log(f"[rankers] {family}: refit in {fit_s:.1f} s (1 thread), matches its trial: "
            f"{same}, params {params_text(chosen[family]['params'])}")
    _limit_threads(threads)

    grid = B.policy_grid(feedback=False)
    by_name = {p.name: p for p in grid}
    scorer_names = ("logistic", *families)
    ranges = {"logistic": "shipped ranker: split 1, 30000 decisions, C on split 2; index: split 0"}
    ranges.update({f: f"split 1, {train_decisions} decisions; hyperparameters on split 2 "
                      f"({tune_decisions} decisions); index: split 0" for f in families})
    limits = {"validation": validation_views, "test": test_views}
    rows = {label: {} for label in SPLIT_LABELS}
    extras = {label: {} for label in SPLIT_LABELS}
    selection: dict = {}
    ranking: dict = {}
    e4_check = None
    boot = B.PairedSessions(reps)
    for label, split in SPLIT_LABELS.items():
        frame = complete_views(views.loc[views.split.eq(split)], limits[label])
        log(f"[rankers] {label}: {len(frame):,} views; scoring every family in a batch")
        full = design(frame, index, workers) if families else None
        for name in scorer_names:
            if name == "logistic":
                spec = build_scorer("logistic", views, split, model_dir, artifacts=artifacts)
            else:
                t0 = time.perf_counter()
                probs = predict(models[name], full.x)
                batch_s = time.perf_counter() - t0
                if label == "test":
                    costs[name].update(batch_us_per_request=batch_s / max(1, len(frame)) * 1e6,
                                       batch_load=_load(None))
                spec = _spec(name, array_scorer(frame, full, probs), ranges[name])
                del probs
            result = B.replay_split(frame, spec, grid, capacities, workers=workers,
                                    session_caps=capacities if label == "test" else ())
            rows[label][name], extras[label][name] = result.rows, result.extra
            if label == "test" and name == "logistic" and len(frame) == E4_CHECK["views"]:
                row = result.rows[E4_CHECK["capacity"]][E4_CHECK["policy"]]
                e4_check = {"hits": row["hits"], "issued": row["issued"], "expected": E4_CHECK,
                            "match": (row["hits"], row["issued"]) == (E4_CHECK["hits"],
                                                                      E4_CHECK["issued"])}
                log(f"[rankers] E4 check: {e4_check}")
                if not e4_check["match"]:
                    raise ValueError(f"the shipped ranker does not reproduce E4: {e4_check}")
            if label == "validation":
                selection[name] = B.select_policies(result.rows, grid)
            else:
                ranking[name] = session_ranking(result.frame, result.table)
                needed = {"fixed0", "fixed1", "fixed4"}
                for cap in capacities:
                    for families_chosen in selection[name][cap].values():
                        needed.update(families_chosen.values())
                for cap in capacities:
                    for policy in sorted(needed):
                        boot.add((name, cap, policy), result.arrays[(cap, policy)])
            log(f"[rankers] {label} {name}: hit@1 {result.extra['ranking_hit_at_1']:.5f}, "
                f"MRR@8 {result.extra['ranking_mrr_at_8']:.5f}")
            del result, spec
        del full
    loads["replay"] = _load(None)

    # Baselines that only the test comparisons need.
    test_frame = complete_views(views.loc[views.split.eq(3)], test_views)
    for name, policies in (("trans_s0", ["fixed0", "fixed1"]), ("trans_online", ["fixed0", "fixed4"])):
        spec = build_scorer(name, views, 3, model_dir, artifacts=artifacts)
        result = B.replay_split(test_frame, spec, [by_name[p] for p in policies], capacities,
                                workers=workers, session_caps=capacities)
        rows["test"][name], extras["test"][name] = result.rows, result.extra
        ranking[name] = session_ranking(result.frame, result.table)
        for cap in capacities:
            for policy in policies:
                boot.add((name, cap, policy), result.arrays[(cap, policy)])
        ranges[name] = spec.train_range
        log(f"[rankers] test {name}: hit@1 {result.extra['ranking_hit_at_1']:.5f}")
        del result

    # Inference cost per request (contended host; recorded with its load).
    timing_frame = test_frame.iloc[:timing_requests]
    shipped_us = timing_per_request(shipped, timing_frame, index, timing_requests)
    fast = FastScorer(shipped)
    t0 = time.perf_counter()
    for row in timing_frame.itertuples(index=False):
        fast.score(row, index)
    fast_us = (time.perf_counter() - t0) / max(1, len(timing_frame)) * 1e6
    costs["logistic"] = {"train_s": None, "train_threads": None, "train_load": None,
                         "params": {"C": float(shipped.steps[1][1].C)},
                         "request_us": fast_us, "request_load": _load(None),
                         "sklearn_request_us": shipped_us}
    for family in families:
        costs[family]["request_us"] = timing_per_request(models[family], timing_frame, index,
                                                         timing_requests)
        costs[family]["request_load"] = _load(None)

    # ------------------------------------------------------------ tables
    metric_rows = []
    for label in SPLIT_LABELS:
        for name, by_cap in rows[label].items():
            metric_rows += B.policy_metric_rows(label, name, by_cap)
    schemas.write_rows("web_rankers/policy_metrics.csv", metric_rows,
                       results_dir / "policy_metrics.csv")

    ranking_rows = []
    for label in SPLIT_LABELS:
        for name, extra in extras[label].items():
            cost = costs.get(name, {})
            ranking_rows.append({
                "split": label, "scorer": name, "train_range": ranges[name],
                "params": cost.get("params"), "eligible": extra["oracle_eligible_pairs"],
                "hit_at_1": extra["ranking_hit_at_1"],
                f"coverage_at_{TOP_K}": extra["oracle_candidate_coverage_at_8"],
                f"mrr_at_{TOP_K}": extra["ranking_mrr_at_8"],
                "novel_eligible": extra["novel_eligible_pairs"],
                "novel_hit_at_1": extra["novel_hit_at_1"],
                f"novel_coverage_at_{TOP_K}": extra["novel_coverage_at_8"],
                "views": extra["replayed_views"],
                "train_s": cost.get("train_s"), "train_threads": cost.get("train_threads"),
                "batch_us_per_request": cost.get("batch_us_per_request") if label == "test" else None,
                "request_us": cost.get("request_us") if label == "test" else None,
                "load_avg": cost.get("request_load") if label == "test" else loads["replay"],
                "timing_note": "CONTENDED shared host; wall time, not a cost claim"})
    schemas.write_rows("web_rankers/ranking_metrics.csv", ranking_rows,
                       results_dir / "ranking_metrics.csv")

    comparisons = comparison_table(families, capacities, selection, rows, ranking, boot, reps)
    schemas.write_rows("web_rankers/comparisons.csv", comparisons, results_dir / "comparisons.csv")
    summary = {
        "families": list(families), "baselines": list(BASELINES), "capacities": list(capacities),
        "trials": dict(TRIALS if trials is None else trials), "search_seed": SEARCH_SEED,
        "train_decisions": train_decisions, "tune_decisions": tune_decisions,
        "train_positives": int(train.y.sum()), "train_candidate_rows": int(len(train.y)),
        "tune_positives": int(tune.y.sum()), "tune_candidate_rows": int(len(tune.y)),
        "validation_views": int(extras["validation"]["logistic"]["replayed_views"]),
        "test_views": int(extras["test"]["logistic"]["replayed_views"]),
        "partial_run": bool(validation_views or test_views),
        "selection": {name: {str(cap): sel for cap, sel in selection[name].items()}
                      for name in selection},
        "selected_family": selected_family(selection, rows, capacities),
        "bootstrap": {"reps": reps, "seed": BOOTSTRAP_SEED, "sessions": boot.sessions},
        "search_s": round(search_s, 1), "costs": costs, "load_avg": loads,
        "e4_check": e4_check,
        "workers": workers, "wall_s": round(time.monotonic() - started, 1),
        "timing_note": "every timing is wall time on a contended shared host",
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                              encoding="utf-8")
    log(f"[rankers] done in {summary['wall_s']} s")
    return summary


def selected_family(selection: dict, rows: dict, capacities=CAPACITIES,
                    cost: float = HEADLINE_COST) -> dict:
    """Per capacity, the scorer whose validated adaptive policy has the highest
    *validation* U(cost); ties go to fewer issued prefetches, then the name."""
    from prefetch_common.utility import row_U
    from .benchmark import cost_key
    out = {}
    for cap in capacities:
        def key(name):
            policy = selection[name][cap][cost_key(cost)]["adaptive"]
            row = rows["validation"][name][cap][policy]
            return (row_U(row, cost), -row["issued"], name)
        out[str(cap)] = max(selection, key=key)
    return out


def comparison_table(families, capacities, selection, rows, ranking, boot, reps) -> list[dict]:
    """Every pre-registered comparison (``results/web_rankers/PREREGISTERED.md``)."""
    from .benchmark import cost_key, verdict
    key = cost_key(HEADLINE_COST)
    out = []

    def add(cid, headline, a, b, cap, metric, cost, ci, label=None, note=None):
        out.append({"comparison": cid, "headline": headline, "scorer": a[0], "policy": a[2],
                    "baseline_scorer": b[0], "baseline_policy": b[2], "split": "test",
                    "capacity": cap, "metric": metric, "cost": cost,
                    "estimate": None if ci is None else ci.estimate,
                    "ci_lo": None if ci is None else ci.low,
                    "ci_hi": None if ci is None else ci.high,
                    "sessions": None if ci is None else ci.n,
                    "verdict": label if ci is None else three_way(ci),
                    "label": label, "reps": reps, "seed": BOOTSTRAP_SEED, "note": note})

    def pair(cid, headline, a, b, cap, note=None, metrics=(("hit_ratio", 0.0),
                                                          ("utility_per_view", HEADLINE_COST))):
        for metric, cost in metrics:
            add(cid, headline, a, b, cap, metric, cost, boot.ci(a, b, cost), note=note)

    # S1: the validation-selected scorer per capacity against the shipped ranker.
    chosen = selected_family(selection, rows, capacities)
    for cap in capacities:
        winner = chosen[str(cap)]
        if winner == "logistic":
            add("S1", True, ("logistic", cap, selection["logistic"][cap][key]["adaptive"]),
                ("logistic", cap, selection["logistic"][cap][key]["adaptive"]), cap,
                "utility_per_view", HEADLINE_COST, None,
                label="no alternative selected: validation chose the shipped logistic ranker")
            continue
        pair("S1", True, (winner, cap, selection[winner][cap][key]["adaptive"]),
             ("logistic", cap, selection["logistic"][cap][key]["adaptive"]), cap,
             note="validation-selected scorer's validated adaptive minus the shipped logistic's")
    for name in ("logistic", *families):
        mine = selection[name]
        for cap in capacities:
            if name != "logistic":
                # K2: fixed depth 1; K3: each scorer's own validated adaptive.
                pair("K2", True, (name, cap, "fixed1"), ("logistic", cap, "fixed1"), cap)
                pair("K3", True, (name, cap, mine[cap][key]["adaptive"]),
                     ("logistic", cap, selection["logistic"][cap][key]["adaptive"]), cap)
            # K4: against transition top-1; K5: against online counts (depth 4).
            pair("K4", name != "logistic", (name, cap, "fixed1"), ("trans_s0", cap, "fixed1"), cap)
            pair("K5", name != "logistic", (name, cap, "fixed4"), ("trans_online", cap, "fixed4"),
                 cap)
            # K6: the controller rule (E4's W1 and W2) for each scorer.
            adaptive, fixed = mine[cap][key]["adaptive"], mine[cap][key]["fixed"]
            vs_best = boot.ci((name, cap, adaptive), (name, cap, fixed), HEADLINE_COST)
            vs_none = boot.ci((name, cap, adaptive), (name, cap, "fixed0"), HEADLINE_COST)
            add("K6", name != "logistic", (name, cap, adaptive), (name, cap, fixed), cap,
                "utility_per_view", HEADLINE_COST, vs_best, label=verdict("adaptive", vs_best, vs_none),
                note="validated adaptive minus validated best fixed depth (same scorer, capacity)")
            add("K6", name != "logistic", (name, cap, adaptive), (name, cap, "fixed0"), cap,
                "utility_per_view", HEADLINE_COST, vs_none, label=verdict("adaptive", vs_best, vs_none),
                note="validated adaptive minus no prefetching (decision rule 1)")
    # K1: ranking quality on the test next views, every scorer against logistic.
    base = ranking["logistic"]
    eligible_keys = base["keys"][base["eligible"] > 0]
    ratio = RatioBootstrap(eligible_keys, reps)
    for name in (*families, "trans_s0", "trans_online"):
        mine = ranking[name]
        if not np.array_equal(mine["keys"], base["keys"]):
            raise ValueError(f"{name} ranked a different session set")
        keep = base["eligible"] > 0
        for metric, denominator in RANKING_METRICS:
            ci = ratio.ci(mine[metric][keep].astype(np.float64), base[metric][keep].astype(np.float64),
                          base[denominator][keep].astype(np.float64))
            add("K1", name in families, (name, None, "ranking"), ("logistic", None, "ranking"), None,
                metric, 0.0, ci, note=f"per {denominator.replace('_', ' ')} next view; "
                                      "sessions with at least one eligible next view")
    return out


# ----------------------------------------------------------------- cli

def main(argv=None) -> int:
    import argparse
    from .paths import MODELS, OUTPUTS, PROCESSED
    parser = argparse.ArgumentParser(prog="python -m webprefetch.rankers", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--results-dir", type=Path, default=OUTPUTS / "rankers")
    parser.add_argument("--artifacts", type=Path, default=None,
                        help="where the fitted rankers go (default <models>/rankers)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--trials", default="", help="family=n,... overrides of TRIALS")
    parser.add_argument("--train-decisions", type=int, default=TRAIN_DECISIONS)
    parser.add_argument("--tune-decisions", type=int, default=TUNE_DECISIONS)
    parser.add_argument("--validation-views", type=int, default=0)
    parser.add_argument("--test-views", type=int, default=0)
    args = parser.parse_args(argv)
    families = tuple(f for f in args.families.split(",") if f)
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        parser.error(f"unknown families {unknown}; expected some of {FAMILIES}")
    trials = dict(TRIALS)
    for part in filter(None, args.trials.split(",")):
        name, _, n = part.partition("=")
        trials[name] = int(n)
    bounded = bool(args.validation_views or args.test_views or trials != TRIALS
                   or args.train_decisions != TRAIN_DECISIONS
                   or args.tune_decisions != TUNE_DECISIONS or args.reps != BOOTSTRAP_REPS
                   or families != FAMILIES)
    if bounded and Path(args.results_dir).resolve().parts[-2:] == ("results", "web_rankers"):
        parser.error("a bounded or reduced run never writes results/web_rankers")
    run_experiment(args.processed / "views.csv.gz", args.models, args.results_dir,
                   artifact_dir=args.artifacts, families=families, trials=trials,
                   train_decisions=args.train_decisions, tune_decisions=args.tune_decisions,
                   validation_views=args.validation_views, test_views=args.test_views,
                   workers=args.workers, reps=args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

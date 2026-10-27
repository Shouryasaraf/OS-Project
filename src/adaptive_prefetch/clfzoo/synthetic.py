"""Part 1: tune every family on training seeds, score it once on test seeds.

Selection (``results/classifier/PREREGISTERED.md`` section 2): for each
setting and family, every random-search candidate is scored on 12 folds;
fold ``i`` fits on training seed ``1000+i`` and validates on training seed
``1000+(i+1) mod 12``. The candidate with the highest mean validation
accuracy is selected (ties: the earlier candidate, so the library default
wins a tie). Test seeds ``900000+i`` are touched only by :func:`evaluate`,
once, with the selected candidate refitted on training seed ``1000+i``.
"""

from __future__ import annotations

import json
import os
from time import perf_counter

from ..trace import CLASSES
from . import data, zoo

#: Filled in the parent before the pool forks, read by the workers.
_VECTORS: dict[tuple, list] = {}


def _load1() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


def _dataset_job(key):
    kind, setting, condition, seed = key
    if kind == "train":
        return key, data.train_set(setting, seed)
    return key, data.test_set(setting, condition, seed)


def dataset_keys(settings) -> list[tuple]:
    keys = []
    for setting in settings:
        for seed in data.TRAIN_SEEDS:
            keys.append(("train", setting, None, seed))
        for condition in data.test_conditions(setting):
            for seed in data.TEST_SEEDS:
                keys.append(("test", setting, condition, seed))
    return keys


def build_vectors(settings, pool=None) -> dict[tuple, list]:
    keys = dataset_keys(settings)
    results = (pool.imap_unordered(_dataset_job, keys, chunksize=4) if pool is not None
               else map(_dataset_job, keys))
    return dict(results)


def _split(rows):
    return [values for values, _ in rows], [label for _, label in rows]


def scores(labels, probs) -> dict:
    """Accuracy, macro-F1, confusion, ECE and Brier of one test set."""
    import numpy as np
    index = {label: i for i, label in enumerate(CLASSES)}
    truth = np.array([index[label] for label in labels])
    probs = np.asarray(probs, dtype=float)
    predicted = probs.argmax(axis=1)
    k = len(CLASSES)
    confusion = np.zeros((k, k), dtype=int)
    for t, p in zip(truth, predicted):
        confusion[t, p] += 1
    f1s = []
    for c in range(k):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    confidence = probs.max(axis=1)
    correct = (predicted == truth).astype(float)
    # Expected calibration error, 15 equal-width bins on the top posterior.
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = 0.0
    for b in range(15):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    onehot = np.eye(k)[truth]
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    return {"accuracy": float(correct.mean()), "macro_f1": float(np.mean(f1s)),
            "confusion": confusion.tolist(), "ece": float(ece), "brier": brier}


def _quiet():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass


def _tune_job(task):
    _quiet()
    setting, family, index, params = task
    accuracies, f1s = [], []
    started = perf_counter()
    error = None
    try:
        for i, seed in enumerate(data.TRAIN_SEEDS):
            rows, labels = _split(_VECTORS[("train", setting, None, seed)])
            val_rows, val_labels = _split(
                _VECTORS[("train", setting, None, data.validation_seed(i))])
            router = zoo.fit(family, params, rows, labels, seed=i)
            result = scores(val_labels, router.proba_batch(val_rows))
            accuracies.append(result["accuracy"])
            f1s.append(result["macro_f1"])
    except Exception as exc:  # noqa: BLE001 - a candidate that cannot fit is recorded
        # A candidate that fails on any fold (QDA with a singular class
        # covariance, say) scores 0 and is never selected over one that fits.
        error = f"{type(exc).__name__}: {exc}"[:200]
        accuracies, f1s = [0.0], [0.0]
    return {"setting": setting, "family": family, "candidate": index,
            "params": params, "folds": data.SEED_PAIRS,
            "val_accuracy": sum(accuracies) / len(accuracies),
            "val_macro_f1": sum(f1s) / len(f1s), "error": error,
            "wall_s": round(perf_counter() - started, 3), "load_avg": _load1()}


def tuning_tasks(settings, families=zoo.FAMILIES, n_candidates=zoo.N_CANDIDATES):
    tasks = []
    for setting in settings:
        for family in families:
            for index, params in enumerate(zoo.candidates(family, n_candidates)):
                tasks.append((setting, family, index, params))
    return tasks


def select(tuning_rows) -> dict[tuple, dict]:
    """(setting, family) -> the selected tuning row (highest mean validation
    accuracy; ties go to the lower candidate index)."""
    best: dict[tuple, dict] = {}
    for row in sorted(tuning_rows, key=lambda r: (r["setting"], r["family"], r["candidate"])):
        key = (row["setting"], row["family"])
        if key not in best or row["val_accuracy"] > best[key]["val_accuracy"]:
            best[key] = row
    return best


def _eval_job(task):
    _quiet()
    setting, family, params = task
    out = []
    for i, seed in enumerate(data.TRAIN_SEEDS):
        rows, labels = _split(_VECTORS[("train", setting, None, seed)])
        start = perf_counter()
        router = zoo.fit(family, params, rows, labels, seed=i)
        train_ms = (perf_counter() - start) * 1000
        for condition in data.test_conditions(setting):
            test_rows, test_labels = _split(
                _VECTORS[("test", setting, condition, data.TEST_SEEDS[i])])
            result = scores(test_labels, router.proba_batch(test_rows))
            # The posterior's argmax must be the deployed predict() label.
            deployed = router.predict_batch(test_rows)
            assert all(CLASSES[max(range(len(CLASSES)), key=p.__getitem__)] == d
                       for p, d in zip(router.proba_batch(test_rows).tolist(), deployed)), \
                f"{family}: predict_batch disagrees with proba_batch"
            result.update({
                "setting": setting, "condition": condition, "family": family,
                "seed_pair": i, "train_seed": seed, "test_seed": data.TEST_SEEDS[i],
                "n_test": len(test_labels), "train_ms": train_ms,
                "predict_us": zoo.predict_cost_us(router, test_rows),
                "load_avg": _load1()})
            out.append(result)
    return out


def run(settings=data.ALL_SETTINGS, families=zoo.FAMILIES, workers: int = 20,
        n_candidates: int = zoo.N_CANDIDATES, progress=print) -> dict:
    """Build vectors, tune, select, evaluate. Returns the raw records."""
    import multiprocessing
    multiprocessing = multiprocessing.get_context("fork")
    global _VECTORS
    started = perf_counter()
    with multiprocessing.Pool(workers) as pool:
        _VECTORS = build_vectors(settings, pool)
    progress(f"vectors: {len(_VECTORS)} sets in {perf_counter() - started:.1f}s")
    tasks = tuning_tasks(settings, families, n_candidates)
    # Longest families first so the pool's tail is short.
    heavy = {"svm_linear": 0, "rf": 1, "extra_trees": 1, "hgb": 0, "mlp": 0}
    tasks.sort(key=lambda t: heavy.get(t[1], 9))
    tuning = []
    with multiprocessing.Pool(workers) as pool:
        for n, row in enumerate(pool.imap_unordered(_tune_job, tasks, chunksize=1), 1):
            tuning.append(row)
            if n % 100 == 0 or n == len(tasks):
                progress(f"tuning {n}/{len(tasks)} ({perf_counter() - started:.0f}s)")
    chosen = select(tuning)
    for row in tuning:
        row["selected"] = chosen[(row["setting"], row["family"])] is row
    eval_tasks = [(s, f, chosen[(s, f)]["params"]) for s in settings for f in families]
    records = []
    with multiprocessing.Pool(workers) as pool:
        for n, rows in enumerate(pool.imap_unordered(_eval_job, eval_tasks, chunksize=1), 1):
            records.extend(rows)
    progress(f"evaluation: {len(records)} seed-pair records "
             f"({perf_counter() - started:.0f}s)")
    return {"tuning": tuning, "chosen": chosen, "records": records}


def params_text(params: dict) -> str:
    return json.dumps(params, sort_keys=True)

"""MSRC light sample (E7): a labelled-window transfer check and a sanity replay.

The MSRC-trace-003 ``.revised`` files carry a per-request ``seq``/``rand``
flag (``Request.pattern``), the only real workload label in the project. This
module asks whether a classifier trained on synthetic windows, as
``adaptive`` is, agrees with that flag, and replays six modes on the same
files. The procedure, label rule and decision words are fixed in
``results/msrc_samples/PREREGISTERED.md``; read it before this code.

The flag is used for scoring only: it never reaches a feature, a classifier
input or candidate generation.

The stdlib parts (windows, labels, metrics, the stdlib bootstrap, the
replay) import nothing optional. The sklearn families and the numpy
resampler are imported lazily, so the disk core stays stdlib-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
from collections import Counter
from pathlib import Path

from .features import WindowContext, dominant_stride
from .simulator import LatencyModel, replay, route_window
from .trace import CLASSES, Request, load_csv, synthetic_window

WINDOW = 32
SEED = 42
REPS = 2000
EXAMPLES_PER_CLASS = 100
LABEL_RULES = {"majority": 17, "quarter": 8}  # min seq-flagged requests of 32
POSITIVE = "seq"
NEGATIVE = "not_seq"
SMALL_N_WINDOWS = 100
SMALL_N_CLASS = 10
CAPACITY = 2048
REPLAY_MODES = (("none", None), ("sequential", None), ("adaptive", None),
                ("adaptive_evidence", None), ("deep", 8), ("linux_ra", 256))
VIEWS = ("native", "writes_as_reads")
LATENCY_MODELS = ("measured", "assumed")
SKLEARN_FAMILIES = ("gaussian_nb", "lda", "qda", "logistic", "knn", "svm_rbf",
                    "decision_tree", "random_forest", "extra_trees",
                    "hist_gradient_boosting", "mlp")
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "samples" / "msrc"


# ---------------------------------------------------------------- data

def discover(root: str | Path = SAMPLE_DIR) -> list[Path]:
    """Every ``*.revised`` file under ``root``, sorted by name."""
    return sorted(Path(root).glob("*.revised"))


def family(stem: str) -> str:
    """MSRC host family: the stem before the last ``_`` (``web_3`` -> ``web``)."""
    return stem.rsplit("_", 1)[0]


@dataclasses.dataclass
class TraceWindows:
    name: str
    vectors: list[tuple[float, ...]]
    seq_counts: list[int]
    routes: list[str]
    evictions: int


def trace_windows(requests: list[Request], name: str = "trace",
                  window: int = WINDOW) -> TraceWindows:
    """Non-overlapping windows with the exact contextual vector ``adaptive``
    builds (one ``WindowContext`` per trace, every window in order)."""
    context = WindowContext()
    vectors, seq_counts, routes = [], [], []
    for start in range(0, len(requests) - window + 1, window):
        chunk = requests[start:start + window]
        stride = dominant_stride(chunk)
        vectors.append(context.observe(chunk, stride))
        seq_counts.append(sum(r.pattern == "seq" for r in chunk))
        routes.append(route_window(chunk, stride))
    return TraceWindows(name, vectors, seq_counts, routes, context.evictions)


def label(seq_count: int, rule: str) -> str:
    return POSITIVE if seq_count >= LABEL_RULES[rule] else NEGATIVE


def to_binary(predicted: str) -> str:
    """Synthetic class or route -> flag space (PREREGISTERED section 1)."""
    return POSITIVE if predicted == "sequential" else NEGATIVE


def synthetic_training_set(seed: int = SEED,
                           examples_per_class: int = EXAMPLES_PER_CLASS):
    """``(X, y)``: exactly the windows ``benchmark.make_model`` fits, in order."""
    X, y = [], []
    for name in CLASSES:
        rng = random.Random(seed)
        context = WindowContext()
        flat: list[Request] = []
        time = 0.0
        for _ in range(examples_per_class):
            chunk = synthetic_window(name, rng, WINDOW, time)
            flat.extend(chunk)
            time = chunk[-1].timestamp_ms
        for start in range(0, len(flat), WINDOW):
            chunk = flat[start:start + WINDOW]
            if len(chunk) < WINDOW:
                break
            X.append(context.observe(chunk, dominant_stride(chunk)))
            y.append(name)
    return X, y


# ------------------------------------------------------------- metrics

def counts(truth, pred) -> tuple[int, int, int, int]:
    """``(tp, fp, fn, tn)`` with ``seq`` positive."""
    tp = fp = fn = tn = 0
    for t, p in zip(truth, pred):
        if p == POSITIVE:
            if t == POSITIVE:
                tp += 1
            else:
                fp += 1
        elif t == POSITIVE:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Accuracy, balanced accuracy and macro-F1; the last two are ``None``
    when the truth holds one class only."""
    n = tp + fp + fn + tn
    pos, neg = tp + fn, fp + tn
    out = {"accuracy": (tp + tn) / n if n else None,
           "balanced_accuracy": None, "macro_f1": None,
           "seq_precision": tp / (tp + fp) if tp + fp else None,
           "seq_recall": tp / pos if pos else None}
    if pos and neg:
        out["balanced_accuracy"] = 0.5 * (tp / pos + tn / neg)
        f_pos = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        f_neg = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
        out["macro_f1"] = 0.5 * (f_pos + f_neg)
    return out


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default rule)."""
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def _interval(draws) -> tuple[float | None, float | None]:
    draws = [d for d in draws if d is not None and not math.isnan(d)]
    if not draws:
        return None, None
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


class Resampler:
    """Window-bootstrap replicates shared by every classifier on one set of
    windows, so paired differences use the same draws.

    ``backend="numpy"`` draws all index arrays at once (the full run);
    ``"stdlib"`` uses :class:`random.Random` (tests, no extras). They give
    different draws for one seed, so the backend is recorded in provenance.
    """

    def __init__(self, n: int, reps: int = REPS, seed: int = SEED,
                 backend: str = "auto"):
        if backend == "auto":
            try:
                import numpy  # noqa: F401
                backend = "numpy"
            except ImportError:
                backend = "stdlib"
        self.n, self.reps, self.backend = n, reps, backend
        if backend == "numpy":
            import numpy as np
            self._np = np
            self.index = np.random.default_rng(seed).integers(0, n, size=(reps, n))
        else:
            rng = random.Random(seed)
            self.index = [[rng.randrange(n) for _ in range(n)] for _ in range(reps)]

    def _cells(self, truth, pred):
        """Per replicate ``(tp, fp, fn, tn)``."""
        if self.backend == "numpy":
            np = self._np
            t = np.asarray([x == POSITIVE for x in truth])[self.index]
            p = np.asarray([x == POSITIVE for x in pred])[self.index]
            tp = (t & p).sum(1)
            fp = (~t & p).sum(1)
            fn = (t & ~p).sum(1)
            tn = (~t & ~p).sum(1)
            return list(zip(tp.tolist(), fp.tolist(), fn.tolist(), tn.tolist()))
        return [counts([truth[i] for i in row], [pred[i] for i in row])
                for row in self.index]

    def draws(self, truth, pred) -> list[dict]:
        return [metrics(*cells) for cells in self._cells(truth, pred)]


def summarise(truth, pred, resampler: Resampler | None, draws=None) -> dict:
    """Point metrics with their window-bootstrap intervals."""
    point = metrics(*counts(truth, pred))
    if draws is None:
        draws = resampler.draws(truth, pred) if resampler is not None else []
    out = dict(point)
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        lo, hi = (_interval([d[key] for d in draws]) if point[key] is not None
                  else (None, None))
        out[f"{key}_lo"], out[f"{key}_hi"] = lo, hi
    return out


def paired_delta(truth, pred_a, pred_b, resampler: Resampler,
                 key: str = "balanced_accuracy", draws_a=None, draws_b=None):
    """``(a - b, lo, hi)`` over the same replicates; ``None`` when undefined."""
    a = metrics(*counts(truth, pred_a))[key]
    b = metrics(*counts(truth, pred_b))[key]
    if a is None or b is None:
        return None, None, None
    da = draws_a if draws_a is not None else resampler.draws(truth, pred_a)
    db = draws_b if draws_b is not None else resampler.draws(truth, pred_b)
    lo, hi = _interval([x[key] - y[key] for x, y in zip(da, db)
                        if x[key] is not None and y[key] is not None])
    return a - b, lo, hi


def small_n(truth) -> bool:
    c = Counter(truth)
    return (len(truth) < SMALL_N_WINDOWS or c[POSITIVE] < SMALL_N_CLASS
            or c[NEGATIVE] < SMALL_N_CLASS)


def verdict(delta, lo, hi, is_small: bool) -> str:
    """PREREGISTERED section 4."""
    if is_small or delta is None or lo is None:
        return "insufficient n"
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "no difference detected"


# ------------------------------------------------------ classifiers

def sklearn_family(name: str):
    """A fresh estimator for one pre-registered family (lazy sklearn import)."""
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    def scaled(model):
        return make_pipeline(StandardScaler(), model)

    factories = {
        "gaussian_nb": lambda: GaussianNB(),
        "lda": lambda: LinearDiscriminantAnalysis(),
        "qda": lambda: QuadraticDiscriminantAnalysis(reg_param=0.01),
        "logistic": lambda: scaled(LogisticRegression(max_iter=1000, random_state=0)),
        "knn": lambda: scaled(KNeighborsClassifier(n_neighbors=5)),
        "svm_rbf": lambda: scaled(SVC(kernel="rbf", random_state=0)),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=0),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0,
                                                        n_jobs=1),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=0,
                                                    n_jobs=1),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=0),
        "mlp": lambda: scaled(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                            random_state=0)),
    }
    return factories[name]()


def _fit_predict(name, X_train, y_train, X_test):
    import warnings
    model = sklearn_family(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
        return [str(p) for p in model.predict(X_test)]


def shipped_predictions(windows: list[TraceWindows], classifier: str):
    """Four-class predictions of ``benchmark.make_model(42, classifier)``.

    The model is frozen (``online_updates=False`` in replay), so one fitted
    instance predicts every window exactly as a replay would.
    """
    from .benchmark import make_model
    model = make_model(SEED, classifier=classifier)
    return {w.name: [model.predict(v)[0] for v in w.vectors] for w in windows}


def classifier_study(windows: list[TraceWindows], *, reps: int = REPS,
                     backend: str = "auto", sklearn: bool = True):
    """``(metric_rows, delta_rows)`` for every protocol, trace and label rule."""
    names = [w.name for w in windows]
    by_name = {w.name: w for w in windows}
    # predictions[(protocol, classifier, rule)][trace] = (binary list, raw list) or skip
    preds: dict[tuple, dict] = {}

    def put(protocol, clf, rule, trace, raw, train_on):
        preds.setdefault((protocol, clf, rule), {})[trace] = (
            [to_binary(p) if p not in (POSITIVE, NEGATIVE) else p for p in raw],
            raw, train_on)

    for rule in LABEL_RULES:
        for clf in ("gnb", "qda"):
            for trace, raw in shipped_predictions(windows, clf).items():
                put("shipped_synthetic", clf, rule, trace, raw,
                    "make_model(seed=42), synthetic")
        for w in windows:
            put("rule", "route_window", rule, w.name, list(w.routes), "none (rule)")
            put("baseline", "always_not_seq", rule, w.name,
                [NEGATIVE] * len(w.vectors), "none (constant)")

    if sklearn:
        X_syn, y_syn = synthetic_training_set()
        for clf in SKLEARN_FAMILIES:
            fitted = None
            for rule in LABEL_RULES:
                for w in windows:
                    if fitted is None:
                        import warnings
                        fitted = sklearn_family(clf)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fitted.fit(X_syn, y_syn)
                    raw = [str(p) for p in fitted.predict(w.vectors)] if w.vectors else []
                    put("sklearn_synthetic", clf, rule, w.name, raw,
                        "make_model windows (seed 42, 100/class), synthetic")
        folds = {"loto_real": [(n, [n]) for n in names],
                 "lofo_real": [(f, [n for n in names if family(n) == f])
                               for f in sorted({family(n) for n in names})]}
        for protocol, fold_list in folds.items():
            for rule in LABEL_RULES:
                for held, tests in fold_list:
                    train = [by_name[n] for n in names if n not in tests]
                    X = [v for w in train for v in w.vectors]
                    y = [label(c, rule) for w in train for c in w.seq_counts]
                    desc = "real windows of " + ",".join(w.name for w in train)
                    majority = Counter(y).most_common(1)[0][0] if y else NEGATIVE
                    for n in tests:
                        put(protocol, "majority", rule, n,
                            [majority] * len(by_name[n].vectors), desc)
                    for clf in SKLEARN_FAMILIES:
                        if len(set(y)) < 2:
                            for n in tests:
                                preds.setdefault((protocol, clf, rule), {})[n] = (
                                    None, f"skipped: training folds hold one class ({desc})",
                                    desc)
                            continue
                        X_test = [v for n in tests for v in by_name[n].vectors]
                        try:
                            out = _fit_predict(clf, X, y, X_test)
                        except Exception as error:  # a degenerate fit is a skip
                            for n in tests:
                                preds.setdefault((protocol, clf, rule), {})[n] = (
                                    None, f"skipped: {type(error).__name__}: {error}", desc)
                            continue
                        offset = 0
                        for n in tests:
                            k = len(by_name[n].vectors)
                            put(protocol, clf, rule, n, out[offset:offset + k], desc)
                            offset += k

    # Truth and resamplers per (rule, scope); scope is a trace or "pooled".
    scopes = names + ["pooled"]
    truth = {}
    resamplers = {}
    for rule in LABEL_RULES:
        for scope in scopes:
            members = names if scope == "pooled" else [scope]
            truth[(rule, scope)] = [label(c, rule) for n in members
                                    for c in by_name[n].seq_counts]
    for scope in scopes:
        n = len(truth[("majority", scope)])
        resamplers[scope] = Resampler(n, reps, SEED, backend) if n else None

    def scope_preds(entry, scope):
        members = names if scope == "pooled" else [scope]
        if any(entry[m][0] is None for m in members):
            reason = next(entry[m][1] for m in members if entry[m][0] is None)
            return None, None, reason, entry[members[0]][2]
        binary = [p for m in members for p in entry[m][0]]
        raw = [p for m in members for p in entry[m][1]]
        train = entry[members[0]][2] if scope != "pooled" else (
            entry[members[0]][2] if len({entry[m][2] for m in members}) == 1
            else "per-trace folds")
        return binary, raw, "ok", train

    cache: dict[tuple, list] = {}

    def draws_for(key, scope, t, binary):
        if (key, scope) not in cache:
            cache[(key, scope)] = resamplers[scope].draws(t, binary)
        return cache[(key, scope)]

    rows, deltas = [], []
    for (protocol, clf, rule), entry in preds.items():
        for scope in scopes:
            t = truth[(rule, scope)]
            binary, raw, status, train = scope_preds(entry, scope)
            flagged = small_n(t)
            row = {"protocol": protocol, "classifier": clf, "label_rule": rule,
                   "trace": scope, "status": status, "train_on": train,
                   "n_windows": len(t), "n_seq": t.count(POSITIVE),
                   "small_n": flagged, "reps": reps, "seed": SEED}
            keys = ("accuracy", "accuracy_lo", "accuracy_hi", "balanced_accuracy",
                    "balanced_accuracy_lo", "balanced_accuracy_hi", "macro_f1",
                    "macro_f1_lo", "macro_f1_hi", "seq_precision", "seq_recall")
            if binary is None or not t:
                row.update({k: None for k in keys})
                row.update(predicted_seq=None, predicted_classes=None, above_chance=None)
                if not t:
                    row["status"] = "n/a: no complete window"
                rows.append(row)
                continue
            mine = draws_for((protocol, clf, rule), scope, t, binary)
            stats = summarise(t, binary, resamplers[scope], mine)
            row.update({k: stats[k] for k in keys})
            row["predicted_seq"] = binary.count(POSITIVE)
            row["predicted_classes"] = dict(sorted(Counter(raw).items()))
            lo = stats["balanced_accuracy_lo"]
            row["above_chance"] = (None if flagged or lo is None else lo > 0.5)
            rows.append(row)
            if rule != "majority":
                continue
            for base_key in (("shipped_synthetic", "gnb"), ("rule", "route_window")):
                if (protocol, clf) == base_key:
                    continue
                base_binary, _, base_status, _ = scope_preds(
                    preds[(*base_key, rule)], scope)
                if base_binary is None:
                    continue
                d, dlo, dhi = paired_delta(
                    t, binary, base_binary, resamplers[scope], draws_a=mine,
                    draws_b=draws_for((*base_key, rule), scope, t, base_binary))
                deltas.append({
                    "protocol": protocol, "classifier": clf,
                    "baseline": "/".join(base_key), "label_rule": rule,
                    "trace": scope, "metric": "balanced_accuracy",
                    "n_windows": len(t), "small_n": flagged,
                    "delta": d, "delta_lo": dlo, "delta_hi": dhi,
                    "verdict": verdict(d, dlo, dhi, flagged),
                    "reps": reps, "seed": SEED})
    return rows, deltas


# ------------------------------------------------------------ replay

def latency_for(name: str, requests: list[Request]) -> LatencyModel:
    if name == "measured":
        return LatencyModel.measured(trace=requests)
    return LatencyModel(5.0, 100.0, 50.0)


def view_of(requests: list[Request], view: str) -> list[Request]:
    if view == "native":
        return requests
    return [dataclasses.replace(r, operation="R") for r in requests]


def replay_one(job) -> dict:
    """One replay row; ``job`` = (path, view, mode, param, latency_name)."""
    path, view, mode, param, latency_name = job
    original = load_csv(path, "revised")
    requests = view_of(original, view)
    latency = latency_for(latency_name, original)
    kwargs = {}
    model = None
    if mode == "deep":
        kwargs["read_ahead_depth"] = param
    elif mode == "linux_ra":
        kwargs["linux_ra_max_blocks"] = param
    elif mode == "adaptive":
        from .benchmark import make_model
        model = make_model(SEED)  # fresh per replay: frozen, never updated
    m, _ = replay(requests, CAPACITY, WINDOW, mode, model, latency_model=latency,
                  **kwargs)
    return {
        "trace": Path(path).stem, "view": view,
        "mode": f"{mode}@{param}" if param is not None else mode,
        "latency_model": latency_name,
        "requests": len(requests),
        "read_requests": sum(r.operation == "R" for r in requests),
        "read_blocks": m.read_blocks, "read_hits": m.read_hits,
        "hit_ratio": m.hit_ratio if m.read_blocks else None,
        "prefetches": m.prefetches, "useful_prefetches": m.useful_prefetches,
        "precision": m.prefetch_precision if m.prefetches else None,
        "device_ios": m.device_ios, "blocks_moved": m.blocks_moved,
        "mean_latency_us": m.mean_access_latency_us if m.read_blocks else None,
        "hit_us": latency.hit_us, "demand_miss_us": latency.demand_miss_us,
        "prefetch_us": latency.prefetch_us,
        "break_even_precision": (latency.break_even_precision()
                                 if math.isfinite(latency.break_even_precision())
                                 else None),
        "capacity": CAPACITY, "seed": SEED,
    }


def replay_jobs(paths) -> list[tuple]:
    return [(str(p), view, mode, param, lat) for p in paths for view in VIEWS
            for lat in LATENCY_MODELS for mode, param in REPLAY_MODES]


def replay_study(paths, workers: int = 8) -> list[dict]:
    jobs = replay_jobs(paths)
    if workers > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            out = pool.map(replay_one, jobs, chunksize=1)
    else:
        out = [replay_one(job) for job in jobs]
    base = {(r["trace"], r["view"], r["latency_model"]): r for r in out
            if r["mode"] == "none"}
    rows = []
    for r in out:
        b = base[(r["trace"], r["view"], r["latency_model"])]
        ok = r["read_blocks"] > 0
        r["status"] = "ok" if ok else "n/a: no read requests in this view"

        def ratio(key):
            return r[key] / b[key] if ok and b[key] else None
        r["hit_ratio_delta_vs_none"] = (r["hit_ratio"] - b["hit_ratio"]) if ok else None
        r["device_ios_ratio_vs_none"] = ratio("device_ios")
        r["blocks_moved_ratio_vs_none"] = ratio("blocks_moved")
        rows.append(r)
    return rows


def trace_profile(path: Path, windows: TraceWindows) -> dict:
    requests = load_csv(path, "revised")
    service = [r.service_ms for r in requests if r.service_ms is not None]
    maj = [label(c, "majority") for c in windows.seq_counts]
    return {
        "flag_matches_contiguity": flag_contiguity_agreement(requests),
        "trace": path.stem, "family": family(path.stem), "path": f"data/samples/msrc/{path.name}",
        "requests": len(requests),
        "read_requests": sum(r.operation == "R" for r in requests),
        "seq_flag_requests": sum(r.pattern == "seq" for r in requests),
        "seq_flag_read_requests": sum(r.pattern == "seq" and r.operation == "R"
                                      for r in requests),
        "windows": len(windows.vectors),
        "seq_windows_majority": maj.count(POSITIVE),
        "seq_windows_quarter": sum(label(c, "quarter") == POSITIVE
                                   for c in windows.seq_counts),
        "small_n_majority": small_n(maj),
        "mean_service_ms": sum(service) / len(service) if service else None,
        "age_table_evictions": windows.evictions,
    }


def flag_contiguity_agreement(requests: list[Request]) -> float | None:
    """Share of requests (after the first) whose ``seq`` flag equals "starts
    where the previous request ended". Near 1 means the flag is an address
    contiguity heuristic, not an independent workload label."""
    pairs = list(zip(requests, requests[1:]))
    if not pairs:
        return None
    same = sum((b.lba == a.lba + a.size_blocks) == (b.pattern == "seq") for a, b in pairs)
    return same / len(pairs)


def load_windows(paths) -> list[TraceWindows]:
    return [trace_windows(load_csv(p, "revised"), Path(p).stem) for p in paths]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", default=str(SAMPLE_DIR))
    parser.add_argument("--out", required=True, help="results directory to write")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=None,
                        help="bounded run: first N requests of each file (never results/)")
    args = parser.parse_args(argv)
    from prefetch_common import provenance, schemas
    start = provenance.snapshot()
    paths = discover(args.samples)
    if not paths:
        print(f"msrc_eval: no *.revised files under {args.samples}")
        return 4
    out = Path(args.out)
    if args.max_requests is not None and out.resolve().is_relative_to(
            schemas.RESULTS_ROOT.resolve()):
        raise SystemExit("a bounded run never writes results/")
    windows = [trace_windows(load_csv(p, "revised")[:args.max_requests], p.stem)
               for p in paths]
    profile = [trace_profile(p, w) for p, w in zip(paths, windows)]
    rows, deltas = classifier_study(windows, reps=args.reps)
    replay_rows = replay_study(paths, args.workers) if args.max_requests is None else []
    out.mkdir(parents=True, exist_ok=True)
    schemas.write_rows("msrc_samples/traces.csv", profile, out / "traces.csv")
    schemas.write_rows("msrc_samples/classifier_metrics.csv", rows,
                       out / "classifier_metrics.csv")
    schemas.write_rows("msrc_samples/classifier_deltas.csv", deltas,
                       out / "classifier_deltas.csv")
    if replay_rows:
        schemas.write_rows("msrc_samples/replay.csv", replay_rows, out / "replay.csv")
    import sklearn
    backend = Resampler(1, 1).backend
    provenance.write_provenance(
        out, ["python", "reproduce.py", "msrc"], inputs=paths, study="msrc_samples",
        start=start, gpu=False,
        notes="E7: labelled-window transfer check and sanity replay on the "
              "size-selected MSRC light sample; see PREREGISTERED.md (A1: 7 traces, LOFO)",
        extra={"bootstrap_backend": backend, "reps": args.reps, "seed": SEED,
               "sklearn": sklearn.__version__, "workers": args.workers,
               "traces": [p.stem for p in paths]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

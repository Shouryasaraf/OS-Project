"""Train, score and analyze stages of the hindsight-router experiment.

Split from :mod:`hindsight_experiment` so the record stage (stdlib only)
never imports numpy or scikit-learn; ``train`` needs both, ``score`` and
``analyze`` need numpy only through the persisted schedules.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import corpora, hindsight
from .hindsight_experiment import (CV_PER_TRACE, LATENCY, LOCO_OBJECTIVES, PER_TRACE, REPO,
                                   SWEEP, _chunks, _Count, _key, _load1,
                                   _write_gz, metrics_row, pool_map, read_gz,
                                   slice_bounds)
from .simulator import replay_stream

# The pool workers read these module globals (fork start method), so the
# training data is never pickled per task.
_DATA: dict = {}


def group_of(trace: str) -> str:
    twin = corpora.twin_group(trace)
    return "twin:" + min(twin) if twin else trace


def _record_path(cache: Path, trace, skip, cap, action) -> Path:
    return cache / "records" / f"{_key(trace, skip, cap, action)}.json.gz"


def _features_path(cache: Path, trace, skip) -> Path:
    return cache / "features" / f"{_key(trace, skip)}.json.gz"


def load_slice(cache: Path, trace: str, skip: int, cap: int, n: int,
               full: bool = True):
    """``(features, {action: record})``; ``None`` for a short slice when
    ``full`` (slice 2). Slice 1 is C1's "first ``n`` requests", which a file
    shorter than ``n`` fills with the whole file (``fiu_madmax``)."""
    feats = read_gz(_features_path(cache, trace, skip))
    if (full and feats["requests"] < n) or feats["requests"] < 2 * 32:
        return None
    records = {a: read_gz(_record_path(cache, trace, skip, cap, a))
               for a in hindsight.ACTIONS}
    return feats["features"], records


def matrix_of(records: dict, objective: str) -> list[tuple]:
    columns = [hindsight.window_utility(records[a]["deltas"], objective)
               for a in hindsight.ACTIONS]
    return list(zip(*columns))


# ------------------------------------------------------------------- train

def _cv_path(task) -> Path:
    cap, objective, family, index, _params = task
    return _DATA["cv_dir"] / f"c{cap}_{objective}_{family}_{index}.json"


def _cv_task(task):
    from . import hindsight_models as hm
    cap, objective, family, index, params = task
    cached = _cv_path(task)
    if cached.exists():
        row = json.loads(cached.read_text(encoding="utf-8"))
        if row["params"] == params and row["per_trace_cv"] == _DATA["per_trace_cv"]:
            return row
    started = time.monotonic()
    datasets = _DATA[(cap, objective)]
    result = hm.cross_validate(family, params, datasets,
                               {d.trace: group_of(d.trace) for d in datasets},
                               _DATA["per_trace_cv"])
    result.update(capacity=cap, objective=objective, family=family, config=index,
                  params=params, wall_s=round(time.monotonic() - started, 3),
                  load=_load1(), per_trace_cv=_DATA["per_trace_cv"])
    # One file per task, so a restarted selection run resumes (JSON turns
    # tuples into lists; the comparison above is on the reloaded form).
    tmp = cached.with_name(cached.name + ".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    result = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.replace(cached)
    return result


def _fit_task(task):
    from . import hindsight_models as hm
    cap, objective, family, params, exclude_corpus, path, record = task
    datasets = [d for d in _DATA[(cap, objective)]
                if exclude_corpus is None or corpora.CORPUS_OF(d.trace) != exclude_corpus]
    model = hm.fit(family, params, datasets, _DATA["per_trace"])
    record = dict(record, traces=[t for t in record["traces"]
                                  if t["trace"] in {d.trace for d in datasets}])
    hm.save(model, path, record)
    return path


def _selection_key(row):
    from . import hindsight_models as hm
    score = row["cv_score"] if row["cv_score"] is not None else float("-inf")
    return (-score, hm.FAMILIES.index(row["family"]), row["config"])


def run_train(args, traces, capacities, cache: Path) -> None:
    from . import hindsight_models as hm
    n = args.requests
    skip1, _ = slice_bounds(1, n)
    per_trace = PER_TRACE if n >= 1_000_000 else max(50, n // 64)
    _DATA["per_trace"] = per_trace
    # Amendment A1: CV fits use a smaller per-trace subsample than the final
    # refit (time); the held-out traces are still scored on every window.
    _DATA["per_trace_cv"] = min(per_trace, CV_PER_TRACE)
    for cap in capacities:
        for objective in hindsight.OBJECTIVES:
            datasets = []
            for trace in traces:
                loaded = load_slice(cache, trace, skip1, cap, n, full=False)
                if loaded is None:
                    raise SystemExit(f"{trace}: slice 1 is empty; training needs all")
                feats, records = loaded
                datasets.append(hm.TraceData(trace, feats, matrix_of(records, objective)))
            _DATA[(cap, objective)] = datasets
    n_configs = args.configs or hm.CONFIGS_PER_FAMILY
    _DATA["cv_dir"] = cache / "train" / "cv"
    _DATA["cv_dir"].mkdir(parents=True, exist_ok=True)
    tasks = [(cap, objective, family, index, json.loads(json.dumps(params)))
             for cap in capacities for objective in hindsight.OBJECTIVES
             for family in hm.FAMILIES
             for index, params in enumerate(hm.search_space(family, n_configs))]
    # Slow families first, so the pool's tail is short.
    order = {"knn": 0, "forest": 1, "extra_trees": 2, "mlp": 3, "hgb": 4}
    tasks.sort(key=lambda t: order.get(t[2], 9))
    results = []
    started = time.monotonic()
    for done, row in enumerate(pool_map(_cv_task, tasks, args.workers), 1):
        results.append(row)
        if not args.quiet:
            print(f"[train cv {done}/{len(tasks)} {time.monotonic() - started:.0f}s "
                  f"load {_load1()}] {row['capacity']} {row['objective']} "
                  f"{row['family']}#{row['config']} score={row['cv_score']}", flush=True)
    results.sort(key=lambda r: (r["capacity"], hindsight.OBJECTIVES.index(r["objective"]),
                                hm.FAMILIES.index(r["family"]), r["config"]))
    chosen = {}
    for cap in capacities:
        for objective in hindsight.OBJECTIVES:
            cell = [r for r in results if r["capacity"] == cap and r["objective"] == objective]
            best = min(cell, key=_selection_key)
            chosen[(cap, objective, "selected")] = best
            for family in hm.FAMILIES:
                rows = [r for r in cell if r["family"] == family]
                chosen[(cap, objective, family)] = min(rows, key=_selection_key)
    for r in results:
        key = (r["capacity"], r["objective"])
        r["selected"] = chosen[(*key, "selected")] is r
        r["family_best"] = chosen[(*key, r["family"])] is r
        r["n_train_windows"] = _DATA["per_trace_cv"] * len(traces)
    (cache / "train").mkdir(parents=True, exist_ok=True)
    (cache / "train" / "cv.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    models = Path(args.models_dir)
    base_record = hindsight.training_record(
        list(traces), skip1, n, window_size=hindsight.WINDOW_SIZE,
        per_trace=per_trace, selection="grouped 5-fold CV over slice-1 traces",
        latency="assumed 5/100/50")
    fits = []
    for (cap, objective, which), row in chosen.items():
        name = f"c{cap}_{objective}_{which}"
        record = dict(base_record, capacity=cap, objective=objective, family=row["family"],
                      params=row["params"], cv_score=row["cv_score"], role=which)
        fits.append((cap, objective, row["family"], row["params"], None,
                     str(models / f"{name}.joblib"), record))
        if which == "selected" and objective in LOCO_OBJECTIVES:
            for corpus in corpora.CORPORA:
                rec = dict(record, role=f"loco:{corpus}", held_out_corpus=corpus)
                fits.append((cap, objective, row["family"], row["params"], corpus,
                             str(models / f"{name}_loco_{corpus}.joblib"), rec))
    for path in pool_map(_fit_task, fits, args.workers):
        if not args.quiet:
            print(f"[train fit] {Path(path).name}", flush=True)
    manifest = {f"{cap}|{objective}|{which}": {"family": row["family"], "params": row["params"],
                                               "cv_score": row["cv_score"]}
                for (cap, objective, which), row in chosen.items()}
    (cache / "train" / "chosen.json").write_text(json.dumps(manifest, indent=1),
                                                 encoding="utf-8")


# ------------------------------------------------------------------- score

def _replay_task(task):
    label, trace, path, skip, n, cap, mode, schedule, out = task
    out = Path(out)
    if out.exists():
        return str(out)
    started, load_start = time.monotonic(), _load1()
    count = _Count()
    kwargs = {}
    if mode == "routed":
        kwargs["router"] = hindsight.ScheduledRouter(schedule)
    model = None
    if mode == "adaptive":
        from .benchmark import make_model
        model = make_model(42)
    if mode == "linux_ra":
        kwargs["linux_ra_max_blocks"] = 256
    metrics = replay_stream(count.wrap(_chunks(path, skip, n)), capacity=cap,
                            window_size=hindsight.WINDOW_SIZE, mode=mode, model=model,
                            latency_model=LATENCY, **kwargs)
    shares = None
    if schedule is not None:
        decided = ["none"] + list(schedule[:-1])
        shares = {a: decided.count(a) / len(decided) for a in hindsight.ACTIONS}
    payload = {"label": label, "trace": trace, "skip": skip, "capacity": cap,
               "mode": mode, "metrics": metrics_row(metrics), "requests": count.requests,
               "action_shares": shares, "load_start": load_start, "load_end": _load1(),
               "wall_s": round(time.monotonic() - started, 3)}
    _write_gz(out, payload)
    return str(out)


def _schedule_task(task):
    """Batch-infer one persisted router's schedule on every test slice."""
    from . import hindsight_models as hm
    artifact, objective, cap, held_out = task
    model, record = hm.load(artifact)
    out = {}
    for trace, (feats, matrix) in _DATA["test"][cap][objective].items():
        if held_out is not None and corpora.CORPUS_OF(trace) != held_out:
            continue
        if held_out is not None and any(t["trace"] == trace for t in record["traces"]):
            raise SystemExit(f"{Path(artifact).name} was trained on {trace}")
        conflict = hindsight.scoring_conflict(record, trace, *_DATA["test_span"])
        if conflict:
            raise SystemExit(f"{Path(artifact).name} on {trace}: {conflict}")
        schedule = hm.predict_schedule(model, feats)
        chosen = [hindsight.ACTIONS.index(a) for a in schedule[:-1]]
        out[trace] = {
            "schedule": schedule, "family": record["family"],
            "headroom": hindsight.headroom_captured(matrix, chosen),
            "proxy_total": hindsight.proxy_totals(matrix, chosen),
            "accuracy": (sum(c == hindsight.best_action_index(matrix[k + 1])
                             for k, c in enumerate(chosen)) / len(chosen)
                         if chosen else None),
            "shares": {a: schedule[:-1].count(a) / max(1, len(chosen))
                       for a in hindsight.ACTIONS}}
    return task, out


def run_score(args, traces, capacities, cache: Path) -> None:
    from . import hindsight_models as hm
    n = args.requests
    skip2, _ = slice_bounds(2, n)
    models = Path(args.models_dir)
    _DATA["test_span"] = (skip2, skip2 + n)
    _DATA["test"] = {cap: {o: {} for o in hindsight.OBJECTIVES} for cap in capacities}
    tasks, proxies, oracle = [], [], {}
    usable = []
    for trace in traces:
        feats_payload = read_gz(_features_path(cache, trace, skip2))
        if feats_payload["requests"] < n:
            continue
        usable.append(trace)
        for cap in capacities:
            records = {a: read_gz(_record_path(cache, trace, skip2, cap, a))
                       for a in hindsight.ACTIONS}
            for objective in hindsight.OBJECTIVES:
                matrix = matrix_of(records, objective)
                _DATA["test"][cap][objective][trace] = (feats_payload["features"], matrix)
                oracle[(trace, cap, f"oracle_{objective}")] = hindsight.oracle_schedule(matrix)
    jobs = []
    for cap in capacities:
        for objective in hindsight.OBJECTIVES:
            for which in ("selected", *hm.FAMILIES):
                jobs.append((str(models / f"c{cap}_{objective}_{which}.joblib"),
                             objective, cap, None))
            if objective in LOCO_OBJECTIVES:
                for corpus in corpora.CORPORA:
                    jobs.append((str(models / f"c{cap}_{objective}_selected_loco_{corpus}.joblib"),
                                 objective, cap, corpus))
    schedules = dict(oracle)
    for (artifact, objective, cap, held_out), out in pool_map(_schedule_task, jobs, args.workers):
        which = Path(artifact).stem.split(f"c{cap}_{objective}_", 1)[1]
        for trace, info in out.items():
            if held_out is not None:
                schedules[(trace, cap, f"routed_{objective}_loco")] = info["schedule"]
                continue
            if which == "selected":
                schedules[(trace, cap, f"routed_{objective}")] = info["schedule"]
            if which == "gnb" and objective in LOCO_OBJECTIVES:
                schedules[(trace, cap, f"routed_{objective}_gnb")] = info["schedule"]
            proxies.append({"trace": trace, "capacity": cap, "objective": objective,
                            "which": which, **{k: v for k, v in info.items()
                                               if k != "schedule"}})
    if not args.quiet:
        print(f"[score] {len(schedules)} schedules on {len(usable)} test slices", flush=True)
    for (trace, cap, label), schedule in sorted(schedules.items()):
        tasks.append((label, trace, str(traces[trace]), skip2, n, cap, "routed", schedule,
                      str(cache / "score" / f"{_key(trace, skip2, cap, label)}.json.gz")))
    for trace in usable:
        for cap in capacities:
            tasks.append(("adaptive_evidence", trace, str(traces[trace]), skip2, n, cap,
                          "adaptive_evidence", None,
                          str(cache / "score" / f"{_key(trace, skip2, cap, 'adaptive_evidence')}.json.gz")))
    # Exact reproduction of one reused sweep row of each reused mode.
    check_trace = usable[0]
    for mode in ("adaptive", "linux_ra"):
        tasks.append((f"check_{mode}", check_trace, str(traces[check_trace]), skip2, n,
                      capacities[0], mode, None,
                      str(cache / "score" / f"{_key(check_trace, skip2, capacities[0], 'check_' + mode)}.json.gz")))
    (cache / "score").mkdir(parents=True, exist_ok=True)
    (cache / "score" / "proxy.json").write_text(json.dumps(proxies), encoding="utf-8")
    todo = [t for t in tasks if not Path(t[-1]).exists()]
    # Deep schedules cost most; start them first.
    todo.sort(key=lambda t: -sum(hindsight.action_depth(a) for a in (t[7] or [])[::64]))
    started = time.monotonic()
    for done, out in enumerate(pool_map(_replay_task, todo, args.workers), 1):
        if not args.quiet:
            print(f"[score {done}/{len(todo)} {time.monotonic() - started:.0f}s "
                  f"load {_load1()}] {Path(out).name}", flush=True)


# ----------------------------------------------------------------- analyze

def run_analyze(args, traces, capacities, cache: Path) -> None:
    from .hindsight_analysis import analyze
    results = Path(args.results_dir) if args.results_dir else cache / "results"
    analyze(cache, results, traces, capacities, args.requests, args.reps,
            sweep=SWEEP if args.requests == 1_000_000 else None)

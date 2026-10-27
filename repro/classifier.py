"""Classifier zoo (track CLF): tuned alternatives to GNB, and the real-trace router benchmark.

``reproduce.py classifier`` runs both parts of
``results/classifier/PREREGISTERED.md`` and writes ``results/classifier/``:

* Part 1: 15 families (the shipped ``OnlineGaussianNB`` / ``OnlineQDA`` and
  13 tuned scikit-learn families) on labelled synthetic windows, every
  candidate selected on training seeds and scored once on test seeds
  (``synthetic_tuning.csv``, ``synthetic_eval.csv``);
* Part 2: each baseline-regime-tuned classifier as the unchanged
  ``adaptive`` router on the 19 full ``PAPER_SUBSET`` slices of C3 slice 2
  at 1 MiB and 64 MiB (``router_runs.csv``, ``router_comparisons.csv``,
  ``router_summary.csv``), with E1's committed baseline rows reused only
  after a re-run reproduced them exactly.

``--quick`` is a bounded smoke path into ``outputs/classifier-quick/``
(never ``results/``): two settings, three candidates per family, two traces
at 100,000 requests. Working caches (window vectors, raw replays) live
under ``outputs/classifier/``; the replay stage is resumable.
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker: the pool supplies the parallelism, and
# the host is shared. Set before numpy is imported anywhere.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

from . import EXIT_UNAVAILABLE  # noqa: E402

OWNER = "track CLF"
IMPLEMENTED = True

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "classifier"
WORK = REPO / "outputs" / "classifier"
QUICK_OUT = REPO / "outputs" / "classifier-quick"
SKIP, SLICE = 1_000_000, 1_000_000
CAPACITIES = (2048, 131072)
#: E1 baselines reused from results/disk/sweep.jsonl (C3, slice 2), by label.
E1_BASELINES = {"none": ("none", 0, None), "sequential": ("sequential", 0, None),
                "deep@8": ("deep", 8, None), "linux_ra@256": ("linux_ra", 0, 256)}
#: The reproduction check: every E1 baseline re-run on this trace.
CHECK_TRACE = "msr_prxy_0"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=20,
                        help="worker processes (default 20, the track's allowance)")
    parser.add_argument("--trace-root", default=str(REPO / "data" / "block-traces"))
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--part", choices=("all", "synthetic", "router"), default="all",
                        help="synthetic computes and caches Part 1 only (results/ untouched); "
                             "router reuses that cache and writes every results file")


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return "about 2-4 min on 20 workers (bounded smoke; outputs/ only)"
    return ("about 25-35 min wall on 20 workers: Part 1 ~5 min, window vectors ~3 min, "
            "replays ~20 min (19 traces x 2 capacities x distinct routings)")


def _available(args) -> str | None:
    for module in ("numpy", "sklearn"):
        if importlib.util.find_spec(module) is None:
            return f"{module} is not installed (use the .venv interpreter)"
    from adaptive_prefetch import corpora
    found = corpora.discover_block_traces(args.trace_root)
    missing = [s for s in corpora.PAPER_SUBSET if s not in found]
    if missing:
        return f"block traces missing under {args.trace_root}: {', '.join(missing[:3])}..."
    return None


def _load1():
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def _quiet_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass


# ------------------------------------------------------------------ workers

def _features_job(task):
    trace, path, skip, cap = task
    from adaptive_prefetch.clfzoo import router
    started = time.monotonic()
    vectors, requests, evictions = router.window_features(router.slice_chunks(path, skip, cap))
    return trace, {"vectors": vectors, "requests": requests, "evictions": evictions,
                   "wall_s": round(time.monotonic() - started, 3), "load_avg": _load1()}


def _replay_job(task):
    kind = task[0]
    started = time.monotonic()
    load = _load1()
    if kind == "labels":
        _, trace, path, skip, cap, capacity, key, coded = task
        from adaptive_prefetch.clfzoo import router
        from adaptive_prefetch.trace import CLASSES
        labels = [CLASSES[int(c)] for c in coded]
        # contextual=False: the router ignores the vector, so the age table
        # is not rebuilt; its eviction count comes from the vector pass.
        metrics = router.replay_labels(router.slice_chunks(path, skip, cap), capacity,
                                       labels, contextual=False)
        return {"kind": kind, "trace": trace, "capacity": capacity, "label_key": key,
                "counters": router.counters(metrics),
                "mean_us": metrics.mean_access_latency_us,
                "wall_s": round(time.monotonic() - started, 3), "load_avg": load}
    _, job = task
    from adaptive_prefetch import multicorpus
    row = multicorpus.run_job(job)
    return {"kind": kind, "trace": job.trace, "capacity": job.capacity,
            "row": row, "wall_s": row["wall_s"], "load_avg": load}


def _job_for(row: dict, path: str, mode: str, depth: int, max_blocks):
    """A multicorpus Job with a committed C3 row's identity and another mode."""
    from adaptive_prefetch import multicorpus
    return multicorpus.Job(
        row["config"], row["trace"], path, row["corpus"], row["skip_requests"],
        row["max_requests"], row["merge"], row["max_merge_sectors"], row["capacity"],
        mode, depth, max_blocks, "assumed", full_slice=True)


# -------------------------------------------------------------------- run

def run(args: argparse.Namespace) -> int:
    reason = _available(args)
    if reason:
        print(f"reproduce.py classifier: unavailable: {reason}")
        return EXIT_UNAVAILABLE
    sys.path.insert(0, str(REPO / "src"))
    _quiet_warnings()
    import multiprocessing

    from adaptive_prefetch import corpora
    from adaptive_prefetch.clfzoo import analysis, data, router, synthetic, zoo
    from prefetch_common import provenance, schemas

    start = provenance.snapshot()
    quick = args.quick
    results = QUICK_OUT if quick else Path(args.results_dir)
    work = (QUICK_OUT / "work") if quick else Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    settings = ("baseline", "transition") if quick else data.ALL_SETTINGS
    n_candidates = 3 if quick else zoo.N_CANDIDATES
    reps = 200 if quick else args.reps
    log = lambda msg: print(f"[classifier {time.strftime('%H:%M:%S')} load={_load1()}] {msg}",
                            flush=True)
    timings = {}

    # ---------------------------------------------------------------- Part 1
    raw_path = work / "synthetic_raw.json"
    if args.part in ("all", "synthetic") or not raw_path.is_file():
        t0 = time.monotonic()
        raw = synthetic.run(settings, zoo.FAMILIES, args.workers, n_candidates, progress=log)
        timings["synthetic_s"] = round(time.monotonic() - t0, 1)
        stored = {"tuning": raw["tuning"], "records": raw["records"],
                  "chosen": {f"{s}|{f}": row["candidate"] for (s, f), row in raw["chosen"].items()},
                  "settings": list(settings), "n_candidates": n_candidates}
        raw_path.write_text(json.dumps(stored), encoding="utf-8")
    stored = json.loads(raw_path.read_text(encoding="utf-8"))
    tuning = stored["tuning"]
    chosen = {}
    for row in tuning:
        if row["selected"]:
            chosen[(row["setting"], row["family"])] = row
    settings = tuple(stored["settings"])
    if args.part == "synthetic" and not quick:
        # A partial run never touches results/: the results directory is
        # rewritten only together with its provenance, at the end of a full
        # (or cached --part router) run.
        log(f"Part 1 cached in {raw_path}; run --part router to write {results}")
        return 0
    schemas.write_rows("classifier/synthetic_tuning.csv", analysis.tuning_rows(tuning),
                       results / "synthetic_tuning.csv")
    schemas.write_rows("classifier/synthetic_eval.csv",
                       analysis.eval_rows(stored["records"], chosen, zoo.FAMILIES, settings,
                                          data.test_conditions, reps),
                       results / "synthetic_eval.csv")
    log("Part 1 written")
    if args.part == "synthetic":
        return 0

    # ---------------------------------------------------------------- Part 2
    found = corpora.discover_block_traces(args.trace_root)
    sweep = [json.loads(line) for line in
             (REPO / "results" / "disk" / "sweep.jsonl").read_text().splitlines() if line.strip()]
    c3 = {(r["trace"], r["capacity"], r["mode"], r["depth"], r["max_blocks"]): r
          for r in sweep if r["config"] == "C3" and r["skip_requests"] == SKIP
          and r["latency"] == "assumed"}
    if quick:
        traces = ("msr_prxy_0", "tencentBlock_1894")
        skip, cap = 0, 100_000
    else:
        # Exactly the slices E1 could fill (two FIU traces are n/a there).
        traces = tuple(t for t in corpora.PAPER_SUBSET
                       if c3.get((t, CAPACITIES[0], "none", 0, None), {}).get("status") == "ok")
        skip, cap = SKIP, SLICE

    # Deployed routers: the baseline-regime selection, fitted on make_model's
    # own training data (seed 42, 100 windows per class), as E1's GNB was.
    train = data.baseline_vectors(42)
    rows_x, rows_y = [v for v, _ in train], [l for _, l in train]
    routers = {f: zoo.fit(f, chosen[("baseline", f)]["params"], rows_x, rows_y, seed=0)
               for f in zoo.FAMILIES}
    from adaptive_prefetch.benchmark import make_model
    shipped = make_model(42)
    assert routers["gnb_online"]._online.mean == shipped.mean, "gnb_online is not make_model(42)"

    feature_dir = work / ("features" if not quick else "features-quick")
    feature_dir.mkdir(parents=True, exist_ok=True)
    features = {}
    todo = []
    for trace in traces:
        cached = feature_dir / f"{trace}.json"
        if cached.is_file():
            features[trace] = json.loads(cached.read_text())
        else:
            todo.append((trace, str(found[trace]), skip, cap))
    t0 = time.monotonic()
    if todo:
        with multiprocessing.get_context("fork").Pool(min(args.workers, len(todo))) as pool:
            for trace, result in pool.imap_unordered(_features_job, todo):
                (feature_dir / f"{trace}.json").write_text(json.dumps(result))
                features[trace] = result
                log(f"vectors {trace}: {len(result['vectors'])} windows ({result['wall_s']}s)")
    timings["vectors_s"] = round(time.monotonic() - t0, 1)

    labels = {}   # (trace, family) -> list of labels
    for trace in traces:
        vectors = features[trace]["vectors"]
        for family, model in routers.items():
            labels[(trace, family)] = model.predict_batch(vectors)
    # GNB's batch labels must be the per-call labels of the shipped model.
    for trace in traces:
        assert labels[(trace, "gnb_online")] == [shipped.predict(tuple(v))[0]
                                                 for v in features[trace]["vectors"]]

    from adaptive_prefetch.trace import CLASSES
    code = {label: str(i) for i, label in enumerate(CLASSES)}
    raw_replays = work / ("router_raw.jsonl" if not quick else "router_raw-quick.jsonl")
    done = {}
    if raw_replays.is_file():
        for line in raw_replays.read_text().splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            ident = (item["kind"], item["trace"], item["capacity"],
                     item.get("label_key") or item.get("mode_label"))
            done[ident] = item
    tasks = []
    for trace in traces:
        for capacity in CAPACITIES:
            keys = {}
            for family in zoo.FAMILIES:
                seq = labels[(trace, family)]
                keys.setdefault(router.label_key(seq), seq)
            for key, seq in keys.items():
                if ("labels", trace, capacity, key) not in done:
                    tasks.append(("labels", trace, str(found[trace]), skip, cap, capacity, key,
                                  "".join(code[l] for l in seq)))
    reference = {}
    if not quick:
        for trace in traces:
            for capacity in CAPACITIES:
                base = c3[(trace, capacity, "none", 0, None)]
                job = _job_for(base, str(found[trace]), "adaptive_evidence", 0, None)
                if ("job", trace, capacity, "adaptive_evidence") not in done:
                    tasks.append(("job", job))
        for label, (mode, depth, mb) in E1_BASELINES.items():
            committed = c3[(CHECK_TRACE, CAPACITIES[0], mode, depth, mb)]
            reference[label] = committed
            if ("job", CHECK_TRACE, CAPACITIES[0], f"check:{label}") not in done:
                tasks.append(("job", _job_for(committed, str(found[CHECK_TRACE]),
                                              mode, depth, mb)))
    else:
        for trace in traces:
            for capacity in CAPACITIES:
                for label, (mode, depth, mb) in {**E1_BASELINES,
                                                 "adaptive_evidence": ("adaptive_evidence", 0, None)}.items():
                    if ("job", trace, capacity, label) in done:
                        continue
                    from adaptive_prefetch import multicorpus
                    tasks.append(("job", multicorpus.Job(
                        "quick", trace, str(found[trace]), corpora.CORPUS_OF(trace), skip, cap,
                        "auto", 1024, capacity, mode, depth, mb, "assumed")))
    # Longest first: the committed adaptive wall time is the cost proxy.
    def cost(task):
        trace, capacity = (task[1], task[5]) if task[0] == "labels" else (task[1].trace,
                                                                          task[1].capacity)
        row = c3.get((trace, capacity, "adaptive", 0, None))
        return -(row["wall_s"] if row and row.get("wall_s") else 0.0)
    tasks.sort(key=cost)
    log(f"replays: {len(tasks)} to run, {len(done)} cached")
    t0 = time.monotonic()
    if tasks:
        with multiprocessing.get_context("fork").Pool(min(args.workers, len(tasks)),
                                                      maxtasksperchild=8) as pool, \
                open(raw_replays, "a", encoding="utf-8") as handle:
            for n, item in enumerate(pool.imap_unordered(_replay_job, tasks), 1):
                if item["kind"] == "job":
                    row = item["row"]
                    label = _label(row)
                    is_check = (not quick and row["trace"] == CHECK_TRACE
                                and row["capacity"] == CAPACITIES[0] and label in E1_BASELINES)
                    item["mode_label"] = f"check:{label}" if is_check else label
                    ident = ("job", item["trace"], item["capacity"], item["mode_label"])
                else:
                    ident = ("labels", item["trace"], item["capacity"], item["label_key"])
                done[ident] = item
                handle.write(json.dumps(item) + "\n")
                handle.flush()
                log(f"[{n}/{len(tasks)}] {ident[0]} {ident[1]} cap={ident[2]} "
                    f"{ident[3]} ({item['wall_s']}s)")
    timings["replays_s"] = round(time.monotonic() - t0, 1)

    # ------------------------------------------------------------ checks
    checks = {}
    if not quick:
        repro_fields = ("requests", "read_blocks", "hit_ratio", "precision",
                        "total_prefetches", "useful_prefetches", "unused", "mean_us",
                        "policy_switches", "demand_ios", "prefetch_ios",
                        "prefetch_ios_mergeable", "device_ios", "blocks_moved")
        for label, committed in reference.items():
            rerun = done[("job", CHECK_TRACE, CAPACITIES[0], f"check:{label}")]["row"]
            diff = [f for f in repro_fields if rerun[f] != committed[f]]
            checks[f"e1_rerun_{label}"] = "identical" if not diff else f"DIFFERS: {diff}"
        gnb_diffs = []
        for trace in traces:
            key = router.label_key(labels[(trace, "gnb_online")])
            for capacity in CAPACITIES:
                item = done[("labels", trace, capacity, key)]
                committed = c3[(trace, capacity, "adaptive", 0, None)]
                mine = dict(item["counters"], requests=features[trace]["requests"],
                            total_prefetches=item["counters"]["prefetches"],
                            mean_us=item["mean_us"])
                diff = [f for f in repro_fields if mine[f] != committed[f]]
                if features[trace]["evictions"] != committed["age_table_evictions"]:
                    diff.append("age_table_evictions")
                if diff:
                    gnb_diffs.append(f"{trace}@{capacity}: {diff}")
        checks["gnb_router_equals_e1_adaptive"] = ("identical on all "
                                                   f"{len(traces) * len(CAPACITIES)} cells"
                                                   if not gnb_diffs else gnb_diffs)
        bad = [k for k, v in checks.items() if v != "identical" and not str(v).startswith("identical")]
        if bad:
            log(f"VALIDITY CHECK FAILED: {bad}: {[checks[k] for k in bad]}")
            (work / "checks_failed.json").write_text(json.dumps(checks, indent=2))
            return 1

    # ------------------------------------------------------------ rows
    runs = []
    for trace in traces:
        gnb_seq = labels[(trace, "gnb_online")]
        gnb_key = router.label_key(gnb_seq)
        for capacity in CAPACITIES:
            for family in zoo.FAMILIES:
                seq = labels[(trace, family)]
                key = router.label_key(seq)
                item = done[("labels", trace, capacity, key)]
                c = item["counters"]
                shares = router.class_shares(seq)
                runs.append({
                    "trace": trace, "corpus": corpora.CORPUS_OF(trace), "capacity": capacity,
                    "policy": family, "source": "replayed_labels", "label_key": key,
                    "gnb_label_key": gnb_key, "requests": features[trace]["requests"],
                    "windows": len(seq), "hit_ratio": c["hit_ratio"],
                    "precision": c["precision"], "total_prefetches": c["prefetches"],
                    "useful_prefetches": c["useful_prefetches"],
                    "device_ios": c["device_ios"], "blocks_moved": c["blocks_moved"],
                    "policy_switches": c["policy_switches"],
                    **{f"share_{l}": shares[l] for l in CLASSES},
                    "agreement_with_gnb": router.agreement(seq, gnb_seq),
                    "age_table_evictions": features[trace]["evictions"],
                    "wall_s": item["wall_s"], "load_avg": item["load_avg"]})
            baselines = {}
            if quick:
                for label in (*E1_BASELINES, "adaptive_evidence"):
                    baselines[label] = (done[("job", trace, capacity, label)]["row"],
                                        "multicorpus_run_job")
            else:
                for label, (mode, depth, mb) in E1_BASELINES.items():
                    baselines[label] = (c3[(trace, capacity, mode, depth, mb)], "e1_sweep_reused")
                baselines["adaptive_evidence"] = (
                    done[("job", trace, capacity, "adaptive_evidence")]["row"],
                    "multicorpus_run_job")
            for label, (row, source) in baselines.items():
                runs.append({
                    "trace": trace, "corpus": corpora.CORPUS_OF(trace), "capacity": capacity,
                    "policy": label, "source": source, "label_key": None,
                    "gnb_label_key": None, "requests": row["requests"], "windows": None,
                    "hit_ratio": row["hit_ratio"], "precision": row["precision"],
                    "total_prefetches": row["total_prefetches"],
                    "useful_prefetches": row["useful_prefetches"],
                    "device_ios": row["device_ios"], "blocks_moved": row["blocks_moved"],
                    "policy_switches": row["policy_switches"],
                    **{f"share_{l}": None for l in CLASSES}, "agreement_with_gnb": None,
                    "age_table_evictions": row.get("age_table_evictions"),
                    "wall_s": row.get("wall_s"), "load_avg": row.get("load_start")})
    policies = (*zoo.FAMILIES, "adaptive_evidence", "deep@8", "linux_ra@256")
    schemas.write_rows("classifier/router_runs.csv", runs, results / "router_runs.csv")
    schemas.write_rows("classifier/router_comparisons.csv",
                       analysis.comparisons(runs, policies, CAPACITIES, reps),
                       results / "router_comparisons.csv")
    schemas.write_rows("classifier/router_summary.csv",
                       analysis.summary(runs, ("none", "sequential", *policies), CAPACITIES),
                       results / "router_summary.csv")
    log("Part 2 written")
    _headlines(results, quick)
    # CPU time of the cached work, so a cached re-run still records it.
    timings["cpu_s_synthetic_tuning"] = round(sum(r["wall_s"] for r in tuning), 1)
    timings["cpu_s_vectors"] = round(sum(features[t]["wall_s"] for t in traces), 1)
    timings["cpu_s_replays"] = round(sum(item["wall_s"] for item in done.values()), 1)

    if not quick:
        extra_previous = {}
        previous = results / "provenance.json"
        if not tasks and args.part == "router" and previous.is_file():
            # A cached re-run (e.g. after a docs-only change in this
            # directory) computed nothing: keep the computing run's timing.
            record = json.loads(previous.read_text())
            extra_previous = {"computing_run": record.get("extra", {}).get("computing_run") or {
                key: record.get(key) for key in ("created_utc", "wall_s", "load_avg", "git_sha")}
                | {"timings_s": record.get("extra", {}).get("timings_s")}}
        inputs = [REPO / "results" / "disk" / "sweep.jsonl"] + [found[t] for t in traces]
        provenance.write_provenance(
            results, ["python", "reproduce.py", "classifier"], inputs, start=start,
            gpu=False, study="classifier",
            notes=("Written after E1-E6 were known (see PREREGISTERED.md). Every hit ratio "
                   "is SIMULATED; predict/train timings are on a shared, contended host."),
            extra={"checks": checks, "timings_s": timings, "workers": args.workers,
                   "traces": list(traces), "settings": list(settings),
                   "n_candidates": n_candidates, "reps": reps,
                   "distinct_replays": len([k for k in done if k[0] == "labels"]),
                   "selected_baseline_params": {f: chosen[("baseline", f)]["params"]
                                                for f in zoo.FAMILIES}, **extra_previous})
        problems = provenance.verify_provenance(results)
        if problems:
            log(f"provenance problems: {problems}")
            return 1
    log(f"done; results in {results}")
    return 0


def _label(row: dict) -> str:
    from adaptive_prefetch.multicorpus import policy_label
    return policy_label(row["mode"], row["depth"], row["max_blocks"])


def _fmt(value, digits=4):
    if value in (None, ""):
        return "-"
    return f"{float(value):.{digits}f}"


def _headlines(results: Path, quick: bool) -> None:
    """``HEADLINES.md``: tables generated from this directory's CSVs only.

    Nothing in it is typed by hand; it is regenerated on every run and hashed
    in ``provenance.json`` like the CSVs.
    """
    from prefetch_common import schemas
    read = lambda name: schemas.read_rows(f"classifier/{name}", results / name)
    evaluation = read("synthetic_eval.csv")
    summary = read("router_summary.csv")
    comparisons = read("router_comparisons.csv")
    lines = ["# Headline tables (generated)", "",
             "Generated by `repro/classifier.py` from the CSVs in this directory; "
             "never edit by hand." + (" QUICK SMOKE RUN: not a result." if quick else ""),
             "Part 1 figures are **synthetic** accuracies; Part 2 hit ratios are "
             "**simulated**. Intervals: 95% bootstrap, 10,000 resamples, seed 42.", "",
             "## Part 1: test accuracy by setting (mean over 12 seed pairs)", ""]
    settings = []
    for row in evaluation:
        key = (row["setting"], row["condition"])
        if key not in settings:
            settings.append(key)
    families = []
    for row in evaluation:
        if row["family"] not in families:
            families.append(row["family"])
    header = "| family | " + " | ".join(f"{s} {c}" if c != "in_regime" else s
                                        for s, c in settings) + " |"
    lines += [header, "|---" * (len(settings) + 1) + "|"]
    cell = {(r["setting"], r["condition"], r["family"]): r for r in evaluation}
    mark = {"better": " (+)", "worse": " (-)", "no difference detected": "", "baseline": " (base)"}
    for family in families:
        lines.append(f"| `{family}` | " + " | ".join(
            _fmt(cell[(s, c, family)]["accuracy"], 3) + mark[cell[(s, c, family)]["verdict_vs_gnb"]]
            for s, c in settings) + " |")
    lines += ["", "(+) better / (-) worse than `gnb_online` by the paired interval over "
              "seed pairs; no mark: no difference detected.", "",
              "## Part 2: routed `adaptive` on C3 slice 2 (19 traces)", ""]
    for capacity in sorted({r["capacity"] for r in summary}):
        lines += [f"### Capacity {capacity} blocks", "",
                  "| policy | mean hit ratio | device-I/O ratio to none | blocks-moved ratio "
                  "| mean precision | agreement with GNB (mean / min) | hit vs sequential "
                  "[paired] [cluster] | verdict | hit vs gnb_online [paired] [cluster] | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        comp = {(r["policy"], r["baseline"], r["metric"]): r for r in comparisons
                if r["capacity"] == capacity}
        for row in (r for r in summary if r["capacity"] == capacity):
            policy = row["policy"]
            cells = []
            for baseline in ("sequential", "gnb_online"):
                c = comp.get((policy, baseline, "hit_ratio"))
                if c is None:
                    cells += ["-", "-"]
                else:
                    cells += [f"{_fmt(c['estimate'])} [{_fmt(c['ci_lo'])}, {_fmt(c['ci_hi'])}] "
                              f"[{_fmt(c['cluster_ci_lo'])}, {_fmt(c['cluster_ci_hi'])}]",
                              c["verdict"]]
            agree = (f"{_fmt(row['mean_agreement_with_gnb'], 3)} / "
                     f"{_fmt(row['min_agreement_with_gnb'], 3)}"
                     if row["mean_agreement_with_gnb"] is not None else "-")
            lines.append(f"| `{policy}` | {_fmt(row['mean_hit_ratio'])} | "
                         f"{_fmt(row['device_io_ratio'])} | {_fmt(row['blocks_moved_ratio'])} | "
                         f"{_fmt(row['mean_precision'], 3)} | {agree} | " + " | ".join(cells) + " |")
        lines.append("")
    (results / "HEADLINES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

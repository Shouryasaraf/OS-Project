"""E2: GPU disk models -- ``learned_depth`` per trace and seed, and the LSTM.

The disk half of ``reproduce.py disk --gpu`` (``repro/gpu.py`` dispatches
here, so the web half of that file stays WP11's). Also runnable alone::

    python -m repro.gpu_disk [--quick] [--traces STEM ...] [--seeds 0 1 ...]

Protocol (docs/MASTER_PLAN.md §2.5, constraints 7 and 14):

* each trace's first ``n`` canonical requests (FIU pages merged) are split
  60/20/20 chronologically; a model is trained on the first 60%, stopped
  early on the next 20%, its thresholds are grid-selected on (a prefix of)
  that validation segment, and it is scored once on the last 20%;
* the baselines get the same treatment: VCF (the fixed depth with the best
  validation hit ratio), the request-relative depth ``k * size`` (k chosen
  on validation), plus ``none``, ``sequential``, ``deep@8`` and
  ``linux_ra`` on the same test slice, so every comparison includes
  ``linux_ra``;
* every artifact's sidecar records its training and validation ranges, and
  scoring refuses any slice that overlaps them.

Scheduling: label building, validation grids and test replays run in a CPU
process pool (``workers``, 16 by default on this shared host); GPU training
runs one job at a time in the main process, with OOM retry at half batch and
an ``nvidia-smi`` sampler around every job. Everything is resumable: each
finished piece is a file under ``work_dir`` and is not recomputed.

Outputs (``results/gpu/`` for the full run, ``outputs/gpu/quick/`` for
``--quick``): ``disk_learned_depth.csv`` (schema-checked),
``disk_depth_baselines.csv``, ``disk_lstm.csv``, ``gpu_env.json`` and
``provenance.json``. The last three CSV/JSON files have no registered schema
yet; their columns are the constants below.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import EXIT_UNAVAILABLE

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

#: The four traces of the LSTM pilot (``$EVAL/gpu-models/v1_full.jsonl``).
LSTM_TRACES = ("msr_hm_0", "msr_prn_0", "fiu_homes-110108-112108",
               "fiu_ikki-110108-112108")
SEEDS = (0, 1, 2, 3, 4)

BASELINE_COLUMNS = (
    "trace", "corpus", "segment", "start", "end", "action_cap", "mode", "param",
    "chosen", "hit_ratio", "precision", "issued", "useful", "device_ios")
LSTM_COLUMNS = (
    "trace", "corpus", "seed", "status", "n_requests", "train_start", "train_end",
    "val_start", "val_end", "test_start", "test_end", "epochs_run", "best_epoch",
    "val_loss", "test_bucket_top1", "test_majority_top1", "test_repeat_top1",
    "lstm_hit_ratio", "lstm_precision", "lstm_issued", "none_hit_ratio",
    "sequential_hit_ratio", "deep8_hit_ratio", "linux_ra_hit_ratio",
    "infer_us_per_request", "device", "train_s", "gpu_contention")


@dataclass
class Config:
    traces: tuple[str, ...] = ()
    seeds: tuple[int, ...] = SEEDS
    n: int = 2_000_000
    epochs: int = 8
    batch_size: int = 8192
    patience: int | None = 2
    capacity: int = 2048
    val_replay: int | None = 100_000
    test_replay: int | None = None
    action_caps: tuple[int, ...] = (64, 256)
    decide_every: int = 32
    lstm_traces: tuple[str, ...] = LSTM_TRACES
    lstm_epochs: int = 8
    lstm_batch_size: int = 4096
    lstm_seed: int = 42
    device: str = "cuda"
    workers: int = 16
    worker_threads: int = 2
    block_root: Path = REPO / "data" / "block-traces"
    results_dir: Path = REPO / "results" / "gpu"
    work_dir: Path = REPO / "outputs" / "gpu" / "work"
    models_dir: Path = REPO / "models" / "depth"
    paths: dict = field(default_factory=dict)
    quick: bool = False

    @classmethod
    def default(cls, **overrides) -> "Config":
        from adaptive_prefetch.corpora import PAPER_SUBSET
        # msr_prn_0 is trained too (plan C2), whether or not the subset
        # constant already lists it.
        traces = tuple(PAPER_SUBSET) + (() if "msr_prn_0" in PAPER_SUBSET
                                        else ("msr_prn_0",))
        return replace(cls(traces=traces), **overrides)

    @classmethod
    def quick_run(cls, **overrides) -> "Config":
        quick = REPO / "outputs" / "gpu"
        # Work files and models stay outside the results directory, whose
        # every file the provenance record hashes.
        base = cls(traces=("msr_hm_0",), seeds=(0,), n=200_000, epochs=1,
                   val_replay=20_000, test_replay=20_000,
                   lstm_traces=("msr_hm_0",), lstm_epochs=1, device="auto",
                   workers=0, results_dir=quick / "quick",
                   work_dir=quick / "quick_work",
                   models_dir=quick / "quick_work" / "models", quick=True)
        return replace(base, **overrides)


def estimate(config: Config) -> str:
    if config.quick:
        return "about 2-3 min (one trace, 200k requests, one epoch, one seed)"
    jobs = len(config.traces) * len(config.seeds)
    # Pilot timings: a GPU job of 20-60 s with its CPU scoring overlapped,
    # and about 4 CPU-minutes of labels and baseline replays per trace.
    minutes = jobs * 1.0 + len(config.traces) * 4 / max(config.workers, 1) \
        + len(config.lstm_traces) * 2
    return (f"about {minutes / 60:.1f} h: {len(config.traces)} traces x "
            f"{len(config.seeds)} seeds = {jobs} GPU jobs run one at a time, "
            f"labels and replays on {config.workers} CPU workers, "
            f"{len(config.lstm_traces)} LSTM traces")


# ------------------------------------------------------------------ files

def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if row.get(c) is None else
                             (json.dumps(row[c]) if isinstance(row[c], (list, dict))
                              else row[c]) for c in columns])
    os.replace(tmp, path)


def _cache_path(config: Config, stem: str) -> Path:
    return config.work_dir / "prepared" / f"{stem}.pkl"


def _load_prepared(config: Config, stem: str):
    with _cache_path(config, stem).open("rb") as handle:
        return pickle.load(handle)


# ------------------------------------------------------------ CPU stages

def _row(metrics_row: dict) -> dict:
    return {k: metrics_row[k] for k in ("hit_ratio", "precision", "issued", "useful",
                                        "device_ios")}


def _prepare_and_baselines(config: Config, stem: str) -> str:
    """Load, label and cache one trace, then score every seed-free baseline."""
    from adaptive_prefetch import depth_model as D
    from adaptive_prefetch.corpora import CORPUS_OF
    D.configure_threads(config.worker_threads)
    cache = _cache_path(config, stem)
    if cache.is_file():
        prepared = _load_prepared(config, stem)
    else:
        prefix = D.load_prefix(config.paths[stem], "oracle", config.n)
        prepared = D.prepare(prefix, config.capacity)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
    out = config.work_dir / "baselines" / f"{stem}.json"
    if out.is_file():
        return stem
    corpus = CORPUS_OF(stem)
    val = D.validation_slice(prepared, config.val_replay)
    test = D.scoring_slice(prepared, config.test_replay)
    test_requests = prepared.prefix.requests(*test)
    rows, summary = [], {"per_cap": {}}

    def add(segment, span, cap, mode, param, chosen, row):
        rows.append({"trace": stem, "corpus": corpus, "segment": segment,
                     "start": span[0], "end": span[1], "action_cap": cap,
                     "mode": mode, "param": param, "chosen": chosen, **_row(row)})

    # Cap-independent references on the test slice.
    for mode, kwargs, key in (("none", {}, "none"), ("sequential", {}, "sequential"),
                              ("deep", {"read_ahead_depth": 8}, "deep8"),
                              ("linux_ra", {}, "linux_ra")):
        row = D._replay_row(test_requests, mode, prepared.capacity, **kwargs)
        add("test", test, None, mode, kwargs.get("read_ahead_depth"), None, row)
        summary[key] = row
    for cap in config.action_caps:
        depth, table = D.select_fixed_depth(prepared, cap, val_replay=config.val_replay)
        for entry in table:
            add("val", val, cap, "deep" if entry["depth"] else "none",
                entry["depth"], entry["depth"] == depth, entry)
        vcf = (D._replay_row(test_requests, "deep", prepared.capacity,
                             read_ahead_depth=depth) if depth
               else summary["none"])
        add("test", test, cap, "vcf", depth, True, vcf)
        k, table = D.select_relative_depth(prepared, cap, val_replay=config.val_replay)
        for entry in table:
            add("val", val, cap, "relsize", entry["k"], entry["k"] == k, entry)
        rel = D._replay_row(test_requests, "learned_depth", prepared.capacity,
                            depth_model=D.RelativeDepth(k, cap))
        add("test", test, cap, "relsize", k, True, rel)
        oracle = D._replay_row(test_requests, "learned_depth", prepared.capacity,
                               depth_model=D.oracle_policy(prepared, *test, cap,
                                                           config.decide_every))
        add("test", test, cap, "oracle_same_actions", None, None, oracle)
        summary["per_cap"][str(cap)] = {"vcf_depth": depth, "vcf": vcf,
                                        "relsize_k": k, "relsize": rel,
                                        "oracle": oracle}
    _write_json(out, {"trace": stem, "corpus": corpus, "n": len(prepared.prefix),
                      "split": prepared.split, "horizon": prepared.horizon,
                      "val_slice": val, "test_slice": test, "prep_s": prepared.prep_s,
                      "rows": rows, "summary": summary})
    return stem


def _artifact_path(config: Config, stem: str, seed: int) -> Path:
    return config.models_dir / f"{stem}_s{seed}.pt"


def _score_depth(config: Config, stem: str, seed: int, info: dict) -> str:
    """Select thresholds on validation, save them, then score the test slice."""
    from adaptive_prefetch import depth_model as D
    D.configure_threads(config.worker_threads)
    out = config.work_dir / "depth" / f"{stem}_s{seed}.json"
    if out.is_file():
        return stem
    prepared = _load_prepared(config, stem)
    artifact = _artifact_path(config, stem, seed)
    network = D.load_artifact(artifact)["network"]
    thresholds, validation = {}, {}
    for cap in config.action_caps:
        chosen, table = D.select_thresholds(network, prepared, cap,
                                            val_replay=config.val_replay,
                                            decide_every=config.decide_every)
        thresholds[str(cap)] = chosen
        validation[str(cap)] = table
    D.save_artifact(artifact, network, prepared, info, thresholds, config.decide_every)
    record = D.training_record(prepared, info, thresholds=thresholds,
                               decide_every=config.decide_every)
    D.write_training_record(artifact, record)
    test = D.scoring_slice(prepared, config.test_replay)
    conflict = D.scoring_conflict(D.read_training_record(artifact), stem, *test)
    if conflict:
        raise RuntimeError(f"refusing to score {artifact}: {conflict}")
    requests = prepared.prefix.requests(*test)
    scored = {}
    for cap in config.action_caps:
        policy = D.load_prefetcher(artifact, action_cap=cap,
                                   num_threads=config.worker_threads)
        row = D._replay_row(requests, "learned_depth", prepared.capacity,
                            depth_model=policy)
        row["decisions"] = policy.decisions
        row["jumps"] = policy.jumps
        row["infer_us_per_decision"] = (row["inference_ms"] * 1000 / policy.decisions
                                        if policy.decisions else None)
        scored[str(cap)] = row
    _write_json(out, {"trace": stem, "seed": seed, "thresholds": thresholds,
                      "validation": validation, "test_slice": test, "test": scored,
                      "info": info})
    return stem


def _score_lstm(config: Config, stem: str, info: dict) -> str:
    from adaptive_prefetch import depth_model as D
    from adaptive_prefetch import lstm as L
    D.configure_threads(config.worker_threads)
    out = config.work_dir / "lstm" / f"{stem}.json"
    if out.is_file():
        return stem
    artifact = config.models_dir / f"lstm_{stem}_s{config.lstm_seed}.pt"
    prefix = D.load_prefix(config.paths[stem], "oracle", config.n)
    (_, c1), _, (c2, n) = D.chronological_split(len(prefix))
    stop = n if config.test_replay is None else min(n, c2 + config.test_replay)
    conflict = D.scoring_conflict(D.read_training_record(artifact), stem, c2, stop)
    if conflict:
        raise RuntimeError(f"refusing to score {artifact}: {conflict}")
    requests = prefix.requests(c2, stop)
    network = L.load_predictor(artifact, num_threads=config.worker_threads).network
    reads = prefix.lba[c2:stop][prefix.read[c2:stop]]
    accuracy = L.bucket_accuracy(network, reads,
                                 reference_lbas=prefix.lba[:c1][prefix.read[:c1]])
    rows = {"lstm": D._replay_row(requests, "lstm", config.capacity,
                                  lstm_model_path=str(artifact))}
    for mode, kwargs in (("none", {}), ("sequential", {}),
                         ("deep", {"read_ahead_depth": 8}), ("linux_ra", {})):
        rows[mode] = D._replay_row(requests, mode, config.capacity, **kwargs)
    _write_json(out, {"trace": stem, "test_slice": [c2, stop], "accuracy": accuracy,
                      "rows": rows, "requests": len(requests), "info": info,
                      "n": len(prefix), "split": D.chronological_split(len(prefix))})
    return stem


class _Executor:
    """A spawn-context process pool, or inline execution when ``workers == 0``."""

    def __init__(self, workers: int):
        self.pool = (multiprocessing.get_context("spawn").Pool(workers)
                     if workers > 0 else None)
        self.pending = []

    def submit(self, fn, *args):
        if self.pool is None:
            self.pending.append(_Done(fn, args))
        else:
            self.pending.append(self.pool.apply_async(fn, args))

    def wait(self):
        errors = []
        for job in self.pending:
            try:
                job.get()
            except Exception as exc:  # reported, never swallowed
                errors.append(exc)
        self.pending = []
        return errors

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


class _Done:
    def __init__(self, fn, args):
        try:
            self.value, self.error = fn(*args), None
        except Exception as exc:  # surfaced by get()
            self.value, self.error = None, exc

    def get(self):
        if self.error is not None:
            raise self.error
        return self.value


# ------------------------------------------------------------- GPU stage

def _train_depth(config: Config, stem: str, seed: int, log) -> dict | None:
    """Fit on the GPU; returns the fit info (cached next to the artifact)."""
    from adaptive_prefetch import depth_model as D
    info_path = config.work_dir / "fit" / f"{stem}_s{seed}.json"
    cached = _read_json(info_path)
    if cached is not None and _artifact_path(config, stem, seed).is_file():
        return cached
    prepared = _load_prepared(config, stem)
    sampler = D.GpuSampler()
    with sampler:
        network, info = D.fit(prepared, seed=seed, epochs=config.epochs,
                              batch_size=config.batch_size, device=config.device,
                              patience=config.patience, progress=log)
    info["gpu_contention"] = sampler.summary() if info["device"] == "cuda" else None
    info["gpu_samples"] = sampler.samples
    D.save_artifact(_artifact_path(config, stem, seed), network, prepared, info,
                    {}, config.decide_every)
    _write_json(info_path, info)
    return info


def _train_lstm(config: Config, stem: str, log) -> dict:
    from adaptive_prefetch import lstm as L
    info_path = config.work_dir / "fit" / f"lstm_{stem}.json"
    artifact = config.models_dir / f"lstm_{stem}_s{config.lstm_seed}.pt"
    cached = _read_json(info_path)
    if cached is not None and artifact.is_file():
        return cached
    info = L.train_lstm_on_trace(artifact, config.paths[stem], format="oracle",
                                 n=config.n, epochs=config.lstm_epochs,
                                 batch_size=config.lstm_batch_size,
                                 seed=config.lstm_seed, device=config.device,
                                 patience=config.patience, progress=log)
    _write_json(info_path, info)
    return info


# ---------------------------------------------------------------- report

def _learned_depth_rows(config: Config, stems) -> list[dict]:
    rows = []
    for stem in stems:
        base = _read_json(config.work_dir / "baselines" / f"{stem}.json")
        if base is None:
            continue
        (a, b), (c, d), _ = base["split"]
        test = base["test_slice"]
        for seed in config.seeds:
            scored = _read_json(config.work_dir / "depth" / f"{stem}_s{seed}.json")
            for cap in config.action_caps:
                per_cap = base["summary"]["per_cap"][str(cap)]
                row = {
                    "trace": stem, "corpus": base["corpus"], "seed": seed,
                    "action_cap": cap, "capacity": config.capacity,
                    "n_requests": base["n"], "train_start": a, "train_end": b,
                    "val_start": c, "val_end": d, "test_start": test[0],
                    "test_end": test[1],
                    "vcf_depth": per_cap["vcf_depth"],
                    "vcf_hit_ratio": per_cap["vcf"]["hit_ratio"],
                    "vcf_issued": per_cap["vcf"]["issued"],
                    "relsize_k": per_cap["relsize_k"],
                    "relsize_hit_ratio": per_cap["relsize"]["hit_ratio"],
                    "deep8_hit_ratio": base["summary"]["deep8"]["hit_ratio"],
                    "none_hit_ratio": base["summary"]["none"]["hit_ratio"],
                    "oracle_same_actions_hit_ratio": per_cap["oracle"]["hit_ratio"],
                }
                if scored is None:
                    row.update(status="skipped: training or scoring did not finish",
                               thresholds=None, test_hit_ratio=None, test_issued=None,
                               test_precision=None, device=config.device, train_s=None,
                               infer_us_per_decision=None, gpu_contention=None)
                else:
                    test_row, info = scored["test"][str(cap)], scored["info"]
                    row.update(status="ok", thresholds=scored["thresholds"][str(cap)],
                               test_hit_ratio=test_row["hit_ratio"],
                               test_issued=test_row["issued"],
                               test_precision=test_row["precision"],
                               device=info["device"], train_s=info["train_s"],
                               infer_us_per_decision=test_row["infer_us_per_decision"],
                               gpu_contention=info.get("gpu_contention"))
                rows.append(row)
    return rows


def _lstm_rows(config: Config, stems) -> list[dict]:
    from adaptive_prefetch.corpora import CORPUS_OF
    rows = []
    for stem in stems:
        scored = _read_json(config.work_dir / "lstm" / f"{stem}.json")
        if scored is None:
            rows.append({"trace": stem, "corpus": CORPUS_OF(stem),
                         "seed": config.lstm_seed,
                         "status": "skipped: training or scoring did not finish"})
            continue
        (a, b), (c, d), _ = scored["split"]
        info, acc, r = scored["info"], scored["accuracy"], scored["rows"]
        rows.append({
            "trace": stem, "corpus": CORPUS_OF(stem), "seed": config.lstm_seed,
            "status": "ok", "n_requests": scored["n"], "train_start": a,
            "train_end": b, "val_start": c, "val_end": d,
            "test_start": scored["test_slice"][0], "test_end": scored["test_slice"][1],
            "epochs_run": info.get("epochs_run"), "best_epoch": info.get("best_epoch"),
            "val_loss": info.get("val_loss"), "test_bucket_top1": acc["top1"],
            "test_majority_top1": acc["majority"], "test_repeat_top1": acc["repeat"],
            "lstm_hit_ratio": r["lstm"]["hit_ratio"],
            "lstm_precision": r["lstm"]["precision"], "lstm_issued": r["lstm"]["issued"],
            "none_hit_ratio": r["none"]["hit_ratio"],
            "sequential_hit_ratio": r["sequential"]["hit_ratio"],
            "deep8_hit_ratio": r["deep"]["hit_ratio"],
            "linux_ra_hit_ratio": r["linux_ra"]["hit_ratio"],
            "infer_us_per_request": r["lstm"]["inference_ms"] * 1000 / scored["requests"],
            "device": info.get("device"), "train_s": info.get("train_s"),
            "gpu_contention": info.get("gpu_contention"),
        })
    return rows


def _gpu_env(config: Config) -> dict:
    env = {"device_requested": config.device, "torch": None, "cuda_device": None,
           "jobs": []}
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    for path in sorted((config.work_dir / "fit").glob("*.json")):
        info = _read_json(path) or {}
        env["jobs"].append({"job": path.stem, "device": info.get("device"),
                            "train_s": info.get("train_s"),
                            "oom_retries": info.get("oom_retries"),
                            "batch_size": info.get("batch_size"),
                            "gpu_contention": info.get("gpu_contention"),
                            "samples": info.get("gpu_samples")})
    env["note"] = ("The GPU is shared with other tenants; nvidia-smi samples include "
                   "this job's own use. Training and inference times are reported "
                   "with these samples and are not quoted as costs without them.")
    return env


def write_outputs(config: Config, stems, lstm_stems, missing, start, argv) -> Path:
    from prefetch_common import schemas
    from prefetch_common.provenance import write_provenance
    out = config.results_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = _learned_depth_rows(config, stems)
    schemas.write_rows("gpu/disk_learned_depth.csv", rows, out / "disk_learned_depth.csv")
    baseline_rows = []
    for stem in stems:
        base = _read_json(config.work_dir / "baselines" / f"{stem}.json")
        if base is not None:
            baseline_rows.extend(base["rows"])
    _write_csv(out / "disk_depth_baselines.csv", BASELINE_COLUMNS, baseline_rows)
    _write_csv(out / "disk_lstm.csv", LSTM_COLUMNS, _lstm_rows(config, lstm_stems))
    env = _gpu_env(config)
    env["missing_traces"] = sorted(missing)
    env["config"] = {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in asdict(config).items() if k != "paths"}
    _write_json(out / "gpu_env.json", env)
    inputs = [config.paths[s] for s in sorted(set(stems) | set(lstm_stems))]
    notes = ("E2 quick smoke run; not a result." if config.quick else
             "E2: learned_depth vs VCF per trace/seed/cap, LSTM on 4 traces. "
             "Simulated hit ratios on oracleGeneral traces replayed as reads.")
    if missing:
        notes += f" Traces not present and not run: {', '.join(sorted(missing))}."
    write_provenance(out, argv, inputs, study="gpu", start=start, notes=notes)
    return out


# ------------------------------------------------------------------- run

def run(config: Config, argv=None, log=print) -> int:
    from adaptive_prefetch.corpora import discover_block_traces
    from prefetch_common.provenance import snapshot
    start = snapshot()
    found = discover_block_traces(config.block_root)
    wanted = tuple(dict.fromkeys(config.traces + config.lstm_traces))
    missing = {s for s in wanted if s not in found}
    config.paths = {s: str(found[s]) for s in wanted if s in found}
    stems = [s for s in config.traces if s in found]
    lstm_stems = [s for s in config.lstm_traces if s in found]
    if not stems and not lstm_stems:
        log(f"no traces under {config.block_root}; nothing to run")
        return EXIT_UNAVAILABLE
    try:
        import torch
    except ImportError:
        log("PyTorch is not installed (pip install -e .[gpu]); E2 cannot run here")
        return EXIT_UNAVAILABLE
    if config.device == "cuda" and not torch.cuda.is_available():
        log("CUDA is not available; E2 needs the GPU (or pass --device cpu)")
        return EXIT_UNAVAILABLE
    for directory in (config.work_dir, config.models_dir, config.results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log(f"E2: {len(stems)} learned_depth traces x {len(config.seeds)} seeds, "
        f"{len(lstm_stems)} LSTM traces; missing: {sorted(missing) or 'none'}")
    executor = _Executor(config.workers)
    errors = []
    try:
        for stem in stems:
            executor.submit(_prepare_and_baselines, config, stem)
        errors += executor.wait()
        for stem in stems:
            if not _cache_path(config, stem).is_file():
                continue
            for seed in config.seeds:
                if (config.work_dir / "depth" / f"{stem}_s{seed}.json").is_file():
                    continue
                t0 = time.time()
                try:
                    info = _train_depth(config, stem, seed,
                                        lambda m, s=stem, k=seed: log(f"  {s} s{k}: {m}"))
                except RuntimeError as exc:
                    log(f"  {stem} seed {seed}: training failed: {exc}")
                    errors.append(exc)
                    continue
                log(f"{stem} seed {seed}: trained in {time.time() - t0:.0f} s "
                    f"({info.get('gpu_contention') or info['device']})")
                executor.submit(_score_depth, config, stem, seed,
                                {k: v for k, v in info.items() if k != "gpu_samples"})
        for stem in lstm_stems:
            try:
                info = _train_lstm(config, stem, lambda m, s=stem: log(f"  lstm {s}: {m}"))
            except RuntimeError as exc:
                log(f"  lstm {stem}: training failed: {exc}")
                errors.append(exc)
                continue
            executor.submit(_score_lstm, config, stem, info)
        errors += executor.wait()
    finally:
        executor.close()
    for error in errors:
        log(f"error: {error!r}")
    out = write_outputs(config, stems, lstm_stems, missing, start,
                        argv if argv is not None else sys.argv)
    log(f"wrote {out}")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--traces", nargs="+")
    parser.add_argument("--lstm-traces", nargs="*")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--n", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--test-replay", type=int)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {}
    for name in ("workers", "device", "n", "epochs", "test_replay", "results_dir",
                 "work_dir", "models_dir"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    for name in ("traces", "lstm_traces", "seeds"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = tuple(value)
    return (Config.quick_run(**overrides) if getattr(args, "quick", False)
            else Config.default(**overrides))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    print(f"estimated runtime: {estimate(config)}", flush=True)
    return run(config, ["python", "-m", "repro.gpu_disk", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    sys.exit(main())

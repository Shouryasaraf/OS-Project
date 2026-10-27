"""Experiment driver for the hindsight router (track HSR, ``results/hindsight``).

Stages, each resumable from its cache under ``--cache-dir``:

``record``  replay every fixed action in ``hindsight.ACTIONS`` over slice 1
            (training) and slice 2 (test) of every trace and capacity, saving
            the end-of-window counters, plus every slice's window features.
``train``   label slice-1 windows per objective, random-search every family
            with grouped CV over traces, fit the selected and per-family-best
            routers on all of slice 1 and the leave-one-corpus-out routers,
            persist them under ``models/hindsight/`` with training records.
``score``   replay the routers (batch-inferred schedules), the oracle
            schedules and ``adaptive_evidence`` on slice 2, plus the exact
            reproduction checks against ``results/disk/sweep.jsonl``.
``analyze`` write the results files through ``prefetch_common.schemas``.

Everything is simulated (LRU block-cache replay under ``LatencyModel(5,
100, 50)``, the multicorpus ``assumed`` model). Run by ``reproduce.py
hindsight``; ``--quick`` caps slices at 20,000 requests and writes
``outputs/hindsight-quick`` only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

from . import corpora, hindsight
from .simulator import LatencyModel, replay_stream
from .trace import iter_oracle_chunks

REPO = Path(__file__).resolve().parents[2]
LATENCY = LatencyModel(5.0, 100.0, 50.0)
CAPACITIES = (2048, 131072)
SLICE_REQUESTS = 1_000_000
CHUNK = 250_000
#: Decision points per trace in every training set (uniform subsample).
PER_TRACE = 4000
#: Decision points per training trace inside each CV fit (amendment A1).
CV_PER_TRACE = 2000
#: Objectives that also get a leave-one-corpus-out and a GNB-family replay.
LOCO_OBJECTIVES = ("hit",)
SWEEP = REPO / "results" / "disk" / "sweep.jsonl"


def _load1() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


class _Count:
    def __init__(self):
        self.requests = 0

    def wrap(self, chunks):
        for chunk in chunks:
            self.requests += len(chunk)
            yield chunk


def slice_bounds(index: int, n: int) -> tuple[int, int]:
    """``(skip, max_requests)`` of slice 1 (training) or 2 (test)."""
    if index not in (1, 2):
        raise ValueError("slice is 1 (training) or 2 (test)")
    return (0 if index == 1 else n), n


def _chunks(path, skip, n):
    return iter_oracle_chunks(path, CHUNK, skip_requests=skip, max_requests=n)


def _key(trace, skip, cap=None, action=None) -> str:
    parts = [trace, f"s{skip}"]
    if cap is not None:
        parts.append(f"c{cap}")
    if action is not None:
        parts.append(action.replace("@", "_"))
    return "__".join(parts)


def _write_gz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def read_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_row(metrics) -> dict:
    return {"read_blocks": metrics.read_blocks, "read_hits": metrics.read_hits,
            "hit_ratio": metrics.hit_ratio, "precision": metrics.prefetch_precision,
            "total_prefetches": metrics.prefetches,
            "useful_prefetches": metrics.useful_prefetches,
            "demand_ios": metrics.demand_ios, "prefetch_ios": metrics.prefetch_ios,
            "prefetch_ios_mergeable": metrics.prefetch_ios_mergeable,
            "device_ios": metrics.device_ios, "blocks_moved": metrics.blocks_moved,
            "policy_switches": metrics.policy_switches}


# ------------------------------------------------------------------ record

def record_job(job: tuple) -> str:
    """One fixed-action replay or feature pass; writes its cache file."""
    kind, trace, path, skip, n, cap, action, out = job
    out = Path(out)
    if out.exists():
        return str(out)
    started, load_start = time.monotonic(), _load1()
    count = _Count()
    if kind == "features":
        feats = hindsight.window_features(count.wrap(_chunks(path, skip, n)))
        payload = {"trace": trace, "skip": skip, "features": feats}
    else:
        metrics, ends = hindsight.record_windows(
            count.wrap(_chunks(path, skip, n)), action, cap, latency_model=LATENCY)
        deltas = hindsight.window_deltas(ends)
        payload = {"trace": trace, "skip": skip, "capacity": cap, "action": action,
                   "metrics": metrics_row(metrics), "deltas": deltas}
    payload.update(requests=count.requests, max_requests=n, load_start=load_start,
                   load_end=_load1(), wall_s=round(time.monotonic() - started, 3))
    _write_gz(out, payload)
    return str(out)


def _cost(job) -> float:
    kind, *_, action, _out = job
    if kind == "features":
        return 1.0
    return 1.0 + hindsight.action_depth(action) / 32.0


def record_jobs(traces: dict, cache: Path, n: int, capacities, slices=(1, 2)) -> list:
    jobs = []
    for trace, path in traces.items():
        for index in slices:
            skip, _ = slice_bounds(index, n)
            jobs.append(("features", trace, str(path), skip, n, None, None,
                         str(cache / "features" / f"{_key(trace, skip)}.json.gz")))
            for cap in capacities:
                for action in hindsight.ACTIONS:
                    jobs.append(("record", trace, str(path), skip, n, cap, action,
                                 str(cache / "records" / f"{_key(trace, skip, cap, action)}.json.gz")))
    return sorted(jobs, key=_cost, reverse=True)


def pool_map(function, items, workers: int):
    if workers <= 1 or len(items) <= 1:
        for item in items:
            yield function(item)
        return
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(min(workers, len(items)),
                                                  maxtasksperchild=20) as pool:
        yield from pool.imap_unordered(function, items, chunksize=1)


def run_record(traces, cache: Path, n: int, capacities, workers: int,
               slices=(1, 2), quiet: bool = False) -> None:
    jobs = [j for j in record_jobs(traces, cache, n, capacities, slices)
            if not Path(j[-1]).exists()]
    started = time.monotonic()
    for done, out in enumerate(pool_map(record_job, jobs, workers), 1):
        if not quiet:
            print(f"[record {done}/{len(jobs)} {time.monotonic() - started:.0f}s "
                  f"load {_load1()}] {Path(out).name}", flush=True)


def select_traces(root, only=None) -> dict:
    found = corpora.discover_block_traces(root)
    stems = [s for s in corpora.PAPER_SUBSET if only is None or s in only]
    missing = [s for s in stems if s not in found]
    if missing:
        raise SystemExit(f"missing traces under {root}: {missing}")
    return {s: found[s] for s in stems}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="all",
                        choices=("record", "train", "score", "analyze", "all"))
    parser.add_argument("--slices", default="1,2", help="record stage: slices to replay")
    parser.add_argument("--trace-root", default=str(REPO / "data" / "block-traces"))
    parser.add_argument("--traces", default=None, help="comma-separated subset of PAPER_SUBSET")
    parser.add_argument("--requests", type=int, default=SLICE_REQUESTS)
    parser.add_argument("--capacities", default=",".join(map(str, CAPACITIES)))
    parser.add_argument("--cache-dir", default=str(REPO / "outputs" / "hindsight"))
    parser.add_argument("--models-dir", default=str(REPO / "models" / "hindsight"))
    parser.add_argument("--results-dir", default=None,
                        help="where analyze writes; default the cache dir's results/")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--configs", type=int, default=None,
                        help="random-search configurations per family")
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    only = tuple(args.traces.split(",")) if args.traces else None
    traces = select_traces(args.trace_root, only)
    capacities = tuple(int(c) for c in args.capacities.split(","))
    cache = Path(args.cache_dir)
    if args.stage in ("record", "all"):
        slices = tuple(int(s) for s in args.slices.split(","))
        run_record(traces, cache, args.requests, capacities, args.workers, slices,
                   args.quiet)
    if args.stage in ("train", "all"):
        from .hindsight_pipeline import run_train
        run_train(args, traces, capacities, cache)
    if args.stage in ("score", "all"):
        from .hindsight_pipeline import run_score
        run_score(args, traces, capacities, cache)
    if args.stage in ("analyze", "all"):
        from .hindsight_pipeline import run_analyze
        run_analyze(args, traces, capacities, cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())

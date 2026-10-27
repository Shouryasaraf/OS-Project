"""Hindsight-labelled router over an expanded action space (track HSR).

``reproduce.py hindsight`` replays every fixed action of
``hindsight.ACTIONS`` over slice 1 and slice 2 of the 21 ``PAPER_SUBSET``
traces at 2048 and 131072 blocks, trains and selects the routers on slice 1
(grouped CV over traces), scores them once on slice 2, and writes
``results/hindsight/`` with ``provenance.json`` last. Pre-registration:
``results/hindsight/PREREGISTERED.md``.

``reproduce.py hindsight --quick`` runs the same pipeline on 4 traces (one
per corpus) at 20,000 requests per slice with 2 configurations per family
and writes ``outputs/hindsight-quick`` only: a smoke run is not a result.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "HSR"
IMPLEMENTED = True

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "hindsight"
QUICK_OUT = REPO / "outputs" / "hindsight-quick"
QUICK_TRACES = ("msr_hm_0", "fiu_homes-110108-112108", "alibabaBlock_19",
                "tencentBlock_1162")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=28,
                        help="processes (default 28, the track's allowance)")
    parser.add_argument("--trace-root", default=str(REPO / "data" / "block-traces"))
    parser.add_argument("--cache-dir", default=str(REPO / "outputs" / "hindsight"))
    parser.add_argument("--models-dir", default=str(REPO / "models" / "hindsight"))
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--reps", type=int, default=10_000)


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return "about 1 min: 4 traces x 20k requests per slice, 2 configs per family"
    return ("about 58 min wall on 28 workers here, measured at host load 32-157 "
            "(slice-1 records ~4.5 min, slice-2 records ~28 min overlapping the "
            "grouped-CV search ~11 min, slice-2 replays ~16 min); a pass over a "
            "complete cache under outputs/hindsight takes about 2.5 min")


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def run(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO / "src"))
    import importlib.util
    from adaptive_prefetch.corpora import discover_block_traces
    if not discover_block_traces(args.trace_root):
        print(f"[hindsight] unavailable: no oracleGeneral traces under {args.trace_root}")
        return EXIT_UNAVAILABLE
    if importlib.util.find_spec("sklearn") is None:
        print("[hindsight] unavailable: scikit-learn is not installed (the [web] extra)")
        return EXIT_UNAVAILABLE
    command = [sys.executable, "-m", "adaptive_prefetch.hindsight_experiment",
               "--trace-root", args.trace_root, "--workers", str(args.workers)]
    if args.quick:
        command += ["--requests", "20000", "--traces", ",".join(QUICK_TRACES),
                    "--configs", "2", "--reps", "1000",
                    "--cache-dir", str(QUICK_OUT / "cache"),
                    "--models-dir", str(QUICK_OUT / "models"),
                    "--results-dir", str(QUICK_OUT / "results"), "--quiet"]
    else:
        command += ["--cache-dir", args.cache_dir, "--models-dir", args.models_dir,
                    "--results-dir", args.results_dir, "--reps", str(args.reps)]
    from prefetch_common.provenance import snapshot, write_provenance
    start = snapshot()
    print(f"[hindsight] {' '.join(command)}", flush=True)
    status = subprocess.call(command, cwd=REPO, env=_env())
    if status:
        print(f"[hindsight] failed (exit {status})")
        return status
    if args.quick:
        print(f"[hindsight] quick run written under {QUICK_OUT} (not results/)")
        return 0
    import json
    traces = sorted(p for p in Path(args.trace_root).rglob("*.oracleGeneral*") if p.is_file())
    from adaptive_prefetch.corpora import PAPER_SUBSET, trace_stem
    traces = [p for p in traces if trace_stem(p) in PAPER_SUBSET]
    cache = Path(args.cache_dir)
    write_provenance(
        args.results_dir, [sys.executable, "reproduce.py", "hindsight", *sys.argv[2:]],
        inputs=[*traces, REPO / "results" / "disk" / "sweep.jsonl"], study="hindsight",
        start=start, gpu=False,
        notes=("Track HSR, simulated. Router trained on slice 1 [0, 1M) of the 21 "
               "PAPER_SUBSET traces with trajectory hindsight labels (an approximation "
               "of the counterfactual), selected by grouped CV over traces, scored once "
               "on slice 2 [1M, 2M). LatencyModel(5, 100, 50) for every row. Oracle rows "
               "are references, never policies. CPU timings are on a shared host and "
               "are not costs. Written after E1-E6 were known."),
        extra={"checks": json.loads((cache / "checks.json").read_text(encoding="utf-8")),
               "chosen": json.loads((cache / "train" / "chosen.json").read_text(
                   encoding="utf-8"))})
    print(f"[hindsight] wrote {args.results_dir}/provenance.json")
    return 0

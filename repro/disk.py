"""Disk study: tests, the main.py --quick path and the multi-corpus sweep (E1).

``reproduce.py disk --quick`` runs the unit suite, ``main.py --quick`` and the
``quick`` multi-corpus configuration (3 traces x every mode x 50k requests)
with its analysis. It writes only under ``outputs/`` (``outputs/disk-quick``),
never ``results/``: a smoke run is not a result.

``reproduce.py disk`` is experiment E1: configurations C1-C4 into
``results/disk/sweep.jsonl`` (resumable: re-running continues where a killed
run stopped), the whole-file corpus profile, the analysis CSVs, and finally
``results/disk/provenance.json`` with the consistency checks. It needs the
block-trace collection under ``data/block-traces`` and reports itself
unavailable without it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "WP10"
IMPLEMENTED = True

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "disk"
QUICK_OUT = REPO / "outputs" / "disk-quick"


def _default_workers() -> int:
    return max(1, min(100, (os.cpu_count() or 9) - 8))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=_default_workers(),
                        help="sweep processes (default min(100, cpus-8))")
    parser.add_argument("--configs", default="C1,C2,C3,C4",
                        help="comma-separated multicorpus configs for the full run")
    parser.add_argument("--trace-root", default=str(REPO / "data" / "block-traces"))
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--reps", type=int, default=10_000,
                        help="bootstrap resamples in the analysis")
    parser.add_argument("--skip-tests", action="store_true",
                        help="--quick: do not run the unit suite first")
    parser.add_argument("--skip-profile", action="store_true",
                        help="full run: do not rebuild corpus_profile.csv")
    parser.add_argument("--cache-dir", default=str(REPO / "outputs" / "multicorpus"),
                        help="full run: survey cache and the checks file")


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return ("about 2-5 min: unit suite (~40 s), main.py --quick (~10 s), "
                "3 traces x every mode x 50k requests plus analysis")
    return ("about 2.5-3 h wall on 100 workers: C1 1-1.5 h, C2 30-45 min, C3 ~20 min, "
            "C4 ~5 min, whole-file corpus profile in parallel, analysis a few minutes; "
            "scales with --workers")


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def _step(name: str, command: list[str]) -> int:
    print(f"[disk] {name}: {' '.join(command)}", flush=True)
    status = subprocess.call(command, cwd=REPO, env=_env())
    if status:
        print(f"[disk] {name} failed (exit {status})")
    return status


def _have_traces(root: str) -> bool:
    sys.path.insert(0, str(REPO / "src"))
    from adaptive_prefetch.corpora import discover_block_traces
    return bool(discover_block_traces(root))


def run(args: argparse.Namespace) -> int:
    python = sys.executable
    available = _have_traces(args.trace_root)
    if args.quick:
        return _quick(args, python, available)
    if not available:
        print(f"[disk] unavailable: no oracleGeneral traces under {args.trace_root}")
        return EXIT_UNAVAILABLE
    return _full(args, python)


def _quick(args: argparse.Namespace, python: str, available: bool) -> int:
    steps = []
    if not args.skip_tests:
        steps.append(("unit tests", [python, "-m", "unittest", "discover", "-s", "tests"]))
    steps.append(("main.py --quick", [python, "main.py", "--quick",
                                      "--output-dir", str(QUICK_OUT / "main")]))
    sweep = QUICK_OUT / "sweep.jsonl"
    workers = str(min(args.workers, 32))
    if available:
        steps += [
            ("multicorpus quick", [python, "-m", "adaptive_prefetch.multicorpus",
                                   "--config", "quick", "--workers", workers,
                                   "--out", str(sweep), "--trace-root", args.trace_root,
                                   "--cache-dir", str(QUICK_OUT / "cache"), "--quiet"]),
            ("analysis", [python, "-m", "adaptive_prefetch.analysis", "--sweep", str(sweep),
                          "--out", str(QUICK_OUT), "--reps", "1000", "--workers", workers]),
        ]
    # A smoke run starts fresh: resuming would keep rows from older code.
    if sweep.exists():
        sweep.unlink()
    for name, command in steps:
        status = _step(name, command)
        if status:
            return status
    if not available:
        print(f"[disk] multicorpus smoke unavailable: no oracleGeneral traces under "
              f"{args.trace_root}")
        return EXIT_UNAVAILABLE
    print(f"[disk] quick run written under {QUICK_OUT} (not results/)")
    return 0


def _full(args: argparse.Namespace, python: str) -> int:
    sys.path.insert(0, str(REPO / "src"))
    from prefetch_common.provenance import snapshot, write_provenance

    start = snapshot()
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    sweep = results / "sweep.jsonl"
    configs = [c for c in args.configs.split(",") if c]
    profile = None
    if not args.skip_profile:
        # Whole-file passes (fiu_cheetah is 442M records) run beside the sweep
        # on a few extra processes rather than after it.
        command = [python, "-m", "adaptive_prefetch.multicorpus", "--profile",
                   "--workers", "8", "--trace-root", args.trace_root,
                   "--out-profile", str(results / "corpus_profile.csv"), "--quiet"]
        print(f"[disk] corpus profile (background): {' '.join(command)}", flush=True)
        profile = subprocess.Popen(command, cwd=REPO, env=_env())
    command = [python, "-m", "adaptive_prefetch.multicorpus", "--workers", str(args.workers),
               "--out", str(sweep), "--trace-root", args.trace_root,
               "--cache-dir", args.cache_dir, "--quiet"]
    for config in configs:
        command += ["--config", config]
    status = _step("sweep " + ",".join(configs), command)
    if profile is not None:
        profile_status = profile.wait()
        if profile_status:
            print(f"[disk] corpus profile failed (exit {profile_status})")
            status = status or profile_status
    if status:
        return status
    # The checks live in provenance, not in a results file of their own.
    checks_path = Path(args.cache_dir) / "checks.json"
    status = _step("analysis", [
        python, "-m", "adaptive_prefetch.analysis", "--sweep", str(sweep),
        "--out", str(results), "--reps", str(args.reps),
        "--workers", str(min(args.workers, 64)), "--checks", str(checks_path)])
    if status:
        return status
    traces = sorted(p for p in Path(args.trace_root).rglob("*.oracleGeneral*") if p.is_file())
    write_provenance(
        results, [python, "reproduce.py", "disk", *sys.argv[2:]],
        inputs=traces, study="disk", start=start, gpu=False,
        notes=("E1 multi-corpus sweep, simulated. Cost model: assumed 5/100/50 us for "
               "every row; the feedback-gated modes also at break-even 1.0101 "
               "(1/100/100 us). Consistency checks are in extra.checks."),
        extra={"configs": configs,
               "checks": json.loads(checks_path.read_text(encoding="utf-8"))})
    print(f"[disk] wrote {results}/provenance.json")
    return 0

"""Web ranker alternatives (track WRK): tuned model families vs the shipped ranker.

Full run: ``python -m webprefetch.rankers run --results-dir
results/web_rankers`` over E4's prepared views and artifacts (it runs
``webprefetch prepare`` and ``train`` first when they are absent): random
search per family on split 1 / split 2, per-capacity policy selection on
split 2, one frozen replay of split 3, paired session bootstraps (10,000
resamples, seed 42), then ``results/web_rankers/provenance.json``.

``--quick`` is the bounded smoke path (two trials per family, 2,000
validation and 4,000 test views, 200 resamples). It writes under
``outputs/web/rankers-quick/`` and never touches ``results/web_rankers``.

Importing this module needs only the standard library.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "WRK"
IMPLEMENTED = True

REPO = Path(__file__).resolve().parents[1]
RAW = Path("data/raw/events.csv")
PROCESSED = Path("data/processed")
MODELS = Path("models/web")
RESULTS = Path("results/web_rankers")
QUICK = Path("outputs/web/rankers-quick")
MODULES = ("numpy", "pandas", "sklearn", "joblib", "scipy")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw", type=Path, default=RAW, help="RetailRocket events.csv")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                        help="processes (16 is this track's CPU allowance)")


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return "about 3-5 min (bounded search and replay)"
    return ("about 20-30 min at 16 workers: random search ~8 min, two full-split replays of "
            "six scorers ~10 min, bootstraps ~3 min")


def _run(arguments: list[str]) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO / "src"), env.get("PYTHONPATH")]))
    command = [sys.executable, "-m", *arguments]
    print("+ " + " ".join(command[1:]), flush=True)
    return subprocess.run(command, cwd=REPO, env=env).returncode


def run(args: argparse.Namespace) -> int:
    raw = args.raw if args.raw.is_absolute() else REPO / args.raw
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    if missing:
        print(f"reproduce.py web_rankers: this interpreter lacks {missing} (the [web] extra).")
        return EXIT_UNAVAILABLE
    prepared = REPO / PROCESSED / "views.csv.gz"
    artifacts = REPO / MODELS / "ranker.joblib"
    if not (prepared.exists() and artifacts.exists()):
        if not raw.is_file():
            print(f"reproduce.py web_rankers: {raw} is missing. Fetch the pinned RetailRocket "
                  "file with `python -m webprefetch download`.")
            return EXIT_UNAVAILABLE
        for step in (["webprefetch", "prepare", "--raw", str(raw)], ["webprefetch", "train"]):
            if _run(step):
                return 1
    sys.path.insert(0, str(REPO / "src"))
    from prefetch_common.provenance import snapshot, write_provenance

    start = snapshot()
    target = QUICK if args.quick else RESULTS
    command = ["webprefetch.rankers", "run", "--results-dir", str(target),
               "--workers", str(args.workers)]
    if args.quick:
        command += ["--trials", "logistic_tuned=2,hgb=2,rf=2,et=2,mlp=2",
                    "--validation-views", "2000", "--test-views", "4000", "--reps", "200",
                    "--train-decisions", "20000", "--tune-decisions", "20000",
                    "--artifacts", str(QUICK / "models")]
    status = _run(command)
    if status:
        print(f"reproduce.py web_rankers: the run failed (exit {status})")
        return status
    inputs = [prepared, artifacts, REPO / MODELS / "candidate_index.joblib"]
    note = ("bounded smoke run (--quick); not research results" if args.quick else
            "WRK: ranker families tuned on split 1/2, policies selected on split 2, split 3 "
            "replayed once; simulated hit ratios, assumed costs; timings contended")
    path = write_provenance(REPO / target, ["reproduce.py", "web_rankers", *sys.argv[2:]],
                            [p for p in inputs if p.exists()], study="web_rankers", start=start,
                            gpu=False, notes=note)
    print(f"reproduce.py web_rankers: done; provenance in {path}")
    return 0

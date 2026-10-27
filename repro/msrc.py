"""MSRC light sample (E7): labelled-window transfer check and sanity replay.

``reproduce.py msrc`` runs ``python -m adaptive_prefetch.msrc_eval`` over
every ``data/samples/msrc/*.revised`` file and writes
``results/msrc_samples/`` (traces, classifier metrics, paired deltas, the
replay) with ``provenance.json`` last. The rules are
``results/msrc_samples/PREREGISTERED.md``. It needs the ``[web]`` extra
(numpy, scikit-learn) and reports itself unavailable without it.

``--quick`` scores the first 4,000 requests of each file with 200 bootstrap
replicates and no replay, into ``outputs/msrc-quick/``, never ``results/``.

Every numeric library runs single-threaded (``OMP_NUM_THREADS=1`` and
friends) in a child process; ``--workers`` replay processes (default 8)
are the only parallelism, because the host is shared.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "WPMSRC"
IMPLEMENTED = True

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "msrc_samples"
QUICK_OUT = REPO / "outputs" / "msrc-quick"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=8, help="replay processes")
    parser.add_argument("--reps", type=int, default=2000, help="bootstrap replicates")
    parser.add_argument("--samples", default=str(REPO / "data" / "samples" / "msrc"))


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return "about 30 s: 4,000 requests per file, 200 replicates, no replay"
    return ("about 4-6 min on 8 workers for the seven-file sample (measured 2026-10-27 "
            "at load ~40): windows and classifiers ~1-2 min, replay ~3 min")


def command(args: argparse.Namespace) -> list[str]:
    out = QUICK_OUT if args.quick else RESULTS
    cmd = [sys.executable, "-m", "adaptive_prefetch.msrc_eval", "--samples", args.samples,
           "--out", str(out), "--workers", str(args.workers),
           "--reps", str(200 if args.quick else args.reps)]
    if args.quick:
        cmd += ["--max-requests", "4000"]
    return cmd


def run(args: argparse.Namespace) -> int:
    missing = [m for m in ("numpy", "sklearn") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"reproduce.py msrc: unavailable here: {', '.join(missing)} not installed "
              "(use .venv/bin/python)")
        return EXIT_UNAVAILABLE
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"),
               **{name: "1" for name in THREAD_VARS})
    return subprocess.call(command(args), cwd=REPO, env=env)

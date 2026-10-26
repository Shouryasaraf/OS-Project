"""Web study: prepare, train and the RetailRocket replay benchmark (E4).

Full run, in order (each step shells out to ``python -m webprefetch``):

1. ``prepare`` and ``train --novel-ranker`` (about 1 min),
2. ``benchmark`` over the logistic, relabelled-logistic and three
   transition scorers, with per-capacity validation selection (about
   10 min at 96 workers),
3. ``freshness``, ``decompose`` and ``parity``, then ``learning-curve``
   (about 10 min; it retrains the ranker at 100k and 300k decisions),
4. ``results/web/provenance.json``, written last so it hashes every output.

``--quick`` is the bounded smoke path: prepare, train and a benchmark over
1,000 validation and 2,000 test views. It writes under
``outputs/web/quick/`` and never touches the tracked ``results/web/``.
The GPU sequence model is ``reproduce.py web --gpu`` (repro/gpu.py).

Importing this module needs only the standard library; the steps need the
``[web]`` extra in the interpreter that runs ``reproduce.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "WP6"
IMPLEMENTED = True

REPO = Path(__file__).resolve().parents[1]
RAW = Path("data/raw/events.csv")
PROCESSED = Path("data/processed")
MODELS = Path("models/web")
OUTPUTS = Path("outputs/web")
RESULTS = Path("results/web")
QUICK = OUTPUTS / "quick"
#: Every scorer except ``seq``, which E5 adds (plan E4). ``logistic_novel`` is
#: the relabelled ranker that ``train --novel-ranker`` fits just before.
DEFAULT_SCORERS = "logistic,logistic_novel,trans_s0,trans_prior,trans_online"
WEB_MODULES = ("numpy", "pandas", "sklearn", "joblib")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw", type=Path, default=RAW, help="RetailRocket events.csv")
    parser.add_argument("--workers", type=int, default=min(40, os.cpu_count() or 1),
                        help="replay processes (40 is the cap when E1/E2 overlap)")
    parser.add_argument("--scorers", default=DEFAULT_SCORERS)
    parser.add_argument("--reuse-prepared", action="store_true",
                        help="skip prepare/train when data/processed and models/web exist")
    parser.add_argument("--no-learning-curve", action="store_true")


def estimate(args: argparse.Namespace) -> str:
    if args.quick:
        return "about 2-3 min (prepare ~35 s, train ~15 s, bounded benchmark)"
    return ("about 25-35 min: prepare+train ~1 min, benchmark ~10 min at 96 workers "
            "(longer at fewer), freshness/decompose/parity ~5 min, learning curve ~10 min")


def _step(arguments: list[str]) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO / "src"), env.get("PYTHONPATH")]))
    command = [sys.executable, "-m", "webprefetch", *arguments]
    print("+ " + " ".join(command[1:]), flush=True)
    return subprocess.run(command, cwd=REPO, env=env).returncode


def run(args: argparse.Namespace) -> int:
    raw = args.raw if args.raw.is_absolute() else REPO / args.raw
    if not raw.is_file():
        print(f"reproduce.py web: {raw} is missing. Fetch the pinned RetailRocket file with "
              "`python -m webprefetch download` (Kaggle, CC BY-NC-SA 4.0; see NOTICE.md).")
        return EXIT_UNAVAILABLE
    missing = [name for name in WEB_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        print(f"reproduce.py web: this interpreter lacks {missing} (the [web] extra).")
        return EXIT_UNAVAILABLE
    sys.path.insert(0, str(REPO / "src"))
    from prefetch_common.provenance import snapshot, write_provenance

    start = snapshot()
    common = ["--raw", str(raw), "--workers", str(args.workers)]
    prepared = (REPO / PROCESSED / "views.csv.gz").exists() and (REPO / MODELS / "ranker.joblib").exists()
    steps = []
    if not (args.reuse_prepared and prepared):
        steps += [["prepare", *common], ["train", "--novel-ranker", *common]]
    if args.quick:
        target = QUICK / "results"
        steps.append(["benchmark", "--quick", "--outputs", str(QUICK),
                      "--results-dir", str(target), "--scorers", "logistic", *common])
    else:
        target = RESULTS
        steps.append(["benchmark", "--scorers", args.scorers, "--results-dir", str(target), *common])
        steps += [["freshness", "--results-dir", str(target)],
                  ["decompose", "--results-dir", str(target), *common],
                  ["parity", "--results-dir", str(target)]]
        if not args.no_learning_curve:
            steps.append(["learning-curve", "--results-dir", str(target), *common])
    for arguments in steps:
        status = _step(arguments)
        if status:
            print(f"reproduce.py web: step `{arguments[0]}` failed (exit {status})")
            return status
    inputs = [raw, REPO / PROCESSED / "views.csv.gz", REPO / MODELS / "ranker.joblib",
              REPO / MODELS / "candidate_index.joblib"]
    note = ("bounded smoke run (--quick); not research results" if args.quick else
            "E4: per-capacity validation selection, frozen test replay; costs are assumed, not measured")
    path = write_provenance(REPO / target, ["reproduce.py", "web", *sys.argv[2:]],
                            [p for p in inputs if p.exists()], study="web", start=start,
                            gpu=False, notes=note)
    print(f"reproduce.py web: done; provenance in {path}")
    return 0

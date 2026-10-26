"""GPU training and scoring: disk learned_depth/LSTM (E2) and the web sequence model (E5).

``reproduce.py disk --gpu`` and ``reproduce.py web --gpu`` call
``run(study, args)`` here instead of the study module's own ``run``. The
disk half lives in ``repro/gpu_disk.py`` (WP9) and the web half, below,
belongs to WP11. ``args`` is the study's namespace (its own options plus
``--quick``); the disk step reads only ``--quick`` from it, and its other
options are ``python -m repro.gpu_disk --help``. See docs/MASTER_PLAN.md
sections 2.5 and 3.2.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import EXIT_UNAVAILABLE, not_implemented

OWNERS = {"disk": "WP9", "web": "WP11"}
IMPLEMENTED = {"disk": True, "web": True}
REPO = Path(__file__).resolve().parents[1]


def _disk_config(args: argparse.Namespace):
    from . import gpu_disk
    return gpu_disk.config_from_args(argparse.Namespace(quick=getattr(args, "quick", False)))


def estimate(study: str, args: argparse.Namespace) -> str:
    if study == "disk":
        from . import gpu_disk
        return gpu_disk.estimate(_disk_config(args))
    if study == "web":
        return _web_estimate(args)
    return "unknown (not yet implemented)"


def run(study: str, args: argparse.Namespace) -> int:
    if study not in OWNERS:
        raise ValueError(f"no GPU step for study {study!r}")
    if study == "disk":
        from . import gpu_disk
        argv = ["reproduce.py", "disk", "--gpu"] + (["--quick"] if getattr(args, "quick", False) else [])
        return gpu_disk.run(_disk_config(args), argv)
    if study == "web":
        return _web_run(args)
    return not_implemented(f"{study} --gpu", OWNERS[study])


# --------------------------------------------------------------- web (E5)
# The work is `python -m webprefetch.seqmodel experiment` (WP6 owns the web
# cli.py); this checks the prerequisites and shells out to it. A full run
# then re-runs E4's decomposition with the sequence-model lever (plan E5):
# `webprefetch decompose` reads E4's outputs/web/benchmark.json and replays
# the test split with every lever, now including seq (the final-protocol
# artifact of seed scorers.SEQ_SEED), into results/web/decomposition.csv.
# The other levers' rows come out as E4 wrote them (same sessions, same
# bootstrap seed). results/web/provenance.json is then rewritten so it
# hashes the new file, with E4's own record kept in extra.e4_provenance.

#: Printed with every E5 run: the history mask makes the plain ranking
#: columns incomparable across rankers.
SEQ_COMPARISON_NOTE = ("seq_metrics.csv: compare the sequence model with other rankers on "
                       "novel_hit_at_1 / novel_recall_at_8. Its history mask never ranks an "
                       "item already in the session window, so hit_at_1 / mrr_at_8 are not "
                       "comparable to an unmasked ranker's.")


def _web_paths():
    sys.path.insert(0, str(REPO / "src"))
    from webprefetch.paths import MODELS, OUTPUTS, PROCESSED, RAW
    return REPO / PROCESSED / "views.csv.gz", REPO / RAW, REPO / MODELS, REPO / OUTPUTS


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH")) if p)
    return env


def _cuda_available() -> bool:
    """Asked in a child process, so reproduce.py itself never imports torch."""
    probe = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    try:
        return subprocess.run([sys.executable, "-c", probe], env=_env(), timeout=300,
                              capture_output=True).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _web_command(args: argparse.Namespace, views: Path, models: Path, outputs: Path) -> list[str]:
    command = [sys.executable, "-m", "webprefetch.seqmodel", "experiment", "--views", str(views)]
    if getattr(args, "quick", False):
        # A smoke run: never written to results/, never a paper number.
        return command + ["--seeds", "0", "--protocols", "validation", "--epochs", "1",
                          "--max-views", "20000", "--out", str(outputs / "seq_quick"),
                          "--models", str(models / "seq_quick"), "--retrain", "--no-provenance"]
    return command + ["--out", str(REPO / "results" / "web"), "--models", str(models / "seq")]


def _web_decompose_command(args: argparse.Namespace, views: Path, models: Path,
                           outputs: Path) -> list[str]:
    """E5's re-run of E4's decomposition, with the seq lever."""
    workers = int(getattr(args, "workers", None) or min(40, os.cpu_count() or 1))
    return [sys.executable, "-m", "webprefetch", "decompose", "--processed", str(views.parent),
            "--models", str(models), "--outputs", str(outputs),
            "--results-dir", str(REPO / "results" / "web"), "--seq-model", str(models / "seq"),
            "--workers", str(workers)]


def _e4_missing(outputs: Path) -> str | None:
    """Why E4's outputs cannot feed the decomposition re-run, or None."""
    import json
    summary = outputs / "benchmark.json"
    if not summary.is_file():
        return f"{summary} is missing (run E4, `reproduce.py web`, first)"
    try:
        record = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"{summary} is unreadable ({error})"
    if record.get("partial_run"):
        return f"{summary} is from a bounded run, not E4's full benchmark"
    if "logistic" not in record.get("validation_selection_per_capacity", {}):
        return f"{summary} has no logistic selection"
    if not (REPO / "results" / "web" / "provenance.json").is_file():
        return "results/web/provenance.json is missing (E4 writes it last)"
    return None


def _seq_lever_written() -> bool:
    import csv
    path = REPO / "results" / "web" / "decomposition.csv"
    if not path.is_file():
        return False
    with open(path, newline="", encoding="utf-8") as handle:
        return any(row.get("lever") == "sequence_model" for row in csv.DictReader(handle))


def _refresh_web_provenance(start: dict, views: Path, models: Path, outputs: Path) -> Path:
    """results/web/provenance.json hashing every file now there, E4's record kept."""
    from prefetch_common import provenance
    from webprefetch.scorers import SEQ_SEED, seq_artifact
    directory = REPO / "results" / "web"
    previous = provenance.read_provenance(directory)
    e4 = (previous.get("extra") or {}).get("e4_provenance", previous)
    artifact = seq_artifact(models / "seq", 3)
    # The decomposition also replays the logistic ranker; config.json pins
    # the seq artifact's weights and vocabulary by SHA-256.
    inputs = [path for path in (views, models / "ranker.joblib", models / "candidate_index.joblib",
                                outputs / "benchmark.json", artifact / "config.json")
              if path.is_file()]
    return provenance.write_provenance(
        directory, ["reproduce.py", *sys.argv[1:]], inputs, study="web",
        start=start, gpu="auto",
        notes=(f"E4's outputs plus E5: seq_metrics.csv (its own record is seq/provenance.json) "
               f"and decomposition.csv re-run with the sequence_model lever ({artifact.name}, "
               f"seed {SEQ_SEED}, fixed before E5). E4's own record is extra.e4_provenance. "
               + SEQ_COMPARISON_NOTE),
        extra={"e4_provenance": e4})


def _web_estimate(args: argparse.Namespace) -> str:
    if getattr(args, "quick", False):
        return "about 2 min (seed 0, validation protocol, 1 epoch, 20k-view replay; needs CUDA)"
    return ("about 1.5-2 h on the shared GPU (5 seeds x 2 protocols x 20 epochs at about "
            "25 s per epoch) plus about 10 min of ranking and replay, then about 10 min to "
            "re-run E4's decomposition with the seq lever")


def _web_run(args: argparse.Namespace) -> int:
    views, raw, models, outputs = _web_paths()
    if not _cuda_available():
        print("reproduce.py web --gpu: unavailable here: torch with CUDA is required")
        return EXIT_UNAVAILABLE
    if not views.is_file():
        if not raw.is_file():
            print(f"reproduce.py web --gpu: unavailable here: neither {views} nor {raw} exists")
            return EXIT_UNAVAILABLE
        print(f"preparing {views} from {raw}", flush=True)
        status = subprocess.run([sys.executable, "-m", "webprefetch", "prepare"],
                                cwd=REPO, env=_env()).returncode
        if status:
            return status
    from prefetch_common.provenance import snapshot
    start = snapshot()
    command = _web_command(args, views, models, outputs)
    print("running: " + " ".join(command), flush=True)
    status = subprocess.run(command, cwd=REPO, env=_env()).returncode
    if status:
        return status
    print(SEQ_COMPARISON_NOTE, flush=True)
    if getattr(args, "quick", False):
        return 0
    missing = _e4_missing(outputs)
    if missing:
        print(f"reproduce.py web --gpu: decomposition re-run with the seq lever unavailable: "
              f"{missing}. seq_metrics.csv is written; run this command again after E4 (the "
              "trained artifacts are reused).")
        return EXIT_UNAVAILABLE
    command = _web_decompose_command(args, views, models, outputs)
    print("running: " + " ".join(command), flush=True)
    status = subprocess.run(command, cwd=REPO, env=_env()).returncode
    if status:
        return status
    if not _seq_lever_written():
        print("reproduce.py web --gpu: decomposition.csv has no sequence_model row (the seq "
              "scorer was skipped; see the [decompose] line above)")
        return 1
    path = _refresh_web_provenance(start, views, models, outputs)
    print(f"reproduce.py web --gpu: done; decomposition.csv now has the seq lever; "
          f"provenance in {path}")
    return 0

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
# cli.py); this only checks the prerequisites and shells out to it.

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


def _web_estimate(args: argparse.Namespace) -> str:
    if getattr(args, "quick", False):
        return "about 2 min (seed 0, validation protocol, 1 epoch, 20k-view replay; needs CUDA)"
    return ("about 1.5-2 h on the shared GPU (5 seeds x 2 protocols x 20 epochs at about "
            "25 s per epoch) plus about 10 min of ranking and replay")


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
    command = _web_command(args, views, models, outputs)
    print("running: " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO, env=_env()).returncode

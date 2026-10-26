"""GPU training and scoring: disk learned_depth/LSTM (E2) and the web sequence model (E5).

``reproduce.py disk --gpu`` and ``reproduce.py web --gpu`` call
``run(study, args)`` here instead of the study module's own ``run``. The
disk half lives in ``repro/gpu_disk.py`` (WP9) and the web half belongs to
WP11. ``args`` is the study's namespace (its own options plus ``--quick``);
the disk step reads only ``--quick`` from it, and its other options are
``python -m repro.gpu_disk --help``. See docs/MASTER_PLAN.md sections 2.5
and 3.2.
"""

from __future__ import annotations

import argparse

from . import not_implemented

OWNERS = {"disk": "WP9", "web": "WP11"}
IMPLEMENTED = {"disk": True, "web": False}


def _disk_config(args: argparse.Namespace):
    from . import gpu_disk
    return gpu_disk.config_from_args(argparse.Namespace(quick=getattr(args, "quick", False)))


def estimate(study: str, args: argparse.Namespace) -> str:
    if study == "disk":
        from . import gpu_disk
        return gpu_disk.estimate(_disk_config(args))
    return "unknown (not yet implemented)"


def run(study: str, args: argparse.Namespace) -> int:
    if study not in OWNERS:
        raise ValueError(f"no GPU step for study {study!r}")
    if study == "disk":
        from . import gpu_disk
        argv = ["reproduce.py", "disk", "--gpu"] + (["--quick"] if args.quick else [])
        return gpu_disk.run(_disk_config(args), argv)
    return not_implemented(f"{study} --gpu", OWNERS[study])

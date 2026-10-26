"""GPU training and scoring: disk learned_depth/LSTM (E2) and the web sequence model (E5).

Stub written by WP0. ``reproduce.py disk --gpu`` and ``reproduce.py web
--gpu`` call ``run(study, args)`` here instead of the study module's own
``run``; WP9 implements the disk half and WP11 the web half. ``args`` is the
study's namespace (its own options plus ``--quick``). See docs/MASTER_PLAN.md
sections 2.5 and 3.2.
"""

from __future__ import annotations

import argparse

from . import not_implemented

OWNERS = {"disk": "WP9", "web": "WP11"}
IMPLEMENTED = {"disk": False, "web": False}


def estimate(study: str, args: argparse.Namespace) -> str:
    return "unknown (not yet implemented)"


def run(study: str, args: argparse.Namespace) -> int:
    if study not in OWNERS:
        raise ValueError(f"no GPU step for study {study!r}")
    return not_implemented(f"{study} --gpu", OWNERS[study])

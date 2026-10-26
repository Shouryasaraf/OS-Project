"""Real Linux prefetch testbed: calibrate, grid and report (E3).

Stub written by WP0; WP8 replaces it. See ``repro/__init__.py`` for the
interface and docs/MASTER_PLAN.md for the experiment.
"""

from __future__ import annotations

import argparse

from . import not_implemented

OWNER = "WP8"
IMPLEMENTED = False


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None,
                        help="directory on the device under test; without it the step is unavailable")


def estimate(args: argparse.Namespace) -> str:
    return "unknown (not yet implemented)"


def run(args: argparse.Namespace) -> int:
    return not_implemented("realio", OWNER)

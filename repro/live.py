"""Real HTTP prefetch experiment with simulator parity (E6).

Stub written by WP0; WP12 replaces it. See ``repro/__init__.py`` for the
interface and docs/MASTER_PLAN.md for the experiment.
"""

from __future__ import annotations

import argparse

from . import not_implemented

OWNER = "WP12"
IMPLEMENTED = False


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store-dir", default=None,
                        help="object-store directory; without it the step is unavailable")


def estimate(args: argparse.Namespace) -> str:
    return "unknown (not yet implemented)"


def run(args: argparse.Namespace) -> int:
    return not_implemented("live", OWNER)

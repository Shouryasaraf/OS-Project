"""Disk study: tests, the main.py --quick path and the multi-corpus sweep (E1).

Stub written by WP0; WP10 replaces it. See ``repro/__init__.py`` for the
interface and docs/MASTER_PLAN.md for the experiment.
"""

from __future__ import annotations

import argparse

from . import not_implemented

OWNER = "WP10"
IMPLEMENTED = False


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def estimate(args: argparse.Namespace) -> str:
    return "unknown (not yet implemented)"


def run(args: argparse.Namespace) -> int:
    return not_implemented("disk", OWNER)

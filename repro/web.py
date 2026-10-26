"""Web study: prepare, train and the RetailRocket replay benchmark (E4).

Stub written by WP0; WP6 replaces it. See ``repro/__init__.py`` for the
interface and docs/MASTER_PLAN.md for the experiment.
"""

from __future__ import annotations

import argparse

from . import not_implemented

OWNER = "WP6"
IMPLEMENTED = False


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def estimate(args: argparse.Namespace) -> str:
    return "unknown (not yet implemented)"


def run(args: argparse.Namespace) -> int:
    return not_implemented("web", OWNER)

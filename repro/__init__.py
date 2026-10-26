"""One reproduce module per study, dispatched by ``reproduce.py``.

Each study module is owned by the work package that implements it, so
parallel work never edits one shared file:

==========  ========================================  =====
module      study                                     owner
==========  ========================================  =====
disk        multi-corpus simulator sweep (E1)         WP10
gpu         GPU training, disk (E2) and web (E5)      WP9 (disk), WP11 (web)
realio      real Linux prefetch testbed (E3)          WP8
web         RetailRocket replay benchmark (E4)        WP6
live        real HTTP prefetch experiment (E6)        WP12
paper       figures, tables, numbers from results/    WP13
==========  ========================================  =====

Interface every study module provides (``gpu`` differs; see its docstring)::

    OWNER: str
    IMPLEMENTED: bool
    def add_arguments(parser: argparse.ArgumentParser) -> None
    def estimate(args: argparse.Namespace) -> str
    def run(args: argparse.Namespace) -> int

``add_arguments`` adds study-specific options only; ``reproduce.py`` adds
``--quick`` (and ``--gpu`` for disk and web) itself. Every option needs a
default, because ``reproduce.py all`` builds each study's arguments without
asking. ``estimate`` is printed before ``run`` starts. ``run`` returns a
process exit status; a step that cannot run on this machine (no device, no
GPU) returns ``EXIT_UNAVAILABLE`` with a reason rather than faking a result.
"""

from __future__ import annotations

STUDIES = ("disk", "realio", "web", "live", "paper")

EXIT_NOT_IMPLEMENTED = 3
EXIT_UNAVAILABLE = 4


def not_implemented(study: str, owner: str) -> int:
    """The stub ``run``: say who implements the step, and fail visibly."""
    print(f"reproduce.py {study}: not yet implemented ({owner})")
    return EXIT_NOT_IMPLEMENTED

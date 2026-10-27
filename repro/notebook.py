"""Notebook: rebuild and execute notebooks/walkthrough.ipynb from results/.

    python reproduce.py notebook                # results/ -> notebooks/walkthrough.ipynb
    python reproduce.py notebook --out PATH     # write elsewhere

Runs no experiment. ``notebooks/build_notebook.py`` holds the cells; every
code cell is executed in order (no Jupyter kernel needed) and its outputs
are stored, so the committed notebook opens pre-rendered. Re-run it after
any change to ``results/`` (and after ``reproduce.py paper``, whose
``paper/decisions.py`` verdicts the notebook prints). Needs nbformat,
pandas and matplotlib (the ``[web,paper]`` extras); without them it
returns ``EXIT_UNAVAILABLE``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "NB"
IMPLEMENTED = True

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "notebooks" / "build_notebook.py"
REQUIRED = ("nbformat", "pandas", "matplotlib")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=None,
                        help="notebook to write (default notebooks/walkthrough.ipynb)")


def estimate(args: argparse.Namespace) -> str:
    return "under 1 min (reads results/ only; executes the notebook's cells)"


def missing_modules() -> list[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def load_builder():
    if "notebook_builder" not in sys.modules:
        spec = importlib.util.spec_from_file_location("notebook_builder", BUILDER)
        module = importlib.util.module_from_spec(spec)
        sys.modules["notebook_builder"] = module
        spec.loader.exec_module(module)
    return sys.modules["notebook_builder"]


def run(args: argparse.Namespace) -> int:
    missing = missing_modules()
    if missing:
        print(f"reproduce.py notebook: unavailable, missing {', '.join(missing)} "
              "(use the venv with the [web,paper] extras and nbformat)")
        return EXIT_UNAVAILABLE
    started = time.monotonic()
    builder = load_builder()
    out = Path(args.out) if args.out else builder.DEFAULT_OUT
    path = builder.build(out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB) "
          f"in {time.monotonic() - started:.1f} s")
    return 0

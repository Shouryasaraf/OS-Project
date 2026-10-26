"""One reproduce command per study.

    python reproduce.py disk [--quick] [--gpu]
    python reproduce.py realio [--device DIR]
    python reproduce.py web [--quick] [--gpu]
    python reproduce.py live [--store-dir DIR]
    python reproduce.py paper
    python reproduce.py all [--quick]

This file is only a dispatcher. Each study lives in ``repro/<study>.py`` and
is owned by the work package that implements it (table in
``repro/__init__.py``), so no two packages edit this file. The runtime
estimate is printed before a step starts. ``all`` runs the studies in the
order above, each with its defaults. It passes over a step that reports
itself unavailable on this machine (no device, no store directory) and
stops at any other failure; if anything was passed over it exits with
``EXIT_UNAVAILABLE`` so an incomplete run never looks like a complete one.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from repro import EXIT_UNAVAILABLE, STUDIES  # noqa: E402

GPU_STUDIES = ("disk", "web")


def _study(name: str):
    return importlib.import_module(f"repro.{name}")


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="study", required=True, metavar="study")
    parsers = {}
    for name in STUDIES:
        module = _study(name)
        summary = (module.__doc__ or name).strip().splitlines()[0]
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.add_argument("--quick", action="store_true", help="the bounded smoke path")
        if name in GPU_STUDIES:
            sub.add_argument("--gpu", action="store_true",
                             help="run the GPU step (repro/gpu.py) instead")
        module.add_arguments(sub)
        parsers[name] = sub
    every = commands.add_parser("all", help="every study in order (see above)")
    every.add_argument("--quick", action="store_true", help="the bounded smoke path")
    return parser, parsers


def run_study(name: str, args: argparse.Namespace) -> int:
    if getattr(args, "gpu", False):
        gpu = _study("gpu")
        print(f"[{name} --gpu] estimated runtime: {gpu.estimate(name, args)}", flush=True)
        return gpu.run(name, args)
    module = _study(name)
    print(f"[{name}] estimated runtime: {module.estimate(args)}", flush=True)
    return module.run(args)


def main(argv: list[str] | None = None) -> int:
    parser, parsers = build_parser()
    args = parser.parse_args(argv)
    if args.study != "all":
        return run_study(args.study, args)
    unavailable = []
    for name in STUDIES:
        status = run_study(name, parsers[name].parse_args(["--quick"] if args.quick else []))
        if status == EXIT_UNAVAILABLE:
            unavailable.append(name)
        elif status:
            print(f"reproduce.py all: stopped at {name} (exit {status})")
            return status
    if unavailable:
        print(f"reproduce.py all: incomplete; unavailable here: {', '.join(unavailable)}")
        return EXIT_UNAVAILABLE
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""``python -m webprefetch.live <command>``; see the package docstring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMANDS = ("plan", "store", "probe", "origin", "run")


def _package_doc() -> str:
    from . import __doc__ as doc
    return doc or ""


def _plan(argv) -> int:
    from . import run
    parser = argparse.ArgumentParser(prog="webprefetch.live plan",
                                     description="build (or reuse) a plan and its sim.json")
    run.add_arguments(parser)
    args = parser.parse_args(argv)
    _, base_names = run.resolve_policies(args.policies, args.baseline)
    directory, plan, sim = run.ensure_plan(args, base_names)
    print(json.dumps({"plan_dir": str(directory), "sessions": plan["sessions_selected"],
                      "views": plan["views"], "objects": len(plan["items"]),
                      "sim_totals": sim["totals"]}, indent=1))
    return 0


def _store(argv) -> int:
    from . import objects
    parser = argparse.ArgumentParser(prog="webprefetch.live store",
                                     description="build the object store for a plan")
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--size-median", type=int, default=objects.SIZE_MEDIAN)
    parser.add_argument("--size-sigma", type=float, default=objects.SIZE_SIGMA)
    args = parser.parse_args(argv)
    plan = json.loads((args.plan_dir / "plan.json").read_text())
    print(json.dumps(objects.build_store(plan["items"], args.store_dir, median=args.size_median,
                                         sigma=args.size_sigma), indent=1))
    return 0


def _probe(argv) -> int:
    from . import objects
    parser = argparse.ArgumentParser(prog="webprefetch.live probe",
                                     description="cold/warm/hinted page-cache probe of a store")
    parser.add_argument("--store", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = objects.probe(args.store, args.index, n=args.n,
                           label=args.label or objects.store_label(args.store))
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: python -m webprefetch.live {{{','.join(COMMANDS)}}} [options]\n\n"
              + (_package_doc()))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    command, rest = argv[0], argv[1:]
    if command == "origin":
        from .origin import main as origin_main
        return origin_main(rest)
    if command == "run":
        from .run import main as run_main
        return run_main(rest)
    return {"plan": _plan, "store": _store, "probe": _probe}[command](rest)


if __name__ == "__main__":
    sys.exit(main())

"""Paper: figures, tables and numbers.tex from results/ (runs no experiment).

    python reproduce.py paper                 # results/ -> paper/
    python reproduce.py paper --fixtures      # synthetic fixtures -> outputs/paper-fixture/

Runs the three generators (``paper/numbers.py``, ``paper/tables/make_tables.py``,
``paper/figures/make_figures.py``) and then ``paper/check.py --strict`` on the
directory they wrote. A missing results file never stops the run: its
figure, table or macro is written as a "MISSING" placeholder, and the
manifests (``numbers.manifest.json``, ``tables/manifest.json``,
``figures/manifest.json``) list what was missing and the SHA-256 of every
input, with whether it matches that study's ``provenance.json``.

Fixture output is never written into ``paper/`` itself: synthetic numbers
must not be committed where the paper reads them. ``--fixtures`` therefore
copies the hand-written sources (``main.tex``, ``refs.bib``, ``sections/``)
to ``outputs/paper-fixture/`` (gitignored) and generates there.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import time
from pathlib import Path

from . import EXIT_UNAVAILABLE

OWNER = "WP13"
IMPLEMENTED = True

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
RESULTS = REPO / "results"
FIXTURES = REPO / "tests" / "fixtures" / "results"
FIXTURE_OUT = REPO / "outputs" / "paper-fixture"
SOURCES = ("main.tex", "refs.bib", "sections")
GENERATORS = (("paper_numbers", "numbers.py"), ("paper_make_tables", "tables/make_tables.py"),
              ("paper_make_figures", "figures/make_figures.py"))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", type=Path, default=None,
                        help=f"results tree to read (default {RESULTS.relative_to(REPO)}/)")
    parser.add_argument("--fixtures", action="store_true",
                        help="read the synthetic tests/fixtures/results/ tree instead")
    parser.add_argument("--out", type=Path, default=None,
                        help="paper directory to write into (default paper/, or "
                             "outputs/paper-fixture/ with --fixtures)")
    parser.add_argument("--no-check", action="store_true",
                        help="skip paper/check.py --strict after generating")


def estimate(args: argparse.Namespace) -> str:
    return "under 1 min (reads results/ only; no experiment runs)"


def load(module: str, relative: str):
    """Import a paper/ script by path; paper/ is not a package, and putting it
    on sys.path would let paper/numbers.py shadow the stdlib ``numbers``."""
    if module not in sys.modules:
        spec = importlib.util.spec_from_file_location(module, PAPER / relative)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module] = loaded
        try:
            spec.loader.exec_module(loaded)
        except BaseException:
            # A half-run module (matplotlib missing) must not be returned
            # by the next load() as if it had imported.
            del sys.modules[module]
            raise
    return sys.modules[module]


def _is_fixture_tree(root: Path) -> bool:
    return (root / "FIXTURE").is_file()


def resolve(args: argparse.Namespace) -> tuple[Path, Path]:
    results = Path(args.results) if args.results else (FIXTURES if args.fixtures else RESULTS)
    if args.out:
        out = Path(args.out)
    else:
        out = FIXTURE_OUT if _is_fixture_tree(results) else PAPER
    return results.resolve(), out.resolve()


def prepare(out: Path) -> None:
    """Give a scratch paper directory the hand-written sources check.py needs."""
    if out == PAPER.resolve():
        return
    out.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        source, target = PAPER / name, out / name
        if source.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    results_root, out = resolve(args)
    if _is_fixture_tree(results_root) and out == PAPER.resolve():
        print("reproduce.py paper: refusing to write fixture-derived output into paper/; "
              "use --out DIR (default with --fixtures: outputs/paper-fixture/)")
        return 2
    prepare(out)
    genlib = load("paper_genlib", "genlib.py")
    results = genlib.Results(results_root)
    print(f"[paper] results: {results.display_root}"
          + ("  (FIXTURE DATA: synthetic, never results)" if results.fixture else ""))
    print(f"[paper] writing: {out}")
    outputs, unavailable = [], []
    for module, relative in GENERATORS:
        try:
            generator = load(module, relative)
        except ImportError as error:  # matplotlib is the paper extra
            unavailable.append(f"{relative}: {error}")
            continue
        produced = generator.generate(results, out)
        outputs += produced
        gaps = [o for o in produced if o.missing]
        print(f"[paper] {relative}: {len(produced)} output(s), {len(gaps)} with missing input")
    missing = sorted({item for o in outputs for item in o.missing})
    for item in missing:
        print(f"[paper]   MISSING: {item}")
    mismatched = sorted({name for o in outputs for name, record in o.inputs.items()
                         if record.get("provenance_match") is False})
    for name in mismatched:
        print(f"[paper]   PROVENANCE MISMATCH: {name} differs from its provenance.json hash")
    status = 0
    if not args.no_check:
        check = load("paper_check", "check.py")
        report = check.run_checks(out, strict=True)
        # An output whose generator could not be imported (matplotlib is the
        # paper extra) is missing because of that, not because of the paper:
        # that is reported once, as "unavailable", and exits with its status.
        skipped = tuple(f"generated by paper/{item.split(':')[0]}" for item in unavailable)
        errors = [f for f in report.errors if not (skipped and any(s in f.message
                                                                    for s in skipped))]
        for finding in errors:
            print(f"[paper]   {finding}")
        print(f"[paper] check.py --strict: {len(errors)} error(s), "
              f"{len(report.warnings) + len(report.errors) - len(errors)} warning(s)")
        status = 0 if not errors else 1
    print(f"[paper] done in {time.monotonic() - started:.1f} s")
    if unavailable:
        for item in unavailable:
            print(f"[paper] unavailable: {item}")
        return status or EXIT_UNAVAILABLE
    return status

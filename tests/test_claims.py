"""The claims ledger (paper/claims.py, results/claims/).

Every number macro the paper and the slides use, and every number cell of the
generated tables, has a ledger row naming the results file, the row, the
column and the raw value it was printed from. These tests hold the ledger to
that: (a) every used macro has a row whose source exists under results/;
(b) re-reading each raw value from its file with its selector, and
re-formatting it with the recorded formatter, gives the printed text; (c) the
committed ledger is exactly what a fresh regeneration writes; (d) a macro
with no traceable source fails loudly, in numbers.Macros and in
paper/check.py --strict.

Stdlib only.
"""

import csv
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
LEDGER = RESULTS / "claims" / "paper_claims.csv"
sys.path.insert(0, str(ROOT / "src"))

from prefetch_common import provenance, schemas  # noqa: E402


def _load(name, relative):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PAPER / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


genlib = _load("paper_genlib", "genlib.py")
numbers_py = _load("paper_numbers", "numbers.py")
claims = _load("paper_claims", "claims.py")
tables_py = _load("paper_make_tables", "tables/make_tables.py")
check = _load("paper_check", "check.py")


def ledger_rows():
    return schemas.read_rows("claims/paper_claims.csv", LEDGER)


def defined_macros():
    text = (PAPER / "numbers.tex").read_text(encoding="utf-8")
    return dict(check._NEWCOMMAND.findall(text))


class _Files:
    """Results files read fresh (plain dicts, no tracing), once each."""

    def __init__(self):
        self.cache = {}

    def rows(self, relative):
        if relative not in self.cache:
            path = RESULTS / relative
            try:
                self.cache[relative] = schemas.read_rows(relative, path)
            except (KeyError, ValueError):
                self.cache[relative] = [json.loads(path.read_text(encoding="utf-8"))]
        return self.cache[relative]


def _column(row, column):
    """A column, a nested JSON path (host.cpu_count, gpu[0].name)."""
    value = row
    for part in column.split("."):
        if "[" in part:
            key, _, index = part.partition("[")
            value = value[key][int(index.rstrip("]"))]
        else:
            value = value[part]
    return value


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _resolve(value):
    if isinstance(value, dict) and set(value) == {"fn"}:
        return genlib.FORMATTERS[value["fn"]]
    return value


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ledger_rows()
        cls.by_id = {r["claim_id"]: r for r in cls.rows}
        cls.defined = defined_macros()

    def test_a_every_used_macro_has_a_traced_row(self):
        used = claims.uses(PAPER, self.defined, claims.SLIDES)
        names = {n for found in used.values() for n in found}
        self.assertTrue(names)
        for name in sorted(names):
            with self.subTest(macro=name):
                row = self.by_id.get(name)
                self.assertIsNotNone(row, f"\\{name} has no ledger row")
                self.assertNotEqual(row["kind"], "missing")
                self.assertEqual(row["printed"], self.defined[name])
                if row["kind"] == "constant":
                    self.assertIn(row["source_file"], numbers_py.CODE_SOURCES)
                    continue
                for relative in row["source_file"].split(";"):
                    self.assertTrue((RESULTS / relative).is_file(), relative)
                if row["kind"] == "decision":
                    self.assertTrue(row["rule"] and row["verdict"])
                    self.assertTrue((ROOT / row["preregistered"]).is_file())
        # Every generated table has its row, and its cells their sources.
        tables = {r["claim_id"] for r in self.rows if r["kind"] == "table"}
        self.assertEqual(tables, {f"tables/{f.__name__}.tex" for f in
                                  tables_py.TABLES})

    def test_b_every_raw_value_rereads_and_reformats_to_the_printed_text(self):
        files = _Files()
        reread = 0
        for row in self.rows:
            with self.subTest(claim=row["claim_id"]):
                if row["selector"] is not None:
                    relative = row["source_file"]
                    selector = row["selector"]
                    candidates = files.rows(relative)
                    if "_row" in selector:
                        match = [candidates[selector["_row"]]]
                    else:
                        match = [r for r in candidates
                                 if all(r.get(k) == v for k, v in selector.items())]
                    self.assertEqual(len(match), 1, f"selector {selector} in {relative}")
                    columns = row["columns"].split(";")
                    self.assertEqual(len(columns), len(row["raw"]))
                    for column, raw in zip(columns, row["raw"]):
                        name, _, scale = column.partition("*")
                        value = _column(match[0], name)
                        if scale:
                            value = int(scale) * value
                        self.assertTrue(_same(value, raw), f"{column}: {value!r} != {raw!r}")
                    reread += 1
                # Re-format the raw values with the recorded formatter calls.
                raws = list(row["raw"])
                for name, args, kwargs, printed in row["formatter"]:
                    function = genlib.FORMATTERS[name]
                    take, raws = raws[:function.traced_raw], raws[function.traced_raw:]
                    text = function(*take, *[_resolve(a) for a in args],
                                    **{k: _resolve(v) for k, v in kwargs.items()})
                    self.assertEqual(text, printed)
                    self.assertIn(printed, row["printed"])
                self.assertEqual(raws, [], "raw values no formatter printed")
                if len(row["formatter"]) == 1 and row["kind"] != "table":
                    if row["printed"] == row["formatter"][0][3]:
                        continue
                    # A unit appended outside the formatter (ms, KiB): the
                    # formatter's text is still the printed number.
                    self.assertTrue(row["printed"].startswith(row["formatter"][0][3]))
        self.assertGreater(reread, len(self.rows) // 2)

    def test_c_committed_ledger_is_current(self):
        results = genlib.Results(RESULTS)
        rows, text = claims.build(results, PAPER)
        with tempfile.TemporaryDirectory() as tmp:
            claims.write(rows, text, Path(tmp))
            for name in (claims.LEDGER, claims.READABLE):
                with self.subTest(file=name):
                    self.assertEqual((Path(tmp) / name).read_bytes(),
                                     (RESULTS / "claims" / name).read_bytes(),
                                     f"results/claims/{name} is stale: run "
                                     "`python reproduce.py paper`")
        self.assertEqual(provenance.verify_provenance(RESULTS / "claims"), [])

    def test_d_a_macro_without_a_source_fails_loudly(self):
        previous = genlib.TRACE.active
        genlib.TRACE.reset()
        genlib.TRACE.active = True
        try:
            m = numbers_py.Macros(False)
            with self.assertRaises(numbers_py.UntracedMacro):
                m.add("newFamilyValue", "$1.00$", "disk/table_main.csv")  # typed, untraced
            with self.assertRaises(numbers_py.UntracedMacro):
                m.add("newFamilyOther", genlib.fnum(1.0), "nowhere/never_read.csv")
            genlib.Results(RESULTS).rows("disk/table_main.csv")
            m.add("newFamilyOk", genlib.fnum(2.0), "disk/table_main.csv")
            self.assertEqual(m.traces["newFamilyOk"]["kind"], "derived")
        finally:
            genlib.TRACE.active = previous
            genlib.TRACE.reset()

    def test_d_check_strict_fails_on_an_untraced_used_macro(self):
        used = [n for n in claims.uses(PAPER, self.defined, None).get("sections/abstract.tex",
                                                                      [])
                if check.NUMBER_MACRO.fullmatch("\\" + n)]
        self.assertTrue(used)
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp) / "paper"
            shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
            self.assertEqual(check.run_checks(paper, strict=True).errors, [],
                             "no ledger beside a scratch copy is not an error")
            (paper / "claims").mkdir()
            with open(LEDGER, newline="", encoding="utf-8") as handle:
                lines = list(csv.reader(handle))
            kept = [lines[0]] + [r for r in lines[1:] if r[0] != used[0]]
            with open(paper / "claims" / "paper_claims.csv", "w", newline="",
                      encoding="utf-8") as handle:
                csv.writer(handle).writerows(kept)
            errors = check.run_checks(paper, strict=True).errors
            self.assertTrue(any("not traced in the claims ledger" in e.message
                                and used[0] in e.message for e in errors), errors)
            with open(paper / "claims" / "paper_claims.csv", "w", newline="",
                      encoding="utf-8") as handle:
                csv.writer(handle).writerows(lines)
            self.assertEqual(check.run_checks(paper, strict=True).errors, [])


if __name__ == "__main__":
    unittest.main()

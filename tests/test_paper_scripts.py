"""The paper generators: paper/numbers.py, tables/make_tables.py, figures/make_figures.py.

They are built against the schema-exact synthetic fixtures in
tests/fixtures/results/ (MASTER_PLAN WP13). These tests pin that every
output is produced from a fixture tree and from an empty one (as a MISSING
placeholder), that macro names are valid TeX control words, that the
numbers printed are the numbers in the results files, that generated LaTeX
is structurally sound for paper/check.py, that figures are deterministic
vector PDFs at IEEE widths with embedded TrueType fonts, and that the
manifests trace every output to its inputs' hashes and provenance.

The numbers and tables tests are stdlib-only; the figure tests skip without
matplotlib.
"""

import contextlib
import importlib.util
import io
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIXTURES = ROOT / "tests" / "fixtures" / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from prefetch_common import schemas  # noqa: E402

HAVE_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _load(name, relative):
    """paper/ is not a package (and must not be on sys.path: numbers.py would
    shadow the stdlib module), so its scripts are loaded by file path."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PAPER / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


genlib = _load("paper_genlib", "genlib.py")
numbers_py = _load("paper_numbers", "numbers.py")
tables_py = _load("paper_make_tables", "tables/make_tables.py")
check = _load("paper_check", "check.py")

TABLE_NAMES = tuple(f.__name__ for f in tables_py.TABLES)
FIGURE_NAMES = ("f1_schematic", "f2_hit_vs_depth", "f3_device_io_vs_hit", "f4_kappa_cdf",
                "f5_sim_vs_measured", "f6_freshness", "f7_live_saved_vs_bytes",
                "f8_utility_envelopes", "f9_action_space")
_NEWCOMMAND = re.compile(r"^\\newcommand\{\\([^}]*)\}\{(.*)\}$")


def _macros(text):
    found = {}
    for line in text.splitlines():
        match = _NEWCOMMAND.match(line)
        if match:
            if match.group(1) in found:
                raise AssertionError(f"\\{match.group(1)} defined twice")
            found[match.group(1)] = match.group(2)
    return found


def _rows(relative):
    return schemas.read_rows(relative, FIXTURES / relative)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def fixture_copy(self) -> Path:
        target = self.tmp / "results"
        shutil.copytree(FIXTURES, target)
        return target


class HelperTests(unittest.TestCase):
    def test_macro_name_parts_are_letters_only(self):
        self.assertEqual(genlib.camel("deep@8"), "DeepAtEight")
        self.assertEqual(genlib.camel("linux_ra@256"), "LinuxRaAtTwoHundredFiftySix")
        self.assertEqual(genlib.camel("msr_hm_0"), "MsrHmZero")
        self.assertEqual(genlib.camel("0.01"), "ZeroPointZeroOne")
        self.assertEqual(genlib.camel("spread:1:5000"), "SpreadOneFiveThousand")
        self.assertEqual(genlib.int_words(131072), "OneHundredThirtyOneThousandSeventyTwo")
        self.assertEqual(genlib.disk_capacity_word(2048), "OneMiB")
        self.assertEqual(genlib.disk_capacity_word(131072), "SixtyFourMiB")
        self.assertEqual(genlib.capacity_label(1048576), "512 MiB")
        self.assertTrue(genlib.is_macro_name("diskHit" + genlib.camel("deep@8")))

    def test_number_formats(self):
        self.assertEqual(genlib.fpp(-0.0089), r"$-0.89$\,pp")
        self.assertEqual(genlib.fpp(0.041), r"$+4.10$\,pp")
        self.assertEqual(genlib.fpp(-0.00001), r"$0.00$\,pp")  # never -0.00
        self.assertEqual(genlib.fnum(-0.0), "$0.00$")
        self.assertEqual(genlib.fpct(0.357231), r"$35.72$\,\%")
        self.assertEqual(genlib.fratio(0.6589), r"$0.659\times$")
        self.assertEqual(genlib.fcount(2756101), "2{,}756{,}101")
        self.assertEqual(genlib.fnum(float("inf")), r"$\infty$")
        self.assertEqual(genlib.fnum(None), genlib.DASH)
        self.assertEqual(genlib.fci(-0.01, 0.02, genlib.fpp), "[$-1.00$, $+2.00$]")
        self.assertEqual(genlib.esc("linux_ra@256 50%"), r"linux\_ra@256 50\%")

    def test_policies_sort_in_mode_order_then_by_parameter(self):
        policies = ["deep@256", "linux_ra@64", "none", "deep@8", "guard", "sequential"]
        self.assertEqual(sorted(policies, key=genlib.policy_key),
                         ["none", "sequential", "guard", "deep@8", "deep@256", "linux_ra@64"])


class FixtureTests(unittest.TestCase):
    def test_fixture_tree_is_marked_and_schema_exact(self):
        self.assertTrue((FIXTURES / genlib.FIXTURE_MARKER).is_file())
        self.assertTrue(genlib.Results(FIXTURES).fixture)
        self.assertFalse(genlib.Results(ROOT / "results").fixture)
        self.assertEqual(schemas.validate_tree(FIXTURES), {})
        # Every schema'd file has a fixture, so every generator path is exercised.
        for name in schemas.SCHEMAS:
            if "*" in name:
                continue
            self.assertTrue((FIXTURES / name).is_file(), name)
        self.assertTrue(list(FIXTURES.glob("realio/device_*.json")))

    def test_fixture_counts_are_physically_possible(self):
        # Synthetic, but never impossible: a negative device-I/O count once
        # produced a negative device-I/O ratio for linux_ra in T2.
        sweep = [r for r in _rows("disk/sweep.jsonl") if r["status"] == "ok"]
        self.assertTrue(all(r["device_ios"] > 0 for r in sweep))
        self.assertTrue(all(r["device_io_ratio"] > 0 for r in _rows("disk/table_main.csv")
                            if r["device_io_ratio"] is not None))

    def test_every_fixture_provenance_matches_its_files(self):
        from prefetch_common.provenance import verify_provenance
        for study in ("disk", "gpu", "realio", "web", "live"):
            self.assertEqual(verify_provenance(FIXTURES / study), [], study)


class NumbersTests(_Tmp):
    def build(self, root):
        numbers_py.generate(genlib.Results(root), self.tmp)
        return (self.tmp / "numbers.tex").read_text(encoding="utf-8")

    def test_fixture_numbers_are_the_results_values(self):
        text = self.build(FIXTURES)
        macros = _macros(text)
        self.assertIn("FIXTURE DATA", text)
        self.assertTrue(all(genlib.is_macro_name(name) for name in macros))
        delta = next(r for r in _rows("disk/deltas.csv")
                     if (r["config"], r["merge"], r["latency"], r["corpus"], r["metric"],
                         r["policy"], r["baseline"], r["capacity"])
                     == ("C1", "auto", "assumed", "all", "hit_ratio", "adaptive",
                         "sequential", 2048))
        expected = genlib.fpp(delta["estimate"])
        self.assertEqual(macros["diskAdaptiveVsSeq"], r"\fixturemark{" + expected + "}")
        self.assertEqual(macros["diskDeltaHitRatioAdaptiveVsSequentialOneMiB"],
                         macros["diskAdaptiveVsSeq"])
        self.assertIn(genlib.fci(delta["ci_lo"], delta["ci_hi"], genlib.fpp),
                      macros["diskAdaptiveVsSeqCI"])
        main = next(r for r in _rows("disk/table_main.csv")
                    if (r["config"], r["capacity"], r["policy"], r["latency"])
                    == ("C1", 131072, "deep@8", "assumed"))
        self.assertEqual(macros["diskHitDeepAtEightSixtyFourMiB"],
                         r"\fixturemark{" + genlib.fpct(main["mean_hit_ratio"]) + "}")
        self.assertIn("diskBreakEvenVsNoneGuardOneMiBCI", macros)
        self.assertIn("realioLatSdbOdirectDeepAtEightClosedMsrHmZeroCI", macros)
        self.assertIn("liveSavedHddFixedFourCI", macros)
        self.assertIn("webCoverageOnlineKEight", macros)
        # The fixture holds one invalid live pass; it is counted, not hidden.
        self.assertEqual(macros["liveInvalidPasses"], r"\fixturemark{1}")

    def test_relative_errors_print_as_percent_and_paired_counts_are_counted(self):
        macros = _macros(self.build(FIXTURES))
        row = next(r for r in _rows("realio/realio_summary.csv") if r["pred_err"] is not None)
        name = ("realioPredErr" + genlib.camel(row["device"]) + genlib.camel(row["backend"])
                + genlib.camel(row["mode"]) + genlib.camel(row["think"])
                + genlib.camel(row["trace"]))
        # (pred - measured) / measured: a percent, never percentage points.
        self.assertEqual(macros[name], r"\fixturemark{" + numbers_py.fsignedpct(row["pred_err"])
                         + "}")
        self.assertNotIn("pp", macros[name])
        traces = {r["trace"] for r in _rows("gpu/disk_learned_depth.csv") if r["status"] == "ok"
                  and r["vcf_hit_ratio"] is not None}
        self.assertEqual(macros["ldVsVcfN"], r"\fixturemark{" + genlib.fcount(len(traces)) + "}")

    def test_live_savings_skip_valid_passes_without_a_valid_baseline(self):
        root = self.fixture_copy()
        (root / "FIXTURE").unlink()
        rows = _rows("live/live_metrics.csv")
        group = [r for r in rows if (r["experiment"], r["store"], r["policy"])
                 == ("headline", "hdd", "fixed4") and r["valid"]]
        self.assertGreater(len(group), 1)
        # webprefetch.live.analyze writes NaN deltas when the baseline pass is invalid.
        for key in ("delta_vs_base_ms", "delta_vs_base_ms_lo", "delta_vs_base_ms_hi",
                    "ideal_delta_vs_base_ms", "bytes_vs_base"):
            group[0][key] = float("nan")
        schemas.write_rows("live/live_metrics.csv", rows, root / "live" / "live_metrics.csv")
        macros = _macros(self.build(root))
        kept = group[1:]
        expected = -sum(r["delta_vs_base_ms"] for r in kept) / len(kept)
        self.assertEqual(macros["liveSavedHddFixedFour"], numbers_py.fms(expected))
        self.assertEqual(macros["liveComparedRepsHddFixedFour"], genlib.fcount(len(kept)))
        self.assertEqual(macros["liveRepsHddFixedFour"], genlib.fcount(len(group)))
        tables = tables_py.generate(genlib.Results(root), self.tmp)
        self.assertTrue(tables)
        text = (self.tmp / "tables" / "t7_live.tex").read_text(encoding="utf-8")
        self.assertNotIn("nan", text.lower())
        self.assertIn("1 valid row(s) have no valid baseline pass", text)

    def test_curated_macros_exist_without_any_results(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        text = self.build(empty)
        macros = _macros(text)
        for alias, _, suffixes, _ in numbers_py.CURATED:
            for suffix in suffixes:
                self.assertEqual(macros[alias + suffix], numbers_py.MISSING, alias + suffix)
        self.assertIn("% MISSING input: disk/deltas.csv", text)
        self.assertNotIn("FIXTURE DATA", text)
        # Constants of the cost models are not results; they are always there.
        self.assertEqual(macros["diskBreakEvenC"], "$1.0101$")

    def test_curated_macro_names_are_the_same_with_and_without_results(self):
        full = set(_macros(self.build(FIXTURES)))
        for alias, _, suffixes, _ in numbers_py.CURATED:
            for suffix in suffixes:
                self.assertIn(alias + suffix, full)

    def test_twin_treatments_and_other_statistics_get_their_own_names(self):
        root = self.fixture_copy()
        rows = _rows("disk/deltas.csv")
        base = next(r for r in rows if (r["policy"], r["baseline"], r["metric"], r["capacity"],
                                        r["corpus"]) == ("adaptive", "sequential", "hit_ratio",
                                                         2048, "all"))
        extra = [dict(base, twins="collapsed", estimate=-0.5, ci_lo=-0.6, ci_hi=-0.4),
                 dict(base, statistic="ratio", estimate=0.97, ci_lo=0.96, ci_hi=0.98)]
        schemas.write_rows("disk/deltas.csv", rows + extra, root / "disk" / "deltas.csv")
        (root / "FIXTURE").unlink()  # plain values are easier to compare
        macros = _macros(self.build(root))
        stem = "diskDeltaHitRatioAdaptiveVsSequentialOneMiB"
        self.assertEqual(macros[stem + "None"], genlib.fpp(base["estimate"]))
        self.assertEqual(macros[stem + "Collapsed"], genlib.fpp(-0.5))
        self.assertEqual(macros[stem + "Ratio"], genlib.fratio(0.97))
        # The headline alias still resolves to the default statistic.
        self.assertEqual(macros["diskAdaptiveVsSeq"], genlib.fpp(base["estimate"]))

    def test_schema_violation_fails_loudly(self):
        root = self.fixture_copy()
        path = root / "disk" / "deltas.csv"
        text = path.read_text().splitlines()
        text[0] = text[0].replace("estimate", "estimat", 1)
        path.write_text("\n".join(text) + "\n")
        with self.assertRaises(schemas.SchemaError):
            self.build(root)

    def test_manifest_records_hashes_and_provenance_matches(self):
        root = self.fixture_copy()
        numbers_py.generate(genlib.Results(root), self.tmp)
        manifest = (self.tmp / numbers_py.MANIFEST).read_text()
        self.assertNotIn("created", manifest)  # no timestamps: regenerating is a no-op
        import json
        record = json.loads(manifest)["outputs"]["numbers.tex"]
        self.assertTrue(record["fixture"])
        self.assertTrue(all(i["provenance_match"] for i in record["inputs"].values()))
        # A results file edited after its provenance was written is flagged.
        path = root / "disk" / "kappa.csv"
        path.write_text(path.read_text().replace("msr_hm_0", "msr_hm_9", 1))
        numbers_py.generate(genlib.Results(root), self.tmp)
        record = json.loads((self.tmp / numbers_py.MANIFEST).read_text())["outputs"]["numbers.tex"]
        self.assertFalse(record["inputs"]["disk/kappa.csv"]["provenance_match"])
        self.assertTrue(record["inputs"]["disk/deltas.csv"]["provenance_match"])


class TablesTests(_Tmp):
    def build(self, root):
        outputs = tables_py.generate(genlib.Results(root), self.tmp)
        return outputs, {name: (self.tmp / "tables" / f"{name}.tex").read_text(encoding="utf-8")
                         for name in TABLE_NAMES}

    def assertSound(self, name, text):
        self.assertEqual(check.brace_problems(text), [], name)
        self.assertEqual(check.environment_problems(text), [], name)
        self.assertEqual(check.bare_underscores(text), [], name)
        self.assertIn(f"\\label{{tab:{name}}}", text)
        self.assertEqual(text.count("\\caption{"), 1, name)

    def test_every_table_from_fixtures(self):
        outputs, texts = self.build(FIXTURES)
        self.assertEqual(len(outputs), len(TABLE_NAMES))
        for name, text in texts.items():
            self.assertSound(name, text)
            self.assertIn("FIXTURE DATA", text, name)
        # Every input has a fixture now, the web dataset profile included.
        incomplete = sorted(o.name for o in outputs if o.missing)
        self.assertEqual(incomplete, [])
        self.assertNotIn("MISSING", texts["t4_web_dataset"])
        self.assertNotIn("MISSING", texts["t2_disk_main"])

    def test_every_table_is_a_placeholder_without_results(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        outputs, texts = self.build(empty)
        for (name, text), out in zip(texts.items(), outputs):
            self.assertSound(name, text)
            self.assertIn("MISSING", text, name)
            self.assertTrue(out.missing, name)
            self.assertNotIn("FIXTURE", text)

    def test_disk_main_table_holds_the_headline_rows(self):
        _, texts = self.build(FIXTURES)
        text = texts["t2_disk_main"]
        rows = [r for r in _rows("disk/table_main.csv") if r["config"] == "C1"]
        for row in rows:
            self.assertIn(genlib.esc(row["policy"]), text)
        # C2-only depth-grid policies stay out of the headline table.
        self.assertNotIn("deep@0", text)
        self.assertNotIn("deep@512", text)
        sequential = next(r for r in rows if r["policy"] == "sequential"
                          and r["capacity"] == 2048 and r["latency"] == "assumed")
        self.assertIn(genlib.fnum(100 * sequential["mean_hit_ratio"], 2), text)
        self.assertIn("skipped", text)  # lstm is skipped in C1, never filled in
        self.assertIn(r"guard$^\dagger$", text)  # the break-even gated row is labelled

    def test_live_table_excludes_invalid_passes_and_reports_them(self):
        _, texts = self.build(FIXTURES)
        text = texts["t7_live"]
        self.assertIn("1 headline pass-policy row(s) failed the validity gate", text)
        rows = _rows("live/live_metrics.csv")
        valid_fixed8_hdd = [r for r in rows if r["valid"] and r["store"] == "hdd"
                            and r["policy"] == "fixed8" and r["experiment"] == "headline"]
        self.assertEqual(len(valid_fixed8_hdd), 2)
        self.assertRegex(text, r"hdd & fixed8 & 2 &")

    def test_web_dataset_table_prints_whatever_the_profile_holds(self):
        root = self.fixture_copy()
        (root / "web" / "dataset_profile.json").write_text(
            '{"events": 2756101, "views": 2664218, "sessions": 1755863,'
            ' "pair_kinds": {"novel_frac": 0.634, "self_repeat_frac": 0.252}}')
        outputs, texts = self.build(root)
        self.assertNotIn("MISSING", texts["t4_web_dataset"])
        self.assertIn("2{,}756{,}101", texts["t4_web_dataset"])
        self.assertIn(r"$63.40$\,\%", texts["t4_web_dataset"])
        self.assertIn("1{,}755{,}863 sessions", texts["t1_datasets"])


class CheckIntegrationTests(_Tmp):
    """Every generated table and figure, included by a paper, passes --strict."""

    def test_generated_files_pass_strict_check_when_included(self):
        paper = self.tmp / "paper"
        (paper / "sections").mkdir(parents=True)
        shutil.copy2(PAPER / "refs.bib", paper / "refs.bib")
        results = genlib.Results(FIXTURES)
        numbers_py.generate(results, paper)
        tables_py.generate(results, paper)
        figures = [f"\\includegraphics[width=\\columnwidth]{{figures/{n}}}" for n in FIGURE_NAMES]
        if HAVE_MATPLOTLIB:
            _load("paper_make_figures", "figures/make_figures.py").generate(results, paper)
        body = "\n".join([f"\\input{{tables/{n}}} See Table~\\ref{{tab:{n}}}." for n in TABLE_NAMES]
                         + (figures if HAVE_MATPLOTLIB else []))
        (paper / "main.tex").write_text(
            "\\documentclass[conference]{IEEEtran}\n\\usepackage{graphicx,xcolor}\n"
            "\\IfFileExists{numbers.tex}{\\input{numbers}}{}\n\\begin{document}\n"
            "\\diskAdaptiveVsSeqCI\n" + body +
            "\n\\bibliographystyle{IEEEtran}\n\\bibliography{refs}\n\\end{document}\n")
        report = check.run_checks(paper, strict=True)
        self.assertEqual(report.errors, [], "\n".join(map(str, report.errors)))
        self.assertIn("numbers.tex", report.files)
        self.assertIn("tables/t8_cross_domain.tex", report.files)


@unittest.skipUnless(HAVE_MATPLOTLIB, "matplotlib is not installed (paper extra)")
class FiguresTests(_Tmp):
    @classmethod
    def setUpClass(cls):
        cls.figures = _load("paper_make_figures", "figures/make_figures.py")

    def build(self, root, directory):
        return self.figures.generate(genlib.Results(root), directory)

    def test_every_figure_from_fixtures_is_a_deterministic_ieee_pdf(self):
        first, second = self.tmp / "a", self.tmp / "b"
        outputs = self.build(FIXTURES, first)
        self.build(FIXTURES, second)
        self.assertEqual([o.name for o in outputs], [f"figures/{n}.pdf" for n in FIGURE_NAMES])
        self.assertEqual([o.name for o in outputs if o.missing], [])
        for name in FIGURE_NAMES:
            data = (first / "figures" / f"{name}.pdf").read_bytes()
            self.assertTrue(data.startswith(b"%PDF"), name)
            self.assertEqual(data, (second / "figures" / f"{name}.pdf").read_bytes(), name)
            self.assertNotIn(b"/CreationDate", data)
            self.assertNotIn(b"/Type3", data, f"{name}: Type 3 fonts fail IEEE PDF checks")
            self.assertIn(b"/FontFile2", data)  # embedded TrueType
            box = re.search(rb"/MediaBox\s*\[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", data)
            width, height = (float(v) / 72 for v in box.groups())
            self.assertAlmostEqual(width, self.figures.SIZES[name][0], places=2)
            self.assertAlmostEqual(height, self.figures.SIZES[name][1], places=2)
            self.assertIn(width, (self.figures.COLUMN_IN, self.figures.TEXT_IN))

    def test_missing_inputs_give_placeholders(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        outputs = self.build(empty, self.tmp / "out")
        for out in outputs:
            path = self.tmp / "out" / out.name
            self.assertTrue(path.read_bytes().startswith(b"%PDF"), out.name)
            if out.name != "figures/f1_schematic.pdf":  # the schematic reads no results
                self.assertTrue(out.missing, out.name)
        manifest = (self.tmp / "out" / "figures" / "manifest.json").read_text()
        self.assertIn("disk/kappa.csv", manifest)
        self.assertIn("width: text", manifest)  # tells the section to use figure*


class ReproducePaperTests(_Tmp):
    def run_reproduce(self, *args):
        import reproduce
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = reproduce.main(["paper", *args])
        return status, out.getvalue()

    def test_fixtures_produce_every_output_and_pass_strict_check(self):
        status, text = self.run_reproduce("--fixtures", "--out", str(self.tmp / "paper"))
        expected = 0 if HAVE_MATPLOTLIB else 4  # repro.EXIT_UNAVAILABLE without matplotlib
        self.assertEqual(status, expected, text)
        self.assertIn("FIXTURE DATA", text)
        self.assertIn("check.py --strict: 0 error(s)", text)
        paper = self.tmp / "paper"
        self.assertTrue((paper / "numbers.tex").is_file())
        for name in TABLE_NAMES:
            self.assertTrue((paper / "tables" / f"{name}.tex").is_file(), name)
        if HAVE_MATPLOTLIB:
            for name in FIGURE_NAMES:
                self.assertTrue((paper / "figures" / f"{name}.pdf").is_file(), name)
        self.assertTrue((paper / "main.tex").is_file())  # sources copied for check.py

    def test_fixture_output_is_never_written_into_paper(self):
        status, text = self.run_reproduce("--fixtures", "--out", str(PAPER))
        self.assertEqual(status, 2)
        self.assertIn("refusing", text)


if __name__ == "__main__":
    unittest.main()

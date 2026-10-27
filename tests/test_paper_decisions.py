"""paper/decisions.py and the committed paper's claims (WP14).

The paper may state a finding only in the words its study's pre-registered
rule gives (results/*/PREREGISTERED.md). decisions.py applies those rules to
the results files; these tests pin the rules themselves on synthetic
intervals, check that the module runs on the fixture tree and on an empty
one, and pin the verdicts the hand-written prose in paper/sections/ relies
on against the committed results. If a result changes so that one of those
verdicts changes, the prose is wrong and this test says so.

They also check that the committed paper/numbers.tex and paper/tables/ are
exactly what the generators write from results/ today (so every number in
the paper traces to a provenance-tagged results file), and that numbers.tex
stays small enough for TeX's memory on Overleaf.

Stdlib only; the figure comparison skips without matplotlib.
"""

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
FIXTURES = ROOT / "tests" / "fixtures" / "results"
sys.path.insert(0, str(ROOT / "src"))

HAVE_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _load(name, relative):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PAPER / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


genlib = _load("paper_genlib", "genlib.py")
decisions = _load("paper_decisions", "decisions.py")
numbers_py = _load("paper_numbers", "numbers.py")
tables_py = _load("paper_make_tables", "tables/make_tables.py")


def _row(lo, hi, clo=None, chi=None):
    return {"ci_lo": lo, "ci_hi": hi, "cluster_ci_lo": lo if clo is None else clo,
            "cluster_ci_hi": hi if chi is None else chi}


class RuleTests(unittest.TestCase):
    def test_disk_verdict_needs_both_intervals(self):
        v = decisions.disk_verdict
        self.assertEqual(v((0.01, 0.02), (0.005, 0.03)), "better")
        self.assertEqual(v((-0.02, -0.01), (-0.03, -0.001)), "worse")
        self.assertEqual(v((-0.02, -0.01), (-0.03, 0.001)), decisions.PAIRED_ONLY)
        self.assertEqual(v((-0.02, 0.01), (-0.03, -0.01)), decisions.NO_DIFFERENCE)
        # A ratio is read against 1, and fewer device I/Os are better.
        self.assertEqual(v((0.6, 0.8), (0.5, 0.9), ratio=True, higher_is_better=False), "better")
        self.assertEqual(v((1.01, 1.2), (1.0, 1.3), ratio=True, higher_is_better=False),
                         decisions.PAIRED_ONLY)

    def test_absolute_interval(self):
        self.assertEqual(decisions.abs_interval(-0.3, 0.1), (0.0, 0.3))
        self.assertEqual(decisions.abs_interval(-0.3, -0.1), (0.1, 0.3))
        self.assertEqual(decisions.abs_interval(0.2, 0.5), (0.2, 0.5))

    def test_web_verdict_has_three_words(self):
        self.assertEqual(decisions.web_verdict(0.001, 0.002), "A better")
        self.assertEqual(decisions.web_verdict(-0.002, -0.001), "A worse")
        self.assertEqual(decisions.web_verdict(-0.001, 0.001), "no detectable difference")

    def test_lever_order_needs_separated_absolute_intervals(self):
        levers = [("predictor", "adaptive", _row(-0.013, -0.006)),
                  ("feedback", "guard", _row(-0.007, -0.001, -0.008, 0.0002)),
                  ("depth", "deep@8", _row(0.03, 0.056))]
        (d,) = decisions._lever_order("disk", "1 MiB", levers, "diskTest")
        self.assertEqual(d.verdict, "the predictor is not the smallest lever")
        self.assertIn("predictor ? feedback", d.note)
        self.assertIn("predictor < depth", d.note)
        smallest = [("predictor", "adaptive", _row(-0.002, 0.001)),
                    ("depth", "deep@8", _row(0.03, 0.056))]
        (d,) = decisions._lever_order("disk", "1 MiB", smallest, "diskTest")
        self.assertEqual(d.verdict, "the predictor is the smallest lever")

    def test_spearman_rule_uses_the_exact_critical_values(self):
        r2 = decisions.r2_verdict
        self.assertEqual(r2(-0.79, 7, sign_consistent=-1), "ranks consistently with measured latency")
        self.assertEqual(r2(-0.78, 7, sign_consistent=-1), "ranking not supported")
        self.assertEqual(r2(0.74, 8, sign_consistent=-1), "ranks against measured latency")
        self.assertEqual(r2(0.98, 8, sign_consistent=+1), "ranks consistently with measured latency")
        self.assertTrue(r2(-1.0, 4, sign_consistent=-1).startswith("descriptive only"))


class DecideTests(unittest.TestCase):
    def test_fixture_tree_gives_unique_macro_keys(self):
        found = decisions.decide(genlib.Results(FIXTURES))
        self.assertTrue(found)
        keys = [d.key for d in found]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(genlib.is_macro_name(k) for k in keys))
        self.assertEqual({d.study for d in found}, {"disk", "realio", "web", "live"})

    def test_empty_tree_gives_nothing_and_a_placeholder_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = genlib.Results(Path(tmp))
            self.assertEqual(decisions.decide(results), [])
            out = results.output("tables/t9_decisions.tex")
            text = tables_py.t9_decisions(results, out)
            self.assertIn("MISSING", text)
            self.assertIn("disk/deltas.csv", out.missing)

    def test_flagged_live_passes_are_not_cited(self):
        rows = [{"valid": True, "foreign_io_frac": 0.0}, {"valid": True, "foreign_io_frac": 0.5},
                {"valid": False, "foreign_io_frac": 0.0}, {"valid": True, "foreign_io_frac": None}]
        self.assertEqual(genlib.live_citable(rows), [rows[0], rows[3]])


class NamingTests(unittest.TestCase):
    def test_comparison_rows_of_each_family_get_their_own_names(self):
        base = {"selection": "per_capacity", "scorer": "logistic", "capacity": 8, "cost": 0.01,
                "split": "test"}
        names = {numbers_py._comparison_tail(dict(base, adaptive_policy=p))
                 for p in ("adaptive_0.01", "adaptive_ca_0.01", "gate_0.01",
                           "feedback_0.01_eps0.005")}
        self.assertEqual(len(names), 4)
        self.assertIn("PerCapacityLogisticCapEightCostZeroPointZeroOne", names)
        self.assertEqual(numbers_py.web_family("adaptive_ca_0.1"), "adaptive_ca")

    def test_split_indexed_profile_keys_are_named_by_role(self):
        self.assertEqual(numbers_py.dataset_key_words(("views_by_split", "3")), "ViewsBySplitTest")
        self.assertEqual(numbers_py.dataset_key_words(("views_by_split", "0")), "ViewsBySplitIndex")
        self.assertEqual(numbers_py.dataset_key_words(("session_length_quantiles", "0.5")),
                         "SessionLengthQuantilesZeroPointFive")


@unittest.skipUnless((RESULTS / "disk" / "deltas.csv").is_file(), "results/ not present")
class CommittedPaperTests(unittest.TestCase):
    """The committed paper against the committed results."""

    @classmethod
    def setUpClass(cls):
        cls.found = {d.key: d for d in decisions.decide(genlib.Results(RESULTS))}

    def verdict(self, key):
        self.assertIn(key, self.found)
        return self.found[key].verdict

    def test_the_verdicts_the_prose_states(self):
        # Each pair is a sentence in paper/sections/*.tex; see the comments.
        claims = {
            # disk_results / abstract: the class router loses to sequential.
            "diskPOneAdaptiveAssumedOneMiB": "worse",
            "diskPOneAdaptiveAssumedSixtyFourMiB": "worse",
            # the gate: paired-only at 1 MiB, worse at 64 MiB and at break-even.
            "diskPTwoGuardAssumedOneMiB": decisions.PAIRED_ONLY,
            "diskPTwoGuardAssumedSixtyFourMiB": "worse",
            "diskPTwoGuardBreakEvenOneMiB": "worse",
            "diskPTwoGuardBreakEvenSixtyFourMiB": "worse",
            # depth and window growth.
            "diskPThreeDeepAtEightAssumedOneMiB": "better",
            "diskPThreeDeepAtEightAssumedSixtyFourMiB": "better",
            "diskPFourLinuxRaAtTwoHundredFiftySixAssumedOneMiB": "better",
            "diskPFiveDeepAtEightOneMiB": "increases device I/Os",
            "diskPFiveLinuxRaAtTwoHundredFiftySixOneMiB": "reduces device I/Os",
            # the lever rule at both capacities.
            "diskOneMiBPredictorSmallest": "the predictor is not the smallest lever",
            "diskSixtyFourMiBPredictorSmallest": "the predictor is not the smallest lever",
            "diskCThreeDeepAtThirtyTwoAssumed": "prefix-dependent",
            # learned depth loses to the validation-chosen depth.
            "diskLearnedDepthVsVcfCapSixtyFour": "worse",
            "diskLearnedDepthVsVcfCapTwoHundredFiftySix": "worse",
            # web: controller wins at caps 8 and 32; counts beat both learned rankers.
            "webWOneCapEight": "A better",
            "webWOneCapThirtyTwo": "A better",
            "webWTwoCapEight": "A better",
            "webRuleOneCapEight": "the adaptive controller beats fixed prefetching",
            "webWThreeHitRanker": "A better",
            "webWFourHitFreshness": "A better",
            "webWFiveHitDepth": "A better",
            "webWFiveUtilityDepth": "A worse",
            "webWSixHitFeedback": "A worse",
            "webWSixUtilityFeedback": "no detectable difference",
            "webWEightHitSequenceModel": "A worse",
            "webPredictorSmallest": "the predictor is not called the smallest lever",
            "webRuleThree": "not described as improving on the adaptive controller",
            # live: parity, every HDD policy lowers the wait, the model understates it.
            "liveHOne": "no mismatch",
            "liveHTwoSsdAdaptiveZeroPointZeroOne": "not established",
            "liveHFiveSsdFixedEight": "demand priority changes the delta",
            "liveHFiveSsdFixedFour": "no change detected by the H5 rule",
        }
        for key, expected in claims.items():
            with self.subTest(key=key):
                self.assertEqual(self.verdict(key), expected)

    def test_the_counts_the_prose_states(self):
        def count(rule, prefix, verdict=None):
            return sum(d.rule == rule and d.scope.startswith(prefix)
                       and (verdict is None or d.verdict.startswith(verdict))
                       for d in self.found.values())
        # realio: no hard-disk O_DIRECT closed-loop cell lowered the latency.
        self.assertEqual(count("R1", "sdb, odirect, closed", "lowers"), 0)
        self.assertGreater(count("R1", "sdb, odirect, closed", "raises"),
                           count("R1", "sdb, odirect, closed") // 2)
        # The highest simulated hit ratio was never the fastest mode.
        fastest = [d for d in self.found.values() if d.rule == "fastest"]
        self.assertTrue(fastest)
        self.assertFalse(any(d.note.endswith(": " + d.verdict) for d in fastest))
        # Simulated hit ratio never ranks the modes consistently (R2).
        self.assertEqual(sum(d.rule == "R2" and "simulated hit," in d.comparison
                             and d.verdict.startswith("ranks consistently")
                             for d in self.found.values()), 0)
        # live H2 on the hard-disk store: every prefetching policy lowers the wait.
        hdd = [d for d in self.found.values() if d.rule == "H2" and d.scope == "hdd store"]
        self.assertTrue(hdd)
        self.assertTrue(all(d.verdict == "lowers the mean wait" for d in hdd))

    def test_committed_numbers_and_tables_are_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results = genlib.Results(RESULTS)
            numbers_py.generate(results, out)
            tables_py.generate(results, out)
            for relative in ["numbers.tex", "numbers.manifest.json", "tables/manifest.json"] + [
                    f"tables/{f.__name__}.tex" for f in tables_py.TABLES]:
                with self.subTest(file=relative):
                    self.assertEqual((out / relative).read_bytes(),
                                     (PAPER / relative).read_bytes(),
                                     f"paper/{relative} is stale: run `python reproduce.py paper`")

    @unittest.skipUnless(HAVE_MATPLOTLIB, "matplotlib is not installed (paper extra)")
    def test_committed_figures_are_current(self):
        figures = _load("paper_make_figures", "figures/make_figures.py")
        with tempfile.TemporaryDirectory() as tmp:
            figures.generate(genlib.Results(RESULTS), Path(tmp))
            for function in figures.FIGURES:
                relative = f"figures/{function.__name__}.pdf"
                with self.subTest(file=relative):
                    self.assertEqual((Path(tmp) / relative).read_bytes(),
                                     (PAPER / relative).read_bytes())

    def test_every_input_matches_its_provenance(self):
        import json
        for manifest in ("numbers.manifest.json", "tables/manifest.json", "figures/manifest.json"):
            payload = json.loads((PAPER / manifest).read_text(encoding="utf-8"))
            self.assertFalse(payload["fixture"])
            for name, record in payload["outputs"].items():
                self.assertEqual(record["missing"], [], name)
                for relative, info in record["inputs"].items():
                    if relative.endswith("provenance.json"):
                        continue  # the provenance record does not hash itself
                    with self.subTest(output=name, input=relative):
                        self.assertIs(info["provenance_match"], True)

    def test_numbers_tex_fits_tex_memory(self):
        # TeX Live's defaults on Overleaf: about 6.25M characters of string
        # pool and 5M words of main memory. Keep numbers.tex well inside both.
        text = (PAPER / "numbers.tex").read_text(encoding="utf-8")
        self.assertLess(text.count("\\newcommand"), 30_000)
        self.assertLess(len(text.encode("utf-8")), 2_500_000)


if __name__ == "__main__":
    unittest.main()

"""E7, the MSRC light sample: windows, labels, metrics and parity with replay.

The stdlib parts run under system python3; the sklearn families skip there.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from adaptive_prefetch import msrc_eval as me
from adaptive_prefetch.benchmark import make_model
from adaptive_prefetch.model import make_classifier
from adaptive_prefetch.simulator import replay
from adaptive_prefetch.trace import Request, load_csv
from prefetch_common import schemas

HAVE_SKLEARN = (importlib.util.find_spec("sklearn") is not None
                and importlib.util.find_spec("numpy") is not None)
SMALL = me.SAMPLE_DIR / "wdev_3.revised"


class DataTests(unittest.TestCase):
    def test_discovers_every_revised_file_sorted(self):
        paths = me.discover()
        self.assertIn(SMALL, paths)
        self.assertEqual(paths, sorted(paths))
        self.assertTrue(all(p.suffix == ".revised" for p in paths))

    def test_family_is_the_host(self):
        self.assertEqual([me.family(s) for s in ("web_3", "rsrch_1", "wdev_2")],
                         ["web", "rsrch", "wdev"])

    def test_label_rules_and_mapping(self):
        self.assertEqual(me.label(17, "majority"), "seq")
        self.assertEqual(me.label(16, "majority"), "not_seq")
        self.assertEqual(me.label(8, "quarter"), "seq")
        self.assertEqual(me.label(7, "quarter"), "not_seq")
        self.assertEqual(me.to_binary("sequential"), "seq")
        for other in ("strided", "random", "mixed", "none"):
            self.assertEqual(me.to_binary(other), "not_seq")

    def test_windows_are_the_vectors_adaptive_classifies(self):
        """The shipped GNB on our vectors predicts exactly what replay predicts."""
        requests = load_csv(SMALL, "revised")
        windows = me.trace_windows(requests, "wdev_3")
        self.assertEqual(len(windows.vectors), len(requests) // me.WINDOW)
        model = make_model(me.SEED)
        ours = [model.predict(v)[0] for v in windows.vectors]
        _, predictions = replay(requests, me.CAPACITY, me.WINDOW, "adaptive",
                                make_model(me.SEED))
        self.assertEqual(ours, [p[1] for p in predictions])

    def test_synthetic_training_set_is_make_models(self):
        X, y = me.synthetic_training_set()
        self.assertEqual(len(X), 4 * me.EXAMPLES_PER_CLASS)
        refit = make_classifier("gnb", len(X[0]))
        for vector, name in zip(X, y):
            refit.update(vector, name)
        shipped = make_model(me.SEED)
        probe = me.trace_windows(load_csv(SMALL, "revised")).vectors + X[::37]
        self.assertEqual([refit.predict(v) for v in probe],
                         [shipped.predict(v) for v in probe])

    def test_flag_contiguity_agreement(self):
        reqs = [Request(0, 0, 8, pattern="rand"), Request(1, 8, 8, pattern="seq"),
                Request(2, 100, 8, pattern="seq"), Request(3, 108, 8, pattern="rand")]
        self.assertAlmostEqual(me.flag_contiguity_agreement(reqs), 1 / 3)
        self.assertIsNone(me.flag_contiguity_agreement(reqs[:1]))


class MetricTests(unittest.TestCase):
    def test_metrics_on_known_counts(self):
        m = me.metrics(tp=3, fp=1, fn=1, tn=5)
        self.assertAlmostEqual(m["accuracy"], 0.8)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5 * (3 / 4 + 5 / 6))
        self.assertAlmostEqual(m["macro_f1"], 0.5 * (6 / 8 + 10 / 12))

    def test_single_class_truth_has_no_balanced_metrics(self):
        m = me.metrics(tp=0, fp=0, fn=0, tn=7)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertIsNone(m["balanced_accuracy"])
        self.assertIsNone(m["macro_f1"])

    def test_stdlib_resampler_is_deterministic_and_paired(self):
        truth = ["seq"] * 20 + ["not_seq"] * 30
        pred = ["seq"] * 15 + ["not_seq"] * 35
        a = me.Resampler(50, 50, 42, "stdlib")
        b = me.Resampler(50, 50, 42, "stdlib")
        self.assertEqual(a.index, b.index)
        stats = me.summarise(truth, pred, a)
        self.assertLessEqual(stats["balanced_accuracy_lo"], stats["balanced_accuracy"])
        self.assertGreaterEqual(stats["balanced_accuracy_hi"], stats["balanced_accuracy"])
        self.assertEqual(me.paired_delta(truth, pred, pred, a), (0.0, 0.0, 0.0))

    def test_verdict_words(self):
        self.assertEqual(me.verdict(0.1, 0.01, 0.2, False), "better")
        self.assertEqual(me.verdict(-0.1, -0.2, -0.01, False), "worse")
        self.assertEqual(me.verdict(0.0, -0.1, 0.1, False), "no difference detected")
        self.assertEqual(me.verdict(0.3, 0.2, 0.4, True), "insufficient n")
        self.assertEqual(me.verdict(None, None, None, False), "insufficient n")

    def test_small_n_flag(self):
        self.assertTrue(me.small_n(["seq"] * 40 + ["not_seq"] * 59))
        self.assertTrue(me.small_n(["seq"] * 5 + ["not_seq"] * 200))
        self.assertFalse(me.small_n(["seq"] * 20 + ["not_seq"] * 200))


class StudyTests(unittest.TestCase):
    def test_stdlib_study_rows_match_their_schemas(self):
        windows = [me.trace_windows(load_csv(SMALL, "revised"), "wdev_3")]
        rows, deltas = me.classifier_study(windows, reps=20, backend="stdlib",
                                           sklearn=False)
        schemas.validate_rows("msrc_samples/classifier_metrics.csv", rows)
        schemas.validate_rows("msrc_samples/classifier_deltas.csv", deltas)
        found = {(r["protocol"], r["classifier"]) for r in rows}
        self.assertIn(("shipped_synthetic", "gnb"), found)
        self.assertIn(("rule", "route_window"), found)
        # wdev_3 has 21 windows: every row is flagged and gets no verdict word.
        self.assertTrue(all(r["small_n"] for r in rows))
        self.assertTrue(all(d["verdict"] == "insufficient n" for d in deltas))

    def test_replay_rows_pair_hit_ratio_with_device_work(self):
        rows = []
        for mode, param in me.REPLAY_MODES:
            for view in me.VIEWS:
                rows.append(me.replay_one((str(SMALL), view, mode, param, "measured")))
        self.assertTrue(all(r["requests"] == 680 for r in rows))
        sim = {(r["view"], r["mode"]): r for r in rows}
        self.assertEqual(sim[("writes_as_reads", "none")]["read_requests"], 680)
        self.assertEqual(sim[("writes_as_reads", "deep@8")]["read_requests"], 680)
        # deep@8 prefetches only after the first window, so it moves more blocks.
        self.assertGreater(sim[("writes_as_reads", "deep@8")]["blocks_moved"],
                           sim[("writes_as_reads", "none")]["blocks_moved"])

    def test_bounded_run_refuses_results(self):
        with self.assertRaises(SystemExit):
            me.main(["--out", str(schemas.RESULTS_ROOT / "msrc_samples"),
                     "--max-requests", "100"])

    def test_profile_counts_windows(self):
        requests = load_csv(SMALL, "revised")
        profile = me.trace_profile(SMALL, me.trace_windows(requests, "wdev_3"))
        schemas.validate_rows("msrc_samples/traces.csv", [profile])
        self.assertEqual(profile["windows"], 21)
        self.assertTrue(profile["small_n_majority"])


@unittest.skipUnless(HAVE_SKLEARN, "needs the [web] extra (numpy, scikit-learn)")
class SklearnTests(unittest.TestCase):
    def test_every_family_fits_the_synthetic_windows(self):
        X, y = me.synthetic_training_set(examples_per_class=20)
        for name in me.SKLEARN_FAMILIES:
            with self.subTest(name):
                out = me._fit_predict(name, X, y, X[:5])
                self.assertEqual(len(out), 5)
                self.assertTrue(set(out) <= set(y))

    def test_full_protocols_on_a_bounded_sample(self):
        paths = [p for p in me.discover() if p.stem in ("wdev_3", "web_3", "rsrch_1")]
        windows = [me.trace_windows(load_csv(p, "revised")[:3200], p.stem) for p in paths]
        rows, deltas = me.classifier_study(windows, reps=30)
        schemas.validate_rows("msrc_samples/classifier_metrics.csv", rows)
        schemas.validate_rows("msrc_samples/classifier_deltas.csv", deltas)
        protocols = {r["protocol"] for r in rows}
        self.assertEqual(protocols, {"shipped_synthetic", "sklearn_synthetic", "loto_real",
                                     "lofo_real", "rule", "baseline"})
        for row in rows:
            if row["status"].startswith("skipped"):
                self.assertIsNone(row["accuracy"])

    def test_bounded_main_writes_outside_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "msrc_samples"
            self.assertEqual(me.main(["--out", str(out), "--max-requests", "640",
                                      "--reps", "10", "--workers", "1"]), 0)
            self.assertEqual(schemas.validate_tree(Path(tmp)), {})
            self.assertFalse((out / "replay.csv").exists())


if __name__ == "__main__":
    unittest.main()

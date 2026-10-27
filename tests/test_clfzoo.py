"""Track CLF: the classifier zoo and the precomputed-label router replay.

The stdlib tests (vector layout, the label-replay identity, analysis rules)
run under both interpreters; the scikit-learn tests skip without it.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import random
import unittest
from pathlib import Path

from adaptive_prefetch.benchmark import make_model
from adaptive_prefetch.clfzoo import analysis, data, router
from adaptive_prefetch.model import OnlineGaussianNB
from adaptive_prefetch.simulator import LatencyModel, replay, replay_stream
from adaptive_prefetch.trace import CLASSES, load_csv, transition_trace
from prefetch_common.stats import CI

HAVE_SKLEARN = (importlib.util.find_spec("sklearn") is not None
                and importlib.util.find_spec("numpy") is not None)
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "samples" / "msr-cambridge-sample-merged.csv"


def _streams():
    """A labelled synthetic phase stream and a real MSR sample."""
    stream, _ = transition_trace(7, windows_per_class=12)
    out = [("transition", stream)]
    if SAMPLE.is_file():
        out.append(("msr_sample", load_csv(SAMPLE, "msr")))
    return out


def _chunks(requests, size):
    return [requests[i:i + size] for i in range(0, len(requests), size)]


def _fields(metrics) -> dict:
    """Every Metrics counter except the wall-clock inference timing."""
    out = dataclasses.asdict(metrics)
    out.pop("inference_ms")
    return out


class VectorLayoutTests(unittest.TestCase):
    def test_baseline_vectors_reproduce_make_model_exactly(self):
        for seed in (42, 1003):
            model = OnlineGaussianNB(data.N_FEATURES)
            for values, label in data.baseline_vectors(seed, 20):
                model.update(values, label)
            shipped = make_model(seed, examples_per_class=20)
            self.assertEqual(model.count, shipped.count)
            self.assertEqual(model.mean, shipped.mean)
            self.assertEqual(model.m2, shipped.m2)

    def test_every_setting_builds_labelled_twelve_feature_vectors(self):
        for setting in data.ALL_SETTINGS:
            rows = data.train_set(setting, data.TRAIN_SEEDS[0])
            per_class = (data.TRANSFER_TRAIN_PER_CLASS if setting == "transfer"
                         else data.TRAIN_PER_CLASS)
            self.assertEqual(len(rows), per_class * len(CLASSES), setting)
            self.assertTrue(all(len(v) == data.N_FEATURES for v, _ in rows))
            self.assertEqual({label for _, label in rows}, set(CLASSES))
            for condition in data.test_conditions(setting):
                test = data.test_set(setting, condition, data.TEST_SEEDS[0])
                self.assertTrue(test)
        # Deterministic, and train/test seeds never coincide.
        self.assertEqual(data.train_set("noisy", 1000), data.train_set("noisy", 1000))
        self.assertFalse(set(data.TRAIN_SEEDS) & set(data.TEST_SEEDS))

    def test_validation_folds_use_training_seeds_only(self):
        seeds = {data.validation_seed(i) for i in range(data.SEED_PAIRS)}
        self.assertEqual(seeds, set(data.TRAIN_SEEDS))
        self.assertTrue(all(data.validation_seed(i) != data.TRAIN_SEEDS[i]
                            for i in range(data.SEED_PAIRS)))


class LabelReplayTests(unittest.TestCase):
    """The Part 2 speed-up is exact: precomputed labels give the same replay."""

    def _per_call(self, requests, capacity, chunk, contextual=True):
        model = make_model(42)
        return replay_stream(_chunks(requests, chunk), capacity=capacity, window_size=32,
                             mode="adaptive", model=copy.deepcopy(model),
                             latency_model=router.ASSUMED, read_ahead_depth=0)

    def test_replayed_labels_equal_per_call_gnb_on_every_counter(self):
        gnb = make_model(42)
        for name, requests in _streams():
            vectors, count, evictions = router.window_features(_chunks(requests, 97))
            self.assertEqual(count, len(requests))
            self.assertEqual(len(vectors), len(requests) // 32)
            labels = [gnb.predict(v)[0] for v in vectors]
            for capacity in (64, 2048):
                expected = self._per_call(requests, capacity, 101)
                for contextual in (True, False):
                    got = router.replay_labels(_chunks(requests, 53), capacity, labels,
                                               contextual=contextual)
                    mine, theirs = _fields(got), _fields(expected)
                    if not contextual:
                        # The age table is not rebuilt; its count comes from
                        # the vector pass instead.
                        self.assertEqual(mine.pop("age_table_evictions"), 0)
                        self.assertEqual(theirs.pop("age_table_evictions"), evictions)
                    self.assertEqual(mine, theirs, f"{name} cap={capacity} ctx={contextual}")
                    self.assertEqual(got.inference_calls, len(labels))

    def test_label_replay_matches_in_memory_replay(self):
        gnb = make_model(42)
        stream = _streams()[0][1]
        vectors, _, _ = router.window_features([stream])
        labels = [gnb.predict(v)[0] for v in vectors]
        memory, _ = replay(stream, 256, 32, "adaptive", copy.deepcopy(gnb),
                           latency_model=router.ASSUMED, read_ahead_depth=0)
        streamed = router.replay_labels([stream], 256, labels)
        expected = _fields(memory)
        got = _fields(streamed)
        # replay_stream does not report recall (its prefetchable-miss counter).
        expected.pop("prefetchable_misses")
        got.pop("prefetchable_misses")
        self.assertEqual(got, expected)

    def test_a_different_routing_changes_the_replay(self):
        stream = _streams()[0][1]
        n = len(stream) // 32
        seq = router.replay_labels([stream], 256, ["sequential"] * n)
        none = router.replay_labels([stream], 256, ["random"] * n)
        self.assertGreater(seq.prefetches, 0)
        self.assertEqual(none.prefetches, 0)

    def test_replayed_refuses_too_few_or_too_many_labels(self):
        stream = _streams()[0][1]
        n = len(stream) // 32
        with self.assertRaises(RuntimeError):
            router.replay_labels([stream], 256, ["random"] * (n - 1))
        with self.assertRaises(RuntimeError):
            router.replay_labels([stream], 256, ["random"] * (n + 1))
        with self.assertRaises(ValueError):
            router.Replayed(["none"])

    def test_label_keys_agreement_and_shares(self):
        a = ["sequential", "random", "mixed", "random"]
        b = ["sequential", "random", "random", "random"]
        self.assertEqual(router.label_key(a), router.label_key(list(a)))
        self.assertNotEqual(router.label_key(a), router.label_key(b))
        self.assertEqual(router.agreement(a, b), 0.75)
        self.assertEqual(router.class_shares(b)["random"], 0.75)
        with self.assertRaises(ValueError):
            router.agreement(a, b[:2])

    def test_committed_row_comparison_names_differing_fields(self):
        stream = _streams()[0][1]
        n = len(stream) // 32
        metrics = router.replay_labels([stream], 256, ["sequential"] * n)
        row = {"requests": len(stream), "read_blocks": metrics.read_blocks,
               "hit_ratio": metrics.hit_ratio, "precision": metrics.prefetch_precision,
               "total_prefetches": metrics.prefetches,
               "useful_prefetches": metrics.useful_prefetches,
               "unused": metrics.unused_prefetches, "mean_us": metrics.mean_access_latency_us,
               "policy_switches": metrics.policy_switches, "demand_ios": metrics.demand_ios,
               "prefetch_ios": metrics.prefetch_ios,
               "prefetch_ios_mergeable": metrics.prefetch_ios_mergeable,
               "device_ios": metrics.device_ios, "blocks_moved": metrics.blocks_moved}
        self.assertEqual(router.committed_row_matches(row, metrics, len(stream)), [])
        row["device_ios"] += 1
        self.assertEqual(router.committed_row_matches(row, metrics, len(stream)),
                         ["device_ios"])


class AnalysisRuleTests(unittest.TestCase):
    def test_verdict_needs_both_intervals_on_one_side(self):
        up, down, across = CI(0.02, 0.01, 0.03, 5), CI(-0.02, -0.03, -0.01, 5), CI(0, -0.01, 0.01, 5)
        self.assertEqual(analysis.verdict(up), "better")
        self.assertEqual(analysis.verdict(down), "worse")
        self.assertEqual(analysis.verdict(up, up), "better")
        self.assertEqual(analysis.verdict(up, across), "no difference detected")
        self.assertEqual(analysis.verdict(down, down), "worse")
        ratio_low = CI(0.9, 0.85, 0.95, 5)
        self.assertEqual(analysis.verdict(ratio_low, ratio_low, higher_is_better=False,
                                          null=1.0), "better")

    def test_comparisons_pair_by_trace_and_report_cluster_intervals(self):
        rng = random.Random(3)
        traces = ["msr_hm_0", "msr_web_2", "fiu_homes-110108-112108", "alibabaBlock_17"]
        runs = []
        for trace in traces:
            for policy, shift in (("none", 0.0), ("sequential", 0.05), ("gnb_online", 0.04),
                                  ("rf", 0.06)):
                runs.append({"trace": trace, "capacity": 2048, "policy": policy,
                             "hit_ratio": 0.3 + shift + rng.uniform(0, 0.001),
                             "device_ios": 1000 + int(shift * 1000),
                             "blocks_moved": 2000 + int(shift * 2000)})
        rows = analysis.comparisons(runs, ("gnb_online", "rf"), (2048,), reps=200)
        rf_vs_gnb = next(r for r in rows if r["policy"] == "rf" and r["baseline"] == "gnb_online")
        self.assertEqual(rf_vs_gnb["n"], 4)
        self.assertEqual(rf_vs_gnb["verdict"], "better")
        self.assertEqual(rf_vs_gnb["wins"], 4)
        ios = next(r for r in rows if r["policy"] == "rf" and r["metric"] == "device_ios")
        self.assertEqual(ios["verdict"], "worse")  # more device I/O than none
        self.assertFalse(any(r["policy"] == "gnb_online" and r["baseline"] == "gnb_online"
                             for r in rows))


class HeadlinesTests(unittest.TestCase):
    def test_headlines_are_generated_from_the_csvs(self):
        import shutil
        import sys
        import tempfile
        sys.path.insert(0, str(ROOT))
        from repro import classifier
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        for path in (ROOT / "tests" / "fixtures" / "results" / "classifier").glob("*.csv"):
            shutil.copy(path, tmp / path.name)
        classifier._headlines(tmp, quick=True)
        text = (tmp / "HEADLINES.md").read_text(encoding="utf-8")
        self.assertIn("QUICK SMOKE RUN", text)
        self.assertIn("| `rf` |", text)
        self.assertIn("no difference detected", text)

    def test_part_synthetic_caches_without_touching_results(self):
        """``--part synthetic`` never writes results/ (no provenance there)."""
        import argparse
        import shutil
        import sys
        import tempfile
        from unittest import mock
        sys.path.insert(0, str(ROOT))
        from repro import classifier
        from adaptive_prefetch.clfzoo import synthetic
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        results, work = tmp / "results", tmp / "work"
        row = {"setting": "baseline", "family": "gnb_online", "candidate": 0, "params": {},
               "folds": 12, "val_accuracy": 1.0, "val_macro_f1": 1.0, "error": None,
               "wall_s": 0.0, "load_avg": None, "selected": True}
        fake = {"tuning": [row], "records": [], "chosen": {("baseline", "gnb_online"): row}}
        args = argparse.Namespace(quick=False, part="synthetic", workers=1,
                                  results_dir=str(results), work_dir=str(work),
                                  trace_root=str(tmp), reps=200)
        with mock.patch.object(classifier, "_available", return_value=None), \
                mock.patch.object(synthetic, "run", return_value=fake):
            self.assertEqual(classifier.run(args), 0)
        self.assertTrue((work / "synthetic_raw.json").is_file())
        self.assertEqual(list(results.iterdir()), [])


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn and numpy are optional ([web] extra)")
class ZooTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        from adaptive_prefetch.clfzoo import zoo
        cls.zoo = zoo
        train = data.baseline_vectors(1000, 25)
        cls.rows = [v for v, _ in train]
        cls.labels = [l for _, l in train]
        cls.test = [v for v, _ in data.baseline_vectors(900000, 10)]

    def test_candidates_are_seeded_distinct_and_start_at_the_default(self):
        zoo = self.zoo
        for family in zoo.FAMILIES:
            first = zoo.candidates(family)
            self.assertEqual(first, zoo.candidates(family))
            if family in zoo.SHIPPED:
                self.assertEqual(first, [{}])
                continue
            self.assertEqual(first[0], zoo.DEFAULTS[family])
            self.assertEqual(len(first), zoo.N_CANDIDATES)
            self.assertEqual(len({repr(c) for c in first}), zoo.N_CANDIDATES)
            self.assertIn(family, zoo.CITATIONS)

    def test_every_family_honours_the_shipped_predict_contract(self):
        zoo = self.zoo
        for family in zoo.FAMILIES:
            params = zoo.candidates(family)[0]
            if family == "qda":
                params = {"reg_param": 0.5}  # the default can be singular on 100 windows
            router_ = zoo.fit(family, params, self.rows, self.labels, seed=0)
            batch = router_.predict_batch(self.test)
            probs = router_.proba_batch(self.test)
            for values, label, p in zip(self.test, batch, probs.tolist()):
                predicted, confidence = router_.predict(values)
                self.assertIn(predicted, CLASSES, family)
                self.assertEqual(predicted, label, family)
                self.assertAlmostEqual(sum(p), 1.0, places=6)
                self.assertAlmostEqual(confidence, max(p), places=6, msg=family)

    def test_gnb_online_is_the_shipped_model(self):
        router_ = self.zoo.fit("gnb_online", {}, self.rows, self.labels)
        shipped = make_model(1000, examples_per_class=25)
        self.assertEqual(router_._online.mean, shipped.mean)
        self.assertEqual(router_.predict_batch(self.test),
                         [shipped.predict(v)[0] for v in self.test])

    def test_online_posterior_matches_online_predict(self):
        for family in ("gnb_online", "qda_online"):
            model = self.zoo.fit(family, {}, self.rows, self.labels)._online
            for values in self.test:
                posterior = self.zoo.online_posterior(model, values)
                label, confidence = model.predict(values)
                self.assertEqual(CLASSES[posterior.index(max(posterior))], label)
                self.assertAlmostEqual(max(posterior), confidence, places=9)

    def test_scores_and_selection(self):
        from adaptive_prefetch.clfzoo import synthetic
        labels = ["sequential", "strided", "random", "mixed"]
        perfect = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        result = synthetic.scores(labels, perfect)
        self.assertEqual((result["accuracy"], result["macro_f1"], result["brier"]),
                         (1.0, 1.0, 0.0))
        self.assertEqual(result["ece"], 0.0)
        rows = [{"setting": "s", "family": "f", "candidate": i, "val_accuracy": acc}
                for i, acc in enumerate((0.9, 0.95, 0.95, 0.1))]
        self.assertEqual(synthetic.select(rows)[("s", "f")]["candidate"], 1)

    def test_a_tuning_fold_set_runs_on_training_seeds_only(self):
        from adaptive_prefetch.clfzoo import synthetic
        saved = synthetic._VECTORS
        try:
            synthetic._VECTORS = {("train", "baseline", None, seed): data.baseline_vectors(seed, 8)
                                  for seed in data.TRAIN_SEEDS}
            row = synthetic._tune_job(("baseline", "cart", 0, {"max_depth": 3,
                                                               "min_samples_leaf": 1}))
            self.assertIsNone(row["error"])
            self.assertEqual(row["folds"], data.SEED_PAIRS)
            self.assertGreater(row["val_accuracy"], 0.5)
            bad = synthetic._tune_job(("baseline", "qda", 0, {"reg_param": 0.0}))
            if bad["error"] is not None:
                self.assertEqual(bad["val_accuracy"], 0.0)
        finally:
            synthetic._VECTORS = saved


if __name__ == "__main__":
    unittest.main()

"""Track HSR: the hindsight-labelled router and its ``routed`` mode.

The stdlib part (labels, routers, replay parity, registration, decision
words) runs under system ``python3``; the scikit-learn part and the
end-to-end pipeline skip without the ``[web]`` extra.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

from adaptive_prefetch import benchmark, hindsight, sweep
from adaptive_prefetch.hindsight import (ACTIONS, FixedRouter, ScheduledRouter,
                                         WindowRouter, contiguity_rule,
                                         feature_router)
from adaptive_prefetch.simulator import (GATED_MODES, MODES_SUPPORTED, LatencyModel,
                                         _make_predictor, replay, replay_stream)
from adaptive_prefetch.trace import Request, transition_trace

HAVE_SKLEARN = (importlib.util.find_spec("sklearn") is not None
                and importlib.util.find_spec("numpy") is not None)

COUNTERS = ("read_blocks", "read_hits", "prefetches", "useful_prefetches",
            "demand_ios", "prefetch_ios", "prefetch_ios_mergeable",
            "prefetch_cost_us", "demand_latency_us", "policy_switches",
            "inference_calls")


def mixed(n=3000, seed=11):
    rng = random.Random(seed)
    out, lba = [], 0
    for i in range(n):
        if rng.random() < 0.65:
            lba += 8
        else:
            lba = rng.randrange(0, 40_000) * 8
        out.append(Request(i * 0.5, lba, rng.choice((1, 8, 16)),
                           "W" if rng.random() < 0.1 else "R"))
    return out


def chunked(requests, size=700):
    return iter([requests[i:i + size] for i in range(0, len(requests), size)])


class RegistrationTests(unittest.TestCase):
    def test_registered_in_all_four_places_and_appended(self):
        self.assertEqual(benchmark.MODES[-1], "routed")
        self.assertEqual(sweep.RANKING_MODES[-1], "routed")
        self.assertEqual(benchmark.MODES[5], "lstm")
        self.assertIn("routed", MODES_SUPPORTED)
        self.assertNotIn("routed", GATED_MODES)
        self.assertIn("routed", benchmark.OPTIONAL_MODES)
        self.assertIn("routed", sweep.OPTIONAL_MODES)
        router = feature_router()
        self.assertIs(_make_predictor("routed", None, router=router), router)

    def test_without_a_router_it_is_a_skipped_row_never_a_substitute(self):
        with self.assertRaises(RuntimeError):
            replay(mixed(200), mode="routed")
        rows = benchmark.benchmark_dataset("t", transition_trace(3, windows_per_class=2)[0])
        self.assertEqual(rows[-1]["mode"], "routed")
        self.assertTrue(str(rows[-1]["status"]).startswith("skipped"))
        self.assertIsNone(rows[-1]["hit_ratio"])


class RouterParityTests(unittest.TestCase):
    def test_fixed_router_reproduces_the_native_modes_bit_for_bit(self):
        requests = mixed()
        native = {"none": ("none", 8), "sequential": ("sequential", 8),
                  "deep@8": ("deep", 8), "deep@32": ("deep", 32),
                  "deep@256": ("deep", 256)}
        for action, (mode, depth) in native.items():
            with self.subTest(action=action):
                expected, _ = replay(requests, capacity=512, mode=mode,
                                     read_ahead_depth=depth)
                got, _ = replay(requests, capacity=512, mode="routed",
                                router=FixedRouter(action))
                for field in COUNTERS[:-2]:
                    self.assertEqual(getattr(got, field), getattr(expected, field), field)

    def test_routed_agrees_across_both_paths(self):
        requests = mixed(seed=3)
        a, _ = replay(requests, capacity=256, mode="routed", router=feature_router())
        b = replay_stream(chunked(requests), capacity=256, mode="routed",
                          router=feature_router())
        for field in COUNTERS:
            self.assertEqual(getattr(a, field), getattr(b, field), field)
        self.assertEqual(a.device_ios, b.device_ios)
        self.assertEqual(b.inference_calls, len(requests))
        self.assertGreater(a.policy_switches, 0)

    def test_batch_schedule_equals_the_in_loop_router(self):
        requests = mixed(seed=4)
        features = hindsight.window_features(chunked(requests, 333))
        self.assertEqual(len(features), len(requests) // 32)
        schedule = [contiguity_rule(f) for f in features]
        self.assertGreater(len(set(schedule)), 1)
        loop = replay_stream(chunked(requests), capacity=256, mode="routed",
                             router=feature_router())
        batch = replay_stream(chunked(requests), capacity=256, mode="routed",
                              router=ScheduledRouter(schedule))
        for field in COUNTERS[:-1]:
            self.assertEqual(getattr(loop, field), getattr(batch, field), field)

    def test_first_window_never_prefetches_and_a_decision_applies_to_the_next(self):
        scan = [Request(float(i), i * 8, 8) for i in range(96)]
        router = ScheduledRouter(["none", "deep@8", "none"], initial="deep@256")
        metrics, _ = replay(scan, capacity=4096, mode="routed", router=router)
        # Window 0 (deep@256, gated) and window 1 (none) issue nothing; only
        # window 2 runs deep@8, decided when window 1 completed.
        self.assertEqual(metrics.prefetches, 8 * 32)
        self.assertEqual(router.counts["deep@8"], 32)

    def test_short_schedule_raises(self):
        with self.assertRaises(ValueError):
            replay(mixed(100), mode="routed", router=ScheduledRouter(["none"]))
        with self.assertRaises(ValueError):
            ScheduledRouter(["deep@x"])


class RecordingTests(unittest.TestCase):
    def test_record_windows_equals_the_replay_loop(self):
        requests = mixed(seed=8)
        latency = LatencyModel(5, 100, 50)
        for action in ACTIONS:
            with self.subTest(action=action):
                metrics, ends = hindsight.record_windows(chunked(requests), action, 512,
                                                         latency_model=latency)
                expected = replay_stream(chunked(requests), capacity=512, mode="routed",
                                         router=FixedRouter(action), latency_model=latency)
                for field in COUNTERS[:-2]:
                    self.assertEqual(getattr(metrics, field), getattr(expected, field))
                self.assertEqual(len(ends["read_hits"]), len(requests) // 32)
                self.assertLessEqual(ends["read_hits"][-1], metrics.read_hits)
                deltas = hindsight.window_deltas(ends)
                self.assertEqual(sum(deltas["prefetches"]), ends["prefetches"][-1])
                self.assertEqual(deltas["prefetches"][0], 0)  # first window gated

    def test_labels_take_the_next_window_and_break_ties_cheaply(self):
        # Three windows; per action: (hits, prefetches, device_ios) per window.
        per = {"none": [(1, 0, 5), (2, 0, 5), (4, 0, 5)],
               "sequential": [(1, 0, 5), (2, 2, 5), (5, 2, 4)],
               "deep@8": [(1, 0, 5), (2, 8, 5), (6, 8, 4)],
               "deep@32": [(1, 0, 5), (1, 32, 6), (6, 32, 9)],
               "deep@256": [(1, 0, 5), (0, 256, 7), (5, 256, 9)]}

        def ends(rows):
            out = {name: [] for name in hindsight.COUNTERS}
            totals = dict.fromkeys(hindsight.COUNTERS, 0)
            for hits, issued, ios in rows:
                totals["read_hits"] += hits
                totals["prefetches"] += issued
                totals["device_ios"] += ios
                for name in hindsight.COUNTERS:
                    out[name].append(totals[name])
            return out
        records = {a: ends(rows) for a, rows in per.items()}
        hit = hindsight.outcome_matrix(records, "hit")
        self.assertEqual(hindsight.label_windows(hit), ["none", "deep@8"])
        # u0.5 on window 2: none 4, sequential 5 - 1 = 4: a tie goes to none.
        self.assertEqual(hindsight.label_windows(hindsight.outcome_matrix(records, "u0.5")),
                         ["none", "none"])
        self.assertEqual(hindsight.label_windows(hindsight.outcome_matrix(records, "dev")),
                         ["none", "sequential"])
        self.assertEqual(hindsight.oracle_schedule(hit), ["none", "deep@8", "none"])
        oracle = [hindsight.best_action_index(hit[k + 1]) for k in range(2)]
        self.assertEqual(hindsight.headroom_captured(hit, oracle), 1.0)
        self.assertEqual(hindsight.headroom_captured(hit, [0, 0]), 0.0)
        flat = [(1.0,) * 5] * 3
        self.assertIsNone(hindsight.headroom_captured(flat, [4, 4]))

    def test_slice_objective(self):
        row = {"read_blocks": 100, "read_hits": 40, "total_prefetches": 20, "device_ios": 30}
        none = {"read_blocks": 100, "read_hits": 10, "total_prefetches": 0, "device_ios": 60}
        self.assertAlmostEqual(hindsight.slice_objective(row, "hit"), 0.4)
        self.assertAlmostEqual(hindsight.slice_objective(row, "u0.5"), 0.3)
        self.assertAlmostEqual(hindsight.slice_objective(row, "dev", none), -0.5)
        with self.assertRaises(ValueError):
            hindsight.slice_objective(row, "dev")

    def test_scoring_refuses_an_overlapping_slice(self):
        record = hindsight.training_record(["msr_hm_0"], 0, 1_000_000)
        self.assertIsNotNone(hindsight.scoring_conflict(record, "msr_hm_0", 999_999, 2_000_000))
        self.assertIsNone(hindsight.scoring_conflict(record, "msr_hm_0", 1_000_000, 2_000_000))
        self.assertIsNone(hindsight.scoring_conflict(record, "msr_web_2", 0, 1_000_000))


class DecisionWordTests(unittest.TestCase):
    def test_verdicts_follow_the_preregistered_rule(self):
        from adaptive_prefetch.hindsight_analysis import verdict
        from prefetch_common.stats import CI
        up, down, across = CI(0.1, 0.05, 0.2, 9), CI(-0.1, -0.2, -0.05, 9), CI(0, -0.1, 0.1, 9)
        self.assertEqual(verdict(up, up, True, False), "better")
        self.assertEqual(verdict(down, down, True, False), "worse")
        self.assertEqual(verdict(across, across, True, False), "no difference detected")
        self.assertTrue(verdict(up, across, True, False).startswith(
            "no difference detected (paired interval only"))
        below = CI(0.9, 0.8, 0.95, 9)
        self.assertEqual(verdict(below, below, False, True), "better")

    def test_headroom_over_vcf_is_undefined_when_the_oracle_does_not_beat_it(self):
        from adaptive_prefetch.hindsight_analysis import headroom_over_vcf
        self.assertAlmostEqual(headroom_over_vcf(0.55, 0.5, 0.6), 0.5)
        self.assertAlmostEqual(headroom_over_vcf(0.45, 0.5, 0.6), -0.5)
        # Policy and oracle both below vcf: two negative gaps must not read
        # as a positive share captured.
        self.assertIsNone(headroom_over_vcf(0.83264, 0.83267, 0.8289))
        self.assertIsNone(headroom_over_vcf(0.5, 0.5, 0.5))


@unittest.skipUnless(HAVE_SKLEARN, "needs numpy and scikit-learn (the [web] extra)")
class ModelTests(unittest.TestCase):
    def _data(self, trace, seed):
        from adaptive_prefetch import hindsight_models as hm
        rng = random.Random(seed)
        features, matrix = [], []
        for _ in range(300):
            contiguity = rng.random()
            features.append((contiguity, *[rng.random() for _ in range(11)]))
            # The best action is deeper the more contiguous the window.
            best = min(4, int(contiguity * 5))
            matrix.append(tuple(10.0 if i == best else rng.random() for i in range(5)))
        # Labels look one window ahead, so align the signal with window k+1.
        return hm.TraceData(trace, features[1:] + [features[0]], matrix)

    def test_search_space_is_deterministic_and_every_family_fits(self):
        from adaptive_prefetch import hindsight_models as hm
        datasets = [self._data(f"t{i}", i) for i in range(3)]
        for family in hm.FAMILIES:
            with self.subTest(family=family):
                space = hm.search_space(family, 3)
                self.assertEqual(space, hm.search_space(family, 3))
                model = hm.fit(family, space[0], datasets, 100)
                schedule = hm.predict_schedule(model, [tuple(d) for d in datasets[0].X[:7]])
                self.assertEqual(len(schedule), 7)
                self.assertTrue(set(schedule) <= set(ACTIONS))

    def test_grouped_cv_keeps_twins_together_and_learns_the_signal(self):
        from adaptive_prefetch import hindsight_models as hm
        traces = ["alibabaBlock_11", "alibabaBlock_15", "msr_hm_0", "msr_web_2", "x"]
        groups = {t: ("twin" if t.startswith("alibabaBlock_1") else t) for t in traces}
        folds = hm.group_folds(traces, groups, folds=3)
        self.assertEqual(sorted(sum(folds, [])), sorted(traces))
        self.assertTrue(any({"alibabaBlock_11", "alibabaBlock_15"} <= set(f) for f in folds))
        datasets = [self._data(t, i) for i, t in enumerate(traces)]
        result = hm.cross_validate("tree", {"max_depth": 4, "min_samples_leaf": 1,
                                            "criterion": "gini", "class_weight": None,
                                            "weighting": "none"},
                                   datasets, groups, 300)
        self.assertGreater(result["cv_score"], 0.8)
        self.assertEqual(result["cv_traces_scored"], len(traces))

    def test_artifact_round_trip_refuses_tampering(self):
        from adaptive_prefetch import hindsight_models as hm
        model = hm.fit("gnb", {"var_smoothing": 1e-9, "weighting": "none"},
                       [self._data("t", 1)], 100)
        with tempfile.TemporaryDirectory() as tmp:
            path = hm.save(model, Path(tmp) / "r.joblib",
                           hindsight.training_record(["t"], 0, 300))
            loaded, record = hm.load(path)
            self.assertEqual(record["traces"][0]["train"], [0, 300])
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                hm.load(path)


@unittest.skipUnless(HAVE_SKLEARN, "needs numpy and scikit-learn (the [web] extra)")
class PipelineTests(unittest.TestCase):
    def test_end_to_end_on_fixture_traces_writes_valid_results(self):
        from test_disk_multicorpus import make_fixture_root

        from adaptive_prefetch import corpora, hindsight_experiment as hx
        from adaptive_prefetch import hindsight_pipeline as hp
        from prefetch_common import schemas
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fixture_root(Path(tmp))
            traces = corpora.discover_block_traces(root)
            cache = Path(tmp) / "cache"
            n, caps = 600, (128,)
            hx.run_record(traces, cache, n, caps, workers=1, quiet=True)
            args = argparse.Namespace(requests=n, configs=1, workers=1, quiet=True,
                                      models_dir=str(Path(tmp) / "models"), reps=50,
                                      results_dir=str(Path(tmp) / "results"))
            hp.run_train(args, traces, caps, cache)
            hp.run_score(args, traces, caps, cache)
            hp.run_analyze(args, traces, caps, cache)
            out = Path(tmp) / "results"
            for name in ("rows", "selection", "families", "summary", "comparisons"):
                rows = schemas.read_rows(f"hindsight/{name}.csv", out / f"{name}.csv")
                self.assertTrue(rows, name)
            rows = schemas.read_rows("hindsight/rows.csv", out / "rows.csv")
            policies = {r["policy"] for r in rows if r["skip_requests"] == n}
            self.assertTrue({"routed_hit", "vcf_hit", "oracle_hit", "routed_hit_loco",
                             "routed_hit_gnb", "adaptive_evidence", "none"} <= policies)
            words = {r["verdict"].split(" (")[0]
                     for r in schemas.read_rows("hindsight/comparisons.csv",
                                                out / "comparisons.csv")}
            self.assertTrue(words <= {"better", "worse", "no difference detected"})


if __name__ == "__main__":
    unittest.main()

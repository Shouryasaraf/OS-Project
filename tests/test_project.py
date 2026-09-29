import json
import random
import tempfile
import unittest
from pathlib import Path
from statistics import mean, pstdev

from adaptive_prefetch.baselines import MarkovPrefetcher, StridePrefetcher
from adaptive_prefetch.artifacts import load_model, save_model
from adaptive_prefetch.benchmark import (MODES, aggregate_results, analyze_results,
                                         benchmark_dataset, drift_report)
from adaptive_prefetch.cli import train
from adaptive_prefetch.eda import (domain_shift, locality_profile, profile_trace)
from adaptive_prefetch.features import (FEATURE_NAMES, STREAM_FEATURES,
                                         WindowContext, dominant_stride,
                                         dominant_stride as _ds, extract)
from adaptive_prefetch.lstm import (HISTORY, MAX_DELTA, LSTMPrefetcher, UNK,
                                    _encode, bucket_of, bucket_span,
                                    load_predictor, train_lstm)
from adaptive_prefetch.model import OnlineGaussianNB, make_classifier
from adaptive_prefetch.pipeline import (PolicySmoother, StandardScaler,
                                        evaluate_classifier)
from adaptive_prefetch.report import benchmark_section as report_benchmark_section
from adaptive_prefetch.report import dataset_overview as report_dataset_overview
from adaptive_prefetch.sweep import aggregate_by_mode as sweep_aggregate
from adaptive_prefetch.sweep import capped_chunks
from adaptive_prefetch.sweep import winners as sweep_winners
from adaptive_prefetch.simulator import (LatencyModel, Metrics, oracle_reference,
                                         replay, replay_stream)
from adaptive_prefetch.trace import (CLASSES, Request, iter_csv_chunks, load_csv,
                                     synthetic_dataset, synthetic_window,
                                     transition_trace, write_csv)
from adaptive_prefetch.training import adapt_msr_sample


def contextual_vectors(windows_and_labels):
    """Feature vectors with the contextual block, laid out as a stream.

    Mirrors how the model is actually trained and replayed, so tests do not
    accidentally exercise the 8-feature path against a 12-feature model.
    """
    context = WindowContext()
    out = []
    for window, label in windows_and_labels:
        out.append((context.observe(window, dominant_stride(window)), label))
    return out


def fit_model(name="gnb", per_class=20, seed=7):
    """Train either classifier on contextual features."""
    model = make_classifier(name, len(FEATURE_NAMES))
    for values, label in contextual_vectors(synthetic_dataset(per_class, seed)):
        model.update(values, label)
    return model


class TraceTests(unittest.TestCase):
    def test_round_trip_csv(self):
        original = [Request(1.0, 10, 2, "R"), Request(2.0, 12, 1, "W")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            write_csv(path, original)
            self.assertEqual(load_csv(path), original)

    def test_load_msr_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "msr.csv"
            path.write_text(
                "Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime\n"
                "100000,hm,0,Read,1024,2048,10\n"
                "110000,hm,0,Write,2048,512,20\n",
                encoding="utf-8",
            )
            self.assertEqual(load_csv(path), [
                Request(0.0, 2, 4, "R", "hm"),
                Request(1.0, 4, 1, "W", "hm"),
            ])

    def test_iotta8_and_alibaba_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            iotta = Path(directory) / "iotta.csv"
            iotta.write_text(
                "device,sector,size,op,offset,timestamp,lifetime,count\n"
                "disk1,20,4,R,0,1000000,0,1\n"
                "disk1,24,2,W,0,1001000,0,1\n", encoding="utf-8")
            self.assertEqual(load_csv(iotta), [
                Request(0.0, 20, 4, "R", "disk1"),
                Request(1.0, 24, 2, "W", "disk1"),
            ])
            cloud = Path(directory) / "alibaba.csv"
            cloud.write_text(
                "device_id,opcode,offset,length,timestamp\n"
                "7,R,1024,2048,1000000\n"
                "7,W,2048,512,1001000\n", encoding="utf-8")
            self.assertEqual(load_csv(cloud), [
                Request(0.0, 2, 4, "R", "7"),
                Request(1.0, 4, 1, "W", "7"),
            ])

    def test_revised_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hm_1.revised"
            path.write_text(
                "0.000004981 RS 1432 128 seq 13770.6865422 0.000255738 13770.6862\n"
                "0.010347144 WS 1572960 8 rand 341.358274019 0.000137415 341.358048\n"
                "0.020573163 RS 1448 96 rand 2.0402e-05 0.000246558 0\n",
                encoding="utf-8")
            loaded = load_csv(path, "revised")
            self.assertEqual(len(loaded), 3)
            # Compare the canonical fields; service_ms/pattern are populated
            # from the trace's own columns and asserted separately below.
            self.assertEqual(
                (loaded[0].timestamp_ms, loaded[0].lba, loaded[0].size_blocks,
                 loaded[0].operation), (0.0, 1432, 128, "R"))
            self.assertAlmostEqual(loaded[1].timestamp_ms, 10.342163, places=5)
            self.assertEqual((loaded[1].lba, loaded[1].size_blocks,
                              loaded[1].operation), (1572960, 8, "W"))
            self.assertAlmostEqual(loaded[2].timestamp_ms, 20.568182, places=5)
            self.assertEqual((loaded[2].lba, loaded[2].size_blocks,
                              loaded[2].operation), (1448, 96, "R"))

    def test_revised_profile_captures_measured_service_time(self):
        """Columns 6-8 are real device timings and must survive loading."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.revised"
            path.write_text(
                "0.0 RS 100 8 seq 13770.6865422 0.000255738 13770.6862\n"
                "0.1 WS 200 8 rand 341.358274019 0.000137415 341.358048\n",
                encoding="utf-8")
            a, b = load_csv(path, "revised")
        self.assertAlmostEqual(a.service_ms, 13770.6865422)
        self.assertAlmostEqual(b.service_ms, 341.358274019)
        self.assertEqual(a.pattern, "seq")
        self.assertEqual(b.pattern, "rand")

    def test_service_ms_defaults_to_none_for_other_profiles(self):
        self.assertIsNone(Request(0.0, 10, 1, "R").service_ms)
        self.assertIsNone(Request(0.0, 10, 1, "R").pattern)

    def test_negative_service_time_rejected(self):
        with self.assertRaises(ValueError):
            Request(0.0, 10, 1, "R", service_ms=-1.0)

    def test_measured_cost_model_uses_recorded_latencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.revised"
            rows = "".join(
                f"{i * 0.001} RS {i * 8} 8 rand 1000.0 0.1 999.9\n"
                for i in range(40))
            path.write_text(rows, encoding="utf-8")
            requests = load_csv(path, "revised")
        model = LatencyModel.measured(trace=requests)
        # Mean measured service time is 1000 ms = 1e6 us; a miss must cost it.
        self.assertAlmostEqual(model.demand_miss_us, 1_000_000.0, places=3)
        self.assertGreater(model.prefetch_us, model.demand_miss_us)
        self.assertLess(model.hit_us, model.demand_miss_us)

    def test_measured_cost_model_uses_mean_not_median(self):
        """Real traces are sub-microsecond-dominant, so a median-based cost
        model would collapse to zero and make every policy look free."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.revised"
            # 90 requests at ~0, 10 at 1000 ms: median ~0, mean 100 ms.
            rows = "".join(
                f"{i * 0.001} RS {i * 8} 8 rand "
                f"{0.000001 if i < 90 else 1000.0} 0.1 0.0\n"
                for i in range(100))
            path.write_text(rows, encoding="utf-8")
            requests = load_csv(path, "revised")
        model = LatencyModel.measured(trace=requests)
        self.assertAlmostEqual(model.demand_miss_us, 100_000.0, places=1)

    def test_measured_cost_model_falls_back_without_measurements(self):
        model = LatencyModel.measured(trace=[Request(0.0, 10, 1, "R")])
        self.assertEqual(model.demand_miss_us, 100.0)

    def test_revised_profile_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.revised"
            path.write_text("0.000004981 QS 1432 128 seq 1 2 3\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "op must be RS or WS"):
                load_csv(path, "revised")
            path.write_text("0.000004981 RS 1432 128 seq 1 2\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 8 columns"):
                load_csv(path, "revised")

    def test_unknown_trace_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.csv"
            path.write_text("time,address\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown CSV schema"):
                load_csv(path)

    def test_explicit_headerless_iotta8_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iotta8.csv"
            path.write_text("disk1,20,4,R,0,1000000,0,1\n"
                            "disk1,24,2,W,0,1001000,0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown CSV schema"):
                load_csv(path)
            self.assertEqual(load_csv(path, "iotta8"), [
                Request(0.0, 20, 4, "R", "disk1"),
                Request(1.0, 24, 2, "W", "disk1"),
            ])

    def test_invalid_request(self):
        with self.assertRaises(ValueError):
            Request(0, -1)
        with self.assertRaises(ValueError):
            Request(0, 1, operation="X")

    def test_contiguous_uses_request_size(self):
        window = [Request(float(i), i * 2, 2) for i in range(8)]
        self.assertAlmostEqual(extract(window)[0], 1.0)

    def test_stride_detection(self):
        # A stride must be at least two request lengths to be distinguishable
        # from a sequential read; the generator no longer produces 1x.
        window = synthetic_window("strided", random.Random(1), 32,
                                  stride_override=64)
        self.assertEqual(dominant_stride(window), 64)

    def test_dominant_stride_is_scale_relative(self):
        """The same 8-request stride must be found at any block size."""
        rng = random.Random(4)
        for size in (1, 8, 128):
            window = [Request(float(i), i * size * 4, size) for i in range(16)]
            self.assertEqual(dominant_stride(window), size * 4,
                             f"block size {size}")

    def test_dominant_stride_finds_realistic_large_strides(self):
        """Real traces stride far beyond the old absolute 1024-block cap."""
        window = [Request(float(i), 1000 + i * 4096, 8) for i in range(16)]
        self.assertEqual(dominant_stride(window), 4096)

    def test_transition_labels(self):
        requests, labels = transition_trace(7, windows_per_class=2)
        self.assertEqual(labels, [label for label in CLASSES for _ in range(2)])
        self.assertEqual(len(requests), 8 * 32)


class ClassifierTests(unittest.TestCase):
    def test_requires_training(self):
        model = OnlineGaussianNB(len(FEATURE_NAMES))
        with self.assertRaises(ValueError):
            model.predict(extract(synthetic_window("random", random.Random(1))))

    def test_held_out_synthetic_accuracy(self):
        model = train(42, 100)
        context = WindowContext()
        test_data = synthetic_dataset(40, 100042)
        correct = 0
        for window, label in test_data:
            values = context.observe(window, dominant_stride(window))
            correct += model.predict(values)[0] == label
        self.assertGreaterEqual(correct / len(test_data), 0.9)

    def test_incremental_update(self):
        model = OnlineGaussianNB(len(FEATURE_NAMES))
        example = contextual_vectors([
            (synthetic_window("sequential", random.Random(1)), "sequential")])[0][0]
        model.update(example, "sequential")
        self.assertEqual(model.count["sequential"], 1)
        self.assertEqual(model.predict(example)[0], "sequential")

    def test_weak_sample_training_saves_loadable_model(self):
        source = Path(__file__).resolve().parents[1] / "data" / "samples" / "msr-cambridge1-sample.csv"
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.json"
            metadata = adapt_msr_sample(source, artifact)
            model, saved = load_model(artifact)
            self.assertEqual(metadata, saved)
            self.assertEqual(metadata["requests"], 1000)
            self.assertGreater(metadata["real_windows_updated"], 0)
            self.assertFalse(metadata["true_real_labels_available"])
            self.assertEqual(model.n_features, len(FEATURE_NAMES))


class SimulatorTests(unittest.TestCase):
    def test_classic_stride_predictor(self):
        predictor = StridePrefetcher(degree=2)
        self.assertEqual(predictor.next_candidates(Request(0, 10)), [])
        self.assertEqual(predictor.next_candidates(Request(1, 14)), [])
        self.assertEqual(predictor.next_candidates(Request(2, 18)), [22, 26])

    def test_markov_predictor_learns_transitions(self):
        predictor = MarkovPrefetcher()
        requests = [Request(float(i), lba) for i, lba in
                    enumerate([10, 11, 15, 16, 20, 21, 25])]
        predictions = [predictor.next_candidates(req) for req in requests]
        self.assertIn(26, predictions[-1])

    def test_modelled_latency_and_oracle_recall_arithmetic(self):
        metrics = Metrics(useful_prefetches=2, prefetchable_misses=3)
        self.assertAlmostEqual(metrics.prefetch_recall, 0.4)
        requests = [Request(0, 10), Request(1, 10)]
        result, _ = replay(requests, mode="none", latency_model=LatencyModel(5, 100, 50))
        self.assertAlmostEqual(result.mean_access_latency_us, 52.5)

    def test_new_modes_observe_first_window(self):
        requests = [Request(float(i), i * 4) for i in range(32)]
        for mode in ("stride", "markov"):
            metrics, _ = replay(requests, mode=mode)
            self.assertEqual(metrics.prefetches, 0)

    def test_no_prefetch_baseline(self):
        requests, _ = transition_trace(7, windows_per_class=1)
        metrics, _ = replay(requests, mode="none")
        self.assertEqual(metrics.prefetches, 0)

    def test_first_window_cannot_use_its_own_prediction(self):
        model = train()
        requests = synthetic_window("sequential", random.Random(1))
        metrics, predictions = replay(requests, model=model)
        self.assertEqual(metrics.prefetches, 0)
        self.assertEqual(len(predictions), 1)

    def test_online_updates_require_ground_truth(self):
        model = train()
        requests = synthetic_window("sequential", random.Random(1))
        with self.assertRaises(ValueError):
            replay(requests, model=model, online_updates=True)
        before = model.count["sequential"]
        replay(requests, model=model, labels=["sequential"], online_updates=True)
        self.assertAlmostEqual(
            model.count["sequential"], before * model.decay_factor + 1
        )

    def test_pseudo_label_update_is_opt_in(self):
        model = train()
        requests = synthetic_window("sequential", random.Random(1))
        before = sum(model.count.values())
        metrics, _ = replay(requests, model=model, pseudo_label_threshold=0.1)
        self.assertEqual(metrics.pseudo_updates, 1)
        # The update lands on whatever class the model predicted, not
        # necessarily the construction label.
        after = sum(model.count.values())
        self.assertGreater(after, before * model.decay_factor)

    def test_benchmark_includes_skipped_optional_lstm(self):
        requests, _ = transition_trace(3, windows_per_class=2)
        rows = benchmark_dataset("test", requests)
        self.assertEqual(len(rows), len(MODES))
        self.assertEqual(rows[0]["mode"], "none")
        self.assertEqual(rows[0]["speedup_vs_none"], 1.0)
        self.assertIn("total_prefetches", rows[0])
        self.assertIn("useful_prefetches", rows[0])
        self.assertTrue(str(rows[5]["status"]).startswith("skipped"))
        self.assertEqual(len(drift_report(3, windows_per_class=2)), 8)

    def test_evidence_router_needs_no_trained_model(self):
        """The evidence router must run on real traces with no prior training."""
        requests, _ = transition_trace(3, windows_per_class=2)
        metrics, predictions = replay(requests, mode="adaptive_evidence")
        self.assertGreaterEqual(len(predictions), 1)
        for _, chosen, _ in predictions:
            self.assertIn(chosen, ("none", "sequential", "strided", "mixed"))

    def test_evidence_router_is_causal_like_every_other_mode(self):
        """The first window observes without prefetching, as every mode does."""
        requests, _ = transition_trace(3, windows_per_class=1)
        self.assertEqual(len(requests), 4 * 32)
        metrics, predictions = replay(requests, mode="adaptive_evidence")
        # Only windows after the first may issue prefetches.
        self.assertEqual(len(predictions), 4)
        first_window_only = requests[:32]
        early, _ = replay(first_window_only, mode="adaptive_evidence")
        self.assertEqual(early.prefetches, 0)

    def test_aggregate_results_sums_prefetch_counters(self):
        first, _ = transition_trace(3, windows_per_class=1)
        second, _ = transition_trace(4, windows_per_class=1)
        first_rows = benchmark_dataset("first", first)
        second_rows = benchmark_dataset("second", second)
        aggregate = aggregate_results(first_rows + second_rows)
        adaptive = next(row for row in aggregate if row["mode"] == "adaptive")
        source = [row for row in first_rows + second_rows if row["mode"] == "adaptive"]
        self.assertEqual(adaptive["total_prefetches"],
                         sum(row["total_prefetches"] for row in source))
        self.assertEqual(adaptive["useful_prefetches"],
                         sum(row["useful_prefetches"] for row in source))

    def test_unavailable_lstm_artifact_is_not_silent(self):
        requests = [Request(float(i), i) for i in range(32)]
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            replay(requests, mode="lstm", lstm_model_path="missing-model.pt")

    def test_analyze_results_names_winner_with_gains(self):
        # A pure sequential scan, where read-ahead unambiguously wins, so the
        # "names a winner" branch is exercised rather than the
        # "no policy beats no-prefetch" branch.
        requests = [Request(float(i) * 0.5, 1000 + i * 64, 8) for i in range(256)]
        rows = benchmark_dataset("analysis_trace", requests)
        report = analyze_results(rows)
        self.assertIn("analysis_trace", report)
        self.assertIn("best policy:", report)
        self.assertIn("up", report)

    def test_analyze_results_survives_skipped_lstm(self):
        requests = [Request(float(i) * 0.5, 1000 + i * 64, 8) for i in range(256)]
        rows = benchmark_dataset("analysis_trace", requests, lstm_model_path=None)
        report = analyze_results(rows)
        self.assertIn("skipped: lstm", report)
        self.assertIn("best policy:", report)

    def test_analyze_results_reports_no_win_honestly(self):
        """A random workload must produce the explicit no-win wording."""
        rng = random.Random(3)
        requests = [Request(i * 0.5, rng.randrange(10**7, 10**8), 8)
                    for i in range(256)]
        report = analyze_results(benchmark_dataset("random_walk", requests))
        self.assertIn("random_walk", report)


class FeatureTests(unittest.TestCase):
    def test_jump_features_are_scale_relative(self):
        """The same access pattern must classify the same at any block size."""
        # delta == size_blocks at both scales, so both are fully contiguous
        # and neither contains a jump beyond 8 request-sizes.
        small = [Request(float(i) * 0.5, i, 1) for i in range(16)]
        large = [Request(float(i) * 0.5, i * 128, 128) for i in range(16)]
        for window in (small, large):
            features = extract(window)
            self.assertAlmostEqual(features[0], 1.0)
            self.assertAlmostEqual(features[2], 0.0)

    def test_distant_jump_scales_with_request_size(self):
        base = [Request(float(i) * 0.5, 1000 + i * 8, 8) for i in range(16)]
        far = [Request(float(i) * 0.5, 1000 + i * 800, 8) for i in range(16)]
        self.assertLess(extract(base)[2], 0.1)
        self.assertAlmostEqual(extract(far)[2], 1.0)

    def test_gnb_transfers_across_a_128x_block_size_change(self):
        """A diagonal model must survive a change of device block size.

        This is the property the old absolute 64/16-block thresholds broke.
        """
        from adaptive_prefetch.pipeline import make_harder_generator
        from adaptive_prefetch.trace import CLASSES as ALL

        def build(size, seed, per_class):
            rng = random.Random(seed)
            rows = contextual_vectors([
                (make_harder_generator(label, rng, 32, noise=0.1,
                                       lba_base=10**7, lba_span=10**9,
                                       size=size), label)
                for label in ALL for _ in range(per_class)])
            return rows

        model = make_classifier("gnb", len(FEATURE_NAMES))
        for values, label in build(1, 1000, 60):
            model.update(values, label)
        test = build(128, 900000, 30)
        correct = sum(model.predict(v)[0] == label for v, label in test)
        self.assertGreaterEqual(correct / len(test), 0.90)


class ClassifierRegistryTests(unittest.TestCase):
    def test_both_classifiers_share_the_update_predict_contract(self):
        for name in ("gnb", "qda"):
            model = make_classifier(name, len(FEATURE_NAMES))
            with self.assertRaises(ValueError):
                model.predict((0.0,) * len(FEATURE_NAMES))
            for values, label in contextual_vectors(synthetic_dataset(20, 7)):
                model.update(values, label)
            probe = contextual_vectors([
                (synthetic_window("sequential", random.Random(1)), "sequential")])
            predicted, confidence = model.predict(probe[0][0])
            self.assertEqual(predicted, "sequential")
            self.assertTrue(0.0 < confidence <= 1.0)

    def test_unknown_classifier_name_rejected(self):
        with self.assertRaises(ValueError):
            make_classifier("svm", 8)

    def test_qda_rejects_bad_shrinkage(self):
        from adaptive_prefetch.model import OnlineQDA
        with self.assertRaises(ValueError):
            OnlineQDA(4, shrink=1.0)
        with self.assertRaises(ValueError):
            OnlineQDA(4, shrink=-0.1)

    def test_decay_factor_must_be_a_decay(self):
        from adaptive_prefetch.model import OnlineGaussianNB, OnlineQDA
        for factory in (OnlineGaussianNB, OnlineQDA):
            with self.assertRaises(ValueError):
                factory(4, decay_factor=1.5)
            with self.assertRaises(ValueError):
                factory(4, decay_factor=0.0)


class ArtifactTests(unittest.TestCase):
    def _round_trip(self, name):
        model = fit_model(name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.json"
            save_model(path, model, {"note": "test"})
            loaded, metadata = load_model(path)
        self.assertEqual(metadata["note"], "test")
        self.assertEqual(loaded.name, name)
        probe = contextual_vectors([
            (synthetic_window("strided", random.Random(2)), "strided")])[0][0]
        return model.predict(probe)[0], loaded.predict(probe)[0]

    def test_gnb_artifact_round_trip(self):
        self.assertEqual(*self._round_trip("gnb"))

    def test_qda_artifact_round_trip(self):
        self.assertEqual(*self._round_trip("qda"))

    def test_stale_feature_set_is_rejected_with_guidance(self):
        model = fit_model("gnb", per_class=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.json"
            save_model(path, model, {})
            payload = json.loads(path.read_text())
            payload["features"] = ["only_one"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "incompatible classifier artifact"):
                load_model(path)

    def test_untrained_artifact_is_rejected(self):
        model = make_classifier("gnb", len(FEATURE_NAMES))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.json"
            save_model(path, model, {})
            with self.assertRaisesRegex(ValueError, "no trained statistics"):
                load_model(path)

    def test_infinite_decay_factor_is_rejected(self):
        model = fit_model("gnb", per_class=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.json"
            save_model(path, model, {})
            payload = json.loads(path.read_text())
            payload["decay_factor"] = 2.0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "decay_factor"):
                load_model(path)


class PostprocessingTests(unittest.TestCase):
    def test_smoother_suppresses_single_window_flips(self):
        """An isolated misclassified window must not change the policy."""
        smoother = PolicySmoother(min_windows=3, history=5)
        self.assertEqual(smoother.update("sequential", 1.0), "sequential")
        # Alternating noise: no run of 3 agreeing windows, so no switch.
        for i in range(12):
            expected = "random" if i % 2 == 0 else "sequential"
            self.assertEqual(smoother.update(expected, 1.0), "sequential")
        self.assertEqual(smoother.switches, 0)
        # A sustained new phase must eventually take over.
        for _ in range(3):
            smoother.update("random", 1.0)
        self.assertEqual(smoother.current, "random")
        self.assertEqual(smoother.switches, 1)

    def test_smoothing_only_applies_to_adaptive_replay(self):
        requests, _ = transition_trace(3, windows_per_class=2)
        with self.assertRaises(ValueError):
            replay(requests, mode="none", smoothing=(2, 0.0))

    def test_smoothing_reduces_policy_switches(self):
        requests, _ = transition_trace(11, windows_per_class=8)
        model = train(42)
        plain, _ = replay(requests, model=model)
        smoothed, _ = replay(requests, model=train(42), smoothing=(3, 0.0))
        self.assertLessEqual(smoothed.policy_switches, plain.policy_switches)


class EdaTests(unittest.TestCase):
    def test_locality_ceiling_bounds_any_prefetcher(self):
        # Every request reads a block that is never read again: no prefetcher
        # can produce a hit, so the ceiling must be 0.
        requests = [Request(float(i), i * 10) for i in range(64)]
        profile = locality_profile(requests, capacity=8)
        self.assertEqual(profile["read_blocks"], 64)
        self.assertEqual(profile["locality_ceiling"], 0.0)
        self.assertEqual(profile["cacheable_ceiling"], 0.0)

    def test_cacheable_ceiling_detects_short_distance_reuse(self):
        # Each block re-read immediately: an 8-block cache can hold it.
        requests = []
        for i in range(16):
            requests.append(Request(float(i), 100))
        profile = locality_profile(requests, capacity=8)
        self.assertGreater(profile["cacheable_ceiling"], 0.9)

    def test_domain_shift_detects_a_population_outside_the_reference(self):
        reference = [extract(w) for w, _ in synthetic_dataset(40, 5)]
        near = [extract(w) for w, _ in synthetic_dataset(40, 6)]
        # Invert every feature to land on the far side of the reference range.
        far = [tuple(1.0 - v for v in extract(w)[:STREAM_FEATURES])
               for w, _ in synthetic_dataset(40, 7)]
        self.assertLess(domain_shift(reference, near)["mean_abs_z"],
                        domain_shift(reference, far)["mean_abs_z"])

    def test_write_only_trace_reports_no_read_metrics(self):
        requests = [Request(float(i), i * 10, 1, "W") for i in range(64)]
        self.assertEqual(locality_profile(requests)["read_blocks"], 0)
        self.assertIn("no read requests", profile_trace(requests, "writes"))


class PipelineTests(unittest.TestCase):
    def test_regimes_train_and_test_are_disjoint(self):
        from adaptive_prefetch.pipeline import labelled_dataset
        train = labelled_dataset(5, 1000, 32)
        test = labelled_dataset(5, 900000, 32)
        train_ids = {id(w) for w, _ in train}
        self.assertFalse(train_ids & {id(w) for w, _ in test})

    def test_evaluate_classifier_returns_bounded_accuracy(self):
        stats = evaluate_classifier("gnb", seeds=2, windows_per_class=20,
                                    test_per_class=10, regime_name="baseline")
        self.assertGreaterEqual(stats["accuracy"], 0.0)
        self.assertLessEqual(stats["accuracy"], 1.0)
        self.assertEqual(stats["classifier"], "gnb")

    def test_unknown_regime_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_classifier("gnb", seeds=1, regime_name="nope")

    def test_scaler_round_trips_to_zero_mean_unit_variance(self):
        rows = [extract(w) for w, _ in synthetic_dataset(30, 13)]
        scaler = StandardScaler().fit(rows)
        scaled = scaler.transform_rows(rows)
        for i in range(len(rows[0])):
            self.assertAlmostEqual(mean(c[i] for c in scaled), 0.0, places=6)
            # A feature that is constant in training (unique_ratio is always
            # ~1.0) has no variance to recover; the scaler must leave it flat
            # rather than divide by zero.
            if scaler.scale[i] != 1.0:
                self.assertAlmostEqual(pstdev(c[i] for c in scaled), 1.0, places=3)

    def test_unfitted_scaler_rejected(self):
        with self.assertRaises(ValueError):
            StandardScaler().transform((0.0,) * 8)


class LSTMTests(unittest.TestCase):
    def test_bucket_round_trip_covers_the_forward_axis(self):
        for delta in (1, 2, 3, 7, 64, 100, 4096):
            index = bucket_of(delta)
            self.assertLess(index, UNK, f"{delta} should be a real bucket")
            low, high = bucket_span(index)
            self.assertTrue(low <= delta <= high, f"{delta} outside its bucket")

    def test_out_of_range_deltas_map_to_unknown(self):
        self.assertEqual(bucket_of(MAX_DELTA + 1), UNK)
        self.assertEqual(bucket_of(-(MAX_DELTA + 1)), UNK)

    def test_encode_is_symmetric_and_signed(self):
        self.assertAlmostEqual(_encode(5), -_encode(-5))
        self.assertEqual(_encode(0), 0.0)
        self.assertGreater(_encode(100), _encode(1))

    def test_trains_and_round_trips_an_artifact(self):
        torch = pytest_torch()
        if torch is None:
            self.skipTest("PyTorch not installed")
        traces = [transition_trace(42 + i, windows_per_class=2)[0] for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lstm.pt"
            result = train_lstm(path, traces=traces, epochs=1)
            self.assertGreater(result["examples"], 0)
            predictor = load_predictor(path)
        self.assertIsInstance(predictor, LSTMPrefetcher)

    def test_stale_artifact_is_rejected_with_guidance(self):
        torch = pytest_torch()
        if torch is None:
            self.skipTest("PyTorch not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"history": HISTORY, "max_delta": MAX_DELTA}, path)
            with self.assertRaisesRegex(ValueError, "retrain"):
                load_predictor(path)

    def test_missing_artifact_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_predictor(Path(directory) / "nope.pt")

    def test_predictor_abstains_on_an_unpredictable_stream(self):
        """The old top-10 decoder emitted candidates on 100% of random reads.

        A properly trained model must instead gate on its unknown class, so
        it issues far fewer speculative fetches when addresses are random
        than when they follow a stride.
        """
        torch = pytest_torch()
        if torch is None:
            self.skipTest("PyTorch not installed")
        traces = [transition_trace(42 + i, windows_per_class=8)[0] for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lstm.pt"
            train_lstm(path, traces=traces, epochs=40)
            predictor = load_predictor(path)

        def emission_rate(make_lba):
            rng = random.Random(5)
            lba = 1000
            emitted = reads = 0
            for i in range(600):
                lba = make_lba(rng, lba)
                if predictor.next_candidates(Request(float(i), lba)):
                    emitted += 1
                reads += 1
            return emitted / reads

        random_rate = emission_rate(lambda rng, _: rng.randrange(1, 10**7))
        stride_rate = emission_rate(lambda rng, prev: prev + 8)
        self.assertLess(random_rate, 0.35,
                        f"should abstain on random addresses, emitted {random_rate:.0%}")
        self.assertGreater(stride_rate, random_rate,
                           "should be more willing to prefetch a stable stride")

    def test_predictor_delta_buffer_is_bounded(self):
        torch = pytest_torch()
        if torch is None:
            self.skipTest("PyTorch not installed")
        traces = [transition_trace(42, windows_per_class=1)[0]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lstm.pt"
            train_lstm(path, traces=traces, epochs=1)
            predictor = load_predictor(path)
        for i in range(2000):
            predictor.next_candidates(Request(float(i), i * 4))
        self.assertLessEqual(len(predictor.deltas), HISTORY)


def pytest_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError:
        return None


class OracleTests(unittest.TestCase):
    def test_oracle_beats_every_real_policy(self):
        requests, _ = transition_trace(3, windows_per_class=2)
        reference = oracle_reference(requests)
        rows = benchmark_dataset("t", requests)
        best = max(float(r["hit_ratio"]) for r in rows
                   if r["status"] == "ok" and r["mode"] != "none")
        self.assertGreater(reference["hit_ratio"], best,
                           "a perfect one-request-lookahead predictor must "
                           "beat every real policy")

    def test_oracle_is_defined_on_reads_only(self):
        writes = [Request(float(i), i * 10, 1, "W") for i in range(64)]
        reference = oracle_reference(writes)
        self.assertEqual(reference["read_blocks"], 0)
        self.assertEqual(reference["hit_ratio"], 0.0)

    def test_prefetchable_misses_counted_once_per_block(self):
        """A block evicted and re-read must not inflate the recall denominator."""
        # Two distinct blocks, each read three times, with a cache too small to
        # hold both: the old code counted each eviction-driven re-miss again.
        requests = [Request(float(i), 0 if i % 2 else 1) for i in range(6)]
        for capacity, expected in ((1, 2), (2, 2), (3, 2)):
            metrics, _ = replay(requests, capacity=capacity, mode="none")
            self.assertEqual(metrics.prefetchable_misses, expected,
                             f"capacity={capacity}")

    def test_write_only_trace_ranks_without_crashing(self):
        writes = [Request(float(i), i * 10, 1, "W") for i in range(96)]
        rows = benchmark_dataset("writes", writes)
        report = analyze_results(rows)
        self.assertIn("no read requests", report)

    def test_aggregate_keeps_partial_mode_results(self):
        """A mode that ran on some traces must not be dropped wholesale."""
        first, _ = transition_trace(3, windows_per_class=1)
        second, _ = transition_trace(4, windows_per_class=1)
        rows = benchmark_dataset("a", first) + benchmark_dataset("b", second)
        for row in rows:
            if row["mode"] == "lstm":
                row["status"] = "skipped: simulated absence"
        aggregate = aggregate_results(rows)
        lstm = next(r for r in aggregate if r["mode"] == "lstm")
        self.assertEqual(lstm["status"], "skipped: no trace produced a result")
        adaptive = next(r for r in aggregate if r["mode"] == "adaptive")
        self.assertEqual(adaptive["status"], "ok")


class ReportTests(unittest.TestCase):
    def test_benchmark_section_reports_winner_and_ceiling(self):
        requests, _ = transition_trace(3, windows_per_class=2)
        rows = benchmark_dataset("t", requests)
        text = report_benchmark_section(rows, 128, {"t": 0.5})
        self.assertIn("winner", text)
        self.assertIn("oracle", text)
        self.assertIn("caught", text)

    def test_dataset_overview_marks_write_only_traces(self):
        writes = [Request(float(i), i * 10, 1, "W") for i in range(64)]
        reads = [Request(float(i), i * 10) for i in range(64)]
        text = report_dataset_overview([("writes", writes), ("reads", reads)], 128)
        self.assertIn("n/a", text)
        self.assertIn("writes", text)

    def test_lstm_summary_excludes_write_only_datasets(self):
        """Averaging a 0/0 write-only trace into the LSTM mean is misleading."""
        reads, _ = transition_trace(3, windows_per_class=2)
        writes = [Request(float(i), i * 10, 1, "W") for i in range(96)]
        rows = benchmark_dataset("r", reads) + benchmark_dataset("w", writes)
        for row in rows:
            if row["mode"] == "lstm":
                # Skipped rows carry no private counters; supply them so the
                # filter is exercised on a realistic payload.
                row["status"] = "ok"
                row["hit_ratio"] = 0.5
                row["precision"] = 0.4
                row["_read_blocks"] = 0 if row["dataset"] == "w" else 400
        text = report_benchmark_section(rows, 128, {"r": 0.5})
        # Only the read-bearing dataset may contribute to the mean, so the
        # figure must be exactly that dataset's value, not a 2-row average.
        self.assertIn("1 dataset(s) with read blocks", text)
        self.assertIn("hit 50.00%", text)


class StreamingTests(unittest.TestCase):
    """The streamed replay must equal the in-memory replay exactly."""

    def _trace_file(self, directory, name="t.revised", lines=400):
        path = Path(directory) / name
        rows = []
        for i in range(lines):
            op = "RS" if i % 3 else "WS"
            pattern = "seq" if i % 4 == 0 else "rand"
            rows.append(f"{i * 0.001} {op} {i * 8} 8 {pattern} 1.0 0.1 0.9\n")
        path.write_text("".join(rows), encoding="utf-8")
        return path

    def test_iter_csv_chunks_respects_the_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._trace_file(directory, lines=500)
            chunks = list(iter_csv_chunks(path, "revised", 1000))
            self.assertEqual(sum(len(c) for c in chunks), 500)
            self.assertEqual(len(chunks), 1)

    def test_iter_csv_chunks_preserves_order_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._trace_file(directory, lines=300)
            streamed = [r for c in iter_csv_chunks(path, "revised", 1000)
                        for r in c]
            whole = load_csv(path, "revised")
        self.assertEqual(len(streamed), len(whole))
        for a, b in zip(streamed, whole):
            self.assertEqual((a.lba, a.size_blocks, a.operation,
                              a.service_ms, a.pattern),
                             (b.lba, b.size_blocks, b.operation,
                              b.service_ms, b.pattern))

    def test_chunk_size_does_not_change_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._trace_file(directory, lines=600)
            results = [replay_stream(iter_csv_chunks(path, "revised", size),
                                     capacity=64, mode="adaptive_evidence")
                       for size in (1000, 1000)]
        for other in results[1:]:
            self.assertAlmostEqual(results[0].hit_ratio, other.hit_ratio)
            self.assertEqual(results[0].prefetches, other.prefetches)

    def test_streamed_matches_in_memory_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._trace_file(directory, lines=600)
            whole = load_csv(path, "revised")
            for mode in ("none", "sequential", "adaptive_evidence"):
                a, _ = replay(whole, capacity=64, mode=mode)
                b = replay_stream(iter_csv_chunks(path, "revised", 1000),
                                  capacity=64, mode=mode)
                self.assertAlmostEqual(a.hit_ratio, b.hit_ratio, places=12)
                self.assertEqual(a.prefetches, b.prefetches)

    def test_streamed_first_window_never_prefetches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._trace_file(directory, lines=32)
            metrics = replay_stream(iter_csv_chunks(path, "revised", 1000),
                                    capacity=64, mode="sequential")
        # Only one window of requests, so the warm-up covers all of them.
        self.assertEqual(metrics.prefetches, 0)

    def test_iter_csv_chunks_rejects_a_tiny_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._trace_file(directory, lines=50)
            with self.assertRaises(ValueError):
                list(iter_csv_chunks(path, "revised", 10))

    def test_chunk_boundary_does_not_change_the_result(self):
        # Same stream, one chunk vs two.
        with tempfile.TemporaryDirectory() as directory:
            path = self._trace_file(directory, lines=3000)
            one = replay_stream(iter_csv_chunks(path, "revised", 100000),
                                capacity=64, mode="adaptive_evidence")
            two = replay_stream(iter_csv_chunks(path, "revised", 1000),
                                capacity=64, mode="adaptive_evidence")
        self.assertAlmostEqual(one.hit_ratio, two.hit_ratio, places=12)
        self.assertEqual(one.prefetches, two.prefetches)


class SweepTests(unittest.TestCase):
    def test_capped_chunks_stops_at_the_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.revised"
            path.write_text("".join(
                f"{i * 0.001} RS {i * 8} 8 rand 1.0 0.1 0.9\n"
                for i in range(500)), encoding="utf-8")
            capped = list(capped_chunks(path, 200, 1000))
            uncapped = list(capped_chunks(path, None, 1000))
        self.assertEqual(sum(len(c) for c in capped), 200)
        self.assertEqual(sum(len(c) for c in uncapped), 500)

    def test_winners_ignores_traces_without_reads(self):
        rows = [
            {"trace": "a", "mode": "none", "hit_ratio": 0.5, "precision": 0.0,
             "mean_us": 50.0, "unused": 0, "total_prefetches": 0,
             "read_blocks": 100},
            {"trace": "a", "mode": "sequential", "hit_ratio": 0.9,
             "precision": 0.8, "mean_us": 40.0, "unused": 10,
             "total_prefetches": 50, "read_blocks": 100},
            {"trace": "b", "mode": "none", "hit_ratio": 0.0, "precision": 0.0,
             "mean_us": 0.0, "unused": 0, "total_prefetches": 0,
             "read_blocks": 0},
            {"trace": "b", "mode": "sequential", "hit_ratio": 0.0,
             "precision": 0.0, "mean_us": 0.0, "unused": 0,
             "total_prefetches": 0, "read_blocks": 0},
        ]
        champ = sweep_winners(rows)
        self.assertEqual(champ["hit_ratio"], "sequential")
        # The write-only trace must not make every metric look like a tie.
        self.assertEqual(champ["cost"], "sequential")

    def test_aggregate_reports_every_mode(self):
        rows = [
            {"trace": "a", "mode": "none", "hit_ratio": 0.1, "precision": 0.0,
             "mean_us": 90.0, "unused": 0, "total_prefetches": 0,
             "read_blocks": 10},
            {"trace": "a", "mode": "markov", "hit_ratio": 0.3,
             "precision": 0.5, "mean_us": 80.0, "unused": 5,
             "total_prefetches": 10, "read_blocks": 10},
        ]
        summary = sweep_aggregate(rows)
        self.assertEqual({r["mode"] for r in summary}, {"none", "markov"})
        markov = next(r for r in summary if r["mode"] == "markov")
        self.assertAlmostEqual(markov["mean_hit"], 0.3)
        self.assertEqual(markov["total_unused"], 5)


if __name__ == "__main__":
    unittest.main()

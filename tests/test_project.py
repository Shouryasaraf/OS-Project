import random
import tempfile
import unittest
from pathlib import Path

from adaptive_prefetch.baselines import MarkovPrefetcher, StridePrefetcher
from adaptive_prefetch.benchmark import benchmark_dataset, drift_report
from adaptive_prefetch.cli import train
from adaptive_prefetch.features import FEATURE_NAMES, dominant_stride, extract
from adaptive_prefetch.model import OnlineGaussianNB
from adaptive_prefetch.simulator import LatencyModel, Metrics, replay
from adaptive_prefetch.trace import (CLASSES, Request, load_csv, synthetic_dataset,
                                     synthetic_window, transition_trace, write_csv)


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
        window = synthetic_window("strided", random.Random(1), stride_override=8)
        self.assertEqual(dominant_stride(window), 8)

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
        test_data = synthetic_dataset(40, 100042)
        accuracy = sum(model.predict(extract(window))[0] == label
                       for window, label in test_data) / len(test_data)
        self.assertGreaterEqual(accuracy, 0.9)

    def test_incremental_update(self):
        model = OnlineGaussianNB(len(FEATURE_NAMES))
        example = extract(synthetic_window("sequential", random.Random(1)))
        model.update(example, "sequential")
        self.assertEqual(model.count["sequential"], 1)
        self.assertEqual(model.predict(example)[0], "sequential")


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
        before = model.count["sequential"]
        metrics, _ = replay(requests, model=model, pseudo_label_threshold=0.1)
        self.assertEqual(metrics.pseudo_updates, 1)
        self.assertGreater(model.count["sequential"], before * model.decay_factor)

    def test_benchmark_includes_skipped_optional_lstm(self):
        requests, _ = transition_trace(3, windows_per_class=2)
        rows = benchmark_dataset("test", requests)
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]["mode"], "none")
        self.assertEqual(rows[0]["speedup_vs_none"], 1.0)
        self.assertTrue(str(rows[5]["status"]).startswith("skipped"))
        self.assertEqual(len(drift_report(3, windows_per_class=2)), 8)

    def test_unavailable_lstm_artifact_is_not_silent(self):
        requests = [Request(float(i), i) for i in range(32)]
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            replay(requests, mode="lstm", lstm_model_path="missing-model.pt")


if __name__ == "__main__":
    unittest.main()

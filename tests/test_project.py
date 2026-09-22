import random
import tempfile
import unittest
from pathlib import Path

from adaptive_prefetch.cli import train
from adaptive_prefetch.features import FEATURE_NAMES, dominant_stride, extract
from adaptive_prefetch.model import OnlineGaussianNB
from adaptive_prefetch.simulator import replay
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


if __name__ == "__main__":
    unittest.main()

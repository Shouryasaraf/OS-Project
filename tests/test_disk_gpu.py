"""GPU disk models (WP9): the retrained LSTM and the ``learned_depth`` mode.

Everything runs on the CPU with tiny models, so the suite needs no GPU; the
one CUDA test is gated on ``torch.cuda.is_available()``. Tests that need
numpy or torch skip without them, so system python3 still runs the
registration tests.
"""

import contextlib
import importlib.util
import io
import random
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from adaptive_prefetch import sweep
from adaptive_prefetch.benchmark import MODES, OPTIONAL_MODES, benchmark_dataset
from adaptive_prefetch.depth_model import (RelativeDepth, chronological_split,
                                           overlaps, read_training_record,
                                           scoring_conflict, training_record_path,
                                           write_training_record)
from adaptive_prefetch.simulator import (GATED_MODES, LatencyModel, LRUCache,
                                         MODES_SUPPORTED, _make_predictor, replay,
                                         replay_stream)
from adaptive_prefetch.trace import Request, transition_trace

REPO = Path(__file__).resolve().parents[1]
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_TORCH = HAVE_NUMPY and importlib.util.find_spec("torch") is not None
HAVE_CUDA = False
if HAVE_TORCH:
    import torch
    HAVE_CUDA = torch.cuda.is_available()

RECORD = struct.Struct("<IQIq")
COUNTERS = ("read_blocks", "read_hits", "prefetches", "useful_prefetches",
            "demand_ios", "prefetch_ios", "prefetch_ios_mergeable",
            "inference_calls", "policy_switches")


def workload(n=3000, seed=3, size=8):
    """Runs of sequential reads broken by jumps: depth matters here."""
    rng = random.Random(seed)
    out, lba = [], 1 << 20
    for i in range(n):
        if rng.random() < 0.15:
            lba = rng.randrange(1 << 16, 1 << 26) * 8
        out.append(Request(i * 10.0, lba, size))
        lba += size
    return out


def write_oracle(path, requests):
    """oracleGeneral records (byte offsets and sizes) for ``requests``."""
    with open(path, "wb") as handle:
        for r in requests:
            handle.write(RECORD.pack(int(r.timestamp_ms // 1000), r.lba * 512,
                                     r.size_blocks * 512, -1))
    return Path(path)


def _quiet(call, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = call(*args)
    return result, out.getvalue()


class RegistrationTests(unittest.TestCase):
    """Stdlib only: the mode is registered everywhere, skipped without an artifact."""

    def test_registered_in_every_place(self):
        self.assertEqual(MODES[-1], "learned_depth")
        self.assertEqual(MODES[5], "lstm")
        self.assertEqual(MODES.index("learned_depth"), MODES.index("linux_ra") + 1)
        self.assertIn("learned_depth", sweep.RANKING_MODES)
        self.assertIn("learned_depth", MODES_SUPPORTED)
        self.assertIn("learned_depth", sweep.OPTIONAL_MODES)
        self.assertIn("learned_depth", OPTIONAL_MODES)
        self.assertNotIn("learned_depth", GATED_MODES)

    def test_benchmark_rows_keep_their_order_and_skip_without_an_artifact(self):
        rows = benchmark_dataset("t", transition_trace(3, windows_per_class=2)[0])
        self.assertEqual(len(rows), len(MODES))
        self.assertEqual(rows[5]["mode"], "lstm")
        self.assertTrue(str(rows[5]["status"]).startswith("skipped"))
        last = rows[-1]
        self.assertEqual(last["mode"], "learned_depth")
        self.assertEqual(last["status"], "skipped: no trained learned_depth artifact")
        self.assertIsNone(last["hit_ratio"])
        self.assertIsNone(last["total_prefetches"])

    def test_sweep_skips_without_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_oracle(Path(directory) / "msr_t.oracleGeneral", workload(400))
            row = sweep.sweep_trace(path, "learned_depth", latency_model=LatencyModel())
        self.assertEqual(row["status"], "skipped: no trained learned_depth artifact")
        self.assertIsNone(row["hit_ratio"])
        self.assertEqual(set(row), set(sweep.ROW_COLUMNS))

    def test_sweep_refuses_an_artifact_trained_on_the_replayed_requests(self):
        # The guard reads the sidecar before anything loads the weights, so
        # a stand-in artifact file is enough and torch is not needed.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_oracle(root / "msr_t.oracleGeneral", workload(400))
            artifact = root / "depth.pt"
            artifact.write_bytes(b"weights")
            write_training_record(artifact, {"traces": [
                {"trace": "msr_t", "train": [0, 240], "val": [240, 320],
                 "test": [320, 400]}]})
            row = sweep.sweep_trace(path, "learned_depth", latency_model=LatencyModel(),
                                    depth_model_path=str(artifact))
            self.assertEqual(row["status"],
                             "skipped: artifact was trained on msr_t requests [0, 240)")
            # A sidecar that no longer matches the artifact proves nothing.
            artifact.write_bytes(b"other weights")
            row = sweep.sweep_trace(path, "learned_depth", latency_model=LatencyModel(),
                                    depth_model_path=str(artifact))
            self.assertIn("no matching training record", row["status"])

    def test_guards_name_traces_as_the_training_record_does(self):
        # Records name traces by corpora.trace_stem; sweep.trace_name differs
        # for .revised files, which once let an in-sample artifact through.
        from adaptive_prefetch import cli
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "depth.pt"
            artifact.write_bytes(b"weights")
            write_training_record(artifact, {"traces": [
                {"trace": "hm_1.csv.revised", "train": [0, 60], "val": [60, 80]},
                {"trace": "msr_hm_0", "train": [0, 600], "val": [600, 800]}]})
            self.assertIn("trained on hm_1.csv.revised", sweep._depth_conflict(
                str(artifact), root / "hm_1.csv.revised", 100))
            # The committed MSR sample is the first 1000 msr_hm_0 requests.
            self.assertIn("trained on msr_hm_0", cli._depth_skip_reason(
                str(artifact), "data/samples/msr-cambridge1-sample.csv", 1000))
            self.assertIsNone(cli._depth_skip_reason(str(artifact), None, 1000))
            # No matching record: the training range is unknown, so refuse.
            artifact.write_bytes(b"other weights")
            self.assertIn("no matching training record",
                          cli._depth_skip_reason(str(artifact), None, 1000))

    def test_make_predictor_needs_an_artifact_or_a_policy(self):
        with self.assertRaisesRegex(RuntimeError, "no trained learned_depth artifact"):
            _make_predictor("learned_depth", None)
        policy = RelativeDepth(1.0)
        self.assertIs(_make_predictor("learned_depth", None, depth_model=policy), policy)

    def test_relative_depth_reads_ahead_k_times_the_request(self):
        policy = RelativeDepth(2.0, action_cap=64)
        self.assertEqual(policy.next_candidates(Request(0.0, 100, 8)), list(range(108, 124)))
        self.assertEqual(len(policy.next_candidates(Request(0.0, 0, 64))), 64)
        self.assertEqual(policy.next_candidates(Request(0.0, 0, 8, "W")), [])

    def test_scoring_conflict_and_split(self):
        self.assertEqual(chronological_split(10), ((0, 6), (6, 8), (8, 10)))
        self.assertTrue(overlaps((0, 6), (5, None)))
        self.assertFalse(overlaps((0, 6), (6, 8)))
        record = {"traces": [{"trace": "a", "train": [0, 60], "val": [60, 80],
                              "test": [80, 100]}]}
        self.assertIsNone(scoring_conflict(record, "a", 80, 100))
        self.assertIsNone(scoring_conflict(record, "b", 0, None))
        self.assertIn("selected", scoring_conflict(record, "a", 79, 100))
        self.assertIn("trained", scoring_conflict(record, "a", 0, None))
        self.assertIsNone(scoring_conflict({"source": "transition_trace"}, "a", 0, 1))
        self.assertIsNone(scoring_conflict(None, "a", 0, 1))

    def test_sidecar_is_bound_to_the_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "m.pt"
            artifact.write_bytes(b"x")
            path = write_training_record(artifact, {"model": "lstm"})
            self.assertEqual(path, training_record_path(artifact))
            self.assertEqual(path.name, "m.training.json")
            self.assertEqual(read_training_record(artifact)["model"], "lstm")
            artifact.write_bytes(b"y")
            self.assertIsNone(read_training_record(artifact))

    def test_cli_formats_come_from_the_loader(self):
        from adaptive_prefetch import cli, trace
        self.assertIs(cli.FORMATS, trace.FORMATS)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class DepthLabelTests(unittest.TestCase):
    def test_seg_labels_are_all_one_on_a_scan(self):
        from adaptive_prefetch.depth_model import SEGMENTS, seg_labels
        n, size, horizon = 400, 8, 256
        lba = [i * size for i in range(n)]
        labels = seg_labels(lba, [size] * n, horizon)
        # A row only sees a full future while the scan continues past 256
        # blocks beyond its end, i.e. 32 more requests.
        reach = max(b for _, b in SEGMENTS) // size
        self.assertTrue((labels[:n - reach] == 1).all())
        self.assertTrue((labels[-1] == 0).all())

    def test_seg_labels_are_rare_on_random_input(self):
        from adaptive_prefetch.depth_model import seg_labels
        rng = random.Random(1)
        n = 3000
        lba = [rng.randrange(0, 10 ** 9) for _ in range(n)]
        self.assertLess(float(seg_labels(lba, [8] * n, 256).mean()), 0.05)

    def test_vectorised_features_match_the_online_ones(self):
        from adaptive_prefetch.depth_model import request_features, step_features
        requests = workload(300) + [Request(5000.0, 3, 1), Request(4990.0, 0, 2)]
        vectorised = step_features([r.timestamp_ms for r in requests],
                                   [r.lba for r in requests],
                                   [r.size_blocks for r in requests])
        previous = None
        for row, request in zip(vectorised, requests):
            online = request_features(previous, request)
            for a, b in zip(row.tolist(), online):
                self.assertAlmostEqual(a, b, places=6)
            previous = request

    def test_training_rows_never_label_from_validation(self):
        from adaptive_prefetch.depth_model import prefix_from_requests, prepare
        prepared = prepare(prefix_from_requests(workload(3000)), 2048)
        (_, c1), (_, c2), _ = prepared.split
        self.assertLess(int(prepared.train_rows.max()) + prepared.horizon, c1)
        self.assertGreaterEqual(int(prepared.val_rows.min()) - 31, c1)
        self.assertLess(int(prepared.val_rows.max()) + prepared.horizon, c2)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class DepthModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adaptive_prefetch import depth_model as D
        cls.D = D
        cls.tmp = tempfile.TemporaryDirectory()
        cls.prepared = D.prepare(D.prefix_from_requests(workload(3000), "tiny"), 512)
        cls.result = D.train_and_select(
            cls.prepared, Path(cls.tmp.name) / "tiny.pt", epochs=1, batch_size=512,
            device="cpu", hidden=16, layers=1, val_replay=None)
        cls.artifact = cls.result.artifact
        cls.network = D.load_artifact(cls.artifact)["network"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def policy(self, threshold=0.0, cap=256, gap=None):
        return self.D.DepthPrefetcher(self.network, self.prepared.segments,
                                      self.prepared.vocab, threshold, gap, cap)

    def test_network_shape_matches_the_plan(self):
        network = self.D._network(4)
        self.assertEqual(self.D.parameter_count(network), 298_500)
        self.assertEqual(self.D.parameter_count(self.D._network(6)), 298_758)

    def test_candidates_ignore_the_future(self):
        requests = workload(400, seed=8)
        reference = self.policy(threshold=0.4)
        expected = [reference.next_candidates(r) for r in requests]
        for cut in (31, 32, 100, 250):
            perturbed = requests[:cut + 1] + [Request(r.timestamp_ms, 7 * r.lba % 10 ** 7, 3)
                                              for r in requests[cut + 1:]]
            policy = self.policy(threshold=0.4)
            got = [policy.next_candidates(r) for r in perturbed]
            self.assertEqual(got[:cut + 1], expected[:cut + 1], f"cut={cut}")

    def test_both_replay_paths_are_bit_identical(self):
        requests = workload(2500, seed=4)
        a, _ = replay(requests, capacity=512, mode="learned_depth",
                      depth_model=str(self.artifact))
        for size in (37, 700, 2500):
            b = replay_stream(iter([requests[i:i + size]
                                    for i in range(0, len(requests), size)]),
                              capacity=512, mode="learned_depth",
                              depth_model=str(self.artifact))
            for field in COUNTERS:
                self.assertEqual(getattr(a, field), getattr(b, field), f"{size}: {field}")
        self.assertEqual(a.inference_calls, len(requests))

    def test_the_first_window_never_prefetches(self):
        # Threshold 0 makes every decision choose the deepest segment, and
        # the first decision happens at index 31, inside the first window.
        issued_at = []

        class Recording(LRUCache):
            def prefetch(self, block):
                issued_at.append(len(seen))
                return super().prefetch(block)

        seen = []
        requests = workload(64, seed=2)
        seen_requests = iter(range(len(requests)))

        def factory(capacity, metrics):
            return Recording(capacity, metrics)

        class Counting:
            def __init__(self, inner):
                self.inner = inner

            def next_candidates(self, request):
                seen.append(next(seen_requests))
                return self.inner.next_candidates(request)

        metrics, _ = replay(requests[:32], mode="learned_depth",
                            depth_model=self.policy(0.0))
        self.assertEqual(metrics.prefetches, 0)
        replay(requests, mode="learned_depth", depth_model=Counting(self.policy(0.0)),
               cache_factory=factory)
        self.assertTrue(issued_at)
        self.assertGreaterEqual(min(issued_at), 33)  # request index 32 onwards

    def test_read_ahead_returns_exactly_depth_blocks(self):
        requests = workload(40, seed=6)
        for cap in (64, 256):
            policy = self.policy(0.0, cap=cap)
            out = [policy.next_candidates(r) for r in requests]
            self.assertEqual(out[30], [])
            for request, candidates in zip(requests[31:], out[31:]):
                end = request.lba + request.size_blocks
                self.assertEqual(candidates, list(range(end, end + cap)))
        never = self.policy(1.01)
        self.assertEqual([c for c in map(never.next_candidates, requests) if c], [])
        self.assertEqual(never.decisions, 1)
        self.assertGreater(never.model_ms, 0.0)

    def test_precomputed_outputs_reproduce_the_online_policy(self):
        D = self.D
        requests = workload(1200, seed=12)
        features = D.step_features([r.timestamp_ms for r in requests],
                                   [r.lba for r in requests],
                                   [r.size_blocks for r in requests])
        outputs = D.decision_probabilities(self.network, features,
                                           D.decision_rows(len(requests)))
        for threshold, gap in ((0.3, None), (0.5, 0.3)):
            online, _ = replay(requests, mode="learned_depth",
                               depth_model=self.policy(threshold, gap=gap))
            pre = D.PrecomputedDepth(*outputs, self.prepared.segments, self.prepared.vocab,
                                     threshold, gap, 256)
            offline, _ = replay(requests, mode="learned_depth", depth_model=pre)
            for field in ("read_hits", "prefetches", "useful_prefetches", "prefetch_ios"):
                self.assertEqual(getattr(online, field), getattr(offline, field), field)

    def test_threshold_selection_never_touches_test_indices(self):
        D = self.D
        import copy
        import numpy as np
        _, _, (c2, n) = self.prepared.split
        chosen, table = D.select_thresholds(self.network, self.prepared, 256,
                                            val_replay=None)
        vcf = D.select_fixed_depth(self.prepared, 256, val_replay=None)
        rel = D.select_relative_depth(self.prepared, 256, val_replay=None)
        poisoned = copy.deepcopy(self.prepared)
        rng = np.random.default_rng(0)
        poisoned.prefix.lba[c2:] = rng.integers(0, 10 ** 9, n - c2)
        poisoned.prefix.size[c2:] = 1
        poisoned.prefix.ts_ms[c2:] = poisoned.prefix.ts_ms[c2 - 1]
        poisoned.labels[c2:] = 1 - poisoned.labels[c2:]
        poisoned.features[c2:] = 0
        def untimed(result):
            # Inference wall time is the only field allowed to differ.
            choice, rows = result
            return choice, [{k: v for k, v in row.items() if k != "inference_ms"}
                            for row in rows]

        self.assertEqual(untimed(D.select_thresholds(self.network, poisoned, 256,
                                                     val_replay=None)),
                         untimed((chosen, table)))
        self.assertEqual(untimed(D.select_fixed_depth(poisoned, 256, val_replay=None)),
                         untimed(vcf))
        self.assertEqual(untimed(D.select_relative_depth(poisoned, 256, val_replay=None)),
                         untimed(rel))
        self.assertEqual(chosen["val_slice"], list(self.prepared.split[1]))

    def test_artifact_carries_thresholds_and_a_bound_record(self):
        D = self.D
        record = read_training_record(self.artifact)
        self.assertEqual(record["model"], "learned_depth")
        self.assertEqual(record["traces"][0]["train"], list(self.prepared.split[0]))
        self.assertEqual(set(record["thresholds"]), {"64", "256"})
        policy = D.load_prefetcher(self.artifact, action_cap=64)
        self.assertEqual(policy.action_cap, 64)
        self.assertEqual(D.load_prefetcher(self.artifact).action_cap, 256)
        with self.assertRaisesRegex(ValueError, "not 128"):
            D.load_prefetcher(self.artifact, action_cap=128)

    def test_stale_or_unselected_artifacts_are_refused(self):
        D = self.D
        import torch
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "stale.pt"
            payload = torch.load(self.artifact, weights_only=True)
            torch.save({**payload, "artifact_version": 0}, stale)
            with self.assertRaisesRegex(ValueError, "retrain"):
                D.load_prefetcher(stale)
            with self.assertRaisesRegex(ValueError, "retrain"):
                replay(workload(100), mode="learned_depth", depth_model=str(stale))
            bare = Path(directory) / "bare.pt"
            torch.save({**payload, "thresholds": {}}, bare)
            with self.assertRaisesRegex(ValueError, "validation-selected"):
                D.load_prefetcher(bare)
            with self.assertRaises(FileNotFoundError):
                D.load_prefetcher(Path(directory) / "missing.pt")

    def test_benchmark_scores_an_artifact_when_given_one(self):
        rows = benchmark_dataset("held_out", workload(600, seed=30),
                                 depth_model_path=str(self.artifact))
        self.assertEqual(rows[-1]["mode"], "learned_depth")
        self.assertEqual(rows[-1]["status"], "ok")

    def test_gpu_sampler_summary_parses_nvidia_smi_rows(self):
        sampler = self.D.GpuSampler()
        sampler.samples = [{"t": 0, "gpus": [{"memory.used": "100", "memory.total": "200",
                                              "utilization.gpu": "50"}]},
                           {"t": 1, "gpus": [{"memory.used": "150", "memory.total": "200",
                                              "utilization.gpu": "90"}]}]
        self.assertIn("memory used 100-150 of 200 MiB, utilisation 50-90%",
                      sampler.summary())
        self.assertIsNone(self.D.GpuSampler().summary())

    @unittest.skipUnless(HAVE_CUDA, "CUDA not available")
    def test_fits_on_the_gpu(self):
        _, info = self.D.fit(self.prepared, epochs=1, batch_size=512, device="cuda",
                             hidden=16, layers=1)
        self.assertEqual(info["device"], "cuda")


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class LSTMTrainingTests(unittest.TestCase):
    def test_vectorised_examples_equal_the_reference_lists(self):
        from adaptive_prefetch import lstm as L
        traces = [transition_trace(42 + i, windows_per_class=2)[0] for i in range(2)]
        reference = L._build_examples(traces)
        inputs, labels = L._example_tensors([L._read_lbas(t) for t in traces])
        import torch
        # The reference stores the float32 values torch.tensor would make.
        expected = torch.tensor([x for x, _ in reference], dtype=torch.float32)
        self.assertTrue(torch.equal(inputs.squeeze(-1), expected))
        self.assertEqual(labels.tolist(), [y for _, y in reference])

    def test_validation_tail_is_held_out_and_threads_are_capped(self):
        from adaptive_prefetch import lstm as L
        import torch
        traces = [transition_trace(42, windows_per_class=2)[0]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "l.pt"
            result = L.train_lstm(path, traces, epochs=3, device="cpu",
                                  record={"source": "test"})
            self.assertLessEqual(torch.get_num_threads(), 4)
            self.assertGreater(result["val_examples"], 0)
            self.assertEqual(len(result["val_loss"]), 3)
            self.assertEqual(read_training_record(path)["model"], "lstm")
            everything = L.train_lstm(Path(directory) / "all.pt", traces, epochs=1,
                                      device="cpu", val_fraction=0)
            self.assertEqual(everything["val_examples"], 0)
            self.assertEqual(everything["examples"],
                             result["examples"] + result["val_examples"] + L.HISTORY + 1)
            self.assertFalse(training_record_path(Path(directory) / "all.pt").exists())

    def test_real_trace_training_records_its_chronological_split(self):
        from adaptive_prefetch import lstm as L
        from adaptive_prefetch.cli import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = write_oracle(root / "msr_t.oracleGeneral", workload(1000))
            save = root / "lstm_t.pt"
            _, out = _quiet(main, ["train-lstm", "--trace", str(trace), "--format",
                                   "oracle", "--device", "cpu", "--epochs", "1",
                                   "--batch-size", "256", "--save", str(save)])
            self.assertIn("[[0, 600], [600, 800], [800, 1000]]", out)
            record = read_training_record(save)
            entry = record["traces"][0]
            self.assertEqual((entry["trace"], entry["train"], entry["val"], entry["test"]),
                             ("msr_t", [0, 600], [600, 800], [800, 1000]))
            self.assertIsNone(scoring_conflict(record, "msr_t", 800, 1000))
            self.assertIsNotNone(scoring_conflict(record, "msr_t", 0, 1000))
            predictor = L.load_predictor(save)
            self.assertIsInstance(predictor, L.LSTMPrefetcher)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TrainDepthCommandTests(unittest.TestCase):
    def test_train_depth_writes_the_artifact_and_its_record(self):
        from adaptive_prefetch.cli import main
        from adaptive_prefetch.depth_model import load_prefetcher
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = write_oracle(root / "msr_t.oracleGeneral", workload(3000))
            _, out = _quiet(main, ["train-depth", "--traces", str(trace), "--device", "cpu",
                                   "--epochs", "1", "--batch-size", "512",
                                   "--val-replay", "300", "--out-dir", str(root / "m")])
            artifact = root / "m" / "msr_t_s0.pt"
            self.assertTrue(artifact.is_file())
            self.assertIn("score it only on requests [2400, 3000] of msr_t", out)
            record = read_training_record(artifact)
            self.assertEqual(record["traces"][0]["test"], [2400, 3000])
            self.assertEqual(record["thresholds"]["64"]["val_slice"], [1800, 2100])
            load_prefetcher(artifact)
            # The replay command refuses the requests the artifact saw.
            _, out = _quiet(main, ["replay", str(trace), "--format", "oracle",
                                   "--depth-model", str(artifact)])
            self.assertIn("learned_depth skipped: artifact was trained on msr_t", out)


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class GpuRunnerTests(unittest.TestCase):
    """repro/gpu_disk.py end to end on a tiny trace, on the CPU."""

    def test_runner_writes_schema_checked_results_and_provenance(self):
        sys.path.insert(0, str(REPO))
        from prefetch_common import schemas
        from prefetch_common.provenance import read_provenance
        from repro import gpu_disk
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "traces" / "msr").mkdir(parents=True)
            write_oracle(root / "traces" / "msr" / "msr_t.oracleGeneral", workload(3000))
            config = gpu_disk.Config(
                traces=("msr_t",), seeds=(0,), n=None, epochs=1, batch_size=512,
                val_replay=300, test_replay=300, lstm_traces=("msr_t",),
                lstm_epochs=1, lstm_batch_size=256, device="cpu", workers=0,
                block_root=root / "traces", results_dir=root / "results" / "gpu",
                work_dir=root / "work", models_dir=root / "models")
            status, _ = _quiet(gpu_disk.run, config, ["test"])
            self.assertEqual(status, 0)
            out = root / "results" / "gpu"
            rows = schemas.validate_file(out / "disk_learned_depth.csv",
                                         "gpu/disk_learned_depth.csv")
            self.assertEqual([(r["seed"], r["action_cap"], r["status"]) for r in rows],
                             [(0, 64, "ok"), (0, 256, "ok")])
            self.assertEqual((rows[0]["test_start"], rows[0]["test_end"]), (2400, 2700))
            baselines = (out / "disk_depth_baselines.csv").read_text()
            self.assertIn("linux_ra", baselines)
            self.assertIn(",ok,", (out / "disk_lstm.csv").read_text())
            self.assertEqual(read_provenance(out)["study"], "gpu")
            # Resumable: a second run recomputes nothing and agrees.
            status, _ = _quiet(gpu_disk.run, config, ["test"])
            self.assertEqual(status, 0)
            self.assertEqual(schemas.validate_file(out / "disk_learned_depth.csv",
                                                   "gpu/disk_learned_depth.csv"), rows)
            # Resuming with a different test slice would mix two runs' files.
            status, log = _quiet(gpu_disk.run, replace(config, test_replay=200), ["test"])
            self.assertEqual(status, 2)
            self.assertIn("test_replay: 300 -> 200", log)


if __name__ == "__main__":
    unittest.main()

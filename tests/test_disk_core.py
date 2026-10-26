"""Disk simulator core: cache hook, device-I/O accounting, linux_ra, sweep fixes.

Everything here is stdlib-only and runs under system python3. The one
real-trace cross-check is gated on ``RUN_SLOW=1``, the data and zstandard.
"""

import importlib.util
import json
import os
import random
import struct
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from adaptive_prefetch import sweep
from adaptive_prefetch.artifacts import load_model, save_model
from adaptive_prefetch.baselines import LinuxReadahead
from adaptive_prefetch.benchmark import MODES, benchmark_dataset, make_model
from adaptive_prefetch.features import (FEATURE_NAMES, WindowContext,
                                        dominant_stride)
from adaptive_prefetch.model import make_classifier
from adaptive_prefetch.simulator import (CacheView, IOCostModel, LatencyModel,
                                         LRUCache, MODES_SUPPORTED, Metrics,
                                         replay, replay_stream)
from adaptive_prefetch.sweep import RANKING_MODES
from adaptive_prefetch.trace import (CLASSES, Request, synthetic_dataset,
                                     synthetic_window, transition_trace)

REPO = Path(__file__).resolve().parents[1]
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HM0 = REPO / "data" / "block-traces" / "msr" / "msr_hm_0.oracleGeneral.zst"

#: Every Metrics counter a replay produces; the parity checks compare all.
COUNTERS = ("read_blocks", "read_hits", "prefetches", "useful_prefetches",
            "policy_switches", "demand_latency_us", "prefetch_cost_us",
            "inference_calls", "observed_precision", "precision_margin",
            "observed_stride", "demand_ios", "prefetch_ios",
            "prefetch_ios_mergeable", "age_table_evictions")


def mixed_trace(n=3000, seed=5):
    """Reads and writes, runs and jumps: exercises every branch of every mode."""
    rng = random.Random(seed)
    out, lba = [], 0
    for i in range(n):
        if rng.random() < 0.6:
            lba += 8
        else:
            lba = rng.randrange(0, 50_000) * 8
        out.append(Request(i * 0.5, lba, rng.choice((1, 8, 16)),
                           "W" if rng.random() < 0.15 else "R"))
    return out


class RecordingLRU(LRUCache):
    """An instrumented cache: logs every operation, changes nothing."""

    def __init__(self, capacity, metrics):
        super().__init__(capacity, metrics)
        self.log = []

    def read(self, block):
        hit = super().read(block)
        self.log.append(("R", block, hit))
        return hit

    def write(self, block):
        super().write(block)
        self.log.append(("W", block, None))

    def prefetch(self, block):
        issued = super().prefetch(block)
        self.log.append(("P", block, issued))
        return issued


class CacheFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lstm_dir = tempfile.TemporaryDirectory()
        cls.lstm_path = None
        cls.depth_path = None
        if HAVE_TORCH:
            import torch
            # A tiny model; the default thread pool would take the whole of
            # a shared host for no speed-up.
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            from adaptive_prefetch.lstm import train_lstm
            path = Path(cls.lstm_dir.name) / "lstm.pt"
            train_lstm(path, traces=[transition_trace(42, windows_per_class=2)[0]],
                       epochs=1)
            cls.lstm_path = str(path)
            # learned_depth (WP9) needs an artifact too; a tiny one on a
            # different trace is enough to pin the parity of both paths.
            from adaptive_prefetch import depth_model
            prepared = depth_model.prepare(
                depth_model.prefix_from_requests(mixed_trace(seed=21)), 256)
            cls.depth_path = str(depth_model.train_and_select(
                prepared, Path(cls.lstm_dir.name) / "depth.pt", epochs=1,
                batch_size=256, device="cpu", hidden=8, layers=1,
                action_caps=(256,), val_replay=200).artifact)

    @classmethod
    def tearDownClass(cls):
        cls.lstm_dir.cleanup()

    def _kwargs(self, mode):
        kwargs = {}
        if mode == "adaptive":
            kwargs["model"] = make_model(42)
        if mode == "lstm":
            kwargs["lstm_model_path"] = self.lstm_path
        if mode == "learned_depth":
            kwargs["depth_model"] = self.depth_path
        return kwargs

    def _modes(self):
        for mode in MODES:
            if mode == "lstm" and self.lstm_path is None:
                continue
            if mode == "learned_depth" and self.depth_path is None:
                continue
            yield mode

    def test_default_cache_factory_is_identity_for_every_mode(self):
        requests = mixed_trace()
        for mode in self._modes():
            with self.subTest(mode=mode):
                implicit, _ = replay(requests, capacity=512, mode=mode,
                                     **self._kwargs(mode))
                explicit, _ = replay(requests, capacity=512, mode=mode,
                                     cache_factory=LRUCache, **self._kwargs(mode))
                caches = []

                def factory(capacity, metrics):
                    caches.append(RecordingLRU(capacity, metrics))
                    return caches[-1]

                recorded, _ = replay(requests, capacity=512, mode=mode,
                                     cache_factory=factory, **self._kwargs(mode))
                streamed = replay_stream(iter([requests[:1000], requests[1000:]]),
                                         capacity=512, mode=mode,
                                         cache_factory=factory,
                                         **self._kwargs(mode))
                for field in COUNTERS:
                    self.assertEqual(getattr(implicit, field),
                                     getattr(explicit, field), field)
                    self.assertEqual(getattr(implicit, field),
                                     getattr(recorded, field), field)
                    self.assertEqual(getattr(implicit, field),
                                     getattr(streamed, field), field)
                # The hook saw exactly the operations the loop performed, and
                # both paths performed the same ones.
                self.assertEqual(len(caches), 2)
                self.assertEqual(caches[0].log, caches[1].log)
                reads = [entry for entry in caches[0].log if entry[0] == "R"]
                self.assertEqual(len(reads), implicit.read_blocks)
                self.assertEqual(sum(hit for _, _, hit in reads), implicit.read_hits)

    def test_every_mode_agrees_across_both_paths_including_inference(self):
        """M13: the streamed path never counted per-request inference calls."""
        requests = mixed_trace(seed=9)
        for mode in self._modes():
            with self.subTest(mode=mode):
                a, _ = replay(requests, capacity=256, mode=mode,
                              **self._kwargs(mode))
                b = replay_stream(iter([requests[i:i + 700]
                                        for i in range(0, len(requests), 700)]),
                                  capacity=256, mode=mode, **self._kwargs(mode))
                for field in COUNTERS:
                    self.assertEqual(getattr(a, field), getattr(b, field),
                                     f"{mode}.{field} differs between paths")
                self.assertEqual(a.device_ios, b.device_ios)
                self.assertEqual(a.blocks_moved, b.blocks_moved)
                if mode in ("stride", "markov", "lstm", "linux_ra", "learned_depth"):
                    self.assertEqual(b.inference_calls, len(requests))
                if mode in ("adaptive", "adaptive_evidence"):
                    self.assertEqual(b.inference_calls, len(requests) // 32)


class DeviceIOTests(unittest.TestCase):
    def test_one_eight_block_miss_is_one_demand_io(self):
        metrics, _ = replay([Request(0.0, 100, 8)], mode="none")
        self.assertEqual(metrics.demand_ios, 1)
        self.assertEqual(metrics.device_ios, 1)
        self.assertEqual(metrics.blocks_moved, 8)

    def test_a_resident_block_splits_the_demand_read(self):
        # Block 104 is resident from the first request (one demand read), so
        # the 8-block read needs two more: 100-103 and 105-107.
        metrics, _ = replay([Request(0.0, 104, 1), Request(1.0, 100, 8)],
                            mode="none")
        self.assertEqual(metrics.demand_ios, 1 + 2)
        self.assertEqual(metrics.blocks_moved, 8)

    def test_a_fully_hit_request_costs_no_device_io(self):
        metrics, _ = replay([Request(0.0, 0, 8), Request(1.0, 0, 8)],
                            mode="none")
        self.assertEqual(metrics.demand_ios, 1)
        self.assertEqual(metrics.read_hits, 8)

    def test_writes_cost_no_demand_io(self):
        metrics, _ = replay([Request(float(i), i * 8, 8, "W") for i in range(50)],
                            mode="deep")
        self.assertEqual(metrics.device_ios, 0)
        self.assertEqual(metrics.blocks_moved, 0)

    def test_readahead_contiguous_with_a_miss_is_mergeable(self):
        # Random 8-block misses: every read misses its last block, and
        # `deep` then prefetches the 8 blocks right after it.
        rng = random.Random(1)
        requests = [Request(float(i), rng.randrange(0, 10 ** 6) * 64, 8)
                    for i in range(200)]
        metrics, _ = replay(requests, mode="deep", read_ahead_depth=8)
        issuing = len(requests) - 32
        self.assertEqual(metrics.prefetch_ios, issuing)
        self.assertEqual(metrics.prefetch_ios_mergeable, issuing)
        self.assertEqual(metrics.device_ios, metrics.demand_ios)
        self.assertEqual(metrics.blocks_moved,
                         metrics.read_blocks - metrics.read_hits + metrics.prefetches)

    def test_readahead_after_a_hit_is_not_mergeable(self):
        # A fully-hit request has no demand read to merge into.
        scan = [Request(float(i), i * 8, 8) for i in range(400)]
        metrics, _ = replay(scan, mode="deep", read_ahead_depth=16)
        self.assertGreater(metrics.prefetch_ios, 0)
        self.assertLess(metrics.prefetch_ios_mergeable, metrics.prefetch_ios)

    def test_a_resident_candidate_splits_the_prefetch_run(self):
        metrics = Metrics()
        cache = LRUCache(64, metrics)
        from adaptive_prefetch.simulator import _issue_prefetches
        cache.read(12)
        _issue_prefetches(cache, metrics, LatencyModel(), None,
                          [10, 11, 12, 13, 14], end=10, tail_missed=True)
        self.assertEqual(metrics.prefetches, 4)
        self.assertEqual(metrics.prefetch_ios, 2)
        self.assertEqual(metrics.prefetch_ios_mergeable, 1)


class IOCostModelTests(unittest.TestCase):
    def _metrics(self, ios, blocks):
        metrics = Metrics()
        metrics.demand_ios = ios
        metrics.read_blocks = blocks
        return metrics

    def test_device_time_is_positioning_plus_transfer(self):
        model = IOCostModel(t_pos_us=5000.0, t_xfer_us_per_block=2.0)
        metrics = self._metrics(10, 400)
        self.assertEqual(model.device_time_us(metrics), 10 * 5000.0 + 400 * 2.0)
        self.assertEqual(model.kappa, 2500.0)

    def test_kappa_star_is_extra_blocks_per_saved_io(self):
        none = self._metrics(100, 1000)
        fewer = self._metrics(60, 1400)
        self.assertEqual(IOCostModel.kappa_star(fewer, none), 10.0)
        # Above kappa* the policy is cheaper, below it dearer.
        for kappa, cheaper in ((20.0, True), (5.0, False)):
            model = IOCostModel(kappa, 1.0)
            self.assertEqual(model.device_time_us(fewer)
                             < model.device_time_us(none), cheaper)
        self.assertEqual(IOCostModel.kappa_star(self._metrics(120, 1400), none),
                         float("inf"))
        self.assertEqual(IOCostModel.kappa_star(self._metrics(90, 900), none), 0.0)

    def test_loads_from_a_device_calibration_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "device_sdb.json"
            path.write_text(json.dumps({"t_pos_us": 5907.0,
                                        "t_xfer_us_per_block": 1.9}))
            model = IOCostModel.from_device_json(path)
            self.assertEqual(model, IOCostModel(5907.0, 1.9))
            path.write_text(json.dumps({"t_pos_us": 5907.0}))
            with self.assertRaisesRegex(ValueError, "t_xfer_us_per_block"):
                IOCostModel.from_device_json(path)

    def test_negative_times_rejected(self):
        with self.assertRaises(ValueError):
            IOCostModel(-1.0, 1.0)

    def test_legacy_latency_model_is_untouched(self):
        self.assertAlmostEqual(LatencyModel(1.0, 100.0, 100.0).break_even_precision(),
                               100 / 99)


class LinuxReadaheadTests(unittest.TestCase):
    def test_sequential_scan_ramps_to_full_windows(self):
        scan = [Request(i * 0.5, i * 8, 8) for i in range(20_000)]
        metrics, _ = replay(scan, mode="linux_ra")
        self.assertGreaterEqual(metrics.hit_ratio, 0.998)
        self.assertLess(metrics.prefetch_ios, 1000)
        # The one policy that cuts device I/Os, not just misses.
        none, _ = replay(scan, mode="none")
        self.assertLess(metrics.device_ios, none.device_ios / 10)

    def test_random_input_issues_nothing(self):
        rng = random.Random(3)
        requests = [Request(i * 0.5, rng.randrange(0, 10 ** 8) * 8, 8)
                    for i in range(5000)]
        metrics, _ = replay(requests, mode="linux_ra")
        self.assertEqual(metrics.prefetches, 0)

    def test_first_window_never_prefetches(self):
        scan = [Request(i * 0.5, i * 8, 8) for i in range(32)]
        metrics, _ = replay(scan, mode="linux_ra")
        self.assertEqual(metrics.prefetches, 0)
        streamed = replay_stream(iter([scan]), mode="linux_ra")
        self.assertEqual(streamed.prefetches, 0)

    def test_window_ceiling_is_respected(self):
        scan = [Request(i * 0.5, i * 8, 8) for i in range(4000)]
        for max_blocks in (64, 1024):
            caches = []

            def factory(capacity, metrics):
                caches.append(RecordingLRU(capacity, metrics))
                return caches[-1]

            replay(scan, capacity=8192, mode="linux_ra",
                   linux_ra_max_blocks=max_blocks, cache_factory=factory)
            # A run is counted within one request's issue: any demand read
            # in between ends it.
            run, longest, last = 0, 0, None
            for op, block, issued in caches[0].log:
                if op != "P":
                    last = None
                    continue
                if not issued:
                    continue
                run = run + 1 if last is not None and block == last + 1 else 1
                longest = max(longest, run)
                last = block
            # A scan ramps all the way up, so the ceiling is reached, not
            # just respected (a run of zero would pass a bare <=).
            self.assertEqual(longest, max_blocks)
        with self.assertRaises(ValueError):
            replay(scan, mode="linux_ra", linux_ra_max_blocks=0)

    def test_is_causal(self):
        """Replacing the future leaves every earlier outcome unchanged."""
        rng = random.Random(4)
        prefix = mixed_trace(1500, seed=4)
        futures = [prefix + [Request(1500 + i * 0.5, rng.randrange(0, 10 ** 6) * 8, 8)
                             for i in range(1500)] for _ in range(2)]
        logs = []
        for requests in [prefix] + futures:
            caches = []

            def factory(capacity, metrics):
                caches.append(RecordingLRU(capacity, metrics))
                return caches[-1]

            replay(requests, capacity=256, mode="linux_ra", cache_factory=factory)
            logs.append(caches[0].log)
        n = len(logs[0])
        self.assertEqual(logs[1][:n], logs[0])
        self.assertEqual(logs[2][:n], logs[0])

    def test_only_observes_a_read_only_view(self):
        view = CacheView(LRUCache(8, Metrics()))
        for name in ("read", "write", "prefetch", "_insert"):
            self.assertFalse(hasattr(view, name), name)
        with self.assertRaises(AttributeError):
            view.blocks = {}

    def test_writes_leave_the_state_alone(self):
        predictor = LinuxReadahead()
        view = CacheView(LRUCache(64, Metrics()))
        self.assertEqual(predictor.observe(Request(0.0, 0, 8, "W"), [], view), [])
        self.assertIsNone(predictor.prev_end)

    def test_registered_in_all_four_places(self):
        self.assertIn("linux_ra", MODES_SUPPORTED)
        self.assertIn("linux_ra", RANKING_MODES)
        self.assertIn("linux_ra", MODES)
        from adaptive_prefetch.simulator import GATED_MODES, _make_predictor
        self.assertNotIn("linux_ra", GATED_MODES)
        self.assertIsInstance(_make_predictor("linux_ra", None), LinuxReadahead)


class ModeOrderTests(unittest.TestCase):
    def test_linux_ra_is_appended_after_deep(self):
        # The exact length is not pinned: a later work package appends
        # `learned_depth`. The order of the first thirteen is.
        self.assertEqual(MODES[:13], (
            "none", "sequential", "strided", "stride", "markov", "lstm",
            "adaptive", "adaptive_evidence", "guard", "depth_adaptive",
            "correlate", "deep", "linux_ra"))
        self.assertEqual(MODES[5], "lstm")
        rows = benchmark_dataset("t", transition_trace(3, windows_per_class=2)[0])
        self.assertEqual(rows[5]["mode"], "lstm")
        self.assertTrue(str(rows[5]["status"]).startswith("skipped"))
        self.assertEqual(rows[12]["mode"], "linux_ra")
        self.assertEqual(rows[12]["status"], "ok")


class BoundedContextTests(unittest.TestCase):
    def test_identical_when_the_bound_exceeds_the_footprint(self):
        requests, _ = transition_trace(7, windows_per_class=8)
        footprint = len({block for r in requests
                         for block in range(r.lba, r.lba + r.size_blocks)})
        windows = [requests[i:i + 32] for i in range(0, len(requests) - 31, 32)]
        bounded, unbounded = WindowContext(footprint), WindowContext(None)
        for window in windows:
            stride = dominant_stride(window)
            self.assertEqual(bounded.observe(window, stride),
                             unbounded.observe(window, stride))
        self.assertEqual(bounded.evictions, 0)
        model = make_model(42)
        a, pa = replay(requests, mode="adaptive", model=model,
                       age_table_bound=footprint)
        b, pb = replay(requests, mode="adaptive", model=model,
                       age_table_bound=None)
        self.assertEqual(pa, pb)
        self.assertEqual(a.hit_ratio, b.hit_ratio)
        self.assertEqual(a.age_table_evictions, 0)

    def test_table_size_never_exceeds_the_bound(self):
        bound = 20_000
        context = WindowContext(bound)
        rng = random.Random(8)
        window = []
        for i in range(100_000):
            window.append(Request(i * 0.1, rng.randrange(0, 10 ** 9), 8))
            if len(window) == 32:
                context.observe(window, None)
                self.assertLessEqual(len(context.last_seen), bound)
                window = []
        self.assertEqual(len(context.last_seen), bound)
        self.assertGreater(context.evictions, 0)

    def test_an_evicted_block_reads_as_never_seen(self):
        old = [Request(float(i), i, 1) for i in range(8)]                # 8 blocks
        new = [Request(8.0 + i, 100 + 2 * i, 2) for i in range(8)]       # 16 more
        bounded, unbounded = WindowContext(16), WindowContext(None)
        for context in (bounded, unbounded):
            context.observe(old, None)
            context.observe(new, None)
        # reuse_ratio (index 8): every block of `old` was pushed out of the
        # 16-entry table, so a revisit reads as a first touch.
        self.assertEqual(bounded.evictions, 8)
        self.assertEqual(bounded.result(old, None)[8], 0.0)
        self.assertEqual(unbounded.result(old, None)[8], 1.0)
        self.assertEqual(bounded.result(new, None)[8], 1.0)

    def test_bound_must_be_positive(self):
        with self.assertRaises(ValueError):
            WindowContext(0)
        with self.assertRaises(ValueError):
            replay(transition_trace(3, 1)[0], mode="none", age_table_bound=0)


class SweepFixTests(unittest.TestCase):
    def _revised(self, directory, lines=3000, name="t.revised"):
        rng = random.Random(1)
        path = Path(directory) / name
        rows, lba = [], 0
        for i in range(lines):
            lba = lba + 8 if rng.random() < 0.7 else rng.randrange(0, 10 ** 5) * 8
            op = "WS" if i % 7 == 0 else "RS"
            rows.append(f"{i * 0.001:.6f} {op} {lba} 8 rand "
                        f"{rng.random():.4f} 0.1 0.9\n")
        path.write_text("".join(rows), encoding="utf-8")
        return path

    def _sweep_modes(self):
        spec = importlib.util.spec_from_file_location(
            "sweep_modes_under_test", REPO / "sweep_modes.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_sweep_modes_full_and_default_modes_exit_zero(self):
        """M3 crashed every --full run; M2 aborted a default run with no LSTM."""
        module = self._sweep_modes()
        with tempfile.TemporaryDirectory() as directory:
            traces = Path(directory) / "traces"
            traces.mkdir()
            self._revised(traces)
            for extra in (["--full", "--modes", "none", "deep"], []):
                out = Path(directory) / ("full" if extra else "default")
                with mock.patch.object(sweep, "LSTM_ARTIFACT",
                                       Path(directory) / "absent.pt"), \
                        mock.patch("sys.stdout"):
                    status = module.main(["--trace-dir", str(traces),
                                          "--output-dir", str(out), *extra])
                self.assertEqual(status, 0)
                self.assertTrue((out / "msrc_sweep.csv").is_file())
                self.assertTrue((out / "msrc_sweep.md").is_file())
            md = (Path(directory) / "full" / "msrc_sweep.md").read_text()
            self.assertIn("requests per trace: all", md)
            text = (Path(directory) / "default" / "msrc_sweep.csv").read_text()
            self.assertIn("lstm", text)
            self.assertIn("skipped: no trained artifact", text)
            self.assertIn("observed_precision", text.splitlines()[0])

    def test_sweep_modes_with_every_mode_skipped_exits_zero(self):
        """Nothing to rank must not crash the summary after the rows exist."""
        module = self._sweep_modes()
        with tempfile.TemporaryDirectory() as directory:
            traces = Path(directory) / "traces"
            traces.mkdir()
            self._revised(traces, lines=400)
            out = Path(directory) / "out"
            with mock.patch.object(sweep, "LSTM_ARTIFACT",
                                   Path(directory) / "absent.pt"), \
                    mock.patch("sys.stdout"):
                status = module.main(["--trace-dir", str(traces),
                                      "--output-dir", str(out),
                                      "--modes", "lstm"])
            self.assertEqual(status, 0)
            self.assertIn("skipped: no trained artifact",
                          (out / "msrc_sweep.csv").read_text())

    def test_missing_lstm_artifact_is_a_skipped_row(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(sweep, "LSTM_ARTIFACT",
                                  Path(directory) / "absent.pt"):
            path = self._revised(directory, lines=400)
            rows, _ = sweep.sweep_collection(
                traces=[path], modes=("none", "lstm"),
                latency_model=LatencyModel())
            stale, _ = sweep.sweep_collection(
                traces=[path], modes=("lstm",), latency_model=LatencyModel(),
                lstm_model_path=str(Path(directory) / "absent.pt"))
        none, lstm = rows
        self.assertEqual(none["status"], "ok")
        self.assertEqual(lstm["status"], "skipped: no trained artifact")
        self.assertIsNone(lstm["hit_ratio"])
        self.assertIsNone(lstm["read_blocks"])
        self.assertTrue(stale[0]["status"].startswith("skipped:"))
        # Skipped rows must not crash or win any aggregate.
        self.assertEqual(sweep.per_trace_winners(rows), {"none": 1})
        summary = {r["mode"]: r for r in sweep.aggregate_by_mode(rows)}
        self.assertEqual(summary["lstm"]["skipped"], 1)
        self.assertEqual(summary["lstm"]["scored"], 0)

    def test_rows_record_depth_and_linux_ra_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._revised(directory, lines=2000)
            rows = {depth: sweep.sweep_trace(path, "deep", read_ahead_depth=depth,
                                             max_blocks=64,
                                             latency_model=LatencyModel())
                    for depth in (2, 32)}
            ra = sweep.sweep_trace(path, "linux_ra", max_blocks=64,
                                   latency_model=LatencyModel())
        self.assertEqual(rows[2]["read_ahead_depth"], 2)
        self.assertEqual(rows[32]["max_blocks"], 64)
        self.assertGreater(rows[32]["total_prefetches"], rows[2]["total_prefetches"])
        self.assertEqual(ra["max_blocks"], 64)
        for row in (rows[2], ra):
            self.assertEqual(set(row), set(sweep.ROW_COLUMNS))
            self.assertEqual(row["requests"], 2000)
            self.assertEqual(row["blocks"], 2000 * 8)
            self.assertEqual(row["format"], "revised")
            self.assertTrue(row["ops_known"])
            self.assertAlmostEqual(row["break_even"], 50 / 95)
            self.assertAlmostEqual(row["hours_covered"], 1.999 / 3600)

    def test_oracle_traces_need_an_explicit_latency_model(self):
        fake = Path("data/none/msr_x_0.oracleGeneral.zst")
        self.assertEqual(sweep.trace_format(fake), "oracle")
        self.assertEqual(sweep.trace_name(fake), "msr_x_0")
        with self.assertRaisesRegex(ValueError, "latency_model"):
            sweep.sweep_trace(fake, "none")
        with self.assertRaisesRegex(ValueError, "latency_model"):
            sweep.sweep_collection(traces=[fake], modes=("none",))

    def test_capped_chunks_routes_by_format(self):
        seen = []

        def fake(path, fmt, chunk):
            seen.append(fmt)
            yield [Request(0.0, 0, 8)] * 5

        with mock.patch.object(sweep, "iter_csv_chunks", fake):
            list(sweep.capped_chunks(Path("a.revised"), None, 100))
            list(sweep.capped_chunks(Path("a.oracleGeneral.zst"), 3, 100))
            list(sweep.capped_chunks(Path("a.dat"), None, 100, format="msr"))
        self.assertEqual(seen, ["revised", "oracle", "msr"])

    def test_discover_traces_finds_both_kinds_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "msr").mkdir()
            (root / "a.revised").write_text("")
            (root / "msr" / "msr_a.oracleGeneral.zst").write_bytes(b"")
            (root / "notes.txt").write_text("")
            found = [p.name for p in sweep.discover_traces(root)]
            only = sweep.discover_traces(root, patterns=("*.revised",))
        self.assertEqual(sorted(found), ["a.revised", "msr_a.oracleGeneral.zst"])
        self.assertEqual([p.name for p in only], ["a.revised"])

    def test_both_weightings_are_reported(self):
        rows = [
            {"trace": "a", "mode": "x", "hit_ratio": 0.9, "precision": 1.0,
             "mean_us": 10.0, "unused": 0, "total_prefetches": 10,
             "read_blocks": 10},
            {"trace": "b", "mode": "x", "hit_ratio": 0.1, "precision": 0.5,
             "mean_us": 90.0, "unused": 45, "total_prefetches": 90,
             "read_blocks": 990},
        ]
        row = sweep.aggregate_by_mode(rows)[0]
        self.assertAlmostEqual(row["mean_hit"], 0.5)
        self.assertAlmostEqual(row["weighted_hit"], (9 + 99) / 1000)
        self.assertAlmostEqual(row["weighted_prec"], 55 / 100)

    def test_degenerate_winners_are_labelled(self):
        rows = [
            {"trace": "a", "mode": "none", "hit_ratio": 0.5, "precision": 0.0,
             "mean_us": 50.0, "unused": 0, "total_prefetches": 0,
             "read_blocks": 100},
            {"trace": "a", "mode": "deep", "hit_ratio": 0.9, "precision": 0.8,
             "mean_us": 60.0, "unused": 10, "total_prefetches": 50,
             "read_blocks": 100},
        ]
        champ = sweep.winners(rows)
        notes = sweep.winner_caveats(rows, champ, 1.0101)
        self.assertIn("degenerate", notes["cost"])
        self.assertIn("degenerate", notes["waste"])
        self.assertNotIn("cost", sweep.winner_caveats(rows, champ, 0.5))

    def test_parse_estimate_uses_the_measured_rate(self):
        module = self._sweep_modes()
        self.assertAlmostEqual(module.parse_minutes(1.0), 1000 / 2.1 / 60)
        self.assertGreater(module.parse_minutes(1.0), 7.9)


class ArtifactAndModelTests(unittest.TestCase):
    def _qda(self, shrink):
        model = make_classifier("qda", len(FEATURE_NAMES), shrink=shrink)
        context = WindowContext()
        for window, label in synthetic_dataset(10, 3):
            model.update(context.observe(window, dominant_stride(window)), label)
        return model

    def test_qda_shrink_round_trips(self):
        model = self._qda(0.3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            save_model(path, model, {})
            self.assertEqual(json.loads(path.read_text())["shrink"], 0.3)
            loaded, _ = load_model(path)
            self.assertEqual(loaded.shrink, 0.3)
            load_model(path, classifier="qda", shrink=0.3)
            with self.assertRaisesRegex(ValueError, "shrink"):
                load_model(path, shrink=0.05)
            with self.assertRaisesRegex(ValueError, "classifier"):
                load_model(path, classifier="gnb")
            with self.assertRaisesRegex(ValueError, "decay_factor"):
                load_model(path, decay_factor=0.9)
            payload = json.loads(path.read_text())
            del payload["shrink"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "regenerate"):
                load_model(path)

    def test_committed_gnb_artifact_still_loads(self):
        model, _ = load_model(REPO / "models" / "msr_sample_gnb.json",
                              classifier="gnb")
        self.assertEqual(model.n_features, len(FEATURE_NAMES))

    def test_measured_validates_prefetch_multiple_without_data(self):
        with self.assertRaises(ValueError):
            LatencyModel.measured(prefetch_multiple=0)
        with self.assertRaises(ValueError):
            LatencyModel.measured(trace=[], prefetch_multiple=-1.0)

    def test_measured_zero_service_time_warns(self):
        zero = [Request(float(i), i * 8, 8, service_ms=0.0) for i in range(10)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LatencyModel.measured(trace=zero)
        self.assertEqual(model.break_even_precision(), float("inf"))
        self.assertTrue(any("infinite" in str(w.message) for w in caught))

    def test_interleaved_training_balances_priors_at_equal_accuracy(self):
        """M16 is opt-in: default priors stay skewed, accuracy is unchanged."""
        held_out = synthetic_dataset(40, 100042)
        for interleave, balanced in ((False, False), (True, True)):
            model = make_model(42, 100, interleave=interleave)
            total = sum(model.count.values())
            priors = [model.count[label] / total for label in CLASSES]
            self.assertEqual(all(abs(p - 0.25) <= 0.05 for p in priors), balanced)
            context = WindowContext()
            correct = sum(model.predict(context.observe(w, dominant_stride(w)))[0] == y
                          for w, y in held_out)
            self.assertEqual(correct / len(held_out), 0.96875)


def _oracle_requests(path, n):
    """First ``n`` records, built as the validated prototype built them."""
    import zstandard
    with open(path, "rb") as handle, \
            zstandard.ZstdDecompressor().stream_reader(handle) as reader:
        data = reader.read(n * 24)
    records = list(struct.iter_unpack("<IQIq", data[: len(data) // 24 * 24]))
    base = min(obj // 512 for _, obj, _, _ in records)
    return [Request(ts * 1000.0, obj // 512 - base, max(1, (size + 511) // 512))
            for ts, obj, size, _ in records]


@unittest.skipUnless(os.environ.get("RUN_SLOW") == "1"
                     and HM0.is_file()
                     and importlib.util.find_spec("zstandard"),
                     "slow real-trace check: set RUN_SLOW=1 (needs data and zstandard)")
class PrototypeCrossCheckTests(unittest.TestCase):
    """msr_hm_0, first 250k requests, 2048-block cache, against sweep2.jsonl."""

    @classmethod
    def setUpClass(cls):
        cls.requests = _oracle_requests(HM0, 250_000)

    def test_deep_depth_8(self):
        metrics, _ = replay(self.requests, mode="deep", read_ahead_depth=8)
        self.assertEqual(round(metrics.hit_ratio, 6), 0.357231)
        self.assertEqual(metrics.prefetches, 901487)

    def test_linux_ra_matches_the_prototype(self):
        metrics, _ = replay(self.requests, mode="linux_ra")
        self.assertEqual(metrics.prefetches, 996191)
        self.assertEqual(metrics.useful_prefetches, 718218)
        self.assertEqual(metrics.prefetch_ios, 12565)
        self.assertEqual(metrics.prefetch_ios_mergeable, 8679)
        streamed = replay_stream(iter([self.requests[i:i + 50_000]
                                       for i in range(0, 250_000, 50_000)]),
                                 mode="linux_ra")
        for field in COUNTERS:
            self.assertEqual(getattr(metrics, field), getattr(streamed, field))


if __name__ == "__main__":
    unittest.main()

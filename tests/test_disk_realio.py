"""Real Linux prefetch testbed (adaptive_prefetch.realio).

Stdlib only. The device tests run on a small temporary file on whichever
filesystem holds this test directory, and skip when the platform is not
Linux or ``O_DIRECT`` is refused there (``EINVAL`` on tmpfs, for example).
They check mechanics -- parity with the simulator, late-hit detection, hint
chunking -- never device performance, which is what E3 measures.
"""

import contextlib
import io
import json
import mmap
import os
import random
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from adaptive_prefetch import realio
from adaptive_prefetch.benchmark import MODES, make_model
from adaptive_prefetch.realio import backend_pagecache, decisions, layout, report
from adaptive_prefetch.realio.calibrate import derive
from adaptive_prefetch.realio.clock import gaps_us, parse_think
from adaptive_prefetch.realio.decisions import (CacheDivergence, PlanStep,
                                                RecordingLRU, parse_mode,
                                                record_plan, verify_plan)
from adaptive_prefetch.realio.layout import (REGION_BLOCKS, RegionAliasError,
                                             RegionMapper)
from adaptive_prefetch.simulator import IOCostModel, LatencyModel, replay
from adaptive_prefetch.trace import Request
from prefetch_common import schemas

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
SUPPORTED, WHY = realio.supported()
COUNTERS = ("read_blocks", "read_hits", "prefetches", "useful_prefetches",
            "demand_ios", "prefetch_ios", "prefetch_ios_mergeable",
            "policy_switches", "observed_precision")
RECORD = struct.Struct("<IQIq")


def mixed_trace(n=2500, seed=7):
    """Runs, jumps, reads and writes: every branch of every mode."""
    rng = random.Random(seed)
    out, lba = [], 0
    for i in range(n):
        lba = lba + 8 if rng.random() < 0.6 else rng.randrange(0, 40_000) * 8
        out.append(Request(i * 0.5, lba, rng.choice((1, 8, 16)),
                           "W" if rng.random() < 0.1 else "R"))
    return out


def sequential_trace(n, size=64, start=0):
    return [Request(float(i), start + i * size, size) for i in range(n)]


def odirect_dir():
    """A temp dir next to the tests where O_DIRECT works, or a skip reason."""
    if not SUPPORTED:
        return None, WHY
    tmp = tempfile.TemporaryDirectory(dir=TESTS, prefix=".realio-")
    probe = Path(tmp.name) / "probe"
    try:
        layout.make_backing_file(probe, 1 / 1024)        # 1 MiB
        fd = os.open(probe, os.O_RDONLY | os.O_DIRECT)
        buf = mmap.mmap(-1, 4096)
        try:
            os.preadv(fd, [buf], 0)
        finally:
            os.close(fd)
            buf.close()
    except OSError as error:
        tmp.cleanup()
        return None, f"O_DIRECT unavailable on {TESTS}: {error}"
    probe.unlink()
    return tmp, ""


class PlanTests(unittest.TestCase):
    """The recorded plan is the simulator's decision sequence, for every mode."""

    def _kwargs(self, mode):
        return {"model": make_model(42)} if mode == "adaptive" else {}

    def test_recording_lru_plan_identity_for_all_modes(self):
        requests = mixed_trace()
        for mode in MODES:
            if mode == "lstm":
                continue   # needs a trained artifact; covered by WP3's cache-hook tests
            with self.subTest(mode=mode):
                expected, _ = replay(requests, capacity=512, mode=mode,
                                     latency_model=LatencyModel(), **self._kwargs(mode))
                metrics, plan = record_plan([requests[:1000], requests[1000:]], mode,
                                            capacity=512, latency_model=LatencyModel(),
                                            **self._kwargs(mode))
                for name in COUNTERS:
                    self.assertEqual(getattr(metrics, name), getattr(expected, name), name)
                self.assertEqual(len(plan), len(requests))
                shadow = verify_plan(requests, plan, 512)
                for name in ("read_blocks", "read_hits", "prefetches", "useful_prefetches"):
                    self.assertEqual(shadow[name], getattr(expected, name), name)
                planned = sum(n for step in plan for _, n in step.prefetch_runs)
                self.assertEqual(planned, expected.prefetches)

    def test_deep_label_sets_the_depth(self):
        self.assertEqual(parse_mode("deep@64"), ("deep", {"read_ahead_depth": 64}))
        self.assertEqual(parse_mode("linux_ra@128"), ("linux_ra", {"linux_ra_max_blocks": 128}))
        self.assertEqual(parse_mode("guard"), ("guard", {}))
        for bad in ("kernel_ra", "guard@2", "nope"):
            with self.assertRaises(ValueError):
                parse_mode(bad)
        requests = sequential_trace(200, size=8)
        deep8, _ = record_plan([requests], "deep", capacity=4096)
        deep64, _ = record_plan([requests], "deep@64", capacity=4096)
        self.assertGreater(deep64.prefetches, deep8.prefetches)

    def test_tampered_plan_raises_cache_divergence(self):
        requests = mixed_trace(600)
        _, plan = record_plan([requests], "deep", capacity=256)
        index = next(i for i, (r, s) in enumerate(zip(requests, plan))
                     if r.operation == "R" and s.hit_flags)
        flags = bytearray(plan[index].hit_flags)
        flags[0] ^= 1
        tampered = list(plan)
        tampered[index] = PlanStep(bytes(flags), plan[index].prefetch_runs)
        with self.assertRaises(CacheDivergence):
            verify_plan(requests, tampered, 256)

    def test_recording_lru_logs_without_changing_behaviour(self):
        metrics_a, metrics_b = decisions.Metrics(), decisions.Metrics()
        plain, recorded = decisions.LRUCache(4, metrics_a), RecordingLRU(4, metrics_b)
        for cache in (plain, recorded):
            for block in (1, 2, 3, 1, 9, 10, 2):
                cache.read(block)
            cache.prefetch(11)
            cache.prefetch(11)          # resident: no-op, not logged
            cache.write(12)
        self.assertEqual(list(plain.blocks.items()), list(recorded.blocks.items()))
        self.assertEqual(sum(op == decisions.PREFETCH for op, _, _ in recorded.log), 1)


class LayoutTests(unittest.TestCase):
    def test_region_mapper_packs_in_first_touch_order(self):
        mapper = RegionMapper(4 * layout.REGION_BYTES)
        requests = [Request(0, 10 * REGION_BLOCKS + 5, 8), Request(1, 3 * REGION_BLOCKS, 8)]
        mapper = RegionMapper.for_plan(4 * layout.REGION_BYTES, requests)
        self.assertEqual(mapper.offset(10 * REGION_BLOCKS + 5), 5 * 512)
        self.assertEqual(mapper.offset(3 * REGION_BLOCKS), layout.REGION_BYTES)
        self.assertEqual(mapper.demand_regions, 2)
        # A run crossing a region boundary splits, and re-merges only when the
        # two regions landed in consecutive slots.
        runs = mapper.range_runs(10 * REGION_BLOCKS + REGION_BLOCKS - 2, 4)
        self.assertEqual(len(runs), 2)
        self.assertEqual(sum(n for _, n, _ in runs), 4 * 512)

    def test_wrap_counter_and_refusal(self):
        requests = [Request(i, i * REGION_BLOCKS * 7, 8) for i in range(5)]
        mapper = RegionMapper.for_plan(3 * layout.REGION_BYTES, requests)
        self.assertEqual(mapper.wrapped, 2)
        with self.assertRaises(RegionAliasError):
            mapper.check()
        allowed = RegionMapper.for_plan(3 * layout.REGION_BYTES, requests, allow_alias=True)
        allowed.check()
        self.assertEqual(allowed.wrapped, 2)
        reserved = RegionMapper.for_plan(6 * layout.REGION_BYTES, requests,
                                         reserved_tail=2 * layout.REGION_BYTES)
        self.assertEqual((reserved.nslots, reserved.wrapped), (4, 1))

    def test_prefetch_only_regions_come_after_demand_regions(self):
        requests = [Request(0, 0, 8), Request(1, 50 * REGION_BLOCKS, 8)]
        plan = [PlanStep(b"\0" * 8, ((20 * REGION_BLOCKS, 4),)), PlanStep(b"\0" * 8, ())]
        mapper = RegionMapper.for_plan(8 * layout.REGION_BYTES, requests, plan)
        self.assertEqual(mapper.demand_regions, 2)
        self.assertEqual(mapper.offset(50 * REGION_BLOCKS), layout.REGION_BYTES)
        self.assertEqual(mapper.offset(20 * REGION_BLOCKS), 2 * layout.REGION_BYTES)


class ClockTests(unittest.TestCase):
    def test_think_models(self):
        requests = [Request(t, i * 8, 8) for i, t in enumerate((0, 0, 0, 0, 1000, 3000))]
        self.assertEqual(list(gaps_us(requests, "closed")), [0.0] * 6)
        self.assertEqual(list(gaps_us(requests, "fixed:250")), [0.0] + [250.0] * 5)
        spread = list(gaps_us(requests, "spread:1:5000"))
        # Every gap (1/4 s inside tick 0, then 1 s and 2 s) hits the 5 ms cap.
        self.assertEqual(spread, [0.0] + [5000.0] * 5)
        uncapped = list(gaps_us(requests, "spread:0.001:1e9"))
        self.assertAlmostEqual(uncapped[1], 250.0)
        self.assertAlmostEqual(uncapped[4], 1000.0)
        self.assertAlmostEqual(uncapped[5], 2000.0)
        for bad in ("open", "fixed", "spread:1", "fixed:-1"):
            with self.assertRaises(ValueError):
                parse_think(bad)


class CalibrationTests(unittest.TestCase):
    FIXTURE = {"seq1m_MBps": 256.0, "rand4k": {"mean": 6016.0, "p50": 5900.0}}

    def test_calibration_arithmetic(self):
        constants = derive(self.FIXTURE)
        self.assertAlmostEqual(constants["t_xfer_us_per_block"], 2.0)
        self.assertAlmostEqual(constants["t_pos_us"], 6000.0)
        self.assertAlmostEqual(constants["kappa"], 3000.0)
        self.assertEqual(derive({"seq1m_MBps": 512.0, "rand4k": {"mean": 1.0}})["t_pos_us"], 0.0)
        with self.assertRaises(ValueError):
            derive({"seq1m_MBps": 0.0, "rand4k": {"mean": 1.0}})

    def test_device_json_round_trips_into_iocostmodel(self):
        row = {"device": "fixture", "path": "/x", "model": None, "rotational": True,
               "read_ahead_kb": 128, "max_sectors_kb": 256, "rand4k_p50_us": 5900.0,
               "rand4k_mean_us": 6016.0, "seq4k_p50_us": 31.0, "seq64k_p50_us": 235.5,
               "seq1m_MBps": 256.0, **derive(self.FIXTURE), "samples": 10,
               "foreign_io_frac": 0.0, "load_avg": [1.0, 1.0, 1.0],
               "created_utc": "2026-01-01T00:00:00+00:00", "extra": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "device_fixture.json"
            schemas.write_rows("realio/device_fixture.json", [row], path)
            model = IOCostModel.from_device_json(path)
        self.assertAlmostEqual(model.kappa, 3000.0)


class ParityArithmeticTests(unittest.TestCase):
    def _run(self, mode, **kw):
        base = {"device": "d", "backend": "odirect", "think": "closed", "trace": "t",
                "requests": 100, "repeat": 1, "mode": mode, "read_requests": 100,
                "full_hits": 20, "read_blocks": 1000, "missed_blocks": 800,
                "prefetches": 0, "lat_mean_us": 820.0, "hit_lat_mean_us": 100.0,
                "miss_lat_mean_us": 1000.0}
        base.update(kw)
        return base

    def test_prediction_and_k_eff(self):
        none = self._run("none")
        deep = self._run("deep", full_hits=50, missed_blocks=500, prefetches=400,
                         lat_mean_us=700.0)
        runs = [none, deep]
        report.attach_parity(runs)
        # pred = 0.5*100 + 0.5*1000; none predicts itself exactly.
        self.assertAlmostEqual(deep["pred_mean_us"], 550.0)
        self.assertAlmostEqual(deep["pred_err"], (550 - 700) / 700)
        self.assertAlmostEqual(none["pred_err"], 0.0)
        # m = 1000*80/800 = 100 us/block; demand = (500*100 + 500*1)/100 = 505.
        self.assertAlmostEqual(deep["k_eff"], (700 - 505) * 100 / (400 * 100))
        self.assertIsNone(none["k_eff"])
        lonely = self._run("deep", repeat=2)
        report.attach_parity([lonely])
        self.assertIsNone(lonely["pred_mean_us"])

    def test_spearman_and_interval(self):
        self.assertAlmostEqual(report.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(report.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(report.spearman([1, 2], [1, 2]))
        self.assertIsNone(report.spearman([1, 1, 1], [1, 2, 3]))
        mean, lo, hi = report.t_interval([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(hi - mean, 4.303 * 1.0 / 3 ** 0.5, places=3)
        self.assertTrue(all(map(lambda v: v != v, report.t_interval([5.0])[1:])))


@unittest.skipUnless(SUPPORTED, WHY)
class CgroupTests(unittest.TestCase):
    def setUp(self):
        backend_pagecache._SYSTEMD_RUN_PROBE = None

    def tearDown(self):
        backend_pagecache._SYSTEMD_RUN_PROBE = None

    def test_reexec_skips_cleanly_without_systemd_run(self):
        with mock.patch.object(backend_pagecache.shutil, "which", return_value=None):
            ok, reason = backend_pagecache.systemd_run_available()
            self.assertFalse(ok)
            self.assertIn("not installed", reason)
            with self.assertRaises(backend_pagecache.CgroupUnavailable):
                backend_pagecache.scope_command(["true"], 65536)

    def test_runner_drops_pagecache_when_unavailable(self):
        from adaptive_prefetch.realio import runner
        with tempfile.TemporaryDirectory() as tmp:
            backing = Path(tmp) / "backing.bin"
            with open(backing, "wb") as handle:
                handle.write(os.urandom(layout.REGION_BYTES))
            args = runner.build_parser().parse_args(
                ["run", "--trace", "x", "--device-dir", tmp, "--modes", "none,kernel_ra",
                 "--backend", "odirect,pagecache", "--results-dir", tmp])
            out = io.StringIO()
            with mock.patch.object(backend_pagecache, "systemd_run_available",
                                   return_value=(False, "systemd-run is not installed")), \
                    contextlib.redirect_stdout(out):
                jobs = runner.plan_jobs(args, ["x"], ["none", "kernel_ra"], ["closed"])
        self.assertIn("skipped: pagecache backend", out.getvalue())
        self.assertEqual([(j["backend"], j["mode"]) for j in jobs], [("odirect", "none")])

    def test_budget_refuses_an_unlimited_cgroup(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)
            (fake / "memory.max").write_text("max\n")
            with self.assertRaises(backend_pagecache.CgroupUnavailable):
                backend_pagecache.CgroupBudget(fake)
            (fake / "memory.max").write_text("268435456\n")
            (fake / "memory.current").write_text("10485760\n")
            (fake / "memory.stat").write_text("anon 7340032\nfile 3145728\n")
            budget = backend_pagecache.CgroupBudget(fake)
            limit = budget.tighten(32 << 20)
            # File pages already charged do not count as baseline: every file
            # page shares the one capacity-sized budget.
            self.assertEqual(limit, (7 << 20) + (32 << 20) + backend_pagecache.BUDGET_OFFSET)
            self.assertEqual(int((fake / "memory.max").read_text()), limit)


@unittest.skipUnless(SUPPORTED, WHY)
class DeviceTests(unittest.TestCase):
    """Mechanics on a 64 MiB temp file; skipped where O_DIRECT is refused."""

    @classmethod
    def setUpClass(cls):
        cls.tmp, reason = odirect_dir()
        if cls.tmp is None:
            raise unittest.SkipTest(reason)
        cls.dir = Path(cls.tmp.name)
        cls.backing = cls.dir / "backing.bin"
        layout.make_backing_file(cls.backing, 64 / 1024)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _odirect(self, requests, plan, think, capacity=4096):
        from adaptive_prefetch.realio.backend_odirect import ODirectBackend
        mapper = RegionMapper.for_plan(layout.check_backing(self.backing), requests, plan)
        mapper.check()
        backend = ODirectBackend(self.backing, mapper, capacity)
        try:
            return backend.run(requests, plan, gaps_us(requests, think))
        finally:
            backend.close()

    def test_backend_b_late_hits_and_parity(self):
        requests = sequential_trace(300, size=64)
        metrics, plan = record_plan([requests], "deep@64", capacity=4096)
        closed = self._odirect(requests, plan, "closed")
        self.assertGreater(closed["late_hits"], 0)
        self.assertEqual(closed["useful_prefetches"], metrics.useful_prefetches)
        self.assertEqual(closed["real_hit_ratio"], metrics.hit_ratio)
        self.assertEqual(closed["prefetches"], metrics.prefetches)
        # With 5 ms of think time every 32 KiB prefetch has long finished.
        # The shared SSD stalls for milliseconds now and then, so a single
        # late hit gets two more tries before it counts as a failure.
        for _ in range(3):
            paced = self._odirect(requests[:120], plan[:120], "fixed:5000")
            if paced["late_hits"] == 0:
                break
        self.assertEqual(paced["late_hits"], 0)
        self.assertEqual(paced["useful_prefetches"],
                         verify_plan(requests[:120], plan[:120], 4096)["useful_prefetches"])

    def test_backend_b_raises_on_a_tampered_plan(self):
        requests = sequential_trace(80, size=16)
        _, plan = record_plan([requests], "none", capacity=4096)
        tampered = list(plan)
        tampered[50] = PlanStep(b"\x01" + plan[50].hit_flags[1:], ())
        with self.assertRaises(CacheDivergence):
            self._odirect(requests, tampered, "closed")

    def test_willneed_chunking_makes_one_mib_resident(self):
        fd = os.open(self.backing, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            offset, length = 8 << 20, 1 << 20
            buf = bytearray(realio.PAGE)
            view = memoryview(buf)
            calls = backend_pagecache.willneed(fd, offset, length, 256 << 10)
            self.assertEqual(calls, 4)
            deadline = time.monotonic() + 10
            pending = list(range(offset, offset + length, realio.PAGE))
            while pending and time.monotonic() < deadline:
                pending = [p for p in pending
                           if backend_pagecache.ready(fd, p, realio.PAGE, view) < realio.PAGE]
            self.assertEqual(pending, [], "pages never became readable without blocking")
        finally:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)

    def _trace_file(self, name, requests):
        path = self.dir / f"{name}.oracleGeneral"
        with open(path, "wb") as handle:
            for r in requests:
                handle.write(RECORD.pack(int(r.timestamp_ms // 1000), r.lba * 512,
                                         r.size_blocks * 512, -1))
        return path

    def _spec(self, trace, mode, backend):
        return {"trace": str(trace), "requests": None, "skip_requests": 0, "mode": mode,
                "backend": backend, "think": "closed", "repeat": 1, "capacity": 4096,
                "window_size": 32, "backing": str(self.backing), "flush_mib": 4,
                "workers": 4, "allow_alias": False, "latency": "5,100,50",
                "device": "test", "disk": None, "no_cgroup": True, "model_path": None,
                "order": 0, "seed": 1}

    def test_csv_columns_are_the_schema(self):
        from adaptive_prefetch.realio import runner
        requests = [Request(float(i // 50) * 1000, (i * 16) % 20_000, 16)
                    for i in range(400)]
        trace = self._trace_file("msr_t", requests)
        outcomes = [runner.run_one(self._spec(trace, mode, backend))
                    for mode, backend in (("none", "odirect"), ("deep", "odirect"),
                                          ("none", "pagecache"), ("kernel_ra", "pagecache"),
                                          ("online-deep", "pagecache"))]
        names = schemas.columns("realio/realio.csv")
        for outcome in outcomes:
            self.assertEqual(tuple(outcome["row"]), names)
        runs = [{**o["row"], **o["detail"]} for o in outcomes]
        report.attach_parity(runs)
        with tempfile.TemporaryDirectory() as tmp:
            rows = runner.write_results(Path(tmp), [o["row"] for o in outcomes])
            back = schemas.read_rows("realio/realio.csv", Path(tmp) / "realio.csv")
            self.assertEqual(len(back), len(rows))
            summary = report.summarize(back)
            schemas.validate_rows("realio/realio_summary.csv", summary)
        kernel = outcomes[3]["row"]
        self.assertIsNone(kernel["sim_hit_ratio"])
        self.assertEqual(outcomes[0]["row"]["real_hit_ratio"],
                         outcomes[0]["row"]["sim_hit_ratio"])

    def test_online_shadow_matches_replay_for_an_ungated_mode(self):
        from adaptive_prefetch.realio import runner
        requests = [Request(float(i // 50) * 1000, (i * 16) % 20_000, 16)
                    for i in range(400)]
        trace = self._trace_file("msr_o", requests)
        outcome = runner.run_one(self._spec(trace, "online-deep", "pagecache"))
        expected, _ = replay(runner.load_requests(trace, None), capacity=4096,
                             mode="deep", latency_model=LatencyModel())
        self.assertEqual(outcome["row"]["sim_hit_ratio"], expected.hit_ratio)
        self.assertEqual(outcome["detail"]["sim_prefetches"], expected.prefetches)
        self.assertEqual(outcome["row"]["sim_device_ios"], expected.device_ios)


class ReproTests(unittest.TestCase):
    def test_repro_realio_without_a_device_is_unavailable(self):
        sys.path.insert(0, str(ROOT))
        import repro
        from repro import realio as repro_realio
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--quick", action="store_true")
        repro_realio.add_arguments(parser)
        args = parser.parse_args([])
        self.assertIsInstance(repro_realio.estimate(args), str)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = repro_realio.run(args)
        self.assertEqual(status, repro.EXIT_UNAVAILABLE)
        self.assertIn("--device", out.getvalue())


if __name__ == "__main__":
    unittest.main()

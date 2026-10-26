"""Live HTTP replay (webprefetch.live): cache parity, timing, hints, plan, end to end.

The origin, client and store are stdlib only, so most of these run under
system python3 too; the plan tests need the [web] extra. Plan and end-to-end
tests use the committed SYNTHETIC ``data/samples/web_fixture.csv``; it is
never research data.

Loopback timing, the link schedule and the injected-delay validity check
compare wall-clock times against the emulated model, so they are the
load-sensitive ("slow") group: they skip when the 1-minute load average is
above 32, the same bound the experiment itself enforces.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from webprefetch.cachecore import Cache, Counters  # noqa: E402
from webprefetch.live import analyze, client, objects  # noqa: E402
from webprefetch.live.origin import Link  # noqa: E402
from webprefetch.live.timing import CLOCK, PreciseTimer  # noqa: E402

HAVE_WEB = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "pandas", "sklearn", "joblib"))
FIXTURE = ROOT / "data" / "samples" / "web_fixture.csv"
LOAD_LIMIT = 32.0
QUIET = os.getloadavg()[0] <= LOAD_LIMIT
load_sensitive = unittest.skipUnless(
    QUIET, f"load-sensitive timing test: load average above {LOAD_LIMIT:g}")
LINUX = sys.platform.startswith("linux")


class _Done:
    """A finished fetch task stand-in for the logical-cache tests."""

    def done(self):
        return True


def _fixed_fetch(kind):
    def start(item=None):
        fetch = client.Fetch(kind, item)
        fetch.task = _Done()
        return fetch
    return start


class OriginServer:
    """``python -m webprefetch.live origin`` as a child process."""

    def __init__(self, store, index, *extra):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "webprefetch.live", "origin", "--store", str(store),
             "--index", str(index), *map(str, extra)],
            stdout=subprocess.PIPE, text=True, env=env)
        ready = json.loads(self.proc.stdout.readline())
        self.port = ready["port"]

    def stats(self, reset=False):
        from webprefetch.live.origin import origin_stats
        return origin_stats("127.0.0.1", self.port, reset)

    def close(self):
        self.proc.kill()
        self.proc.wait()
        self.proc.stdout.close()


def _store(directory, items, median, sigma=0.0):
    meta = objects.build_store(items, directory, median=median, sigma=sigma)
    return meta["store"], meta["index"]


class SessionCacheParityTests(unittest.TestCase):
    def test_random_sequences_match_cachecore_over_300_trials(self):
        rng = random.Random(0)
        for trial in range(300):
            capacity = rng.choice((1, 2, 4, 8))
            reference = Cache(capacity, Counters())
            live = client.SessionCache(capacity)
            for _ in range(rng.randint(1, 60)):
                item = rng.randint(0, 12)
                self.assertEqual(reference.demand(item),
                                 live.demand(item, _fixed_fetch("demand"))[0])
                for candidate in rng.sample(range(13), rng.randint(0, 4)):
                    self.assertEqual(reference.prefetch(candidate),
                                     live.prefetch(candidate, _fixed_fetch("prefetch"))
                                     is not None)
                self.assertEqual(list(reference.blocks.items()),
                                 list(live.cache.blocks.items()))
                live.check()  # every resident object still has its fetch
            for name in ("views", "hits", "issued", "useful", "prefetch_hits",
                         "evicted_unused", "demand_evicted_by_prefetch"):
                self.assertEqual(getattr(reference.counters, name),
                                 getattr(live.counters, name), (trial, name))
            self.assertEqual(len(live.used_prefetches), live.counters.useful)

    def test_planned_requests_follow_the_logical_cache(self):
        views = [{"item": 1, "gap_ms": 0, "pf": {"p": [2, 3]}},
                 {"item": 2, "gap_ms": 10, "pf": {"p": [3, 4]}},
                 {"item": 1, "gap_ms": 10, "pf": {}}]
        # miss + 2 prefetches; hit + 1 new prefetch (3 resident); hit.
        self.assertEqual(client.planned_requests(views, "p", 8), [3, 1, 0])
        # Hints prefetch nothing into the browser cache, so view 2 still misses.
        self.assertEqual(client.planned_requests(views, "p", 8, hint=True), [2, 2, 0])


@unittest.skipUnless(LINUX, "page-cache hints need Linux")
class PageCacheTests(unittest.TestCase):
    def test_chunked_willneed_makes_a_1mib_range_resident(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "big.bin"
            path.write_bytes(os.urandom(1 << 20))
            objects.drop_page_cache(path)
            fd = os.open(path, os.O_RDONLY)
            try:
                if objects.resident(fd, 0, 1 << 20):
                    self.skipTest("this filesystem keeps pages resident after DONTNEED")
                self.assertEqual(objects.willneed(fd, 0, 1 << 20), 4)
                deadline = time.monotonic() + 5
                while not objects.resident(fd, 0, 1 << 20) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(objects.resident(fd, 0, 1 << 20))
            finally:
                os.close(fd)

    def test_store_is_deterministic_aligned_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            first = objects.build_store([5, 3, 9, 3], directory, median=6000, sigma=1.0)
            self.assertFalse(first["reused"])
            index = objects.load_index(first["index"])
            self.assertEqual(sorted(index), [3, 5, 9])
            for item, (offset, size) in index.items():
                self.assertEqual(offset % objects.ALIGN, 0)
                self.assertEqual(size, objects.object_size(item, 6000, 1.0))
            digest = Path(first["store"]).read_bytes()
            self.assertTrue(objects.build_store([3, 5, 9], directory, median=6000,
                                                sigma=1.0)["reused"])
            self.assertTrue(objects.build_store([5], directory, median=6000,
                                                sigma=1.0)["reused"])  # a subset reads it too
            rebuilt = objects.build_store([3, 5, 9], directory, median=6000, sigma=1.0,
                                          reuse=False)
            self.assertFalse(rebuilt["reused"])
            self.assertEqual(Path(rebuilt["store"]).read_bytes(), digest)

    def test_sizes_are_clipped_lognormal_around_the_median(self):
        sizes = sorted(objects.object_size(i) for i in range(4001))
        self.assertTrue(all(objects.SIZE_MIN <= s <= objects.SIZE_MAX for s in sizes))
        self.assertLess(abs(sizes[2000] / objects.SIZE_MEDIAN - 1), 0.1)


class PoolOrderingTests(unittest.TestCase):
    def _admissions(self, priority, releases):
        """Two demands hold both slots; A(p) B(d) C(p) D(d) queue; then ``releases``."""
        async def scenario():
            pool = client.Pool("127.0.0.1", 0, limit=2, prefetch_limit=1, priority=priority)
            await pool.acquire("demand")
            await pool.acquire("demand")
            order = []

            async def wait(kind, name):
                await pool.acquire(kind)
                order.append(name)

            tasks = []
            for kind, name in (("prefetch", "A"), ("demand", "B"), ("prefetch", "C"),
                               ("demand", "D")):
                tasks.append(asyncio.ensure_future(wait(kind, name)))
                await asyncio.sleep(0)
            snapshots = []
            for kind in releases:
                pool.release(kind)
                for _ in range(3):
                    await asyncio.sleep(0)
                snapshots.append("".join(order))
            for task in tasks:
                task.cancel()
            return snapshots
        return asyncio.run(scenario())

    def test_waiting_demands_are_admitted_before_waiting_prefetches(self):
        # B and D take the two freed slots; A gets the next; C waits for A
        # (one speculative connection) even though a slot is free.
        snapshots = self._admissions(True, ["demand"] * 4 + ["prefetch"])
        self.assertEqual(snapshots, ["B", "BD", "BDA", "BDA", "BDAC"])

    def test_fifo_pool_admits_in_arrival_order(self):
        snapshots = self._admissions(False, ["demand", "demand", "prefetch", "demand"])
        self.assertEqual(snapshots, ["A", "AB", "ABC", "ABCD"])

    def test_prefetch_limit_holds_with_free_slots(self):
        async def scenario():
            pool = client.Pool("127.0.0.1", 0, limit=6, prefetch_limit=2)
            for _ in range(2):
                await pool.acquire("prefetch")
            third = asyncio.ensure_future(pool.acquire("prefetch"))
            await asyncio.sleep(0)
            blocked = not third.done()
            await pool.acquire("demand")  # demands still get the free slots
            pool.release("prefetch")
            await asyncio.sleep(0)
            return blocked, third.done(), pool.speculative_active
        self.assertEqual(asyncio.run(scenario()), (True, True, 2))


class LinkScheduleTests(unittest.TestCase):
    @load_sensitive
    def test_back_to_back_sends_follow_the_schedule_without_drift(self):
        async def scenario():
            timer = PreciseTimer(asyncio.get_running_loop())
            link = Link(1e6, timer, priority=True)  # 10 kB take exactly 10 ms
            link.schedule = []

            async def send():
                sent = await link.send(10_000, True)
                return sent, CLOCK()

            try:
                return await asyncio.gather(*(send() for _ in range(20))), link.schedule
            finally:
                timer.close()
        sent, schedule = asyncio.run(scenario())
        self.assertEqual(len(schedule), 20)  # one server, however many sends in one tick
        for (start, end), (previous_start, previous_end) in zip(schedule[1:], schedule):
            # Each start is the previous *scheduled* end, so jitter cannot accumulate.
            self.assertEqual(start, previous_end)
            self.assertAlmostEqual(end - start, 0.01, delta=1e-6)
        self.assertAlmostEqual(schedule[-1][1] - schedule[0][0], 0.2, delta=1e-6)
        for ((_, start, end), done), scheduled in zip(sent, schedule):
            self.assertEqual((start, end), scheduled)
            self.assertLess(done - end, 0.002)  # each completion is on time, none drifts

    def test_priority_link_sends_demand_before_queued_prefetch(self):
        class Instant:
            def sleep_until(self, when):
                future = asyncio.get_running_loop().create_future()
                future.set_result(None)
                return future

        async def scenario():
            link = Link(1e9, Instant(), priority=True)
            order = []

            async def send(name, demand):
                await link.send(10, demand)
                order.append(name)

            await asyncio.gather(send("p1", False), send("p2", False), send("d", True))
            return order
        self.assertEqual(asyncio.run(scenario()), ["d", "p1", "p2"])


@unittest.skipUnless(LINUX, "the origin's page-cache endpoints need Linux")
class LoopbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store, cls.index = _store(cls.tmp.name, [1, 2, 3], median=100_000)
        cls.origin = OriginServer(cls.store, cls.index, "--rtt-ms", 20, "--mbps", 8)

    @classmethod
    def tearDownClass(cls):
        cls.origin.close()
        cls.tmp.cleanup()

    def _pass(self, views, policy="p", **kwargs):
        config = client.ClientConfig(port=self.origin.port, gap_cap_s=10, **kwargs)
        results, lag = client.run_pass({"1": views}, policy, config, {"1": 0.0})
        return results[0], lag

    @load_sensitive
    def test_miss_is_within_5ms_of_the_model_and_a_ready_hit_waits_zero(self):
        size = objects.load_index(self.index)[1][1]
        model_ms = 20 + 1e3 * size / 1e6  # 8 Mbps is 1e6 bytes/s
        views = [{"item": 1, "gap_ms": 0, "pf": {}},
                 {"item": 1, "gap_ms": 300, "pf": {"p": [2]}},
                 {"item": 2, "gap_ms": 30, "pf": {}}]
        record, _ = self._pass(views)
        (_, _, c0, w0, _, _), (_, _, c1, w1, _, _), (_, _, c2, w2, _, _) = record["views"]
        self.assertEqual((c0, c1, c2), ("miss", "ready", "inflight"))
        self.assertLess(abs(w0 * 1e3 - model_ms), 5.0, (w0, model_ms))
        self.assertEqual(w1, 0.0)
        # The prefetch of 2 was issued 30 ms before view 2 and needs ~120 ms.
        self.assertGreater(w2, 0.0)
        self.assertLess(w2 * 1e3, model_ms)
        self.assertEqual(record["counts"], [2, 1, 3, 1])  # hits, issued, views, useful
        self.assertEqual(record["errors"], 0)

    def test_hint_issues_chunked_willneed_and_no_client_prefetch(self):
        self.origin.stats(reset=True)
        views = [{"item": 1, "gap_ms": 0, "pf": {"p": [2, 3]}},
                 {"item": 2, "gap_ms": 50, "pf": {}}]
        config = client.ClientConfig(port=self.origin.port, gap_cap_s=10)
        results, _ = client.run_pass({"1": views}, "p", config, {"1": 0.0}, hint=True)
        stats = self.origin.stats()
        self.assertEqual(results[0]["counts"], [0, 0, 2, 0])
        self.assertEqual(stats["hints"], 2)
        self.assertEqual(stats["hint_calls"], 2)  # each 100 kB object fits one 256 KiB call
        self.assertEqual(results[0]["prefetch_bytes"], 0)


class ValidityGateTests(unittest.TestCase):
    def _summary(self, lag_us=100.0, errors=0):
        return {"req_lag_p99_us": lag_us, "conn_errors": errors}

    def test_gate_passes_a_clean_pass_and_names_every_failure(self):
        server = {"loop_lag_us_p99": 300.0, "connection_errors": 0}
        self.assertEqual(analyze.validity(self._summary(), server, {"loop_lag_us_p99": 200.0},
                                          0), (True, []))
        valid, reasons = analyze.validity(self._summary(lag_us=2500.0, errors=1),
                                          {"loop_lag_us_p99": 4000.0},
                                          {"loop_lag_us_p99": None}, 3)
        self.assertFalse(valid)
        self.assertEqual(len(reasons), 5, reasons)

    def test_invalid_pass_is_never_compared(self):
        summary = {"sessions": 1, "views": 2, "hits": 1, "counted_views": 2, "ready": 1,
                   "inflight": 0, "miss": 1, "mean_wait_ms": 50.0,
                   "model_ideal_wait_ms": 40.0, "p95_wait_ms": 99.0,
                   "per_session": {"1": [100.0, 80.0, 2]}, "demand_bytes": 10,
                   "prefetch_bytes": 0, "useful_prefetch_bytes": 0,
                   "demand_disk_us_mean": 5.0, "req_lag_p99_us": 10.0, "conn_errors": 0}
        meta = {"experiment": "smoke", "store": "ssd", "config": "c", "link": "priority",
                "rtt_ms": 50.0, "mbps": 10.0, "size_median_kib": 64.0, "gap_cap_s": 2.0,
                "ramp_s": 1.0, "rep": 0}
        row = analyze.metrics_row(meta, "fixed4", "fixed0", summary, summary, valid=False,
                                  base_valid=True, mismatched=0, server={},
                                  client_lag={}, foreign_io_frac=None, load_avg=1.0)
        from prefetch_common.schemas import validate_row
        validate_row("live/live_metrics.csv", row)
        self.assertFalse(row["valid"])
        self.assertNotEqual(row["delta_vs_base_ms"], row["delta_vs_base_ms"])  # NaN

    @load_sensitive
    @unittest.skipUnless(LINUX, "the origin's page-cache endpoints need Linux")
    def test_gate_trips_on_an_origin_that_stalls_its_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            store, index = _store(directory, range(1, 9), median=4096)
            origin = OriginServer(store, index, "--rtt-ms", 2, "--mbps", 1000,
                                  "--inject-block-ms", 5)
            try:
                origin.stats(reset=True)
                sessions = {str(s): [{"item": 1 + (s + k) % 8, "gap_ms": 0, "pf": {}}
                                     for k in range(4)] for s in range(8)}
                config = client.ClientConfig(port=origin.port, gap_cap_s=0)
                results, lag = client.run_pass(sessions, "p", config,
                                               {sid: 0.0 for sid in sessions})
                server = origin.stats()
                sizes = {item: size for item, (_, size) in objects.load_index(index).items()}
            finally:
                origin.close()
        summary = analyze.summarise_pass(results, sizes, rtt_ms=2, mbps=1000)
        valid, reasons = analyze.validity(summary, server, lag, 0)
        self.assertFalse(valid)
        self.assertTrue(any("lag" in reason for reason in reasons), reasons)


@unittest.skipUnless(HAVE_WEB, "the plan needs numpy, pandas, scikit-learn and joblib")
class PlanTests(unittest.TestCase):
    POLICIES = ("fixed0@trans_s0", "fixed4@trans_s0", "adaptive_ca_0.05@trans_s0",
                "gate_0.1@trans_s0", "feedback_0.05_e0.1@trans_s0")

    @classmethod
    def setUpClass(cls):
        from webprefetch.data import build_views
        from webprefetch.index import fit_index
        from webprefetch.simulator import TransitionScorer
        cls.views, _ = build_views(FIXTURE, verify_source=False)
        cls.scorers = {"trans_s0": (fit_index(cls.views), TransitionScorer())}

    def _build(self, **kwargs):
        from webprefetch.live import plan
        options = {"split": (1, 2, 3), "sessions": 20, "min_views": 1, "seed": 42,
                   "capacity": 4}
        options.update(kwargs)
        return plan.build_plan(self.views, list(self.POLICIES), self.scorers, **options)

    def test_policy_names_parse_and_bad_ones_are_refused(self):
        from webprefetch.live.plan import parse_policy
        p = parse_policy("feedback_0.01_e0.05_w0.5@trans_online")
        self.assertEqual((p.policy.kind, p.policy.parameter, p.policy.epsilon,
                          p.policy.pollution, p.scorer), ("feedback", 0.01, 0.05, 0.5,
                                                          "trans_online"))
        self.assertEqual(parse_policy("adaptive_ca_0.002").policy.kind, "adaptive_ca")
        self.assertEqual(parse_policy("fixed8").policy.parameter, 8)
        self.assertEqual(parse_policy("fixed8").scorer, "logistic")
        for bad in ("fixed", "deep8", "gate_0.1@seq", "adaptive_x"):
            with self.assertRaises(ValueError):
                parse_policy(bad)

    def test_plan_is_deterministic_and_seeded(self):
        from webprefetch.live.plan import _dump
        plan_a, sim_a = self._build()
        plan_b, sim_b = self._build()
        self.assertEqual(_dump(plan_a), _dump(plan_b))
        self.assertEqual(_dump(sim_a), _dump(sim_b))
        self.assertEqual(plan_a["sessions_selected"], 20)
        other, _ = self._build(seed=7)
        self.assertNotEqual(sorted(other["sessions"]), sorted(plan_a["sessions"]))
        # Every requestable object is in the store item list.
        items = set(plan_a["items"])
        for views in plan_a["sessions"].values():
            for view in views:
                self.assertIn(view["item"], items)
                for ids in view["pf"].values():
                    self.assertTrue(set(ids) <= items)
        self.assertEqual(set(sim_a["sessions"]), set(self.POLICIES))

    def test_a_walk_that_disagrees_with_the_simulator_is_refused(self):
        from webprefetch.live import plan
        real = plan._walk

        def tampered(frame, ranked, live, capacity, gap):
            decisions, sessions = real(frame, ranked, live, capacity, gap)
            if live.name == "fixed4@trans_s0":
                first = next(iter(sessions))
                sessions[first] = [x + 1 for x in sessions[first]]
            return decisions, sessions

        with mock.patch.object(plan, "_walk", tampered):
            with self.assertRaises(plan.PlanParityError):
                self._build()


@unittest.skipUnless(HAVE_WEB, "the plan needs numpy, pandas, scikit-learn and joblib")
@unittest.skipUnless(LINUX, "the origin's page-cache endpoints need Linux")
class EndToEndParityTests(unittest.TestCase):
    """Real HTTP over loopback must reproduce the simulator's counters exactly."""

    def test_20_sessions_every_policy_matches_the_simulator(self):
        from webprefetch.live import plan
        PlanTests.setUpClass()
        policies = list(PlanTests.POLICIES)
        plan_, sim = plan.build_plan(PlanTests.views, policies, PlanTests.scorers,
                                     split=(1, 2, 3), sessions=20, min_views=1, capacity=4)
        self.assertEqual(plan_["sessions_selected"], 20)
        with tempfile.TemporaryDirectory() as directory:
            store, index = _store(directory, plan_["items"], median=4096, sigma=0.5)
            origin = OriginServer(store, index, "--rtt-ms", 2, "--mbps", 1000)
            try:
                config = client.ClientConfig(port=origin.port, capacity=4, gap_cap_s=0.01)
                starts = client.session_starts(plan_["sessions"], ramp_s=0.3)
                issued = 0
                for label in policies + ["hint:fixed4@trans_s0"]:
                    name, hint = analyze.reference_policy(label)
                    results, _ = client.run_pass(plan_["sessions"], name, config, starts,
                                                 hint=hint)
                    record = analyze.parity(results, sim, label, "fixed0@trans_s0")
                    self.assertEqual(record["mismatched_sessions"], 0, (label, record))
                    self.assertEqual(sum(r["errors"] for r in results), 0, label)
                    issued += record["live_issued"]
                self.assertGreater(issued, 0)
            finally:
                origin.close()

    def test_run_command_writes_schema_valid_results(self):
        from prefetch_common.schemas import read_rows
        from webprefetch.live.run import main
        PlanTests.setUpClass()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            views = directory / "views.csv.gz"
            PlanTests.views.to_csv(views, index=False, compression="gzip")
            argv = ["--views", str(views), "--policies", "fixed0@trans_s0,fixed2@trans_s0",
                    "--baseline", "fixed0@trans_s0", "--sessions", "8", "--min-views", "1",
                    "--capacity", "4", "--size-median", "4096", "--rtt-ms", "2",
                    "--mbps", "1000", "--gap-cap-s", "0.01", "--ramp-s", "0.2",
                    "--reps", "2", "--experiment", "smoke", "--no-probe",
                    "--work-dir", str(directory / "work"),
                    "--results-dir", str(directory / "results"), "--max-load", "1000",
                    "--bootstrap-reps", "50"]
            with mock.patch("sys.stdout"):
                self.assertEqual(main(argv), 0)
            results = directory / "results"
            metrics = read_rows("live/live_metrics.csv", results / "live_metrics.csv")
            parity = read_rows("live/live_parity.csv", results / "live_parity.csv")
            self.assertEqual(len(metrics), 4)
            self.assertEqual(sum(r["mismatched_sessions"] for r in parity), 0)
            config = json.loads((results / "live_config.json").read_text())
            from prefetch_common.schemas import validate_file
            validate_file(results / "live_config.json", "live/live_config.json")
            validate_file(results / "provenance.json", "live/provenance.json")
            run = next(iter(config["runs"].values()))
            self.assertEqual(run["assumptions"]["rtt_ms"]["tag"], "ASSUMED")
            self.assertEqual(run["assumptions"]["store"]["tag"], "MEASURED")
            self.assertTrue((results / "provenance.json").is_file())
            self.assertIn("Parity: 4 passes", (results / "live_report.md").read_text())
            # A second run replaces its own rows instead of duplicating them.
            with mock.patch("sys.stdout"):
                self.assertEqual(main(argv), 0)
            self.assertEqual(len(read_rows("live/live_metrics.csv",
                                           results / "live_metrics.csv")), 4)
            # ... and its config entry keeps the replaced passes as superseded.
            run = next(iter(json.loads((results / "live_config.json").read_text())
                            ["runs"].values()))
            self.assertEqual((len(run["passes"]), len(run["superseded"]),
                              len(run["invocations"])), (4, 4, 2))
            # A changed views file builds a new plan instead of reusing the old one.
            from webprefetch.live.run import add_arguments, ensure_plan
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(argv)
            names = ["fixed0@trans_s0", "fixed2@trans_s0"]
            with mock.patch("sys.stdout"):
                first, _, _ = ensure_plan(args, names)
                stat = views.stat()
                os.utime(views, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
                second, _, _ = ensure_plan(args, names)
            self.assertNotEqual(first, second)
            # A run whose planned rate is too high is refused and writes no results.
            refused = argv[:argv.index("--results-dir")] + [
                "--results-dir", str(directory / "refused"), "--max-load", "1000",
                "--bootstrap-reps", "50", "--max-rps-per-shard", "0.001"]
            refused.remove("--no-probe")
            with mock.patch("sys.stdout"):
                self.assertEqual(main(refused), 2)
            self.assertFalse((directory / "refused").exists())


if __name__ == "__main__":
    unittest.main()

"""prefetch_common: the shared objective, statistics, provenance and results contract.

Stdlib only, so it runs under system python3 too; the numpy/scipy/pandas
comparisons skip when those are missing.
"""

import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_prefetch.simulator import LatencyModel, Metrics, replay  # noqa: E402
from adaptive_prefetch.trace import Request  # noqa: E402
from prefetch_common import provenance, schemas, stats, utility  # noqa: E402

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_WEB = all(importlib.util.find_spec(n) is not None
               for n in ("numpy", "pandas", "sklearn", "joblib"))
FIXTURES = ROOT / "tests" / "fixtures" / "results"
PIVOT_ORIGINAL = ROOT / "results" / "pivot_original"


class UtilityTests(unittest.TestCase):
    def test_cost_to_c_at_the_two_cost_models(self):
        # Measured shape (hit 1% of a miss, prefetch = one miss) and the
        # 5/100/50 us assumption.
        self.assertEqual(utility.cost_to_c(1, 100, 100), 100 / 99)
        self.assertEqual(utility.cost_to_c(5, 100, 50), 50 / 95)
        self.assertEqual(utility.BREAK_EVEN_C, 100 / 99)

    def test_cost_to_c_is_the_simulator_break_even(self):
        for model in (LatencyModel(), LatencyModel(1, 100, 100),
                      LatencyModel(75.9, 7590.0, 0.9 * 7590.0), LatencyModel(3, 10, 0)):
            self.assertEqual(utility.model_c(model), model.break_even_precision())
        self.assertEqual(utility.cost_to_c(100, 100, 5), math.inf)
        with self.assertRaises(ValueError):
            utility.cost_to_c(-1, 100, 100)

    def test_U(self):
        self.assertEqual(utility.U(30, 10, 100, 0.5), 0.25)
        self.assertEqual(utility.U(30, 10, 100, 0.0), 0.30)
        with self.assertRaises(ValueError):
            utility.U(1, 1, 0, 0.1)
        with self.assertRaises(ValueError):
            utility.U(1, 1, 10, -0.1)

    def test_row_counts_reads_both_studies_rows(self):
        metrics = Metrics(read_blocks=200, read_hits=50, prefetches=40)
        self.assertEqual(utility.row_counts(metrics), (50.0, 40.0, 200.0))
        web = {"policy": "fixed1", "views": 1000, "hits": 120, "issued": 800,
               "hit_ratio": 0.12, "issued_per_view": 0.8}
        self.assertEqual(utility.row_counts(web), (120.0, 800.0, 1000.0))
        # A disk sweep row: counts per block, never per I/O request.
        sweep = {"mode": "deep", "hit_ratio": 0.25, "total_prefetches": 30,
                 "read_blocks": 400, "requests": 50}
        self.assertEqual(utility.row_counts(sweep), (100.0, 30.0, 400.0))
        # A skipped sweep row keeps its per-I/O requests; it must not become
        # the denominator in place of the missing read_blocks.
        partial = dict(sweep, read_blocks=None)
        with self.assertRaises(ValueError):
            utility.row_counts(partial)
        ratios = {"views": 10, "hit_ratio": 0.5, "issued_per_view": 2.0}
        self.assertEqual(utility.row_counts(ratios), (5.0, 20.0, 10.0))
        with self.assertRaises(KeyError):
            utility.row_counts({"hits": 1, "issued": 1})

    def test_c_star_is_the_tie_point(self):
        none = {"views": 1000, "hits": 100, "issued": 0}
        policy = {"views": 1000, "hits": 120, "issued": 800}
        c = utility.c_star(policy, none)
        self.assertAlmostEqual(c, 0.025)
        self.assertAlmostEqual(utility.row_U(policy, c), utility.row_U(none, c))
        self.assertGreater(utility.row_U(policy, c / 2), utility.row_U(none, c / 2))
        self.assertLess(utility.row_U(policy, 2 * c), utility.row_U(none, 2 * c))
        self.assertIsNone(utility.c_star(none, none))
        worse = {"views": 1000, "hits": 90, "issued": 500}
        self.assertLess(utility.c_star(worse, none), 0)

    def test_envelope_and_utility_columns(self):
        rows = [{"policy": "none", "views": 100, "hits": 10, "issued": 0},
                {"policy": "fixed1", "views": 100, "hits": 20, "issued": 50},
                {"policy": "fixed1_copy", "views": 100, "hits": 20, "issued": 50}]
        env = utility.envelope(rows, (0.0, 0.1, 0.5))
        self.assertEqual([e["policy"] for e in env], ["fixed1", "fixed1", "none"])
        # c*=0.2: exactly tied there, and the earlier row wins a tie.
        self.assertEqual(utility.envelope(rows, (0.2,))[0]["policy"], "none")
        names = list(utility.utility_columns(rows[1]))
        self.assertEqual(names[0], "U_0")
        self.assertEqual(names[-1], "U_1.0101")
        self.assertEqual(len(names), len(utility.UTILITY_CS))
        with self.assertRaises(ValueError):
            utility.envelope([], (0.1,))


class D11CostEquivalenceTests(unittest.TestCase):
    """argmax U(c) == argmin modelled cost, on the rows D11 is about.

    The rows are real ``replay`` outputs: the contiguous scan and the random
    trace of ``test_prefetch_multiple_does_change_the_cost_ranking`` and
    ``test_cost_ranking_is_invariant_to_the_calibration_scale``.
    """

    SCAN = [Request(float(i) * 0.5, i * 8, 8) for i in range(3000)]

    @staticmethod
    def _random():
        rng = random.Random(11)
        return [Request(float(i) * 0.5, rng.randrange(0, 4000) * 16, 8) for i in range(4000)]

    def _rows(self, requests, modes, model):
        rows = []
        for mode in modes:
            metrics, _ = replay(requests, 2048, 32, mode, latency_model=model)
            rows.append((mode, metrics))
        return rows

    def _check(self, requests, modes, model):
        c = utility.model_c(model)
        rows = self._rows(requests, modes, model)
        for _, metrics in rows:
            u = utility.row_U(metrics, c)
            self.assertAlmostEqual(
                utility.modelled_cost_us(u, model.hit_us, model.demand_miss_us),
                metrics.mean_access_latency_us, delta=1e-9 * model.demand_miss_us)
        cheapest = min(rows, key=lambda r: r[1].mean_access_latency_us)[0]
        best = utility.envelope([{"mode": m, **{
            "read_blocks": x.read_blocks, "read_hits": x.read_hits,
            "prefetches": x.prefetches}} for m, x in rows], (c,))[0]["policy"]
        self.assertEqual(best, cheapest)
        return best

    def test_break_even_model_picks_none_and_overlapped_picks_deep(self):
        modes = ("none", "sequential", "deep")
        picks = {}
        for multiple in (1.0, 0.9, 0.5):
            model = LatencyModel(hit_us=75.9, demand_miss_us=7590.0,
                                 prefetch_us=7590.0 * multiple)
            picks[multiple] = self._check(self.SCAN, modes, model)
        self.assertEqual(picks[1.0], "none")  # c = 1.0101: nothing pays
        self.assertEqual(picks[0.9], "deep")  # D11's measured flip
        self.assertEqual(picks[0.5], "deep")

    def test_equivalence_holds_on_a_random_trace_and_the_assumed_model(self):
        modes = ("none", "sequential", "strided", "adaptive_evidence")
        for model in (LatencyModel(), LatencyModel(1, 100, 100),
                      LatencyModel(0.01 * 1488, 1488, 1488)):
            self._check(self._random(), modes, model)


class StatsTests(unittest.TestCase):
    A = [0.30, 0.42, 0.25, 0.61, 0.33, 0.50, 0.47, 0.29, 0.38, 0.55]
    B = [0.28, 0.40, 0.26, 0.52, 0.30, 0.49, 0.41, 0.30, 0.31, 0.50]

    def test_quantile_is_linear_interpolation(self):
        self.assertEqual(stats.quantile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(stats.quantile([4, 1, 3, 2], 0.0), 1)
        self.assertAlmostEqual(stats.quantile(list(range(11)), 0.025), 0.25)

    @unittest.skipUnless(HAVE_NUMPY, "needs numpy")
    def test_quantile_matches_numpy(self):
        import numpy as np
        rng = random.Random(3)
        values = [rng.gauss(0, 1) for _ in range(997)]
        for q in (0.025, 0.5, 0.975):
            self.assertAlmostEqual(stats.quantile(values, q), float(np.quantile(values, q)),
                                   places=12)

    def test_paired_bootstrap_is_deterministic_for_a_seed(self):
        first = stats.paired_bootstrap(self.A, self.B, reps=2000, seed=42)
        again = stats.paired_bootstrap(self.A, self.B, reps=2000, seed=42)
        other = stats.paired_bootstrap(self.A, self.B, reps=2000, seed=7)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        mean_diff = sum(a - b for a, b in zip(self.A, self.B)) / len(self.A)
        self.assertAlmostEqual(first.estimate, mean_diff)
        self.assertLessEqual(first.low, first.estimate)
        self.assertGreaterEqual(first.high, first.estimate)
        self.assertEqual(first.n, 10)

    def test_paired_bootstrap_options(self):
        self.assertEqual(stats.paired_bootstrap([0.5], [0.25]), stats.CI(0.25, 0.25, 0.25, 1))
        ratio = stats.paired_bootstrap([2, 4], [1, 4], reps=500, statistic="ratio")
        self.assertAlmostEqual(ratio.estimate, 1.5)
        weighted = stats.paired_bootstrap([1, 0], [0, 0], reps=500, weights=[3, 1])
        self.assertAlmostEqual(weighted.estimate, 0.75)
        for bad in (dict(a=[1], b=[1, 2]), dict(a=[], b=[]),
                    dict(a=[1, 2], b=[0, 1], statistic="ratio"),
                    dict(a=[1, 2], b=[1, 2], statistic="median"),
                    dict(a=[1, 2], b=[1, 2], weights=[1]),
                    dict(a=[1, 2], b=[1, 2], level=1.0),
                    dict(a=[1, 2], b=[1, 2], reps=0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                stats.paired_bootstrap(**bad)

    def test_cluster_bootstrap_widens_with_a_cluster_effect(self):
        rng = random.Random(5)
        values, clusters = [], []
        for corpus, centre in (("msr", 0.0), ("fiu", 1.0), ("alibaba", 2.0), ("tencent", 3.0)):
            for _ in range(14):
                values.append(centre + rng.gauss(0, 0.01))
                clusters.append(corpus)
        flat = stats.paired_bootstrap(values, [0.0] * len(values), reps=2000)
        nested = stats.cluster_bootstrap(values, clusters, reps=2000)
        self.assertAlmostEqual(nested.estimate, flat.estimate)
        self.assertGreater(nested.high - nested.low, 3 * (flat.high - flat.low))
        self.assertEqual(nested, stats.cluster_bootstrap(values, clusters, reps=2000))

    def test_cluster_bootstrap_ignores_cluster_listing_order(self):
        values = [0.1, 0.2, 0.3, 1.1, 1.3, 2.0, 2.4, 2.2]
        clusters = ["a", "a", "a", "b", "b", "c", "c", "c"]
        reordered = [5, 6, 7, 0, 1, 2, 3, 4]  # whole clusters moved, members in order
        first = stats.cluster_bootstrap(values, clusters, reps=1000)
        moved = stats.cluster_bootstrap([values[i] for i in reordered],
                                        [clusters[i] for i in reordered], reps=1000)
        self.assertEqual(first, moved)

    def test_twin_weights_count_a_family_once(self):
        names = ["alibaba_13", "alibaba_16", "alibaba_1", "msr_hm_0"]
        groups = [("alibaba_13", "alibaba_16", "alibaba_18"), ("alibaba_1", "alibaba_5")]
        self.assertEqual(stats.twin_weights(names, groups), [0.5, 0.5, 1.0, 1.0])
        weighted = stats.cluster_bootstrap([1.0, 1.0, 0.0, 0.0], ["a"] * 4,
                                           stats.twin_weights(names, groups), reps=200)
        self.assertAlmostEqual(weighted.estimate, 1 / 3)
        with self.assertRaises(ValueError):
            stats.twin_weights(["x"], [("x", "y"), ("x", "z")])

    @staticmethod
    def _sessions(seed=0, n=400):
        rng = random.Random(seed)
        sessions = {"adaptive": {}, "fixed1": {}}
        for sid in range(n):
            views = rng.randint(1, 12)
            for name, lift in (("adaptive", 1), ("fixed1", 0)):
                hits = min(views, rng.randint(0, views // 2) + lift * rng.randint(0, 1))
                issued = rng.randint(0, 2 * views)
                sessions[name][sid] = [hits, issued, views, rng.randint(0, issued)]
        del sessions["fixed1"][7]  # a session one policy never saw counts as zeros
        return sessions

    def test_session_bootstrap_stdlib(self):
        sessions = self._sessions()
        ci = stats.session_bootstrap(sessions, "adaptive", "fixed1", cost=0.01,
                                     backend="stdlib")
        self.assertEqual(ci, stats.session_bootstrap(sessions, "adaptive", "fixed1",
                                                     cost=0.01, backend="stdlib"))
        left, right = sessions["adaptive"], sessions["fixed1"]
        num = sum((left[k][0] - right.get(k, [0, 0, 0])[0])
                  - 0.01 * (left[k][1] - right.get(k, [0, 0, 0])[1]) for k in left)
        self.assertAlmostEqual(ci.estimate, num / sum(left[k][2] for k in left))
        self.assertLess(ci.low, ci.estimate)
        self.assertEqual(ci.n, 400)
        with self.assertRaises(ValueError):
            stats.session_bootstrap(sessions, "adaptive", "fixed1", backend="R")

    @unittest.skipUnless(HAVE_WEB, "needs the web study's dependencies")
    def test_session_bootstrap_numpy_reproduces_webprefetch(self):
        from webprefetch.simulator import paired_bootstrap
        sessions = self._sessions(seed=1)
        for cost in (0.0, 0.01, 0.25):
            ours = stats.session_bootstrap(sessions, "adaptive", "fixed1", cost=cost,
                                           backend="numpy")
            theirs = paired_bootstrap(sessions, "adaptive", "fixed1", cost=cost)
            self.assertEqual((ours.low, ours.high), theirs)

    def test_wins(self):
        self.assertEqual(stats.wins([1, 2, 3, 4], [1, 1, 5, 4]), (1, 1, 2))
        self.assertEqual(stats.wins([1.0, 2.0], [1.05, 1.0], tol=0.1), (1, 0, 1))

    def test_wilcoxon_without_scipy_returns_none(self):
        with mock.patch.object(stats, "HAVE_SCIPY", False):
            self.assertIsNone(stats.wilcoxon(self.A, self.B))
        self.assertIsNone(stats.wilcoxon([1.0], [0.0]))

    @unittest.skipUnless(stats.HAVE_SCIPY, "needs scipy")
    def test_wilcoxon_matches_scipy(self):
        from scipy.stats import wilcoxon
        expected = float(wilcoxon([a - b for a, b in zip(self.A, self.B)]).pvalue)
        self.assertAlmostEqual(stats.wilcoxon(self.A, self.B), expected)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def _write(self, **kwargs):
        out = self.tmp / "results" / "disk"
        out.mkdir(parents=True)
        (out / "table_main.csv").write_text("a,b\n1,2\n")
        (out / "sub").mkdir()
        (out / "sub" / "x.json").write_text("{}\n")
        source = self.tmp / "input.bin"
        source.write_bytes(b"trace bytes")
        start = provenance.snapshot()
        path = provenance.write_provenance(out, ["reproduce.py", "disk"], [source],
                                           start=start, gpu=False, **kwargs)
        return out, path, source

    def test_record_has_the_contract_keys_and_hashes(self):
        out, path, source = self._write(notes="unit test")
        record = json.loads(path.read_text())
        self.assertEqual(tuple(record), provenance.PROVENANCE_KEYS)
        self.assertEqual(record["format"], provenance.FORMAT)
        self.assertEqual(record["study"], "disk")
        self.assertEqual(record["command"], ["reproduce.py", "disk"])
        self.assertEqual(list(record["inputs"].values()),
                         [provenance.sha256_file(source)])
        self.assertEqual(sorted(record["outputs"]), ["sub/x.json", "table_main.csv"])
        self.assertIsNotNone(record["wall_s"])
        self.assertIsNone(record["gpu"])
        self.assertIn("end", record["load_avg"])
        self.assertEqual(provenance.verify_provenance(out), [])
        schemas.validate_file(path)

    def test_git_sha_is_head(self):
        if shutil.which("git") is None:
            self.skipTest("git not installed")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=False)
        if head.returncode != 0:
            self.skipTest("not a git checkout")
        sha, dirty = provenance.git_state()
        self.assertEqual(sha, head.stdout.strip())
        self.assertIsInstance(dirty, bool)

    def test_verify_detects_edits(self):
        out, _, _ = self._write()
        (out / "table_main.csv").write_text("a,b\n1,3\n")
        (out / "new.csv").write_text("x\n")
        (out / "sub" / "x.json").unlink()
        problems = provenance.verify_provenance(out)
        self.assertIn("changed output table_main.csv", problems)
        self.assertIn("unlisted output new.csv", problems)
        self.assertIn("missing output sub/x.json", problems)

    def test_inputs_must_exist_and_be_files(self):
        with self.assertRaises(FileNotFoundError):
            provenance.write_provenance(self.tmp / "o", "cmd", [self.tmp / "absent"], gpu=False)
        with self.assertRaises(IsADirectoryError):
            provenance.write_provenance(self.tmp / "o", "cmd", [self.tmp], gpu=False)

    def test_gpu_is_optional(self):
        with mock.patch.object(provenance.shutil, "which", return_value=None):
            self.assertIsNone(provenance.gpu_info())
            with self.assertRaises(RuntimeError):
                provenance.write_provenance(self.tmp / "o", "cmd", gpu=True)
            path = provenance.write_provenance(self.tmp / "o", "cmd", gpu="auto")
        self.assertIsNone(json.loads(path.read_text())["gpu"])

    def test_extra_and_key_check(self):
        out, path, _ = self._write(extra={"config": "C1"})
        record = provenance.read_provenance(out)
        self.assertEqual(record["extra"], {"config": "C1"})
        record["surprise"] = 1
        self.assertTrue(provenance.check_keys(record))

    def test_pivot_original_provenance_is_reconciled(self):
        """The carried-over pivot outputs use the same keys, and still hash."""
        record = provenance.read_provenance(PIVOT_ORIGINAL)
        self.assertEqual(record["git_sha"], "8569cdf476ab49442ed9077c9617ab0574b48fd5")
        self.assertTrue(record["copied_verbatim"])
        self.assertEqual(len(record["outputs"]), 17)
        self.assertEqual(provenance.verify_provenance(PIVOT_ORIGINAL), [])
        schemas.validate_file(PIVOT_ORIGINAL / "provenance.json")


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    @staticmethod
    def _row(name, **overrides):
        """A type-valid row: placeholder values by dtype."""
        sample = {"str": "x", "int": 1, "float": 0.5, "bool": False, "json": {"k": [1]}}
        schema = schemas.schema_for(name)
        row = {c.name: sample[c.base] for c in schema.columns if c.required}
        row.update(overrides)
        return row

    def test_every_schema_is_well_formed(self):
        expected = {"disk/sweep.jsonl", "disk/table_main.csv", "disk/table_by_corpus.csv",
                    "disk/deltas.csv", "disk/winners.csv", "disk/kappa.csv",
                    "disk/corpus_profile.csv",
                    "gpu/disk_learned_depth.csv", "realio/realio.csv",
                    "realio/realio_summary.csv", "realio/device_*.json",
                    "web/policy_metrics.csv", "web/comparisons.csv",
                    "web/decomposition.csv", "web/freshness.csv", "web/seq_metrics.csv",
                    "live/live_metrics.csv", "live/live_parity.csv", "*/provenance.json",
                    # WPF1: files their producers write without write_rows.
                    "gpu/disk_lstm.csv", "gpu/disk_depth_baselines.csv", "gpu/gpu_env.json",
                    "web/calibration.csv", "web/learning_curve.csv",
                    "web/dataset_profile.json", "web/think_time.json",
                    "web/session_parity.json", "live/live_config.json",
                    "live/disk_probe_*.json",
                    # WPMSRC (E7): the MSRC light sample.
                    "msrc_samples/traces.csv", "msrc_samples/classifier_metrics.csv",
                    "msrc_samples/classifier_deltas.csv", "msrc_samples/replay.csv",
                    # The claims ledger (paper/claims.py, reproduce.py paper).
                    "claims/paper_claims.csv"}
        self.assertEqual(set(schemas.SCHEMAS), expected)
        for name, schema in schemas.SCHEMAS.items():
            self.assertIn(schema.fmt, schemas.FORMATS, name)
            self.assertEqual(len(set(schema.names)), len(schema.names), name)
            for column in schema.columns:
                self.assertIn(column.base, schemas.DTYPES, f"{name}.{column.name}")
                if schema.fmt == "jsonl":
                    self.assertTrue(column.required, f"{name}.{column.name}")
                elif schema.fmt == "csv" and not column.required:
                    # An absent optional csv column reads back as no key; a
                    # present one fills the rows that lack it with None.
                    self.assertTrue(column.nullable, f"{name}.{column.name}")
        self.assertTrue(set(schemas.SWEEP_KEY) <= set(schemas.columns("disk/sweep.jsonl")))
        self.assertEqual(schemas.columns("disk/table_main.csv"),
                         schemas.columns("disk/table_by_corpus.csv"))

    def test_realio_columns_are_the_plan_list(self):
        self.assertEqual(schemas.columns("realio/realio.csv"), (
            "device", "backend", "mode", "think", "repeat", "trace", "requests",
            "sim_hit_ratio", "sim_precision", "sim_device_ios",
            "real_hit_ratio", "full_hit_request_frac", "ready_hits", "late_hits", "late_wait_us",
            "lat_mean_us", "lat_p50_us", "lat_p95_us", "lat_p99_us", "lat_max_us",
            "device_MiB", "prefetch_MiB", "think_total_s", "think_error_us_p99", "wall_s",
            "cg_file_MiB", "cg_events_max", "foreign_io_frac", "wrapped",
            "pred_mean_us", "pred_err", "k_eff", "load_avg"))

    def test_policy_metrics_carry_every_utility_column(self):
        names = schemas.columns("web/policy_metrics.csv")
        for c in utility.UTILITY_CS:
            self.assertIn(utility.utility_column(c), names)
        self.assertIn("c_star", names)

    def test_pattern_lookup(self):
        self.assertIs(schemas.schema_for("realio/device_sdb.json"),
                      schemas.SCHEMAS["realio/device_*.json"])
        self.assertIs(schemas.schema_for("pivot_original/provenance.json"),
                      schemas.SCHEMAS["*/provenance.json"])
        with self.assertRaises(schemas.SchemaError):
            schemas.schema_for("disk/unknown.csv")
        with self.assertRaises(schemas.SchemaError):
            schemas.default_path("realio/device_*.json")

    def test_write_rows_rejects_unknown_missing_and_mistyped(self):
        target = self.tmp / "kappa.csv"
        good = self._row("disk/kappa.csv", kappa_star=math.inf)
        unknown = dict(good, extra=1)
        missing = {k: v for k, v in good.items() if k != "trace"}
        for bad in (unknown, missing, dict(good, capacity="2048"), dict(good, capacity=True),
                    dict(good, ios_none=None), dict(good, delta_ios=1.5)):
            with self.assertRaises(schemas.SchemaError):
                schemas.write_rows("disk/kappa.csv", [good, bad], target)
            self.assertFalse(target.exists(), "nothing is written when a row fails")
        schemas.write_rows("disk/kappa.csv", [good], target)
        self.assertTrue(target.exists())
        # '' and None are one empty field in csv, so a nullable str refuses ''.
        decomposition = self._row("web/decomposition.csv", note="")
        with self.assertRaises(schemas.SchemaError):
            schemas.write_rows("web/decomposition.csv", [decomposition], self.tmp / "d.csv")
        self.assertFalse((self.tmp / "d.csv").exists())

    def test_csv_round_trip_is_typed(self):
        target = self.tmp / "learned.csv"
        rows = [self._row("gpu/disk_learned_depth.csv", thresholds={"seg": [0.1, 0.2]},
                          test_hit_ratio=0.1 + 0.2, vcf_depth=None, gpu_contention=None),
                self._row("gpu/disk_learned_depth.csv", thresholds=None, train_s=math.inf,
                          status="skipped: no trained learned_depth artifact",
                          test_hit_ratio=None)]
        for column in schemas.schema_for("gpu/disk_learned_depth.csv").columns:
            rows[1].setdefault(column.name, None)
        schemas.write_rows("gpu/disk_learned_depth.csv", rows, target)
        self.assertEqual(target.read_text().splitlines()[0],
                         ",".join(schemas.columns("gpu/disk_learned_depth.csv")))
        back = schemas.read_rows("gpu/disk_learned_depth.csv", target)
        self.assertEqual(back, [{c: r.get(c) for c in schemas.columns(
            "gpu/disk_learned_depth.csv")} for r in rows])
        self.assertEqual(back[0]["test_hit_ratio"], 0.1 + 0.2)  # repr round-trips exactly

    def test_csv_header_must_match_in_order(self):
        target = self.tmp / "freshness.csv"
        names = list(schemas.columns("web/freshness.csv"))
        target.write_text(",".join(reversed(names)) + "\n")
        with self.assertRaises(schemas.SchemaError):
            schemas.read_rows("web/freshness.csv", target)

    def test_jsonl_appends_and_refuses_non_finite(self):
        target = self.tmp / "sweep.jsonl"
        row = self._row("disk/sweep.jsonl", max_requests=None)
        schemas.write_rows("disk/sweep.jsonl", [row], target)
        schemas.write_rows("disk/sweep.jsonl", [dict(row, mode="deep")], target, append=True)
        back = schemas.read_rows("disk/sweep.jsonl", target)
        self.assertEqual([r["mode"] for r in back], ["x", "deep"])
        self.assertIsNone(back[0]["max_requests"])
        with self.assertRaises(schemas.SchemaError):
            schemas.write_rows("disk/sweep.jsonl", [dict(row, hit_ratio=math.nan)],
                               target, append=True)
        self.assertEqual(len(schemas.read_rows("disk/sweep.jsonl", target)), 2)
        with self.assertRaises(schemas.SchemaError):
            schemas.write_rows("disk/kappa.csv", [self._row("disk/kappa.csv")],
                               self.tmp / "k.csv", append=True)

    def test_json_object_with_optional_keys(self):
        target = self.tmp / "realio" / "device_sdb.json"
        device = self._row("realio/device_sdb.json", model=None)
        schemas.write_rows("realio/device_sdb.json", [device], target)
        self.assertEqual(schemas.validate_file(target), [device])
        schemas.write_rows("realio/device_sdb.json", [dict(device, extra={"n": 3})], target)
        with self.assertRaises(schemas.SchemaError):
            schemas.write_rows("realio/device_sdb.json", [dict(device, surprise=1)], target)
        with self.assertRaises(schemas.SchemaError):
            schemas.write_rows("realio/device_sdb.json", [device, device], target)

    @unittest.skipUnless(HAVE_NUMPY, "needs numpy")
    def test_numpy_scalars_are_accepted(self):
        import numpy as np
        row = self._row("web/freshness.csv", k=np.int64(8), coverage=np.float64(0.25),
                        eligible_pairs=np.int32(10), sources_with_transitions=None)
        target = self.tmp / "freshness.csv"
        schemas.write_rows("web/freshness.csv", [row], target)
        back = schemas.read_rows("web/freshness.csv", target)[0]
        self.assertEqual((back["k"], back["coverage"]), (8, 0.25))
        with self.assertRaises(schemas.SchemaError):
            schemas.write_rows("realio/device_sdb.json",
                               [self._row("realio/device_sdb.json", rotational=np.bool_(True),
                                          samples=np.float64(2.0))], self.tmp / "d.json")
        schemas.write_rows("realio/device_sdb.json",
                           [self._row("realio/device_sdb.json", rotational=np.bool_(True))],
                           self.tmp / "d.json")

    def test_validate_tree_flags_bad_files_and_ignores_others(self):
        root = self.tmp / "results"
        schemas.write_rows("live/live_parity.csv", [self._row("live/live_parity.csv")],
                           root / "live" / "live_parity.csv")
        (root / "README.md").write_text("fixture\n")
        self.assertEqual(schemas.validate_tree(root), {})
        (root / "web").mkdir()
        (root / "web" / "freshness.csv").write_text("split,index\nvalidation,online\n")
        (root / "web" / "mystery.csv").write_text("a\n1\n")
        self.assertEqual(sorted(schemas.validate_tree(root)),
                         ["web/freshness.csv", "web/mystery.csv"])

    def test_optional_csv_column_may_be_absent_or_present_in_order(self):
        name = "gpu/disk_learned_depth.csv"
        without = self._row(name, thresholds=None)
        target = self.tmp / "learned.csv"
        schemas.write_rows(name, [without], target)
        header = target.read_text().splitlines()[0].split(",")
        self.assertNotIn("linux_ra_hit_ratio", header)
        self.assertNotIn("linux_ra_hit_ratio", schemas.read_rows(name, target)[0])
        # One row carrying it puts the column in the file, None elsewhere.
        schemas.write_rows(name, [without, dict(without, linux_ra_hit_ratio=0.25)], target)
        header = target.read_text().splitlines()[0].split(",")
        self.assertEqual(header, list(schemas.columns(name)))
        back = schemas.read_rows(name, target)
        self.assertEqual([r["linux_ra_hit_ratio"] for r in back], [None, 0.25])
        # Present out of schema order is still refused.
        lines = target.read_text().splitlines()
        moved = header[:-1]
        moved.insert(0, header[-1])
        target.write_text(",".join(moved) + "\n")
        with self.assertRaises(schemas.SchemaError):
            schemas.read_rows(name, target)
        # A required column cannot be left out.
        target.write_text(",".join(header[1:]) + "\n" + lines[1].split(",", 1)[1] + "\n")
        with self.assertRaises(schemas.SchemaError):
            schemas.read_rows(name, target)

    def test_validate_tree_checks_only_provenance_of_a_verbatim_copy(self):
        root = self.tmp / "results"
        (root / "pivot_original").mkdir(parents=True)
        (root / "pivot_original" / "benchmark.json").write_text('{"pivot": 1}\n')
        self.assertEqual(schemas.validate_tree(root), {})
        (root / "pivot_original" / "provenance.json").write_text('{"format": 1}\n')
        self.assertEqual(list(schemas.validate_tree(root)), ["pivot_original/provenance.json"])

    def test_committed_results_tree_matches_its_schemas(self):
        """results/ as committed (E0 calibration, pivot provenance) passes."""
        self.assertEqual(schemas.validate_tree(schemas.RESULTS_ROOT), {})

    def test_committed_fixtures_match_their_schemas(self):
        """Every fixture WP13 builds under tests/fixtures/results/ obeys the contract."""
        if not FIXTURES.is_dir():
            self.skipTest("no results fixtures yet")
        self.assertEqual(schemas.validate_tree(FIXTURES), {})


class ProducerOutputTests(unittest.TestCase):
    """Producers that write without ``write_rows`` emit what their schema says.

    The end-to-end writer tests (web benchmark, live run, GPU runner) validate
    their real output files; these pin the column constants and the writers
    that have no end-to-end test here.
    """

    def test_gpu_runner_columns_are_the_schema(self):
        sys.path.insert(0, str(ROOT))
        from repro import gpu_disk
        self.assertEqual(gpu_disk.LSTM_COLUMNS, schemas.columns("gpu/disk_lstm.csv"))
        self.assertEqual(gpu_disk.BASELINE_COLUMNS,
                         schemas.columns("gpu/disk_depth_baselines.csv"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "disk_lstm.csv"
            # A skipped row as _lstm_rows writes it, then an ok one.
            skipped = {"trace": "msr_t", "corpus": "msr", "seed": 42,
                       "status": "skipped: training or scoring did not finish"}
            ok = {c: 1 for c in gpu_disk.LSTM_COLUMNS}
            ok.update(trace="msr_u", corpus="msr", status="ok", val_loss=[1.25, 1.2],
                      lstm_hit_ratio=0.5, device="cpu", gpu_contention=None)
            gpu_disk._write_csv(target, gpu_disk.LSTM_COLUMNS, [skipped, ok])
            back = schemas.validate_file(target, "gpu/disk_lstm.csv")
            self.assertIsNone(back[0]["lstm_hit_ratio"])
            self.assertEqual(back[1]["val_loss"], [1.25, 1.2])

    @unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
    def test_web_diagnostics_columns_are_the_schema(self):
        from webprefetch import diagnostics
        self.assertEqual(diagnostics.CALIBRATION_COLUMNS, schemas.columns("web/calibration.csv"))
        self.assertEqual(diagnostics.LEARNING_CURVE_COLUMNS,
                         schemas.columns("web/learning_curve.csv"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "the probe uses posix_fadvise")
    def test_live_disk_probe_validates(self):
        from webprefetch.live import objects
        with tempfile.TemporaryDirectory() as directory:
            meta = objects.build_store(range(40), directory, median=4096, sigma=0.0)
            probe = objects.probe(meta["store"], meta["index"], n=20, hinted_n=5,
                                  leads_ms=(0.0, 1.0), label="ssd")
            # Written exactly as webprefetch.live.run writes it.
            target = Path(directory) / "live" / "disk_probe_ssd.json"
            target.parent.mkdir()
            target.write_text(json.dumps(probe, indent=1) + "\n")
            self.assertEqual(schemas.validate_file(target)[0]["tag"], "MEASURED")


if __name__ == "__main__":
    unittest.main()

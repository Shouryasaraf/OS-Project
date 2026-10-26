"""Multi-corpus driver (E1) and its analysis, on tiny oracleGeneral fixtures.

The fixtures are uncompressed ``struct.pack`` files in a temp directory, laid
out like ``data/block-traces/<corpus>/``, so everything except the real-trace
checks runs under system python3 without zstandard. The real-trace checks
(the msr_hm_0 cross-check and the smoke timing) need ``RUN_SLOW=1``, the data
and zstandard.
"""

import importlib.util
import json
import os
import random
import struct
import sys
import tempfile
import time
import unittest
from itertools import chain
from pathlib import Path
from unittest import mock

from adaptive_prefetch import analysis, corpora, eda, multicorpus, sweep
from adaptive_prefetch.multicorpus import (CONFIGS, E2_ONLY_REASON, Job,
                                           belady_hit_ratio, build_jobs,
                                           read_sweep, row_key, run_job,
                                           run_jobs, run_survey)
from adaptive_prefetch.simulator import GATED_MODES, LatencyModel, replay_stream
from adaptive_prefetch.trace import Request, iter_oracle_chunks, load_csv
from prefetch_common import schemas

REPO = Path(__file__).resolve().parents[1]
HM0 = REPO / "data" / "block-traces" / "msr" / "msr_hm_0.oracleGeneral.zst"
HAVE_REAL = (os.environ.get("RUN_SLOW") == "1" and HM0.is_file()
             and importlib.util.find_spec("zstandard") is not None)

RECORD = struct.Struct("<IQIq")
#: Fixture traces: stem -> (corpus directory, records, size unit). Two of the
#: alibaba stems are a real twin pair (corpora.TWIN_GROUPS) on purpose.
FIXTURES = {
    "msr_fix_0": ("msr", 2400, "bytes"),
    "fiu_fix-110108-112108": ("fiu", 3000, "sectors"),
    "alibabaBlock_11": ("alibaba", 2000, "bytes"),
    "alibabaBlock_15": ("alibaba", 2000, "bytes"),
    "tencentBlock_9001": ("tencent", 1800, "bytes"),
}


def _records(seed: int, n: int, unit: str, tag: int = 0):
    """Sequential runs and random jumps, at most a few records per second."""
    rng = random.Random(seed)
    out, ts, offset = [], 1_000, 0
    page = 4096 if unit == "bytes" else 8  # 4 KiB in the file's own unit
    while len(out) < n:
        if rng.random() < 0.35:
            offset = rng.randrange(0, 40_000) * 4096
        run = rng.choice((1, 2, 4, 8, 16))
        same_second = unit == "sectors"  # FIU writes one record per page
        for _ in range(run):
            if len(out) >= n:
                break
            size = page * (1 if same_second else rng.choice((1, 2, 4)))
            out.append((ts, tag + offset, size, -1))
            offset += size * (512 if unit == "sectors" else 1)
            if not same_second or rng.random() < 0.3:
                ts += rng.choice((0, 1))
        ts += 1
    return out


def make_fixture_root(directory: Path) -> Path:
    root = Path(directory) / "block-traces"
    for index, (stem, (corpus, n, unit)) in enumerate(FIXTURES.items()):
        folder = root / corpus
        folder.mkdir(parents=True, exist_ok=True)
        tag = (130 << 40) if corpus in ("alibaba", "tencent") else 0
        data = b"".join(RECORD.pack(*r) for r in _records(index + 1, n, unit, tag))
        (folder / f"{stem}.oracleGeneral").write_bytes(data)
    return root


def tiny_config(**overrides) -> dict:
    spec = dict(description="test", traces="all", slices=((0, 1500, None),),
                merges=(("auto", 1024, None),), capacities=(256,),
                modes=("none", "sequential", "lstm", "guard", "depth_adaptive",
                       "deep", "linux_ra"),
                deep_depths=(0, 8), ra_max=(64,), belady=True)
    spec.update(overrides)
    return spec


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.root = make_fixture_root(cls.tmp)
        cls.paths = corpora.discover_block_traces(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def plan(self, config="T", **overrides):
        with mock.patch.dict(CONFIGS, {config: tiny_config(**overrides)}):
            specs = multicorpus.plan_slices(config, self.root)
            survey = run_survey(specs, cache_dir=None)
            return build_jobs((config,), root=self.root, survey=survey)


class PlanningTests(FixtureCase):
    def test_paper_subset_has_21_traces_including_msr_prn_0(self):
        self.assertEqual(len(corpora.PAPER_SUBSET), 21)
        self.assertEqual(len(set(corpora.PAPER_SUBSET)), 21)
        self.assertIn("msr_prn_0", corpora.PAPER_SUBSET)

    def test_c1_covers_every_ranking_mode_and_skips_e2_only_modes(self):
        policies = multicorpus.config_policies(CONFIGS["C1"], 2048)
        labels = [multicorpus.policy_label(*p) for p in policies]
        for mode in sweep.RANKING_MODES:
            self.assertTrue(any(l == mode or l.startswith(mode + "@") for l in labels), mode)
        self.assertEqual([l for l in labels if l.startswith("deep@")],
                         ["deep@8", "deep@32", "deep@256"])
        self.assertEqual([l for l in labels if l.startswith("linux_ra@")],
                         ["linux_ra@64", "linux_ra@256", "linux_ra@1024"])
        with mock.patch.dict(CONFIGS, {"C1x": dict(CONFIGS["C1"], traces="all")}):
            survey = run_survey(multicorpus.plan_slices("C1x", self.root), cache_dir=None)
            jobs = build_jobs(("C1x",), root=self.root, survey=survey)
        per_cell = len(policies) + sum(1 for m, _, _ in policies if m in GATED_MODES)
        self.assertEqual(len(jobs), len(FIXTURES) * 2 * per_cell)
        for job in jobs:
            if job.mode in multicorpus.E2_ONLY:
                self.assertEqual(job.skip_reason, E2_ONLY_REASON)
            else:
                self.assertIsNone(job.skip_reason)
            self.assertEqual(job.latency == "break_even",
                             job.mode in GATED_MODES and job.latency != "assumed")
            self.assertEqual(job.belady, job.mode == "none")
            self.assertEqual(job.depth != 0, job.mode == "deep")
            self.assertEqual(job.max_blocks is not None, job.mode == "linux_ra")
        gated = {(j.trace, j.capacity, j.mode) for j in jobs if j.latency == "break_even"}
        self.assertEqual(len(gated), len(FIXTURES) * 2 * len(GATED_MODES))

    def test_keys_are_unique_and_match_the_schema_key(self):
        jobs = self.plan()
        keys = [job.key() for job in jobs]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys[0]), len(schemas.SWEEP_KEY))

    def test_slice_rules_mark_short_and_long_slices_na(self):
        jobs = self.plan(slices=((900, 1500, "full"), (0, None, ("whole_if_at_most", 2100)),
                                 (10**7, 100, None)))
        by = {}
        for job in jobs:
            by.setdefault((job.trace, job.skip_requests, job.max_requests), set()).add(
                job.skip_reason or "run")
        # msr_fix_0 has 2400 requests: slice 2 [900, 2400) is full, whole file > 2100.
        self.assertEqual(by[("msr_fix_0", 900, 1500)], {"run", E2_ONLY_REASON})
        self.assertTrue(all(r.startswith("n/a: whole file") for r in by[("msr_fix_0", 0, None)]))
        # tencent has 1800 requests: no full second slice, but a whole-file row.
        self.assertTrue(all(r.startswith("n/a: no full slice")
                            for r in by[("tencentBlock_9001", 900, 1500)]))
        self.assertEqual(by[("tencentBlock_9001", 0, None)], {"run", E2_ONLY_REASON})
        self.assertTrue(all(r.startswith("n/a: slice is empty")
                            for r in by[("msr_fix_0", 10**7, 100)]))

    def test_a_rule_without_a_survey_refuses_to_plan(self):
        with mock.patch.dict(CONFIGS, {"T": tiny_config(slices=((5, 5, "full"),))}):
            with self.assertRaisesRegex(ValueError, "survey"):
                build_jobs(("T",), root=self.root)

    def test_merge_rule_limited_to_one_corpus(self):
        jobs = self.plan(merges=(("auto", 1024, None), ("none", None, "fiu")))
        none_traces = {j.trace for j in jobs if j.merge == "none"}
        self.assertEqual(none_traces, {"fiu_fix-110108-112108"})

    def test_survey_cache_is_reused_and_invalidated_by_a_changed_file(self):
        cache = self.tmp / "cache"
        with mock.patch.dict(CONFIGS, {"T": tiny_config()}):
            specs = multicorpus.plan_slices("T", self.root)
        first = run_survey(specs, cache_dir=cache)
        with mock.patch.object(multicorpus, "survey_slice",
                               side_effect=AssertionError("re-surveyed")):
            self.assertEqual(run_survey(specs, cache_dir=cache), first)
        target = self.paths["msr_fix_0"]
        saved = target.read_bytes()
        try:
            # Shorter than the 1500-request slice, so a stale cache hit
            # would still say 1500.
            target.write_bytes(saved[: RECORD.size * 1000])
            again = run_survey(specs, cache_dir=cache)
            key = multicorpus.survey_key(next(s for s in specs if s.trace == "msr_fix_0"))
            self.assertEqual(first[key]["requests"], 1500)
            self.assertEqual(again[key]["requests"], 1000)
            self.assertEqual(multicorpus.load_survey(cache)[key]["requests"], 1000)
        finally:
            target.write_bytes(saved)


class RunTests(FixtureCase):
    def test_row_equals_a_direct_replay_of_the_same_slice(self):
        job = next(j for j in self.plan(slices=((0, 800, None),))
                   if j.trace == "fiu_fix-110108-112108" and j.mode == "linux_ra")
        row = run_job(job)
        schemas.validate_row("disk/sweep.jsonl", row)
        requests = list(chain(*iter_oracle_chunks(job.path, 1000, max_requests=800)))
        metrics = replay_stream(iter([requests]), capacity=256, mode="linux_ra",
                                latency_model=LatencyModel(5, 100, 50),
                                linux_ra_max_blocks=64, read_ahead_depth=0)
        self.assertEqual(row["requests"], 800)
        self.assertEqual(row["blocks"], sum(r.size_blocks for r in requests))
        self.assertEqual(row["hit_ratio"], metrics.hit_ratio)
        self.assertEqual(row["total_prefetches"], metrics.prefetches)
        self.assertEqual(row["device_ios"], metrics.device_ios)
        self.assertEqual(row["size_unit"], "sectors")
        self.assertFalse(row["ops_known"])
        self.assertEqual((row["hit_us"], row["miss_us"], row["prefetch_us"]), (5.0, 100.0, 50.0))
        self.assertIsNone(row["belady_hit_ratio"])

    def test_skipped_rows_carry_no_metrics(self):
        job = next(j for j in self.plan() if j.mode == "lstm")
        row = run_job(job)
        self.assertEqual(row["status"], f"skipped: {E2_ONLY_REASON}")
        for column in ("hit_ratio", "requests", "device_ios", "belady_hit_ratio"):
            self.assertIsNone(row[column])

    def test_break_even_rows_use_the_measured_shape(self):
        job = next(j for j in self.plan() if j.mode == "guard" and j.latency == "break_even")
        row = run_job(job)
        self.assertAlmostEqual(row["break_even"], 100 / 99)
        self.assertEqual((row["hit_us"], row["prefetch_us"]), (1.0, 100.0))

    def test_resume_skips_done_keys_and_repairs_a_torn_line(self):
        jobs = self.plan()
        out = self.tmp / "resume" / "sweep.jsonl"
        first = run_jobs(jobs, out, max_jobs=7)
        self.assertEqual(first["ran"], 7)
        with open(out, "ab") as handle:  # a kill in the middle of a write
            handle.write(b'{"config": "T", "trace": "msr_fi')
        second = run_jobs(jobs, out, workers=2)
        self.assertEqual(second["already_done"], 7)
        self.assertEqual(second["ran"], len(jobs) - 7)
        rows = read_sweep(out)
        keys = [row_key(r) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), {j.key() for j in jobs})
        third = run_jobs(jobs, out)
        self.assertEqual((third["ran"], third["duplicates_in_file"]), (0, 0))
        self.assertEqual(len(schemas.read_rows("disk/sweep.jsonl", out)), len(jobs))

    def test_guard_equals_depth_adaptive_at_break_even(self):
        rows = [run_job(j) for j in self.plan() if j.mode in ("guard", "depth_adaptive")]
        report = analysis.checks(rows)["guard_eq_depth_adaptive_at_break_even"]
        self.assertEqual(report["pairs"], len(FIXTURES))
        self.assertEqual(report["identical"], len(FIXTURES))


class BeladyTests(unittest.TestCase):
    def test_matches_the_eda_reference_with_and_without_numpy(self):
        rng = random.Random(7)
        backends = [False] + ([True] if importlib.util.find_spec("numpy") else [])
        for trial in range(25):
            requests = [Request(i, rng.randrange(0, 200), rng.randrange(1, 9), "R")
                        for i in range(rng.randrange(1, 300))]
            lbas = [r.lba for r in requests]
            sizes = [r.size_blocks for r in requests]
            for capacity in (1, 4, 32, 1000):
                expected = eda.belady_demand_hit_ratio(requests, capacity)
                for use_numpy in backends:
                    self.assertAlmostEqual(
                        belady_hit_ratio(lbas, sizes, capacity, use_numpy=use_numpy),
                        expected, places=12, msg=(trial, capacity, use_numpy))

    def test_empty_and_single_block_streams(self):
        backends = [False] + ([True] if importlib.util.find_spec("numpy") else [])
        for use_numpy in backends:
            self.assertEqual(belady_hit_ratio([], [], 4, use_numpy=use_numpy), 0.0)
            self.assertEqual(belady_hit_ratio([5], [1], 1, use_numpy=use_numpy), 0.0)
            self.assertEqual(belady_hit_ratio([5, 5], [1, 1], 1, use_numpy=use_numpy), 0.5)

    def test_heap_compaction_keeps_the_count(self):
        rng = random.Random(3)
        lbas = [rng.randrange(0, 64) for _ in range(5000)]
        sizes = [1] * len(lbas)
        requests = [Request(i, b, 1, "R") for i, b in enumerate(lbas)]
        self.assertEqual(belady_hit_ratio(lbas, sizes, 2, use_numpy=False, chunk=7),
                         eda.belady_demand_hit_ratio(requests, 2))

    def test_none_rows_carry_belady_and_it_is_not_below_lru(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_fixture_root(Path(tmp))
            with mock.patch.dict(CONFIGS, {"T": tiny_config()}):
                survey = run_survey(multicorpus.plan_slices("T", root), cache_dir=None)
                jobs = build_jobs(("T",), root=root, survey=survey)
            for job in jobs:
                if job.mode == "none":
                    row = run_job(job)
                    self.assertGreaterEqual(row["belady_hit_ratio"], row["hit_ratio"])
                    self.assertLessEqual(row["belady_hit_ratio"], 1.0)


def _row(trace, policy, hit, *, corpus=None, prefetches=0, unused=0, ios=100,
         moved=1000, config="C1", capacity=2048, latency="assumed", status="ok",
         skip=0, max_requests=1_000_000, merge="auto", mms=1024, belady=None):
    mode, _, arg = policy.partition("@")
    return {"config": config, "trace": trace,
            "corpus": corpus or corpora.CORPUS_OF(trace),
            "skip_requests": skip, "max_requests": max_requests, "merge": merge,
            "max_merge_sectors": mms, "capacity": capacity, "mode": mode,
            "depth": int(arg) if mode == "deep" else 0,
            "max_blocks": int(arg) if mode == "linux_ra" else None,
            "latency": latency, "status": status, "read_blocks": 1000,
            "hit_ratio": hit, "precision": (prefetches - unused) / prefetches if prefetches else 0.0,
            "total_prefetches": prefetches, "unused": unused,
            "device_ios": ios, "blocks_moved": moved, "belady_hit_ratio": belady}


class AnalysisTests(FixtureCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with mock.patch.dict(CONFIGS, {"T": tiny_config(capacities=(256, 1024))}):
            survey = run_survey(multicorpus.plan_slices("T", cls.root), cache_dir=None)
            jobs = build_jobs(("T",), root=cls.root, survey=survey)
        cls.rows = [run_job(job) for job in jobs]

    def test_outputs_match_their_schemas_and_are_deterministic(self):
        first = analysis.write_analysis(self.rows, self.tmp / "a1", reps=300)
        second = analysis.write_analysis(self.rows, self.tmp / "a2", reps=300, workers=2)
        for name in analysis.OUTPUTS:
            self.assertEqual(first[name].read_bytes(), second[name].read_bytes(), name)
            schemas.validate_file(first[name], f"disk/{name}")
        deltas = schemas.read_rows("disk/deltas.csv", first["deltas.csv"])
        self.assertTrue(deltas)
        for row in deltas:
            self.assertLessEqual(row["ci_lo"], row["ci_hi"])
            self.assertLessEqual(row["cluster_ci_lo"], row["cluster_ci_hi"])
            self.assertEqual(row["wins"] + row["losses"] + row["ties"], row["n"])
            self.assertEqual((row["reps"], row["seed"]), (300, 42))
        table = schemas.read_rows("disk/table_main.csv", first["table_main.csv"])
        policies = {r["policy"] for r in table}
        self.assertTrue({"none", "sequential", "deep@0", "deep@8", "linux_ra@64",
                         "belady_min"} <= policies)
        self.assertNotIn("lstm", policies)  # skipped rows are never scored
        gated = {r["policy"] for r in table if r["latency"] == "break_even"}
        self.assertEqual(gated, {"guard", "depth_adaptive"})
        report = analysis.checks(self.rows)
        self.assertEqual(report["deep0_eq_none"]["pairs"], report["deep0_eq_none"]["identical"])
        self.assertEqual(report["duplicate_keys"], 0)

    def test_twin_family_is_down_weighted_in_the_cluster_estimate(self):
        rows = [_row("msr_fix_0", "none", 0.1), _row("msr_fix_0", "deep@8", 0.4),
                _row("alibabaBlock_11", "none", 0.1), _row("alibabaBlock_11", "deep@8", 0.2),
                _row("alibabaBlock_15", "none", 0.1), _row("alibabaBlock_15", "deep@8", 0.2)]
        deltas = analysis.compute_deltas(analysis.group_rows(rows), reps=200)
        hit = next(d for d in deltas if d["metric"] == "hit_ratio" and d["corpus"] == "all")
        self.assertAlmostEqual(hit["estimate"], (0.3 + 0.1 + 0.1) / 3)
        # Twins 11 and 15 weigh 1/2 each, so the family counts as one trace.
        self.assertAlmostEqual(hit["cluster_estimate"], (0.3 + 0.1) / 2)
        self.assertEqual((hit["wins"], hit["losses"], hit["n"]), (3, 0, 3))

    def test_lower_is_better_metrics_count_wins_the_right_way(self):
        rows = [_row("msr_fix_0", "none", 0.1, ios=100),
                _row("msr_fix_0", "linux_ra@256", 0.3, prefetches=50, unused=5, ios=60)]
        deltas = analysis.compute_deltas(analysis.group_rows(rows), reps=50)
        ios = next(d for d in deltas if d["metric"] == "device_ios" and d["corpus"] == "all")
        self.assertEqual((ios["statistic"], ios["estimate"]), ("ratio", 0.6))
        self.assertEqual((ios["wins"], ios["losses"]), (1, 0))
        kappa = analysis.kappa_rows(analysis.group_rows(rows))
        self.assertEqual(kappa[0]["kappa_star"], 0.0)  # fewer I/Os, no extra blocks

    def test_cells_keep_slices_and_merge_caps_apart(self):
        self.assertEqual(analysis.cell_label(_row("msr_fix_0", "none", 0, config="C3",
                                                  skip=1_000_000)), "C3:skip=1000000")
        self.assertEqual(analysis.cell_label(_row("msr_fix_0", "none", 0, config="C3",
                                                  max_requests=None)), "C3:whole")
        self.assertEqual(analysis.cell_label(_row("msr_fix_0", "none", 0, config="C4",
                                                  mms=256)), "C4:mms=256")
        self.assertEqual(analysis.cell_label(_row("msr_fix_0", "none", 0, config="C2",
                                                  merge="same_second", mms=None)), "C2")

    def test_winners_count_ties_for_every_tied_policy(self):
        rows = [_row("msr_fix_0", "none", 0.2), _row("msr_fix_0", "deep@8", 0.5),
                _row("msr_fix_0", "linux_ra@256", 0.5),
                _row("tencentBlock_9001", "none", 0.1), _row("tencentBlock_9001", "deep@8", 0.3),
                _row("tencentBlock_9001", "linux_ra@256", 0.2)]
        won = {(w["policy"]): w for w in analysis.winners_rows(analysis.group_rows(rows))
               if w["metric"] == "hit_ratio" and w["corpus"] == "all"}
        self.assertEqual(won["deep@8"]["traces_won"], 2)
        self.assertEqual(won["linux_ra@256"]["traces_won"], 1)
        self.assertEqual(won["deep@8"]["rank_by_mean"], 1)
        self.assertIsNotNone(won["linux_ra@256"]["note"])
        self.assertEqual(won["none"]["traces_won"], 0)

    def test_break_even_policies_pair_with_assumed_baselines(self):
        rows = [_row("msr_fix_0", "none", 0.1), _row("msr_fix_0", "sequential", 0.2),
                _row("msr_fix_0", "guard", 0.15, latency="break_even")]
        deltas = analysis.compute_deltas(analysis.group_rows(rows), reps=50)
        guard = [d for d in deltas if d["policy"] == "guard" and d["corpus"] == "all"
                 and d["metric"] == "hit_ratio"]
        self.assertEqual({(d["baseline"], d["latency"]) for d in guard},
                         {("none", "break_even"), ("sequential", "break_even")})

    def test_checks_carry_an_overall_pass_flag(self):
        same = [_row("msr_fix_0", "guard", 0.2, latency="break_even"),
                _row("msr_fix_0", "depth_adaptive", 0.2, latency="break_even"),
                _row("msr_fix_0", "none", 0.1), _row("msr_fix_0", "deep@0", 0.1)]
        for row in same:
            row.update({f: row.get(f, 0) for f in analysis.IDENTITY_FIELDS})
        report = analysis.checks(same)
        self.assertEqual((report["pass"], report["failures"], report["not_checked"]),
                         (True, [], ["hm0_deep8_250k"]))
        # Nothing to compare is not a pass of that gate, and not a failure.
        self.assertEqual(analysis.checks([])["not_checked"],
                         ["guard_eq_depth_adaptive_at_break_even", "deep0_eq_none",
                          "hm0_deep8_250k"])
        differ = same[:1] + [dict(same[1], hit_ratio=0.3)] + same[2:]
        report = analysis.checks(differ)
        self.assertFalse(report["pass"])
        self.assertEqual(report["failures"], ["guard_eq_depth_adaptive_at_break_even"])
        skipped = _row("msr_fix_0", "lstm", None, status="skipped: evaluated in E2 only")
        report = analysis.checks(same + [skipped, dict(skipped)])
        self.assertEqual((report["pass"], report["failures"]), (False, ["duplicate_keys"]))
        bad_hm0 = _row("msr_hm_0", "deep@8", 0.36, prefetches=901487, max_requests=250_000,
                       config="XCHECK", merge="none", mms=None)
        bad_hm0.update({f: bad_hm0.get(f, 0) for f in analysis.IDENTITY_FIELDS})
        report = analysis.checks(same + [bad_hm0])
        self.assertEqual((report["pass"], report["failures"]), (False, ["hm0_deep8_250k"]))

    def test_hm0_check_and_prototype_crosscheck(self):
        hm0 = _row("msr_hm_0", "deep@8", 0.357230590985335, prefetches=901487,
                   max_requests=250_000, config="XCHECK", merge="none", mms=None)
        hm0["useful_prefetches"] = 1
        hm0["policy_switches"] = 0
        hm0["prefetch_ios"] = 7
        self.assertTrue(analysis.checks([hm0])["hm0_deep8_250k"][0]["pass"])
        cells = [{"trace": "msr_hm_0", "variant": "raw", "cap": 2048, "mode": "deep",
                  "depth": 8, "hit": 0.357230590985335, "prefetches": 901487, "useful": 1},
                 {"trace": "msr_hm_0", "variant": "raw", "cap": 2048, "mode": "deep",
                  "depth": 8, "hit": 0.36, "prefetches": 901487},
                 {"trace": "msr_hm_0", "variant": "merged", "cap": 2048, "mode": "deep",
                  "depth": 8, "hit": 0.1},
                 {"trace": "msr_hm_0", "variant": "raw", "cap": 2048, "mode": "linuxra",
                  "depth": 256, "hit": 0.1}]
        path = self.tmp / "sweep2.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in cells))
        report = analysis.crosscheck([hm0], sweep2=path)
        self.assertEqual((report["compared"], report["matched"], report["not_comparable"],
                          report["missing"]), (2, 1, 1, 1))


class ProfileTests(FixtureCase):
    def test_profile_rows_follow_the_schema_and_merge_fiu_pages(self):
        out = multicorpus.write_corpus_profile(self.root, self.tmp / "profile.csv")
        rows = {r["trace"]: r for r in schemas.read_rows("disk/corpus_profile.csv", out)}
        self.assertEqual(set(rows), set(FIXTURES))
        fiu = rows["fiu_fix-110108-112108"]
        self.assertEqual(fiu["records"], 3000)
        self.assertLess(fiu["requests"], 3000)  # same-second pages are re-assembled
        self.assertEqual(fiu["size_unit"], "sectors")
        msr = rows["msr_fix_0"]
        self.assertEqual((msr["records"], msr["requests"]), (2400, 2400))
        requests = load_csv(self.paths["msr_fix_0"], "oracle")
        sizes = sorted(r.size_blocks for r in requests)
        self.assertEqual(msr["median_request_blocks"], (sizes[1199] + sizes[1200]) / 2)
        self.assertEqual(rows["alibabaBlock_11"]["twin_group"],
                         "alibabaBlock_11+alibabaBlock_15")
        self.assertTrue(rows["alibabaBlock_11"]["tagged"])
        self.assertIsNone(msr["twin_group"])


class OracleSweepTraceTests(FixtureCase):
    """Follow-up #2: an oracle trace end to end through ``sweep.sweep_trace``."""

    def test_sweep_trace_on_an_oracle_file(self):
        path = self.paths["fiu_fix-110108-112108"]
        latency = LatencyModel(1.0, 100.0, 100.0)
        row = sweep.sweep_trace(path, "deep", latency_model=latency, capacity=256,
                                requests_per_trace=800, chunk_requests=1000)
        requests = load_csv(path, "oracle")[:800]
        metrics = replay_stream(iter([requests]), capacity=256, mode="deep",
                                latency_model=latency)
        self.assertEqual(set(row), set(sweep.ROW_COLUMNS))
        self.assertEqual((row["status"], row["format"], row["corpus"]), ("ok", "oracle", "fiu"))
        self.assertEqual((row["merge"], row["size_unit"]), ("pages", "sectors"))
        self.assertFalse(row["ops_known"])
        self.assertEqual(row["ts_resolution_s"], 1)
        self.assertEqual(row["requests"], 800)
        self.assertAlmostEqual(row["break_even"], 100 / 99)
        for column, value in (("hit_ratio", metrics.hit_ratio),
                              ("total_prefetches", metrics.prefetches),
                              ("device_ios", metrics.device_ios)):
            self.assertEqual(row[column], value, column)


class EntryPointTests(FixtureCase):
    def test_cli_dry_run_and_run(self):
        out = self.tmp / "cli" / "sweep.jsonl"
        with mock.patch.dict(CONFIGS, {"T": tiny_config(modes=("none", "deep"),
                                                       deep_depths=(8,))}):
            with mock.patch("builtins.print") as printed:
                multicorpus.main(["--config", "T", "--trace-root", str(self.root),
                                  "--out", str(out), "--workers", "1", "--quiet",
                                  "--cache-dir", str(self.tmp / "cli-cache"), "--dry-run"])
            self.assertEqual(json.loads(printed.call_args[0][0])["planned"], 2 * len(FIXTURES))
            self.assertFalse(out.exists())
            with mock.patch("builtins.print"):
                multicorpus.main(["--config", "T", "--trace-root", str(self.root),
                                  "--out", str(out), "--workers", "1", "--quiet",
                                  "--cache-dir", str(self.tmp / "cli-cache")])
        self.assertEqual(len(read_sweep(out)), 2 * len(FIXTURES))

    def test_datasets_formats_is_the_loader_tuple(self):
        from adaptive_prefetch import datasets, trace
        self.assertIs(datasets.FORMATS, trace.FORMATS)
        self.assertIn("oracle", datasets.FORMATS)

    def test_trace_head_equals_the_head_of_a_full_load(self):
        from adaptive_prefetch.datasets import trace_head
        path = self.tmp / "head.revised"
        path.write_text("".join(f"{i * 0.001:.6f} {'RS' if i % 3 else 'WS'} {i * 8} 8 "
                                f"seq {0.5 + i % 7:.3f} 0.01 0.0\n" for i in range(2500)))
        self.assertEqual(trace_head(path, 1234), load_csv(path, "revised")[:1234])
        self.assertEqual(len(trace_head(path, 10_000)), 2500)

    def test_repro_disk_full_run_is_unavailable_without_traces(self):
        sys.path.insert(0, str(REPO))
        import repro
        from repro import disk
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--quick", action="store_true")
        disk.add_arguments(parser)
        args = parser.parse_args(["--trace-root", str(self.tmp / "nothing")])
        self.assertIn("100 workers", disk.estimate(args))
        with mock.patch("builtins.print"):
            self.assertEqual(disk.run(args), repro.EXIT_UNAVAILABLE)


class ReproduceDiskTests(FixtureCase):
    def test_full_run_on_fixtures_writes_every_results_file(self):
        """``reproduce.py disk`` end to end: sweep, profile, analysis, provenance."""
        import subprocess
        results = self.tmp / "results" / "disk"
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        done = subprocess.run(
            [sys.executable, str(REPO / "reproduce.py"), "disk", "--configs", "C1",
             "--trace-root", str(self.root), "--results-dir", str(results),
             "--cache-dir", str(self.tmp / "repro-cache"), "--workers", "2",
             "--reps", "100"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        for name in ("sweep.jsonl", "corpus_profile.csv", *analysis.OUTPUTS):
            schemas.validate_file(results / name, f"disk/{name}")
        from prefetch_common.provenance import read_provenance, verify_provenance
        record = read_provenance(results)
        self.assertEqual(verify_provenance(results), [])
        checks = record["extra"]["checks"]
        self.assertEqual(checks["duplicate_keys"], 0)
        gate = checks["guard_eq_depth_adaptive_at_break_even"]
        self.assertEqual((gate["pairs"], gate["identical"]), (2 * len(FIXTURES),) * 2)
        self.assertEqual(len(record["inputs"]), len(FIXTURES))


@unittest.skipUnless(HAVE_REAL, "slow real-trace check: set RUN_SLOW=1 (needs data and zstandard)")
class RealTraceTests(unittest.TestCase):
    def test_hm0_deep8_through_the_driver(self):
        job = Job("C2", "msr_hm_0", str(HM0), "msr", 0, 250_000, "auto", 1024,
                  2048, "deep", 8, None, "assumed")
        row = run_job(job)
        self.assertEqual(round(row["hit_ratio"], 6), 0.357231)
        self.assertEqual(row["total_prefetches"], 901487)

    def test_smoke_config_finishes_in_under_three_minutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            counts = multicorpus.run(("smoke",), out=Path(tmp) / "sweep.jsonl",
                                     root=REPO / "data" / "block-traces", workers=3,
                                     cache_dir=None)
            self.assertLess(time.monotonic() - started, 180)
        self.assertEqual(counts["ran"], 9)


if __name__ == "__main__":
    unittest.main()

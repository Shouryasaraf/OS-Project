"""Web benchmark: per-capacity selection, comparisons, metrics, scorers, parity.

Fixture tests replay the committed SYNTHETIC ``data/samples/web_fixture.csv``
(``build_views(..., verify_source=False)``) with a tiny logistic ranker fit
on its own candidates; nothing here is research data. Tests on the real
prepared views skip when ``data/processed`` or ``models/web`` is absent, and
the full-split reproduction of ``$EVAL/pivot-code/exp/percap.json`` also
needs ``RUN_SLOW=1``.
"""

import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
# The web study needs the [web] extra; system python3 runs only the disk suite.
HAVE_WEB = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "pandas", "sklearn", "joblib"))
if HAVE_WEB:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from prefetch_common import schemas, stats, utility
    from webprefetch import benchmark, cli, diagnostics, scorers
    from webprefetch.benchmark import (FAMILIES, PairedSessions, cost_key, needed_policies,
                                       policy_grid, replay_policies, select_policies,
                                       transfer_selection, verdict)
    from webprefetch.data import build_views
    from webprefetch.demo_server import DemoData, run_parity, session_parity
    from webprefetch.index import fit_index
    from webprefetch.model import features, save_artifacts
    from webprefetch.simulator import replay_capacities

import repro  # noqa: E402
import repro.web as repro_web  # noqa: E402

FIXTURE = ROOT / "data" / "samples" / "web_fixture.csv"
PREPARED = ROOT / "data" / "processed" / "views.csv.gz"
MODELS = ROOT / "models" / "web"
RUN_SLOW = os.environ.get("RUN_SLOW") == "1"


#: Every row of $EVAL/pivot-code/exp/percap.json (selection on validation at
#: each capacity, test utility rounded to 5 places):
#: (cap, cost): (fixed, U, adaptive, U, gate, U, fixed0 U).
PERCAP = {
    (2, 0.0): ("fixed0", 0.10529, "adaptive_0.1", 0.10569,
               "gate_0.1", 0.10225, 0.10529),
    (2, 0.01): ("fixed0", 0.10529, "adaptive_0.1", 0.10564,
               "gate_0.1", 0.10195, 0.10529),
    (2, 0.05): ("fixed0", 0.10529, "adaptive_0.2", 0.10547,
               "gate_0.1", 0.10077, 0.10529),
    (2, 0.25): ("fixed0", 0.10529, "adaptive_0.2", 0.10532,
               "gate_0.1", 0.09485, 0.10529),
    (4, 0.0): ("fixed2", 0.12145, "adaptive_0.005", 0.11949,
               "gate_0.01", 0.12041, 0.11218),
    (4, 0.01): ("fixed0", 0.11218, "adaptive_0.02", 0.11415,
               "gate_0.02", 0.11336, 0.11218),
    (4, 0.05): ("fixed0", 0.11218, "adaptive_0.1", 0.11258,
               "gate_0.1", 0.11177, 0.11218),
    (4, 0.25): ("fixed0", 0.11218, "adaptive_0.2", 0.11223,
               "gate_0.1", 0.10644, 0.11218),
    (8, 0.0): ("fixed4", 0.13172, "adaptive_0.002", 0.13081,
               "gate_0.01", 0.12634, 0.11453),
    (8, 0.01): ("fixed1", 0.11539, "adaptive_0.01", 0.11897,
               "gate_0.01", 0.1185, 0.11453),
    (8, 0.05): ("fixed0", 0.11453, "adaptive_0.05", 0.11502,
               "gate_0.1", 0.11462, 0.11453),
    (8, 0.25): ("fixed0", 0.11453, "adaptive_0.2", 0.11458,
               "gate_0.1", 0.10945, 0.11453),
    (16, 0.0): ("fixed8", 0.13953, "adaptive_0.002", 0.1406,
               "gate_0.01", 0.12849, 0.11536),
    (16, 0.01): ("fixed1", 0.11699, "adaptive_0.01", 0.12225,
               "gate_0.01", 0.12078, 0.11536),
    (16, 0.05): ("fixed0", 0.11536, "adaptive_0.05", 0.11592,
               "gate_0.1", 0.11557, 0.11536),
    (16, 0.25): ("fixed0", 0.11536, "adaptive_0.2", 0.11542,
               "gate_0.1", 0.11045, 0.11536),
    (32, 0.0): ("fixed8", 0.14367, "adaptive_0.002", 0.14367,
               "gate_0.01", 0.12924, 0.11563),
    (32, 0.01): ("fixed1", 0.11749, "adaptive_0.01", 0.12331,
               "gate_0.01", 0.12158, 0.11563),
    (32, 0.05): ("fixed0", 0.11563, "adaptive_0.05", 0.11621,
               "gate_0.1", 0.11587, 0.11563),
    (32, 0.25): ("fixed0", 0.11563, "adaptive_0.2", 0.11568,
               "gate_0.1", 0.11076, 0.11563),
}


def fixture_views():
    views, _ = build_views(FIXTURE, verify_source=False)
    return views


def tiny_ranker(views, index):
    """A logistic pipeline fit on the fixture's own candidate rows."""
    x, y = [], []
    for row in views.loc[views.split.le(2)].itertuples(index=False):
        for rank, (item, count) in enumerate(index.candidates(int(row.itemid))):
            x.append(features(row, item, count, rank, index))
            y.append(int(bool(row.eligible) and item == row.next_item))
    y[0] = 1 - y[0] if len(set(y)) == 1 else y[0]
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
        np.asarray(x, dtype=float), y)


def row(name, hits, issued, views=100):
    return {"policy": name, "hits": hits, "issued": issued, "views": views,
            "hit_ratio": hits / views, "issued_per_view": issued / views}


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class SelectionTests(unittest.TestCase):
    def grid(self):
        return policy_grid(feedback=False)

    def test_per_capacity_selection_differs_from_cap8_when_constructed_to(self):
        # At cap 2 prefetching only pollutes, so fixed0 wins; at cap 8 depth 4 pays.
        rows = {2: {"fixed0": row("fixed0", 10, 0), "fixed4": row("fixed4", 8, 300),
                    "adaptive_0.1": row("adaptive_0.1", 11, 5)},
                8: {"fixed0": row("fixed0", 10, 0), "fixed4": row("fixed4", 20, 300),
                    "adaptive_0.1": row("adaptive_0.1", 12, 5)}}
        selection = select_policies(rows, self.grid())
        key = cost_key(0.0)
        self.assertEqual(selection[2][key]["fixed"], "fixed0")
        self.assertEqual(selection[8][key]["fixed"], "fixed4")
        transfer = transfer_selection(selection, (2, 8))
        self.assertEqual(transfer[2], selection[8])
        self.assertNotEqual(transfer[2], selection[2])
        needed = needed_policies(selection, transfer)
        self.assertEqual(needed[2], ["adaptive_0.1", "fixed0", "fixed4"])

    def test_ties_go_to_fewer_prefetches_then_name(self):
        rows = {8: {"fixed1": row("fixed1", 10, 50), "fixed2": row("fixed2", 10, 20),
                    "fixed4": row("fixed4", 10, 20)}}
        self.assertEqual(select_policies(rows, self.grid())[8][cost_key(0.0)]["fixed"], "fixed4")

    def test_a_missing_family_is_left_out(self):
        selection = select_policies({8: {"fixed0": row("fixed0", 1, 0)}}, self.grid())
        self.assertEqual(set(selection[8][cost_key(0.01)]), {"fixed"})

    def test_labels_never_call_a_loser_to_no_prefetch_a_win(self):
        ci = stats.CI
        self.assertEqual(verdict("adaptive", ci(1, .5, 2, 9), ci(1, .5, 2, 9)), "adaptive wins")
        self.assertEqual(verdict("adaptive", ci(1, .5, 2, 9), ci(-1, -2, -.5, 9)),
                         "adaptive wins; loses to no prefetch")
        self.assertEqual(verdict("gate", ci(-1, -2, -.5, 9), ci(0, -1, 1, 9)), "gate loses")
        self.assertEqual(verdict("feedback", ci(0, -1, 1, 9), ci(0, -1, 1, 9)),
                         "feedback within noise")

    def test_grid_names_are_unique_and_cover_every_family(self):
        grid = policy_grid()
        self.assertEqual(len({p.name for p in grid}), len(grid))
        self.assertEqual({p.kind for p in grid}, set(FAMILIES))


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class PairedSessionTests(unittest.TestCase):
    def sessions(self, seed=5, n=300):
        rng = random.Random(seed)
        out = {}
        for name in ("a", "b", "c"):
            out[name] = {}
            for key in range(n):
                views = 1 + (key * 7) % 5
                hits = rng.randrange(views + 1)
                issued = rng.randrange(9)
                out[name][key] = [hits, issued, views, rng.randrange(issued + 1)]
        return out

    def paired(self, sessions, reps=200):
        paired = PairedSessions(reps, flush_every=2)
        for name, records in sessions.items():
            paired.add(name, benchmark._session_arrays(records))
        return paired

    def test_matches_the_shared_session_bootstrap(self):
        sessions = self.sessions()
        paired = self.paired(sessions)
        for cost in (0.0, 0.01, 0.25):
            for a, b in (("a", "b"), ("c", "a")):
                expected = stats.session_bootstrap(sessions, a, b, cost=cost, reps=200,
                                                   backend="numpy")
                actual = paired.ci(a, b, cost)
                self.assertEqual(actual.n, expected.n)
                for x, y in zip(actual[:3], expected[:3]):
                    if cost == 0:
                        self.assertEqual(x, y)
                    else:
                        self.assertAlmostEqual(x, y, places=12)

    def test_a_different_session_set_is_refused(self):
        sessions = self.sessions()
        paired = self.paired({"a": sessions["a"]})
        shorter = dict(list(sessions["b"].items())[:-1])
        with self.assertRaises(ValueError):
            paired.add("b", benchmark._session_arrays(shorter))


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class ScorerRegistryTests(unittest.TestCase):
    def test_parse_rejects_unknown_names_and_deduplicates(self):
        self.assertEqual(scorers.parse_scorers("logistic, trans_s0,logistic"),
                         ("logistic", "trans_s0"))
        self.assertEqual(scorers.parse_scorers(None), scorers.DEFAULT_SCORERS)
        with self.assertRaises(ValueError):
            scorers.parse_scorers("logistic,bogus")

    def test_optional_scorers_skip_without_artifacts(self):
        views = fixture_views()
        index = fit_index(views)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = (index, tiny_ranker(views, index))
            novel = scorers.build_scorer("logistic_novel", views, 3, tmp, artifacts=artifacts)
            self.assertTrue(novel.status.startswith("skipped: "), novel.status)
            self.assertIsNone(novel.scorer)
            seq = scorers.build_scorer("seq", views, 3, tmp, artifacts=artifacts)
            self.assertTrue(seq.status.startswith("skipped: "), seq.status)

    def test_transition_scorers_pick_index_and_order(self):
        views = fixture_views()
        online = scorers.build_scorer("trans_online", views, 3, ".")
        self.assertEqual((online.status, online.order), ("ok", "global"))
        self.assertTrue(getattr(online.index, "online", False))
        prior = scorers.build_scorer("trans_prior", views, 3, ".")
        self.assertEqual(prior.order, "session")
        self.assertGreaterEqual(sum(prior.index.source_total.values()),
                                sum(scorers.build_scorer("trans_s0", views, 3, ".")
                                    .index.source_total.values()))

    def test_table_scorer_looks_up_session_and_position(self):
        table = scorers.TableScorer({(4, 1): ((9, .5), (3, .1))})
        hit = pd.Series({"session_id": 4, "position": 1})
        miss = pd.Series({"session_id": 4, "position": 2})
        self.assertEqual(table.score(hit), [(9, .5), (3, .1)])
        self.assertEqual(table.score(miss), [])


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class FixtureBenchmarkTests(unittest.TestCase):
    """One end-to-end benchmark on the fixture, shared by the checks below."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.views = fixture_views()
        cls.index = fit_index(cls.views)
        cls.model = tiny_ranker(cls.views, cls.index)
        cls.models, cls.processed = root / "models", root / "processed"
        cls.outputs, cls.results = root / "outputs", root / "results"
        save_artifacts(cls.models, cls.index, cls.model, {})
        # evaluate_models reads two diagnostic rankers; the tiny one stands in.
        for name in ("ranker_no_context.joblib", "hist_gradient_boosting.joblib"):
            joblib.dump(cls.model, cls.models / name)
        cls.processed.mkdir()
        cls.views.to_csv(cls.processed / "views.csv.gz", index=False, compression="gzip")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.summary = benchmark.run_benchmark(
                cls.processed / "views.csv.gz", cls.models, cls.outputs, results_dir=cls.results,
                scorers=("logistic", "trans_online", "seq"), workers=1, reps=50)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_schema_files_validate(self):
        for name in ("policy_metrics.csv", "comparisons.csv"):
            schemas.read_rows(f"web/{name}", self.results / name)
        for name in ("calibration.csv", "dataset_profile.json", "think_time.json"):
            self.assertTrue((self.results / name).exists(), name)

    def test_unavailable_scorer_is_recorded_not_substituted(self):
        self.assertTrue(self.summary["scorers"]["seq"]["status"].startswith("skipped: "))
        rows = schemas.read_rows("web/policy_metrics.csv", self.results / "policy_metrics.csv")
        self.assertNotIn("seq", {r["scorer"] for r in rows})

    def test_comparisons_carry_every_baseline_with_cis(self):
        rows = schemas.read_rows("web/comparisons.csv", self.results / "comparisons.csv")
        self.assertEqual({r["selection"] for r in rows}, {"per_capacity", "capacity_transfer"})
        self.assertEqual({r["scorer"] for r in rows}, {"logistic", "trans_online"})
        for r in rows:
            for column in ("vs_best_fixed_same_cap", "vs_fixed0", "vs_transition_top1"):
                self.assertIsNotNone(r[column], column)
                self.assertLessEqual(r[f"{column}_lo"], r[f"{column}_hi"])
            self.assertAlmostEqual(r["vs_best_fixed_same_cap"],
                                   r["adaptive_utility"] - r["best_fixed_utility"], places=12)
            self.assertAlmostEqual(r["vs_fixed0"], r["adaptive_utility"] - r["fixed0_utility"],
                                   places=12)
            self.assertAlmostEqual(r["vs_transition_top1"],
                                   r["adaptive_utility"] - r["transition_top1_utility"], places=12)
        # The transfer setting reuses the cap-8 choice, so at cap 8 it is identical.
        key = ("scorer", "capacity", "cost", "adaptive_policy", "best_fixed_policy",
               "vs_best_fixed_same_cap", "vs_fixed0")
        per = {tuple(r[k] for k in key) for r in rows
               if r["selection"] == "per_capacity" and r["capacity"] == 8}
        moved = {tuple(r[k] for k in key) for r in rows
                 if r["selection"] == "capacity_transfer" and r["capacity"] == 8}
        self.assertEqual(per, moved)
        selection = self.summary["validation_selection_per_capacity"]["logistic"]
        for r in rows:
            if r["scorer"] == "logistic" and r["selection"] == "capacity_transfer":
                self.assertEqual(r["best_fixed_policy"], selection["8"][cost_key(r["cost"])]["fixed"])

    def test_policy_metrics_fill_c_star_and_utilities(self):
        rows = schemas.read_rows("web/policy_metrics.csv", self.results / "policy_metrics.csv")
        keys = [(r["split"], r["scorer"], r["capacity"], r["policy"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        none = {(r["split"], r["scorer"], r["capacity"]): r for r in rows if r["policy"] == "fixed0"}
        for r in rows:
            base = none[(r["split"], r["scorer"], r["capacity"])]
            if r["policy"] == "fixed0" or r["issued_per_view"] == base["issued_per_view"]:
                self.assertIsNone(r["c_star"])
            else:
                expected = (r["hit_ratio"] - base["hit_ratio"]) / (r["issued_per_view"] - base["issued_per_view"])
                self.assertAlmostEqual(r["c_star"], expected, places=12)
            self.assertAlmostEqual(r["U_0.01"], r["hit_ratio"] - 0.01 * r["issued_per_view"], places=12)
        # The transition baseline is recorded even when trans_s0 was not asked for.
        self.assertIn(("test", "trans_s0", 8, "fixed1"), keys)

    def test_legacy_summary_keeps_the_cap8_selection(self):
        per_capacity = self.summary["validation_selection_per_capacity"]["logistic"]
        self.assertEqual(self.summary["validation_selection"], per_capacity["8"])
        self.assertEqual(sorted(per_capacity), sorted(str(c) for c in benchmark.CAPACITIES))
        self.assertFalse(self.summary["partial_run"])
        self.assertTrue((self.outputs / "capacity_transfer.csv").exists())

    def test_calibration_brier_matches_a_direct_computation(self):
        rows = [r for r in pd.read_csv(self.results / "calibration.csv").to_dict("records")
                if r["scorer"] == "logistic" and r["split"] == "test"]
        spec = scorers.build_scorer("logistic", self.views, 3, self.models)
        test = self.views.loc[self.views.split.eq(3)]
        squared, count, positives, ranked_rows = 0.0, 0, 0, 0
        for view in test.itertuples(index=False):
            for item, p in spec.scorer.score(view, spec.index):
                y = int(bool(view.eligible) and item == view.next_item)
                squared += (p - y) ** 2
                count += 1
                positives += y
            ranked_rows += 1
        self.assertEqual(sum(r["count"] for r in rows), count)
        self.assertAlmostEqual(rows[0]["brier"], squared / count, places=12)
        rate = positives / count
        self.assertAlmostEqual(rows[0]["brier_skill_score"], 1 - (squared / count) / (rate * (1 - rate)),
                               places=10)
        # The same candidates the replay counts in its calibration bins.
        replayed = replay_capacities(test, spec.index, None, [policy_grid()[0]], (8,),
                                     scorer=spec.scorer)[8][1]["calibration_bins"]
        self.assertEqual([r["count"] for r in rows], [b[0] for b in replayed])

    def test_split_replay_equals_the_serial_replay(self):
        for name in ("logistic", "trans_online"):
            spec = scorers.build_scorer(name, self.views, 3, self.models)
            grid = policy_grid()
            test = self.views.loc[self.views.split.eq(3)]
            serial = replay_capacities(test, spec.index, None, grid, (2, 8), scorer=spec.scorer,
                                       order="global", with_sessions=True)
            mine = benchmark.replay_split(test, spec, grid, (2, 8), workers=3, session_caps=(2, 8))
            for cap in (2, 8):
                self.assertEqual([mine.rows[cap][p.name] for p in grid], serial[cap][0])
                for p in grid:
                    self.assertEqual(benchmark.session_records(mine.arrays[(cap, p.name)]),
                                     dict(serial[cap][1]["sessions"][p.name]), (name, cap, p.name))
            # Counts are exact; probability sums are added in another order.
            for ours, theirs in zip(mine.extra["calibration_bins"], serial[2][1]["calibration_bins"]):
                self.assertEqual((ours[0], ours[2]), (theirs[0], theirs[2]))
                self.assertAlmostEqual(ours[1], theirs[1], places=9)

    def test_feedback_pool_equals_one_process(self):
        spec = scorers.build_scorer("logistic", self.views, 3, self.models)
        grid = [p for p in policy_grid() if p.kind in ("fixed", "feedback")][:9]
        test = self.views.loc[self.views.split.eq(3)]
        one, _ = replay_policies(test, spec, grid, (2, 8), workers=1)
        two, _ = replay_policies(test, spec, grid, (2, 8), workers=3)
        strip = lambda rows: {c: {n: {k: v for k, v in r.items() if k != "depth_counts"}
                                  for n, r in by.items()} for c, by in rows.items()}
        self.assertEqual(strip(one), strip(two))

    def test_demo_parity_holds_at_every_selected_cost(self):
        data = DemoData(ROOT, models=self.models, processed=self.processed, outputs=self.outputs)
        cases = session_parity(data, data.ids, capacity=8)
        self.assertEqual({c["cost"] for c in cases}, {0.0, 0.01, 0.05, 0.25})
        self.assertTrue(all(case["match"] for case in cases), cases)
        selection = self.summary["validation_selection_per_capacity"]["logistic"]
        for case in cases:
            self.assertEqual(case["resolved_policy"], selection["8"][cost_key(case["cost"])]["adaptive"])
        record = run_parity(ROOT, self.results / "session_parity.json", models=self.models,
                            processed=self.processed, outputs=self.outputs, sessions=5)
        self.assertEqual(record["mismatches"], 0)
        self.assertEqual(json.loads((self.results / "session_parity.json").read_text())["format"],
                         "webprefetch.session_parity/2")

    def test_demo_adaptive_is_the_validated_policy_not_penalty_equals_cost(self):
        data = DemoData(ROOT, models=self.models, processed=self.processed, outputs=self.outputs)
        selection = self.summary["validation_selection_per_capacity"]["logistic"]
        session = data.ids[0]
        for cap in (2, 32):
            result = data.step({"session_id": session, "capacity": cap, "cost": 0.0, "policy": "adaptive"})
            self.assertEqual(result["resolved_policy"], selection[str(cap)][cost_key(0.0)]["adaptive"])
        plain = data.step({"session_id": session, "cost": 0.0, "policy": "adaptive_cost"})
        self.assertEqual((plain["resolved_policy"], plain["penalty"]), ("adaptive_cost", 0.0))
        with self.assertRaises(ValueError):
            data.step({"session_id": session, "cost": 0.03, "policy": "adaptive"})

    def test_decomposition_uses_the_validated_choices(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rows = benchmark.run_decomposition(self.processed / "views.csv.gz", self.models,
                                               self.outputs, self.results, reps=50)
        schemas.read_rows("web/decomposition.csv", self.results / "decomposition.csv")
        levers = {r["lever"] for r in rows}
        self.assertEqual(levers, {"ranker", "freshness", "depth", "controller", "feedback", "cache_scope"})
        selection = self.summary["validation_selection_per_capacity"]["logistic"]
        controller = [r for r in rows if r["lever"] == "controller" and r["capacity"] == 8]
        for r in controller:
            self.assertEqual(r["baseline"], selection["8"][cost_key(0.01)]["fixed"])
        scope = [r for r in rows if r["lever"] == "cache_scope"]
        self.assertEqual([(r["capacity"], r["cache_scope"], r["metric"]) for r in scope],
                         [(None, "visitor", "hit_ratio")])
        self.assertGreaterEqual(scope[0]["estimate"], 0.0)


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class DiagnosticTests(unittest.TestCase):
    def frame(self, rows):
        """Views from (session, timestamp, item, split) rows, prepared-view shaped."""
        frame = pd.DataFrame(rows, columns=["session_id", "timestamp", "itemid", "split"])
        frame["source_row"] = range(len(frame))
        ordered = frame.sort_values(["session_id", "timestamp"], kind="mergesort")
        group = ordered.groupby("session_id", sort=False)
        ordered["prev_item"] = group.itemid.shift(1).fillna(-1).astype(int)
        ordered["next_item"] = group.itemid.shift(-1).fillna(-1).astype(int)
        ordered["next_timestamp"] = group.timestamp.shift(-1).fillna(-1).astype(int)
        ordered["eligible"] = ordered.next_item.ge(0)
        return ordered.sort_values(["timestamp", "source_row"], kind="mergesort")

    def test_online_coverage_uses_only_transitions_already_observed(self):
        # 5 -> 6 is first seen in split 3 itself. The second test decision
        # (t=30) may use it; the first one (t=10) may not.
        views = self.frame([(1, 1, 1, 0), (1, 2, 2, 0), (2, 3, 1, 1), (2, 4, 2, 1),
                            (3, 5, 1, 2), (3, 6, 2, 2),
                            (4, 10, 5, 3), (4, 20, 6, 3), (5, 30, 5, 3), (5, 40, 6, 3)])
        rows = {(r["index"], r["k"]): r["coverage"]
                for r in diagnostics.freshness_rows(views, 3, "test")}
        # Item 6 never occurs before split 3, so no frozen index or popular
        # fallback can offer it.
        self.assertEqual(rows[("split0", 1)], 0.0)
        self.assertEqual(rows[("split012", None)], 0.0)
        self.assertEqual(rows[("online", 1)], 0.5)

    def test_coverage_grows_with_k(self):
        rows = diagnostics.freshness_rows(fixture_views(), 3, "test")
        for label in ("split0", "split012", "online"):
            values = [r["coverage"] for r in rows if r["index"] == label]
            self.assertEqual(values, sorted(values))

    def test_pair_kinds(self):
        views = self.frame([(1, 1, 7, 3), (1, 2, 7, 3), (1, 3, 8, 3), (1, 4, 7, 3), (1, 5, 9, 3)])
        kinds = diagnostics.pair_kinds(views, 3)
        self.assertEqual(kinds["kinds"], {"self_repeat": 1, "novel": 2, "earlier_in_session": 1})
        self.assertEqual(kinds["eligible"], 4)

    def test_think_time(self):
        views = self.frame([(1, 0, 1, 3), (1, 1500, 2, 3), (1, 1600, 3, 3)])
        result = diagnostics.think_time(views, 3)
        self.assertEqual(result["eligible"], 2)
        self.assertEqual(result["share_below_s"]["1"], 0.5)


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class CliTests(unittest.TestCase):
    def test_a_bounded_run_never_writes_the_tracked_results(self):
        parser = cli.build_parser()
        full = parser.parse_args(["benchmark"])
        self.assertEqual(cli._results_dir(full), cli.RESULTS)
        partial = parser.parse_args(["benchmark", "--test-views", "10", "--outputs", "x"])
        self.assertEqual(cli._results_dir(partial), Path("x") / "results")
        explicit = parser.parse_args(["benchmark", "--test-views", "10", "--results-dir", "r"])
        self.assertEqual(cli._results_dir(explicit), Path("r"))

    def test_every_subcommand_parses(self):
        for command in ("parity", "freshness", "decompose", "learning-curve", "benchmark"):
            self.assertEqual(cli.build_parser().parse_args([command]).command, command)


class ReproWebTests(unittest.TestCase):
    """Stdlib only: the reproduce step's arguments and its unavailable path."""

    def test_defaults_and_estimate(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--quick", action="store_true")
        repro_web.add_arguments(parser)
        args = parser.parse_args([])
        self.assertTrue(repro_web.IMPLEMENTED)
        self.assertIn("min", repro_web.estimate(args))
        self.assertIn("min", repro_web.estimate(parser.parse_args(["--quick"])))

    def test_missing_events_file_is_unavailable_not_a_fake_result(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--quick", action="store_true")
        repro_web.add_arguments(parser)
        with tempfile.TemporaryDirectory() as tmp:
            args = parser.parse_args(["--quick", "--raw", str(Path(tmp) / "absent.csv")])
            with contextlib.redirect_stdout(io.StringIO()) as out:
                status = repro_web.run(args)
        self.assertEqual(status, repro.EXIT_UNAVAILABLE)
        self.assertIn("absent.csv", out.getvalue())


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
@unittest.skipUnless(PREPARED.exists() and (MODELS / "ranker.joblib").exists(),
                     "real prepared views or web models absent")
@unittest.skipUnless(RUN_SLOW, "slow: set RUN_SLOW=1")
class RealDataTests(unittest.TestCase):
    def test_per_capacity_selection_reproduces_percap(self):
        """$EVAL/pivot-code/exp/percap.json, every (cap, cost), to 1e-5."""
        from webprefetch.model import load_artifacts
        views = pd.read_csv(PREPARED)
        artifacts = load_artifacts(MODELS)
        grid = [p for p in policy_grid(False) if p.kind in ("fixed", "adaptive", "gate")]
        rows = {}
        for label, split in benchmark.SPLITS.items():
            spec = scorers.build_scorer("logistic", views, split, MODELS, artifacts=artifacts)
            rows[label], _ = replay_policies(views.loc[views.split.eq(split)], spec, grid,
                                             benchmark.CAPACITIES, workers=min(64, os.cpu_count()))
        selection = select_policies(rows["validation"], grid)
        for (cap, cost), (fixed, fixed_u, adaptive, adaptive_u, gate, gate_u, none_u) in PERCAP.items():
            chosen = selection[cap][cost_key(cost)]
            self.assertEqual((chosen["fixed"], chosen["adaptive"], chosen["gate"]),
                             (fixed, adaptive, gate), (cap, cost))
            test = rows["test"][cap]
            for name, value in ((fixed, fixed_u), (adaptive, adaptive_u), (gate, gate_u),
                                ("fixed0", none_u)):
                self.assertAlmostEqual(utility.row_U(test[name], cost), value, delta=5e-6,
                                       msg=(cap, cost, name))

if __name__ == "__main__":
    unittest.main()

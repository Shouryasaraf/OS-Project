"""Track WRK: alternative web rankers (webprefetch.rankers).

Every test runs on the committed SYNTHETIC ``data/samples/web_fixture.csv``
(``build_views(..., verify_source=False)``); nothing here is research data.
The module needs the [web] extra, so system python3 skips these tests.
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
HAVE_WEB = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "pandas", "sklearn", "joblib", "scipy"))
if HAVE_WEB:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from prefetch_common import schemas, stats
    from webprefetch import benchmark, rankers
    from webprefetch.data import build_views
    from webprefetch.index import fit_index
    from webprefetch.model import features, save_artifacts
    from webprefetch.simulator import complete_views

import repro  # noqa: E402

FIXTURE = ROOT / "data" / "samples" / "web_fixture.csv"
TINY_TRIALS = {"logistic_tuned": 2, "hgb": 2, "rf": 1, "et": 1, "mlp": 1}


def fixture_views():
    views, _ = build_views(FIXTURE, verify_source=False)
    return views


def tiny_ranker(views, index):
    x, y = [], []
    for row in views.loc[views.split.le(2)].itertuples(index=False):
        for rank, (item, count) in enumerate(index.candidates(int(row.itemid))):
            x.append(features(row, item, count, rank, index))
            y.append(int(bool(row.eligible) and item == row.next_item))
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
        np.asarray(x, dtype=np.float32), y)


@unittest.skipUnless(HAVE_WEB, "needs the [web] extra")
class DesignAndRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.views = fixture_views()
        cls.index = fit_index(cls.views)
        cls.frame = cls.views.loc[cls.views.split.eq(2)]

    def test_design_is_train_rankers_matrix(self):
        # The exact loop of model.train_ranker, in the same row order.
        x, y = [], []
        for row in self.frame.itertuples(index=False):
            for rank, (item, count) in enumerate(self.index.candidates(int(row.itemid))):
                x.append(features(row, item, count, rank, self.index))
                y.append(int(bool(row.eligible) and item == row.next_item))
        for workers in (1, 3):
            d = rankers.design(self.frame, self.index, workers)
            np.testing.assert_array_equal(d.x, np.asarray(x, dtype=np.float32))
            np.testing.assert_array_equal(d.y, np.asarray(y, dtype=np.int8))
            self.assertEqual(int(d.lengths.sum()), len(y))
            self.assertEqual(len(d.lengths), len(self.frame))

    def test_sample_is_train_rankers_linspace(self):
        sample = rankers.sample_decisions(self.frame, 7)
        positions = np.linspace(0, len(self.frame) - 1, 7, dtype=np.int64)
        self.assertTrue(sample.equals(self.frame.iloc[positions]))
        self.assertIs(rankers.sample_decisions(self.frame, 0), self.frame)

    def test_rank_order_breaks_ties_by_item(self):
        probs = np.array([0.2, 0.5, 0.5, 0.1, 0.3, 0.3])
        items = np.array([9, 7, 3, 1, 8, 2])
        lengths = np.array([4, 2])
        order = rankers.rank_order(probs, items, lengths)
        self.assertEqual(items[order].tolist(), [3, 7, 9, 1, 2, 8])

    def test_ranking_summary_matches_a_direct_loop(self):
        d = rankers.design(self.frame, self.index)
        probs = np.random.default_rng(3).random(len(d.y))
        summary = rankers.ranking_summary(probs, d)
        hits = top1 = 0
        rr = 0.0
        for r, (a, b) in enumerate(zip(d.offsets[:-1], d.offsets[1:])):
            if not d.eligible[r]:
                continue
            ranked = sorted(zip(probs[a:b], d.items[a:b], d.y[a:b]), key=lambda t: (-t[0], t[1]))[:8]
            rank = next((n for n, t in enumerate(ranked, 1) if t[2]), None)
            if rank:
                hits += 1
                top1 += rank == 1
                rr += 1 / rank
        eligible = int(d.eligible.sum())
        self.assertAlmostEqual(summary["hit_at_1"], top1 / eligible)
        self.assertAlmostEqual(summary["coverage_at_8"], hits / eligible)
        self.assertAlmostEqual(summary["mrr_at_8"], rr / eligible)

    def test_array_scorer_returns_the_per_request_ranking(self):
        d = rankers.design(self.views.loc[self.views.split.le(1)], self.index)
        model = rankers.make_estimator("hgb", {"max_iter": 20, "max_leaf_nodes": 7,
                                               "min_samples_leaf": 5}).fit(d.x, d.y)
        full = rankers.design(self.frame, self.index)
        table = rankers.array_scorer(self.frame, full, rankers.predict(model, full.x))
        online = rankers.ModelScorer(model)
        for row in self.frame.itertuples(index=False):
            expected, actual = online.score(row, self.index), table.score(row)
            self.assertEqual([i for i, _ in actual], [i for i, _ in expected])
            np.testing.assert_allclose([p for _, p in actual], [p for _, p in expected], rtol=1e-12)

    def test_replay_through_the_table_equals_the_per_request_replay(self):
        d = rankers.design(self.views.loc[self.views.split.le(1)], self.index)
        model = rankers.make_estimator("rf", {"n_estimators": 5, "max_depth": 4}).fit(d.x, d.y)
        full = rankers.design(self.frame, self.index)
        table = rankers.array_scorer(self.frame, full, rankers.predict(model, full.x))
        from webprefetch.scorers import ScorerSpec
        grid = benchmark.policy_grid(feedback=False)
        a = benchmark.replay_split(self.frame, ScorerSpec("t", "ok", table, None), grid, (2, 8))
        b = benchmark.replay_split(self.frame, ScorerSpec("m", "ok", rankers.ModelScorer(model),
                                                          self.index), grid, (2, 8))
        for cap in (2, 8):
            for name in a.rows[cap]:
                for key in ("hits", "issued", "views"):
                    self.assertEqual(a.rows[cap][name][key], b.rows[cap][name][key])

    def test_every_family_fits_and_predicts_probabilities(self):
        d = rankers.design(self.views.loc[self.views.split.le(1)], self.index)
        for family in rankers.FAMILIES:
            params = rankers.sample_trials(family, 1)[0]
            if family in ("rf", "et"):
                params["n_estimators"] = 5
            model = rankers.make_estimator(family, params).fit(d.x, d.y)
            probs = rankers.predict(model, d.x)
            self.assertEqual(probs.shape, (len(d.y),), family)
            self.assertTrue(((probs >= 0) & (probs <= 1)).all(), family)

    def test_trials_are_seeded_and_inside_the_space(self):
        for family in rankers.FAMILIES:
            first, again = rankers.sample_trials(family), rankers.sample_trials(family)
            self.assertEqual(first, again)
            self.assertEqual(len(first), rankers.TRIALS[family])
            space = rankers.search_spaces()[family]
            for trial in first:
                self.assertEqual(set(trial), set(space))
                for key, value in trial.items():
                    if isinstance(space[key], list):
                        self.assertIn(value, space[key])
        with self.assertRaises(ValueError):
            rankers.make_estimator("knn", {})

    def test_selection_prefers_mrr_then_log_loss_then_trial(self):
        rows = [{"mrr_at_8": 0.1, "log_loss": 0.2, "trial": 0},
                {"mrr_at_8": 0.1, "log_loss": 0.1, "trial": 1},
                {"mrr_at_8": 0.05, "log_loss": 0.01, "trial": 2}]
        self.assertEqual(min(rows, key=rankers.selection_key)["trial"], 1)

    def test_random_search_marks_one_selected_row_per_family(self):
        d = rankers.design(self.views.loc[self.views.split.le(1)], self.index)
        tune = rankers.design(self.frame, self.index)
        rows = rankers.random_search(d, tune, ("logistic_tuned", "hgb"),
                                     {"logistic_tuned": 2, "hgb": 2}, workers=1, log=lambda *_: None)
        self.assertEqual(len(rows), 4)
        for family in ("logistic_tuned", "hgb"):
            mine = [r for r in rows if r["family"] == family]
            self.assertEqual(sum(r["selected"] for r in mine), 1)
            chosen = next(r for r in mine if r["selected"])
            self.assertEqual(chosen, min(mine, key=rankers.selection_key))


@unittest.skipUnless(HAVE_WEB, "needs the [web] extra")
class StatisticsTests(unittest.TestCase):
    def test_ratio_bootstrap_matches_the_shared_session_bootstrap(self):
        rng = np.random.default_rng(1)
        n = 120
        views = rng.integers(1, 6, n)
        a, b = rng.integers(0, 4, n), rng.integers(0, 4, n)
        sessions = {"a": {k: [int(a[k]), 0, int(views[k]), 0] for k in range(n)},
                    "b": {k: [int(b[k]), 0, int(views[k]), 0] for k in range(n)}}
        expected = stats.session_bootstrap(sessions, "a", "b", reps=300, backend="numpy")
        actual = rankers.RatioBootstrap(np.arange(n), reps=300).ci(
            a.astype(float), b.astype(float), views.astype(float))
        self.assertAlmostEqual(actual.estimate, expected.estimate, places=12)
        self.assertAlmostEqual(actual.low, expected.low, places=12)
        self.assertAlmostEqual(actual.high, expected.high, places=12)
        self.assertEqual(actual.n, expected.n)

    def test_three_way_verdicts(self):
        CI = stats.CI
        self.assertEqual(rankers.three_way(CI(1, 0.1, 2, 5)), "better")
        self.assertEqual(rankers.three_way(CI(-1, -2, -0.1, 5)), "worse")
        self.assertEqual(rankers.three_way(CI(0, -1, 1, 5)), "no difference detected")


@unittest.skipUnless(HAVE_WEB, "needs the [web] extra")
class ExperimentTests(unittest.TestCase):
    """One end-to-end run on the fixture, shared by the checks below."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.views = fixture_views()
        index = fit_index(cls.views)
        cls.models, cls.results = root / "models", root / "results"
        save_artifacts(cls.models, index, tiny_ranker(cls.views, index), {})
        (root / "processed").mkdir()
        cls.prepared = root / "processed" / "views.csv.gz"
        cls.views.to_csv(cls.prepared, index=False, compression="gzip")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.summary = rankers.run_experiment(
                cls.prepared, cls.models, cls.results, trials=TINY_TRIALS, train_decisions=0,
                tune_decisions=0, workers=1, reps=60, timing_requests=20, min_positives=1,
                log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_results_file_validates(self):
        for name in ("tuning.csv", "ranking_metrics.csv", "policy_metrics.csv",
                     "comparisons.csv", "summary.json"):
            schemas.validate_file(self.results / name, f"web_rankers/{name}")

    def test_every_family_has_a_saved_artifact_and_one_selected_trial(self):
        tuning = schemas.read_rows("web_rankers/tuning.csv", self.results / "tuning.csv")
        for family in rankers.FAMILIES:
            self.assertTrue((self.models / "rankers" / f"{family}.joblib").is_file())
            self.assertEqual(sum(r["selected"] for r in tuning if r["family"] == family), 1)
        self.assertEqual([r["trial"] for r in tuning if r["family"] == "logistic"], [-1])
        # The refit model is the one its trial scored.
        for family in rankers.FAMILIES:
            self.assertTrue(self.summary["costs"][family]["refit_matches_trial"], family)

    def test_comparisons_cover_the_preregistered_ids(self):
        rows = schemas.read_rows("web_rankers/comparisons.csv", self.results / "comparisons.csv")
        ids = {r["comparison"] for r in rows}
        self.assertEqual(ids, {"S1", "K1", "K2", "K3", "K4", "K5", "K6"})
        for row in rows:
            if row["estimate"] is not None:
                self.assertLessEqual(row["ci_lo"], row["ci_hi"])
                self.assertIn(row["verdict"], ("better", "worse", "no difference detected"))
        k2 = {(r["scorer"], r["capacity"], r["metric"]) for r in rows if r["comparison"] == "K2"}
        for family in rankers.FAMILIES:
            for cap in rankers.CAPACITIES:
                self.assertIn((family, cap, "hit_ratio"), k2)
                self.assertIn((family, cap, "utility_per_view"), k2)

    def test_k2_matches_the_policy_metrics(self):
        metrics = schemas.read_rows("web_rankers/policy_metrics.csv",
                                    self.results / "policy_metrics.csv")
        by = {(r["split"], r["scorer"], r["capacity"], r["policy"]): r for r in metrics}
        rows = schemas.read_rows("web_rankers/comparisons.csv", self.results / "comparisons.csv")
        for row in rows:
            if row["comparison"] == "K2" and row["metric"] == "hit_ratio":
                a = by[("test", row["scorer"], row["capacity"], "fixed1")]
                b = by[("test", "logistic", row["capacity"], "fixed1")]
                self.assertAlmostEqual(row["estimate"], a["hit_ratio"] - b["hit_ratio"], places=12)

    def test_policies_are_selected_on_validation_only(self):
        metrics = schemas.read_rows("web_rankers/policy_metrics.csv",
                                    self.results / "policy_metrics.csv")
        for name, by_cap in self.summary["selection"].items():
            for cap, by_cost in by_cap.items():
                chosen = by_cost[benchmark.cost_key(rankers.HEADLINE_COST)]["adaptive"]
                validation = {r["policy"]: r for r in metrics if r["split"] == "validation"
                              and r["scorer"] == name and r["capacity"] == int(cap)
                              and r["policy"].startswith("adaptive_")
                              and not r["policy"].startswith("adaptive_ca")}
                best = max(validation, key=lambda p: (validation[p]["U_0.01"],
                                                      -validation[p]["issued_per_view"], p))
                self.assertEqual(chosen, best)

    def test_session_ranking_agrees_with_the_replay_statistics(self):
        rows = schemas.read_rows("web_rankers/ranking_metrics.csv",
                                 self.results / "ranking_metrics.csv")
        test = {r["scorer"]: r for r in rows if r["split"] == "test"}
        self.assertEqual(set(test), {"logistic", *rankers.FAMILIES, "trans_s0", "trans_online"})
        # Recompute logistic's per-session counts directly and compare totals.
        from webprefetch.scorers import RANK_ROW, build_scorer, rank_table
        frame = complete_views(self.views.loc[self.views.split.eq(3)], 0).copy()
        frame[RANK_ROW] = np.arange(len(frame))
        spec = build_scorer("logistic", self.views, 3, self.models)
        counts = rankers.session_ranking(frame, rank_table(frame, spec))
        eligible = counts["eligible"].sum()
        self.assertEqual(eligible, test["logistic"]["eligible"])
        self.assertAlmostEqual(counts["hit_at_1"].sum() / eligible, test["logistic"]["hit_at_1"])
        self.assertAlmostEqual(counts["mrr_at_8"].sum() / eligible, test["logistic"]["mrr_at_8"])
        self.assertAlmostEqual(counts["novel_coverage_at_8"].sum() / counts["novel_eligible"].sum(),
                               test["logistic"]["novel_coverage_at_8"])

    def test_a_bounded_cli_run_refuses_the_tracked_results(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            rankers.main(["run", "--test-views", "100", "--results-dir",
                          str(ROOT / "results" / "web_rankers")])


class ReproRankersTests(unittest.TestCase):
    def test_registered_and_estimated(self):
        import argparse
        import repro.web_rankers as module
        self.assertIn("web_rankers", repro.STUDIES)
        parser = argparse.ArgumentParser()
        parser.add_argument("--quick", action="store_true")
        module.add_arguments(parser)
        args = parser.parse_args([])
        self.assertLessEqual(args.workers, 16)
        self.assertIn("min", module.estimate(args))
        self.assertIn("min", module.estimate(parser.parse_args(["--quick"])))


if __name__ == "__main__":
    unittest.main()

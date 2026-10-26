"""Pre-experiment wiring of E5 and E6: seq artifact paths, validated live policies, budget.

* ``scorers.seq`` finds the artifacts E5 writes (``models/web/seq/
  gru_<protocol>_s0``), picks the protocol that never trained on the
  evaluated split, and refuses one that did.
* ``reproduce.py web --gpu`` re-runs E4's decomposition with the seq lever,
  leaving the other levers' rows as they were, and keeps E4's provenance.
* The live plan builds its scorers through the benchmark's registry and
  replays the controllers E4 validated per capacity (``validated_*``), read
  from ``comparisons.csv``, with an explicit error when that file is absent.
* ``reproduce.py live`` runs the design of ``results/live/PREREGISTERED.md``
  amendment 1 by default.

Everything runs on the committed SYNTHETIC ``data/samples/web_fixture.csv``
or hand-written selection files labelled as such; none of it is research data.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

HAVE_WEB = all(importlib.util.find_spec(name) is not None
               for name in ("numpy", "pandas", "sklearn", "joblib"))
HAVE_TORCH = HAVE_WEB and importlib.util.find_spec("torch") is not None
FIXTURE = ROOT / "data" / "samples" / "web_fixture.csv"
LINUX = sys.platform.startswith("linux")

import repro.gpu as repro_gpu  # noqa: E402
import repro.live as repro_live  # noqa: E402
from repro import EXIT_UNAVAILABLE  # noqa: E402


def fixture_views():
    from webprefetch.data import build_views
    views, _ = build_views(FIXTURE, verify_source=False)
    return views


def tiny_ranker(views, index):
    """A logistic pipeline fit on the fixture's own candidate rows."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from webprefetch.model import features
    x, y = [], []
    for row in views.loc[views.split.le(2)].itertuples(index=False):
        for rank, (item, count) in enumerate(index.candidates(int(row.itemid))):
            x.append(features(row, item, count, rank, index))
            y.append(int(bool(row.eligible) and item == row.next_item))
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
        np.asarray(x, dtype=float), y)


def train_tiny_artifact(views, root, protocol):
    """A tiny CPU sequence model saved where E5 would save it."""
    from webprefetch import seqmodel as sm
    directory = sm.artifact_dir("gru", protocol, 0, root)
    with contextlib.redirect_stderr(io.StringIO()):
        sm.train_artifact(views, protocol=protocol, epochs=1, seed=0, device="cpu", batch=64,
                          d=8, h=16, out=directory)
    return directory


def on_cpu():
    """Keep artifact loading off the shared GPU in tests."""
    import torch
    from webprefetch import seqmodel as sm
    return mock.patch.object(sm, "resolve_device", lambda name="auto": torch.device("cpu"))


def comparison_row(scorer, capacity, cost, policy, fixed, selection="per_capacity"):
    """A schema-valid ``web/comparisons.csv`` row; only the names matter here."""
    return {"split": "test", "selection": selection, "scorer": scorer, "capacity": capacity,
            "cost": cost, "adaptive_policy": policy, "best_fixed_policy": fixed,
            "adaptive_utility": 0.1, "best_fixed_utility": 0.1, "fixed0_utility": 0.1,
            "transition_top1_utility": None,
            **{f"{c}{s}": 0.0 for c in ("vs_best_fixed_same_cap", "vs_fixed0")
               for s in ("", "_lo", "_hi")},
            "vs_transition_top1": None, "vs_transition_top1_lo": None,
            "vs_transition_top1_hi": None, "adaptive_hit_ratio": 0.2,
            "best_fixed_hit_ratio": 0.2, "adaptive_issued_per_view": 1.0,
            "best_fixed_issued_per_view": 1.0, "label": "fixture", "reps": 1, "seed": 42}


def write_selection(path, scorer="logistic"):
    """A hand-written selection file: the cap-8, cost-0.01 rows plus decoys."""
    from prefetch_common.schemas import write_rows
    rows = [comparison_row(scorer, 8, 0.01, "adaptive_0.02", "fixed4"),
            comparison_row(scorer, 8, 0.01, "feedback_0.01_eps0.002", "fixed4"),
            comparison_row(scorer, 8, 0.01, "gate_0.05", "fixed4"),
            comparison_row("trans_online", 8, 0.01, "adaptive_0.005", "fixed8"),
            # Decoys: another capacity, another cost, the cap-8 transfer setting.
            comparison_row(scorer, 4, 0.01, "adaptive_0.2", "fixed1"),
            comparison_row(scorer, 8, 0.0, "adaptive_0.002", "fixed8"),
            comparison_row(scorer, 16, 0.01, "adaptive_0.1", "fixed2", "capacity_transfer")]
    write_rows("web/comparisons.csv", rows, path)
    return path


# ------------------------------------------------------------------ seq (E5)

@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class SeqArtifactPathTests(unittest.TestCase):
    def test_a_root_resolves_per_split_and_an_artifact_is_used_as_is(self):
        from webprefetch import scorers
        root = Path("models/web/seq")
        self.assertEqual(scorers.seq_artifact(root, 2), root / "gru_validation_s0")
        self.assertEqual(scorers.seq_artifact(root, 3), root / "gru_final_s0")
        self.assertEqual(scorers.SEQ_SEED, 0)
        with self.assertRaises(ValueError):
            scorers.seq_artifact(root, 1)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}")
            self.assertEqual(scorers.seq_artifact(tmp, 2), Path(tmp))

    def test_default_path_is_the_e5_final_artifact(self):
        from webprefetch import scorers
        views = fixture_views()
        with tempfile.TemporaryDirectory() as tmp:
            spec = scorers.build_scorer("seq", views, 3, tmp, artifacts=(None, None))
            self.assertTrue(spec.status.startswith("skipped: "), spec.status)
            self.assertIn(str(Path(tmp) / "seq" / "gru_final_s0"), spec.status)

    @unittest.skipUnless(HAVE_TORCH, "the sequence model needs torch")
    def test_e5_artifacts_load_by_default_and_leakage_is_refused(self):
        import torch
        from webprefetch import scorers
        from webprefetch import seqmodel as sm
        torch.set_num_threads(2)
        views = fixture_views()
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            final = train_tiny_artifact(views, models / "seq", "final")
            train_tiny_artifact(views, models / "seq", "validation")
            with on_cpu():
                test = scorers.build_scorer("seq", views, 3, models, artifacts=(None, None))
                validation = scorers.build_scorer("seq", views, 2, models, artifacts=(None, None))
                leaked = scorers.build_scorer("seq", views, 2, models, artifacts=(None, None),
                                              seq_path=final)
                own = sm.SequenceScorer.load(final).precompute(views, (3,))
            self.assertEqual(test.status, "ok")
            self.assertIn("gru_final_s0", test.train_range)
            self.assertIn("novel_", test.train_range)
            self.assertEqual(validation.status, "ok")
            self.assertIn("gru_validation_s0", validation.train_range)
            self.assertTrue(leaked.status.startswith("skipped: "), leaked.status)
            self.assertIn("trained on splits", leaked.status)
            rows = views.loc[views.split.eq(3)].head(20)
            for row in rows.itertuples(index=False):
                self.assertEqual(test.scorer.score(row), own.score(row))


@unittest.skipUnless(HAVE_TORCH, "the decomposition re-run with seq needs torch")
class SeqDecompositionTests(unittest.TestCase):
    def test_seq_lever_is_added_and_every_other_row_is_unchanged(self):
        import joblib
        import torch
        from prefetch_common import schemas
        from webprefetch import benchmark
        from webprefetch.index import fit_index
        from webprefetch.model import save_artifacts
        torch.set_num_threads(2)
        views = fixture_views()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models, outputs, processed = root / "models", root / "outputs", root / "processed"
            index = fit_index(views)
            model = tiny_ranker(views, index)
            save_artifacts(models, index, model, {})
            for name in ("ranker_no_context.joblib", "hist_gradient_boosting.joblib"):
                joblib.dump(model, models / name)
            processed.mkdir()
            prepared = processed / "views.csv.gz"
            views.to_csv(prepared, index=False, compression="gzip")
            quiet = contextlib.redirect_stdout(io.StringIO())
            with quiet:
                benchmark.run_benchmark(prepared, models, outputs, results_dir=root / "e4",
                                        scorers=("logistic", "trans_s0", "trans_online"),
                                        workers=1, reps=50)
                before = benchmark.run_decomposition(prepared, models, outputs, root / "e4",
                                                     reps=50)
            train_tiny_artifact(views, models / "seq", "final")
            with on_cpu(), contextlib.redirect_stdout(io.StringIO()):
                after = benchmark.run_decomposition(prepared, models, outputs, root / "e5",
                                                    reps=50, seq_path=models / "seq")
            self.assertNotIn("sequence_model", {r["lever"] for r in before})
            seq_rows = [r for r in after if r["lever"] == "sequence_model"]
            self.assertEqual(len(seq_rows), 2)
            self.assertEqual({r["scorer"] for r in seq_rows}, {"seq"})
            self.assertEqual([r for r in after if r["lever"] != "sequence_model"], before)
            schemas.read_rows("web/decomposition.csv", root / "e5" / "decomposition.csv")


class ReproGpuWebTests(unittest.TestCase):
    def test_decompose_command_reads_the_seq_root_and_writes_results(self):
        command = repro_gpu._web_decompose_command(SimpleNamespace(workers=7),
                                                   Path("/v/data/processed/views.csv.gz"),
                                                   Path("/m"), Path("/o"))
        self.assertEqual(command[1:4], ["-m", "webprefetch", "decompose"])
        for flag, value in (("--seq-model", "/m/seq"), ("--outputs", "/o"),
                            ("--processed", "/v/data/processed"), ("--workers", "7"),
                            ("--results-dir", str(repro_gpu.REPO / "results" / "web"))):
            self.assertEqual(command[command.index(flag) + 1], value, flag)

    def test_e4_outputs_are_required_for_the_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(repro_gpu, "REPO", tmp):
                self.assertIn("missing", repro_gpu._e4_missing(tmp / "outputs"))
                (tmp / "outputs").mkdir()
                summary = tmp / "outputs" / "benchmark.json"
                summary.write_text(json.dumps({"partial_run": True}))
                self.assertIn("bounded", repro_gpu._e4_missing(tmp / "outputs"))
                summary.write_text(json.dumps({"partial_run": False,
                                               "validation_selection_per_capacity":
                                                   {"logistic": {}}}))
                self.assertIn("provenance", repro_gpu._e4_missing(tmp / "outputs"))
                (tmp / "results" / "web").mkdir(parents=True)
                (tmp / "results" / "web" / "provenance.json").write_text("{}")
                self.assertIsNone(repro_gpu._e4_missing(tmp / "outputs"))

    def _run(self, tmp, e4_missing):
        views = tmp / "views.csv.gz"
        views.write_text("")
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=0)

        out = io.StringIO()
        with mock.patch.object(repro_gpu, "_cuda_available", return_value=True), \
                mock.patch.object(repro_gpu, "_web_paths",
                                  return_value=(views, tmp / "raw", tmp / "m", tmp / "o")), \
                mock.patch.object(repro_gpu.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(repro_gpu, "_e4_missing", return_value=e4_missing), \
                mock.patch.object(repro_gpu, "_seq_lever_written", return_value=True), \
                mock.patch.object(repro_gpu, "_refresh_web_provenance",
                                  return_value=tmp / "p.json") as refresh, \
                contextlib.redirect_stdout(out):
            status = repro_gpu.run("web", SimpleNamespace(quick=False, workers=3))
        return status, commands, out.getvalue(), refresh

    def test_full_run_trains_then_reruns_the_decomposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, commands, out, refresh = self._run(Path(tmp), None)
        self.assertEqual(status, 0)
        self.assertEqual([c[3] for c in commands], ["experiment", "decompose"])
        self.assertIn("novel_recall_at_8", out)
        refresh.assert_called_once()

    def test_without_e4_the_run_is_reported_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, commands, out, refresh = self._run(Path(tmp), "benchmark.json is missing")
        self.assertEqual(status, EXIT_UNAVAILABLE)
        self.assertEqual([c[3] for c in commands], ["experiment"])
        self.assertIn("decomposition re-run", out)
        refresh.assert_not_called()

    @unittest.skipUnless(HAVE_WEB, "the refresh names the seq artifact via webprefetch.scorers")
    def test_provenance_refresh_hashes_the_new_file_and_keeps_e4(self):
        from prefetch_common import provenance
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            web = tmp / "results" / "web"
            web.mkdir(parents=True)
            (web / "decomposition.csv").write_text("e4\n")
            provenance.write_provenance(web, ["reproduce.py", "web"], study="web", gpu=False,
                                        notes="E4")
            (web / "decomposition.csv").write_text("e5\n")
            with mock.patch.object(repro_gpu, "REPO", tmp):
                for _ in range(2):  # a second E5 run keeps the original E4 record
                    repro_gpu._refresh_web_provenance(provenance.snapshot(), tmp / "v",
                                                      tmp / "m", tmp / "o")
            record = provenance.read_provenance(web)
            self.assertEqual(record["outputs"]["decomposition.csv"],
                             provenance.sha256_file(web / "decomposition.csv"))
            self.assertEqual(record["extra"]["e4_provenance"]["notes"], "E4")
            self.assertNotIn("extra", record["extra"]["e4_provenance"])


# ------------------------------------------------------------------ live (E6)

@unittest.skipUnless(HAVE_WEB, "the live plan needs numpy, pandas, scikit-learn and joblib")
class LiveSelectionTests(unittest.TestCase):
    def test_validated_names_resolve_per_capacity_and_cost(self):
        from webprefetch.live import selection
        with tempfile.TemporaryDirectory() as tmp:
            source = write_selection(Path(tmp) / "comparisons.csv")
            labels, record = selection.resolve(
                ["fixed0", "validated_adaptive", "validated_feedback",
                 "validated_adaptive@trans_online", "hint:validated_fixed"],
                source=source, capacity=8, cost=0.01)
        self.assertEqual(labels, ["fixed0", "adaptive_0.02", "feedback_0.01_eps0.002",
                                  "adaptive_0.005@trans_online", "hint:fixed4"])
        self.assertEqual((record["capacity"], record["cost"]), (8, 0.01))
        self.assertEqual(len(record["sha256"]), 64)
        self.assertEqual(record["resolved"]["validated_feedback"], "feedback_0.01_eps0.002")

    def test_missing_selection_is_an_explicit_error_and_concrete_names_need_none(self):
        from webprefetch.live import selection
        missing = Path("/nonexistent/comparisons.csv")
        self.assertEqual(selection.resolve(["fixed0", "adaptive_0.01"], source=missing,
                                           capacity=8), (["fixed0", "adaptive_0.01"], None))
        with self.assertRaises(selection.SelectionError) as caught:
            selection.resolve(["validated_adaptive"], source=missing, capacity=8)
        self.assertIn("no built-in fallback", str(caught.exception))
        with tempfile.TemporaryDirectory() as tmp:
            source = write_selection(Path(tmp) / "comparisons.csv")
            for bad in ("validated_adaptive_ca", "validated_feedback@trans_online",
                        "validated_bogus"):
                with self.assertRaises(selection.SelectionError):
                    selection.resolve([bad], source=source, capacity=8)
            with self.assertRaises(selection.SelectionError):  # no capacity-2 rows
                selection.resolve(["validated_adaptive"], source=source, capacity=2)
            with self.assertRaises(selection.SelectionError):  # resolves onto a listed name
                selection.resolve(["adaptive_0.02", "validated_adaptive"], source=source,
                                  capacity=8)

    def test_every_benchmark_policy_name_parses_to_the_same_policy(self):
        from webprefetch.benchmark import policy_grid
        from webprefetch.live.plan import parse_policy
        for policy in policy_grid(feedback=True):
            self.assertEqual(parse_policy(policy.name).policy, policy, policy.name)

    def test_make_scorers_uses_the_benchmark_registry(self):
        from webprefetch import scorers
        from webprefetch.live import plan
        views = fixture_views()
        built = plan.make_scorers(views, {"trans_s0", "trans_online"}, split=3)
        rows = list(views.loc[views.split.eq(3)].head(30).itertuples(index=False))
        for name in ("trans_s0", "trans_online"):
            spec = scorers.build_scorer(name, views, 3, ".")
            index, scorer = built[name]
            self.assertIs(type(index), type(spec.index))
            if getattr(index, "online", False):
                self.assertEqual(spec.order, "global")
                continue
            for row in rows:
                self.assertEqual(scorer.score(row, index), spec.scorer.score(row, spec.index))
        with self.assertRaises(ValueError):
            plan.make_scorers(views, {"logistic"})  # needs models_dir
        with self.assertRaises(ValueError):
            plan.make_scorers(views, {"seq"}, models_dir=".")
        with tempfile.TemporaryDirectory() as tmp:
            from webprefetch.index import fit_index
            from webprefetch.model import save_artifacts
            index = fit_index(views)
            save_artifacts(tmp, index, tiny_ranker(views, index), {})
            with self.assertRaises(ValueError) as caught:  # no relabelled ranker trained
                plan.make_scorers(views, {"logistic_novel"}, models_dir=tmp)
            self.assertIn("skipped", str(caught.exception))
            self.assertIn("logistic", plan.make_scorers(views, {"logistic"}, models_dir=tmp))

    def test_store_dirs_outside_the_ssd_and_sdb_are_refused(self):
        from webprefetch.live import run
        for bad in ("/mnt/9792e947-f73e-45c8-a77f-0ba5cd89e363/os-project-web",
                    "/mnt/c609e3de-d917-4ea6-8519-cd5416d1154d/x",
                    "/mnt/b18edc65-4395-4a85-976a-bed3428c073b/elsewhere"):
            with self.assertRaises(SystemExit):
                run.check_store_dir(bad)
        run.check_store_dir(run.HDD_STORE)
        run.check_store_dir(tempfile.gettempdir())
        run.check_store_dir(None)
        self.assertEqual(str(run.HDD_STORE), repro_live.HDD_STORE)


@unittest.skipUnless(HAVE_WEB, "the live plan needs numpy, pandas, scikit-learn and joblib")
@unittest.skipUnless(LINUX, "the origin's page-cache endpoints need Linux")
class LiveValidatedRunTests(unittest.TestCase):
    def _argv(self, directory, policies, selection_file):
        return ["--views", str(directory / "views.csv.gz"), "--policies", policies,
                "--baseline", "fixed0@trans_s0", "--sessions", "8", "--min-views", "1",
                "--capacity", "8", "--size-median", "4096", "--rtt-ms", "2", "--mbps", "1000",
                "--gap-cap-s", "0.01", "--ramp-s", "0.2", "--reps", "1",
                "--experiment", "smoke", "--no-probe", "--work-dir", str(directory / "work"),
                "--results-dir", str(directory / "results"), "--max-load", "1000",
                "--bootstrap-reps", "20", "--selection", str(selection_file)]

    def test_validated_policies_replay_with_parity_and_are_recorded(self):
        from prefetch_common.schemas import read_rows
        from webprefetch.live.run import main
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture_views().to_csv(directory / "views.csv.gz", index=False, compression="gzip")
            source = write_selection(directory / "comparisons.csv", scorer="trans_s0")
            # The baseline is listed too, as in the headline default.
            policies = "fixed0@trans_s0,validated_adaptive@trans_s0,validated_feedback@trans_s0"
            with mock.patch("sys.stdout"):
                self.assertEqual(main(self._argv(directory, policies, source)), 0)
            results = directory / "results"
            metrics = read_rows("live/live_metrics.csv", results / "live_metrics.csv")
            self.assertEqual({r["policy"] for r in metrics},
                             {"fixed0@trans_s0", "adaptive_0.02@trans_s0",
                              "feedback_0.01_eps0.002@trans_s0"})
            self.assertTrue(all("foreign_io_frac" in r for r in metrics))
            parity = read_rows("live/live_parity.csv", results / "live_parity.csv")
            self.assertEqual(sum(r["mismatched_sessions"] for r in parity), 0)
            run = next(iter(json.loads((results / "live_config.json").read_text())
                            ["runs"].values()))
            self.assertEqual(run["selection"]["resolved"],
                             {"validated_adaptive@trans_s0": "adaptive_0.02@trans_s0",
                              "validated_feedback@trans_s0": "feedback_0.01_eps0.002@trans_s0"})
            feedback = next(p for p in run["plan"]["params"]["policies"]
                            if p["kind"] == "feedback")
            self.assertEqual((feedback["parameter"], feedback["epsilon"]), (0.01, 0.002))
            self.assertIn("foreign I/O", (results / "live_report.md").read_text())
            # The selection record rides inside a run, so the WPF1 schema still holds.
            from prefetch_common.schemas import validate_file
            validate_file(results / "live_config.json", "live/live_config.json")

    def test_a_missing_selection_stops_the_run_before_any_output(self):
        from webprefetch.live.run import main
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            argv = self._argv(directory, "validated_adaptive@trans_s0",
                              directory / "absent.csv")
            with mock.patch("sys.stdout"), self.assertRaises(SystemExit) as caught:
                main(argv)
            self.assertIn("no built-in fallback", str(caught.exception))
            self.assertFalse((directory / "results").exists())
            self.assertFalse((directory / "work").exists())


class ReproLiveBudgetTests(unittest.TestCase):
    def _args(self, *extra):
        import reproduce
        _, parsers = reproduce.build_parser()
        return parsers["live"].parse_args(["--store-dir", repro_live.HDD_STORE, *extra])

    def test_default_design_is_amendment_1(self):
        from webprefetch.live import plan
        self.assertEqual(repro_live.HEADLINE, ",".join(plan.HEADLINE_POLICIES))
        args = self._args()
        self.assertEqual(repro_live.plan_passes(args),
                         [("headline/ssd", 18), ("headline/hdd", 18), ("fifo/ssd", 3),
                          ("hint/hdd", 3)])
        steps = dict(repro_live._steps(args))
        for label, argv in steps.items():
            self.assertEqual(argv[argv.index("--selection") + 1], "results/web/comparisons.csv")
            self.assertEqual(argv[argv.index("--sessions") + 1], "2000")
        for label in ("headline/hdd", "hint/hdd"):
            argv = steps[label]
            self.assertEqual(argv[argv.index("--store-dir") + 1], repro_live.HDD_STORE)
            self.assertEqual(argv[argv.index("--store-label") + 1], "hdd")
        self.assertIn("--no-probe", steps["fifo/ssd"])
        self.assertNotIn("--no-probe", steps["headline/ssd"])
        text = repro_live.estimate(args)
        self.assertIn("42 passes", text)
        hours = float(text.split("about ")[1].split(" h")[0])
        self.assertLessEqual(hours, 3.3)

    def test_the_amendment_is_registered(self):
        text = (ROOT / "results" / "live" / "PREREGISTERED.md").read_text()
        log = text[text.index("## Amendment log"):]
        self.assertIn("### Amendment 1 (2026-10-26): runtime budget", log)
        for phrase in ("results/web/comparisons.csv", "42 in all", "no fallback"):
            self.assertIn(phrase, log)

    def test_validated_policies_need_e4_before_any_pass(self):
        args = self._args("--selection", "/nonexistent/comparisons.csv")
        args.quick = False
        self.assertIn("no built-in fallback", repro_live._selection_missing(args))
        concrete = self._args("--policies", "fixed0,fixed4", "--selection", "/nonexistent")
        self.assertIsNone(repro_live._selection_missing(concrete))
        with mock.patch.object(repro_live, "_inputs_missing", return_value=None), \
                mock.patch("webprefetch.live.run.main") as live_main, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(repro_live.run(args), EXIT_UNAVAILABLE)
        live_main.assert_not_called()
        self.assertIn("comparisons.csv", out.getvalue())


if __name__ == "__main__":
    unittest.main()

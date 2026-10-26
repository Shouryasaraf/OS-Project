"""The web GPU sequence model: windows, vocabulary, masking, table parity, artifacts.

Everything except the ``slow`` class runs on the CPU with a tiny vocabulary,
on the committed SYNTHETIC ``data/samples/web_fixture.csv`` or on hand-built
toy sessions; none of it is research data. The real-data checks need
``RUN_SLOW=1`` and ``data/raw/events.csv`` (or prepared views); the GPU smoke
run also needs CUDA.
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
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
if HAVE_WEB:
    import numpy as np
    import pandas as pd

    from webprefetch import seqmodel as sm
    from webprefetch.data import build_views
    from webprefetch.index import fit_index
    from webprefetch.simulator import TransitionScorer, replay_capacities
if HAVE_TORCH:
    import torch

FIXTURE = ROOT / "data" / "samples" / "web_fixture.csv"
RAW = ROOT / "data" / "raw" / "events.csv"
PREPARED = ROOT / "data" / "processed" / "views.csv.gz"
RUN_SLOW = os.environ.get("RUN_SLOW") == "1"
TINY = {"d": 8, "h": 16}


def fixture_views():
    views, _ = build_views(FIXTURE, verify_source=False)
    return views


def toy_views(chains, sessions_per_split, splits=(0, 1, 2, 3)):
    """Prepared-view records for repeated deterministic chains (A -> B -> C).

    Every session is one chain; sessions of each split lie in their own time
    band, so the chronological protocol cuts behave as on real data.
    """
    rows, session = [], 0
    for split in splits:
        for n in range(sessions_per_split):
            chain = chains[n % len(chains)]
            start = split * 10_000_000 + n * 10_000
            for position, item in enumerate(chain):
                rows.append({"session_id": session, "visitorid": session,
                             "timestamp": start + position * 1000, "itemid": item,
                             "split": split, "position": position, "cart_count": 0,
                             "gap_ms": 1000 if position else 0})
            session += 1
    frame = pd.DataFrame(rows)
    frame["source_row"] = range(len(frame))
    group = frame.groupby("session_id", sort=False)
    frame["prev_item"] = group.itemid.shift(1).fillna(-1).astype(int)
    frame["next_item"] = group.itemid.shift(-1).fillna(-1).astype(int)
    frame["next_timestamp"] = group.timestamp.shift(-1).fillna(-1).astype(int)
    frame["eligible"] = frame.next_item.ge(0)
    return frame.sort_values(["timestamp", "source_row"], kind="mergesort").reset_index(drop=True)


def tiny_scorer(views, protocol="validation", epochs=2, seed=0, mask="history", **kwargs):
    with contextlib.redirect_stderr(io.StringIO()):
        return sm.train_artifact(views, protocol=protocol, epochs=epochs, seed=seed,
                                 device="cpu", mask=mask, batch=64, **{**TINY, **kwargs})


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class WindowAndVocabTests(unittest.TestCase):
    def test_no_item_seen_only_in_split_two_or_later_is_in_the_vocabulary(self):
        views = fixture_views()
        # The fixture's catalogue is small, so give splits 2 and 3 a new item each.
        for split, item in ((2, 999_998), (3, 999_999)):
            views.loc[views.index[views.split.eq(split)][0], "itemid"] = item
        vocab = sm.build_vocab(views, (0, 1))
        early = set(views.loc[views.split.le(1), "itemid"])
        late_only = set(views.loc[views.split.ge(2), "itemid"]) - early
        self.assertEqual(late_only, {999_998, 999_999})
        self.assertFalse(late_only & set(vocab.tolist()))
        self.assertEqual(set(vocab.tolist()), early)
        # Such an item encodes as OOV, never as a trainable class.
        self.assertTrue((sm.encode(vocab, sorted(late_only)) == sm.OOV).all())

    def test_encode_and_decode_round_trip(self):
        vocab = np.array([5, 9, 40])
        ids = sm.encode(vocab, [9, 5, 41, 40, 0])
        self.assertEqual(ids.tolist(), [4, 3, sm.OOV, 5, sm.OOV])
        self.assertEqual(sm.decode(vocab, [3, 5, sm.END, -1]).tolist(), [5, 40, -1, -1])

    def test_window_holds_the_session_prefix_and_targets_end_on_the_last_view(self):
        views = toy_views([(10, 11, 12)], 2, splits=(0,))
        vocab = sm.build_vocab(views, (0,))
        w = sm.build_windows(views, vocab, k=4)
        self.assertEqual(w.history[:3].tolist(),
                         [[0, 0, 0, 3], [0, 0, 3, 4], [0, 3, 4, 5]])
        # The second session starts again from padding: sessions never mix.
        self.assertEqual(w.history[3].tolist(), [0, 0, 0, 3])
        self.assertEqual(w.target[:3].tolist(), [4, 5, sm.END])
        self.assertEqual(w.novel[:3].tolist(), [True, True, False])

    def test_windows_are_causal(self):
        views = fixture_views()
        vocab = sm.build_vocab(views, (0, 1))
        before = sm.build_windows(views, vocab)
        # Rewrite every view after position 1 of every session (and the next
        # item that names it): windows at positions 0-1 must not move.
        changed = views.copy()
        late = changed.position.ge(2)
        changed.loc[late, "itemid"] = 999_999
        changed.loc[changed.position.ge(1) & changed.eligible, "next_item"] = 999_999
        after = sm.build_windows(changed, vocab)
        early = before.frame.position.le(1).to_numpy()
        self.assertTrue(np.array_equal(before.history[early], after.history[early]))
        self.assertTrue(np.array_equal(before.context[early], after.context[early]))

    def test_novel_matches_the_simulators_definition(self):
        views = fixture_views()
        test = views.loc[views.split.ge(1)]
        out = replay_capacities(test, fit_index(views), None, [], (4,), scorer=TransitionScorer())
        extra = out[4][1]
        self.assertEqual(int(sm.novel_next(test).sum()), extra["novel_eligible_pairs"])

    def test_protocol_rows_are_chronological(self):
        views = fixture_views()
        w = sm.build_windows(views, sm.build_vocab(views, (0, 1)))
        split, ts = w.frame.split.to_numpy(), w.frame.timestamp.to_numpy()
        rows = sm.protocol_rows(w, "validation")
        self.assertTrue(set(split[rows["train"]]) <= {0, 1})
        self.assertTrue((split[rows["early"]] == 1).all())
        self.assertLess(ts[rows["train"]][split[rows["train"]] == 1].max(),
                        ts[rows["early"]].min())
        self.assertTrue(w.eligible[rows["early"]].all())
        self.assertTrue((split[rows["evaluate"]] == 2).all())
        self.assertFalse((w.target[rows["train"]] == sm.OOV).any())
        final = sm.protocol_rows(w, "final")
        self.assertTrue(set(split[final["train"]]) <= {0, 1, 2})
        self.assertEqual(len(final["early"]), 0)
        self.assertTrue((split[final["evaluate"]] == 3).all())

    def test_ranking_metrics_match_the_scorer_harness_and_the_replay(self):
        views = fixture_views()
        index = fit_index(views)
        validation = views.loc[views.split.eq(2)]
        metrics = sm.scorer_metrics(validation, TransitionScorer(), index,
                                    sm.novel_next(validation))
        extra = replay_capacities(validation, index, None, [], (4,),
                                  scorer=TransitionScorer())[4][1]
        self.assertEqual(metrics["rows"], extra["oracle_eligible_pairs"])
        self.assertAlmostEqual(metrics["hit_at_1"], extra["ranking_hit_at_1"], places=12)
        self.assertAlmostEqual(metrics["mrr_at_8"], extra["ranking_mrr_at_8"], places=12)
        self.assertAlmostEqual(metrics["coverage_at_8"],
                               extra["oracle_candidate_coverage_at_8"], places=12)
        self.assertAlmostEqual(metrics["novel_hit_at_1"], extra["novel_hit_at_1"], places=12)
        self.assertAlmostEqual(metrics["novel_recall_at_8"], extra["novel_coverage_at_8"],
                               places=12)

    def test_ranking_metrics_on_no_rows_is_zero_not_an_error(self):
        metrics = sm.ranking_metrics(np.zeros((0, 8), dtype=np.int64), [], novel=[])
        self.assertEqual((metrics["rows"], metrics["hit_at_1"], metrics["novel_rows"]), (0, 0.0, 0))
        short = sm.ranking_metrics([[5], [6]], [6, 6])
        self.assertEqual((short["hit_at_1"], short["coverage_at_8"]), (0.5, 0.5))

    def test_pivot_sample_is_train_rankers_validation_sample(self):
        views = pd.DataFrame({"split": [2] * 30 + [1] * 5, "x": range(35)})
        sample = sm.pivot_validation_sample(views, n=10)
        self.assertEqual(sample.x.tolist(),
                         np.linspace(0, 29, 10, dtype=np.int64).tolist())

    def test_rank_table_keys_and_missing_rows(self):
        table = sm.RankTable(sm._keys([7, 7], [0, 1]),
                             np.array([[4, -1], [5, 6]]), np.array([[0.5, 0], [0.2, 0.1]],
                                                                   dtype=np.float32))
        self.assertEqual(table[(7, 0)], [(4, 0.5)])
        self.assertEqual(sorted(table), [(7, 0), (7, 1)])
        row = SimpleNamespace(session_id=7, position=2)
        with self.assertRaises(KeyError):
            table.score(row, None)

    def test_module_imports_without_torch_and_says_why_it_cannot_train(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(sm.SeqModelUnavailable):
                sm._torch()


@unittest.skipUnless(HAVE_WEB, "web study needs numpy, pandas, scikit-learn and joblib")
class SharedGpuTests(unittest.TestCase):
    class FakeOOM(RuntimeError):
        def __init__(self):
            super().__init__("CUDA out of memory. Tried to allocate 2.00 GiB")

    def test_oom_is_retried_and_other_errors_are_not(self):
        calls, waits, seen = [], [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise self.FakeOOM()
            return "done"

        with contextlib.redirect_stderr(io.StringIO()):
            result = sm.oom_retry(flaky, retries=4, wait_s=7, sleep=waits.append,
                                  on_oom=lambda attempt, error: seen.append(attempt))
        self.assertEqual((result, len(calls), waits, seen), ("done", 3, [7, 7], [0, 1]))
        with self.assertRaises(ValueError):
            sm.oom_retry(lambda: (_ for _ in ()).throw(ValueError("bug")), sleep=waits.append)
        self.assertEqual(len(waits), 2)

    def test_oom_retry_gives_up_after_its_budget(self):
        def always():
            raise self.FakeOOM()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                sm.oom_retry(always, retries=2, wait_s=0, sleep=lambda s: None)

    def test_gpu_sampler_summarises_contention(self):
        readings = iter([[{"utilization.gpu": "90", "memory.used": "39000",
                           "memory.total": "97887"}],
                         [{"utilization.gpu": "100", "memory.used": "41000",
                           "memory.total": "97887"}]])
        with sm.GpuSampler(interval_s=3600, query=lambda: next(readings)) as sampler:
            pass
        self.assertEqual(len(sampler.samples), 2)
        self.assertIn("util mean 95% max 100%", sampler.summary())
        self.assertIn("41000 of 97887 MiB", sampler.summary())
        with sm.GpuSampler(interval_s=3600, query=lambda: None) as none:
            pass
        self.assertIsNone(none.summary())


@unittest.skipUnless(HAVE_TORCH, "the sequence model needs torch")
class SequenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.views = fixture_views()
        cls.scorer, cls.config, cls.windows, cls.rows = tiny_scorer(cls.views)

    def test_the_current_item_and_the_session_window_are_never_ranked(self):
        table = self.scorer.precompute(self.views)
        w = sm.build_windows(self.views, self.scorer.vocab)
        window_items = sm.decode(self.scorer.vocab, w.history)
        current = w.frame.itemid.to_numpy()
        for n, key in enumerate(zip(w.frame.session_id, w.frame.position)):
            ranked = [item for item, _ in table[key]]
            self.assertNotIn(int(current[n]), ranked)
            self.assertFalse(set(ranked) & set(window_items[n].tolist()) - {-1})
        # The "current" mask drops only the current item.
        loose = sm.SequenceScorer(self.scorer.net, self.scorer.vocab,
                                  {**self.config, "mask": "current"})
        loose_table = loose.precompute(self.views)
        for n, key in enumerate(zip(w.frame.session_id, w.frame.position)):
            self.assertNotIn(int(current[n]), [item for item, _ in loose_table[key]])

    def test_probabilities_are_a_sub_distribution_in_rank_order(self):
        table = self.scorer.precompute(self.views)
        for key in table:
            probs = [p for _, p in table[key]]
            self.assertLessEqual(sum(probs), 1.0 + 1e-6)
            self.assertTrue(all(p > 0 for p in probs))
            self.assertEqual(probs, sorted(probs, reverse=True))

    def test_precomputed_table_equals_live_scoring(self):
        table = self.scorer.precompute(self.views, (2, 3), batch_rows=7)
        self.scorer.reset()
        selected = self.views.loc[self.views.split.isin([2, 3])]
        # The replay's session order: each session's views in position order.
        for row in selected.sort_values(["session_id", "position"]).itertuples(index=False):
            live = self.scorer.score(row, None)
            stored = table.score(row, None)
            self.assertEqual([i for i, _ in live], [i for i, _ in stored])
            for (_, a), (_, b) in zip(live, stored):
                self.assertAlmostEqual(a, b, places=6)

    def test_live_scoring_refuses_a_session_without_its_prefix(self):
        self.scorer.reset()
        row = self.views.loc[self.views.position.eq(1)].iloc[0]
        with self.assertRaises(ValueError):
            self.scorer.score(SimpleNamespace(**row.to_dict()), None)

    def test_rankings_are_causal(self):
        # Changing views after position 1 cannot move a ranking at position 0 or 1.
        changed = self.views.copy()
        changed.loc[changed.position.ge(2), "itemid"] = int(self.scorer.vocab[0])
        before, after = self.scorer.precompute(self.views), self.scorer.precompute(changed)
        for key in before:
            if key[1] <= 1:
                self.assertEqual(before[key], after[key])

    def test_artifact_round_trip_has_no_pickle_and_rejects_stale_or_tampered_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "art"
            sm.save_artifact(directory, self.scorer.net, self.scorer.vocab, self.config)
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             ["config.json", "vocab.npy", "weights.npz"])
            with np.load(directory / "weights.npz", allow_pickle=False) as weights:
                self.assertIn("out.weight", weights.files)
            loaded = sm.SequenceScorer.load(directory, device="cpu")
            a = self.scorer.precompute(self.views, (2,))
            b = loaded.precompute(self.views, (2,))
            self.assertEqual({k: a[k] for k in a}, {k: b[k] for k in b})
            config = json.loads((directory / "config.json").read_text())
            (directory / "config.json").write_text(json.dumps({**config, "artifact_version": 0}))
            with self.assertRaisesRegex(sm.ArtifactError, "retrain"):
                sm.SequenceScorer.load(directory, device="cpu")
            (directory / "config.json").write_text(json.dumps(config))
            with open(directory / "vocab.npy", "ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(sm.ArtifactError, "vocab.npy"):
                sm.SequenceScorer.load(directory, device="cpu")
            with self.assertRaises(sm.ArtifactError):
                sm.read_config(Path(tmp) / "missing")

    def test_evaluation_matches_the_replay_on_the_same_table(self):
        metrics, table, costs = sm.evaluate_ranking(self.scorer, self.views, 2)
        replayed = sm.replay_table(self.views, table, 2, capacities=(4,))
        extra = replayed[4][1]
        self.assertEqual(metrics["rows"], extra["oracle_eligible_pairs"])
        self.assertAlmostEqual(metrics["hit_at_1"], extra["ranking_hit_at_1"], places=12)
        self.assertAlmostEqual(metrics["mrr_at_8"], extra["ranking_mrr_at_8"], places=12)
        self.assertAlmostEqual(metrics["novel_recall_at_8"], extra["novel_coverage_at_8"],
                               places=12)
        self.assertGreater(costs["infer_us_per_request"], 0)
        # Replay through a table is the same in parallel and serial.
        parallel = sm.replay_table(self.views, table, 2, capacities=(4,), workers=2)
        self.assertEqual(replayed[4][0], parallel[4][0])

    def test_metric_rows_follow_the_results_schema(self):
        from prefetch_common import schemas
        metrics, table, costs = sm.evaluate_ranking(self.scorer, self.views, 2)
        replayed = sm.replay_table(self.views, table, 2)
        base = {"model": "gru", "protocol": "validation", "seed": 0, "split": "2",
                "train_range": self.config["train_range"], "epochs": 2, "device": "cpu",
                "train_s": 0.1, "infer_us_per_request": 1.0, "gpu_contention": None}
        rows = sm.metric_rows(base, metrics, replayed)
        self.assertEqual(len(rows), len(sm.EVAL_CAPACITIES) * len(sm.eval_policies())
                         * len(sm.EVAL_COSTS))
        schemas.validate_rows("web/seq_metrics.csv", rows)
        skipped = sm.skipped_rows({k: v for k, v in base.items()
                                   if k not in ("train_s", "infer_us_per_request")}, "no GPU")
        schemas.validate_rows("web/seq_metrics.csv", skipped)
        self.assertTrue(all(r["hit_ratio"] is None and r["status"] == "skipped: no GPU"
                            for r in skipped))
        fixed0 = [r for r in rows if r["policy"] == "fixed0"]
        self.assertTrue(all(r["issued_per_view"] == 0 for r in fixed0))


@unittest.skipUnless(HAVE_TORCH, "the sequence model needs torch")
class ToyCorpusTests(unittest.TestCase):
    def test_deterministic_chains_reach_hit_at_one(self):
        torch.set_num_threads(2)
        chains = [(10, 11, 12), (20, 21, 22), (30, 31, 32), (40, 41, 42)]
        views = toy_views(chains, 40)
        scorer, config, _, _ = tiny_scorer(views, epochs=40, lr=0.02, d=16, h=32)
        metrics, _, _ = sm.evaluate_ranking(scorer, views, 2)
        self.assertEqual(metrics["rows"], 2 * 40)
        self.assertEqual(metrics["hit_at_1"], 1.0)
        self.assertEqual(config["early_stop_metric"], "mrr_at_8")


@unittest.skipUnless(HAVE_TORCH, "the sequence model needs torch")
class CommandTests(unittest.TestCase):
    def test_train_and_experiment_commands_on_the_fixture(self):
        from prefetch_common import provenance, schemas
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            views_path = tmp / "views.csv.gz"
            fixture_views().to_csv(views_path, index=False, compression="gzip")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                sm.main(["train", "--views", str(views_path), "--device", "cpu",
                         "--epochs", "1", "--out", str(tmp / "art")])
            summary = json.loads((tmp / "art" / "summary.json").read_text())
            self.assertEqual(summary["protocol"], "validation")
            self.assertIn("split2_all_eligible", summary)
            self.assertIn("split2_pivot_10k_sample", summary)
            args = ["experiment", "--views", str(views_path), "--device", "cpu",
                    "--epochs", "1", "--seeds", "0", "--models", str(tmp / "models"),
                    "--out", str(tmp / "results"), "--workers", "1",
                    "--sample-interval-s", "3600"]
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sm.main(args), 0)
            rows = schemas.read_rows("web/seq_metrics.csv", tmp / "results" / "seq_metrics.csv")
            self.assertEqual({r["protocol"] for r in rows}, {"validation", "final"})
            self.assertEqual({r["split"] for r in rows}, {"2", "3"})
            record = provenance.read_provenance(tmp / "results" / "seq")
            self.assertIn("seq_metrics.csv", record["extra"]["outputs_in_parent"])
            self.assertEqual(len(record["extra"]["runs"]), 2)
            # A second run reuses both artifacts instead of retraining.
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                sm.main(args + ["--no-provenance"])
            again = schemas.read_rows("web/seq_metrics.csv", tmp / "results" / "seq_metrics.csv")
            self.assertEqual([r["utility"] for r in rows], [r["utility"] for r in again])
            # Other training settings are never served from a stale artifact.
            log = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(log):
                sm.main(args + ["--no-provenance", "--protocols", "validation", "--lr", "0.002"])
            self.assertIn("trained with other settings", log.getvalue())


class ReproGpuWebTests(unittest.TestCase):
    def test_web_half_is_implemented_and_reports_unavailable_without_cuda(self):
        from repro import EXIT_UNAVAILABLE, gpu
        self.assertTrue(gpu.IMPLEMENTED["web"])
        args = SimpleNamespace(quick=True)
        self.assertIn("min", gpu.estimate("web", args))
        with mock.patch.object(gpu, "_cuda_available", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(gpu.run("web", args), EXIT_UNAVAILABLE)
        self.assertIn("CUDA", out.getvalue())

    def test_quick_run_never_writes_results(self):
        from repro import gpu
        command = gpu._web_command(SimpleNamespace(quick=True), Path("v"), Path("m"), Path("o"))
        self.assertNotIn(str(ROOT / "results" / "web"), command)
        self.assertIn("--no-provenance", command)
        full = gpu._web_command(SimpleNamespace(quick=False), Path("v"), Path("m"), Path("o"))
        self.assertIn(str(ROOT / "results" / "web"), full)


@unittest.skipUnless(HAVE_TORCH and RUN_SLOW and (PREPARED.is_file() or RAW.is_file()),
                     "real-data checks need RUN_SLOW=1 and RetailRocket events or prepared views")
class RealDataSlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        if PREPARED.is_file():
            cls.views_path = PREPARED
        else:
            views, _ = build_views(RAW)
            cls.views_path = Path(cls.tmp.name) / "views.csv.gz"
            views.to_csv(cls.views_path, index=False, compression="gzip")
        from webprefetch.data import load_views
        cls.views = load_views(cls.views_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_harness_reproduces_the_logistic_ranker_on_pivots_sample(self):
        from webprefetch.model import FastScorer, train_ranker
        index = fit_index(self.views)
        model, summary = train_ranker(self.views, index)
        sample = sm.pivot_validation_sample(self.views)
        metrics = sm.scorer_metrics(sample, FastScorer(model), index)
        self.assertEqual(metrics["rows"], 3266)
        self.assertAlmostEqual(metrics["hit_at_1"], 0.038885, places=6)
        self.assertAlmostEqual(metrics["mrr_at_8"], 0.055767, places=6)
        self.assertAlmostEqual(metrics["coverage_at_8"], 0.093386, places=6)
        self.assertAlmostEqual(metrics["hit_at_1"], summary["validation_hit_at_1"], places=12)

    @unittest.skipUnless(HAVE_TORCH and torch.cuda.is_available(), "needs CUDA")
    def test_one_epoch_gpu_smoke_run_finishes_in_two_minutes(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "webprefetch.seqmodel", "train", "--protocol", "validation",
             "--epochs", "1", "--device", "cuda", "--views", str(self.views_path),
             "--out", str(Path(self.tmp.name) / "smoke")],
            capture_output=True, text=True, env=env, timeout=600)
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertLess(elapsed, 120)
        summary = json.loads((Path(self.tmp.name) / "smoke" / "summary.json").read_text())
        self.assertEqual(summary["split2_all_eligible"]["rows"],
                         int((self.views.split.eq(2) & self.views.eligible).sum()))


if __name__ == "__main__":
    unittest.main()

"""CLI, report and EDA regressions for the disk study (WP1).

The original suite never ran an entry point, which is how a demo that always
crashed, a cost-model label that contradicted ``--cost-model``, files written
at a different cache size than the console, and a narrative quoting another
run's numbers all survived a green suite. These tests run the real entry
points in a subprocess with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from adaptive_prefetch.benchmark import benchmark_dataset  # noqa: E402
from adaptive_prefetch.cli import FORMATS, build_parser  # noqa: E402
from adaptive_prefetch.datasets import (  # noqa: E402
    BENCHMARK_SEEDS, LSTM_TRAINING_SEEDS, default_datasets, lstm_training_set,
)
from adaptive_prefetch.eda import (  # noqa: E402
    _block_accesses, belady_demand_hit_ratio, locality_profile,
    lru_demand_curve, lru_demand_hit_ratio, profile_trace,
)
from adaptive_prefetch.report import (  # noqa: E402
    benchmark_section, caveats_section, real_world_section,
)
from adaptive_prefetch.simulator import LatencyModel, replay  # noqa: E402
from adaptive_prefetch.trace import Request, load_csv, transition_trace  # noqa: E402

#: Narrative figures that used to be typed into report.py from earlier runs.
STALE_LITERALS = ("18.99", "4.10", "1.0101")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return env


def _run(*args: str, timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=REPO, env=_env(),
                          capture_output=True, text=True, timeout=timeout)


def _write_revised(path: Path, lines: int = 3000) -> None:
    """A small MSRC-trace-003 ``.revised`` file with recorded service times."""
    rng = random.Random(7)
    rows, lba = [], 10_000
    for i in range(lines):
        if i % 50 == 0:
            lba = rng.randrange(1, 4_000_000) * 8
        op = "WS" if i % 7 == 0 else "RS"
        pattern = "seq" if i % 50 else "rand"
        service = 0.0004 if i % 3 else 2.5 + (i % 11) * 0.3
        rows.append(f"{i * 0.001:.6f} {op} {lba} 8 {pattern} "
                    f"{service:.4f} 0.0100 {max(0.0, service - 0.01):.4f}\n")
        lba += 8
    path.write_text("".join(rows), encoding="utf-8")


def _oracle_column(text: str) -> dict[str, str]:
    """dataset -> oracle cell of the first DATASETS block in ``text``."""
    block = text[text.index("DATASETS"):]
    block = block[:block.index("oracle =")]
    cells = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[-1].endswith("%") and parts[0] != "dataset":
            cells[parts[0]] = parts[-1]
    return cells


def _has_literal(text: str, literal: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(literal)}(?!\d)", text) is not None


class CliTests(unittest.TestCase):
    def test_demo_exits_zero(self):
        # The demo fed the 8 stream features to a 12-feature model and always
        # failed with "incorrect feature count".
        result = _run("-m", "adaptive_prefetch", "demo")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Accuracy:", result.stdout)
        self.assertNotIn("incorrect feature count", result.stderr)

    def test_oracle_is_a_format_choice_on_every_trace_command(self):
        self.assertIn("oracle", FORMATS)
        parser = build_parser()
        for argv in (["replay", "t", "--format", "oracle"],
                     ["eda", "t", "--format", "oracle"],
                     ["shift", "t", "--format", "oracle"],
                     ["normalize", "--input", "a", "--output", "b",
                      "--format", "oracle"],
                     ["train-lstm", "--format", "oracle"],
                     ["benchmark", "--format", "oracle"]):
            self.assertEqual(parser.parse_args(argv).format, "oracle")

    def test_oracle_format_fails_as_a_usage_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.oracleGeneral.zst"
            result = _run("-m", "adaptive_prefetch", "eda", str(missing),
                          "--format", "oracle")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("invalid choice", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class MainQuickTests(unittest.TestCase):
    """One ``main.py --quick`` run per cost model, on a fixture MSRC trace."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.msrc = root / "msrc"
        cls.msrc.mkdir()
        _write_revised(cls.msrc / "hm_1.revised")
        cls.out = {}
        processes = {}
        for model in ("assumed", "measured"):
            cls.out[model] = root / model
            processes[model] = subprocess.Popen(
                [sys.executable, str(REPO / "main.py"), "--quick",
                 "--cache-blocks", "2048", "--cost-model", model,
                 "--output-dir", str(cls.out[model]),
                 "--msrc-dir", str(cls.msrc)],
                cwd=REPO, env=_env(), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
        cls.console, cls.returncode = {}, {}
        for model, process in processes.items():
            stdout, stderr = process.communicate(timeout=300)
            cls.console[model] = stdout + stderr
            cls.returncode[model] = process.returncode

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _results_md(self, model: str) -> str:
        return (self.out[model] / "results.md").read_text(encoding="utf-8")

    def test_both_runs_succeed(self):
        for model in ("assumed", "measured"):
            self.assertEqual(self.returncode[model], 0, self.console[model])

    def test_results_md_oracle_column_equals_the_console(self):
        # results.md was rendered at a hard-coded 128-block cache.
        for model in ("assumed", "measured"):
            console = _oracle_column(self.console[model])
            written = _oracle_column(self._results_md(model))
            self.assertIn("msrc_hm_1", console)
            self.assertEqual(console, written)
            self.assertIn("capacity 2048 blocks, window 32 requests",
                          self._results_md(model))
            eda = (self.out[model] / "eda.md").read_text(encoding="utf-8")
            self.assertIn("cache capacity 2048 blocks", eda)
            self.assertNotIn("cache capacity 128 blocks", eda)

    def test_assumed_cost_model_is_never_labelled_measured(self):
        # The fixture records service times, which is what used to flip the
        # label to "MEASURED" whatever --cost-model said.
        for text in (self.console["assumed"], self._results_md("assumed")):
            self.assertNotIn("MEASURED", text)
            self.assertNotIn("measured from", text)
        self.assertIn("assumed 5/100/50 us", self.console["assumed"])
        # Positive control: the measured run is labelled measured.
        self.assertIn("MEASURED", self.console["measured"])
        self.assertIn("source            : measured from 1 trace(s)",
                      self.console["measured"])

    def test_report_quotes_no_figure_it_did_not_compute(self):
        trace = load_csv(self.msrc / "hm_1.revised", "revised")
        computed = {
            "assumed": {f"{LatencyModel().break_even_precision():.4f}"},
            "measured": {f"{LatencyModel.measured(trace=trace).break_even_precision():.4f}"},
        }
        for model in ("assumed", "measured"):
            text = self.console[model] + self._results_md(model)
            for literal in STALE_LITERALS:
                if literal in computed[model]:
                    continue
                self.assertFalse(_has_literal(text, literal),
                                 f"{literal!r} in the {model} report")
            self.assertNotIn("web_3", text)
            self.assertNotIn("287-5491", text)
        # The measured model really is the degenerate 1.0101 one, and the
        # report says so from the model rather than from a typed constant.
        self.assertIn("1.0101", computed["measured"])
        self.assertIn("break-even precision is 1.0101", self.console["measured"])
        self.assertIn("Break-even precision is 0.5263", self.console["assumed"])

    def test_synthetic_and_real_rows_are_pooled_separately(self):
        rows = (self.out["assumed"] / "results.csv").read_text(encoding="utf-8")
        datasets = {line.split(",")[0] for line in rows.splitlines()[1:]}
        self.assertIn("ALL_SYNTHETIC", datasets)
        self.assertIn("ALL_REAL", datasets)
        self.assertNotIn("ALL_TRACES", datasets)
        # The stale pre-D10 demo trace is no longer pooled into the run.
        self.assertNotIn("demo_normalized", datasets)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = []
        for name, seed in (("real_a", 11), ("real_b", 12)):
            requests, _ = transition_trace(seed, windows_per_class=3)
            cls.rows += benchmark_dataset(name, requests)

    def _hit(self, name, mode):
        return next(float(r["hit_ratio"]) for r in self.rows
                    if r["dataset"] == name and r["mode"] == mode)

    def test_depth_figure_is_computed_from_the_rows(self):
        deltas = [self._hit(n, "deep") - self._hit(n, "sequential")
                  for n in ("real_a", "real_b")]
        text = real_world_section(self.rows, 2048)
        self.assertIn(f"{100 * sum(deltas) / 2:+.2f} pp", text)
        for literal in ("+0.4 points", "4.10", "18.99", "web_3", "49%"):
            self.assertNotIn(literal, text)

    def test_break_even_sentence_follows_the_cost_model(self):
        degenerate = real_world_section(self.rows, 2048, LatencyModel(1, 100, 100))
        self.assertIn("break-even precision is 1.0101", degenerate)
        self.assertIn("degenerate", degenerate)
        payable = real_world_section(self.rows, 2048, LatencyModel())
        self.assertIn("0.5263", payable)
        self.assertNotIn("degenerate", payable)
        self.assertNotIn("1.0101", payable)
        self.assertNotIn("break-even", real_world_section(self.rows, 2048))

    def test_caveats_quote_this_runs_service_times(self):
        text = caveats_section("measured from 2 trace(s)",
                               {"a": 400.0, "b": 1200.0})
        self.assertIn("400-1,200 us", text)
        self.assertIn("4.0-12.0x", text)
        self.assertNotIn("287", caveats_section())
        self.assertIn("No trace in this run recorded", caveats_section())

    def test_pooled_rows_are_not_listed_as_datasets(self):
        from adaptive_prefetch.benchmark import aggregate_results
        rows = self.rows + aggregate_results(self.rows, "ALL_REAL")
        text = benchmark_section(rows, 2048)
        self.assertIn("pooled ALL_REAL", text)
        self.assertNotIn("  ALL_REAL ", text.split("vs none   =")[0])


class EdaReferenceTests(unittest.TestCase):
    def test_eda_no_longer_calls_read_reuse_an_upper_bound(self):
        requests, _ = transition_trace(5, windows_per_class=2)
        text = profile_trace(requests, "t", 32, 2048)
        self.assertNotIn("upper bound", text.lower())
        self.assertIn("not a bound", text)
        self.assertIn("Belady MIN", text)

    def test_mattson_lru_equals_the_none_mode(self):
        requests, _ = transition_trace(3, windows_per_class=4)
        mixed = [Request(r.timestamp_ms, r.lba, r.size_blocks,
                         "W" if i % 5 == 0 else r.operation)
                 for i, r in enumerate(requests)]
        sample = load_csv(REPO / "data" / "samples" / "msr-cambridge2-sample.csv", "msr")
        capacities = (1, 8, 64, 512, 2048)
        for trace in (requests, mixed, sample):
            curve = lru_demand_curve(trace, capacities)
            for capacity in capacities:
                none, _ = replay(trace, capacity, mode="none")
                self.assertAlmostEqual(curve[capacity], none.hit_ratio, places=12)
            self.assertEqual(list(curve.values()), sorted(curve.values()))

    def test_belady_is_optimal_on_small_traces(self):
        def exhaustive(requests, capacity):
            accesses = _block_accesses(requests)

            @lru_cache(maxsize=None)
            def best(t, cache):
                if t == len(accesses):
                    return 0
                block, is_read = accesses[t]
                hit = int(is_read and block in cache)
                if block in cache:
                    return hit + best(t + 1, cache)
                if len(cache) < capacity:
                    return hit + best(t + 1, cache | {block})
                options = [best(t + 1, cache)]
                options += [best(t + 1, (cache - {v}) | {block}) for v in cache]
                return hit + max(options)

            reads = sum(is_read for _, is_read in accesses)
            return best(0, frozenset()) / reads if reads else 0.0

        rng = random.Random(3)
        for _ in range(80):
            requests = [Request(float(i), rng.randrange(6), 1, rng.choice("RRW"))
                        for i in range(rng.randrange(1, 13))]
            capacity = rng.randrange(1, 4)
            optimal = belady_demand_hit_ratio(requests, capacity)
            self.assertAlmostEqual(optimal, exhaustive(requests, capacity), places=12)
            self.assertGreaterEqual(optimal + 1e-12,
                                    lru_demand_hit_ratio(requests, capacity))

    def test_belady_hand_case_beats_lru_on_a_loop(self):
        # A-B-C looped through a 2-block cache: LRU never hits, MIN keeps two
        # of the three and hits on them.
        requests = [Request(float(i), lba) for i, lba in enumerate([1, 2, 3, 1, 2, 3])]
        self.assertEqual(lru_demand_hit_ratio(requests, 2), 0.0)
        self.assertAlmostEqual(belady_demand_hit_ratio(requests, 2), 2 / 6)

    def test_locality_profile_reports_both_references(self):
        requests, _ = transition_trace(4, windows_per_class=2)
        profile = locality_profile(requests, capacity=256)
        self.assertEqual(profile["cacheable_ceiling"],
                         profile["read_reuse_within_capacity"])
        self.assertGreaterEqual(profile["belady_demand_hit_ratio"] + 1e-12,
                                profile["lru_demand_hit_ratio"])


def _load_main():
    spec = importlib.util.spec_from_file_location("main_under_test", REPO / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signature(requests):
    return tuple((r.lba, r.size_blocks, r.operation) for r in requests)


class LstmTrainingSetTests(unittest.TestCase):
    def test_training_set_is_disjoint_from_the_benchmark(self):
        benchmark = default_datasets(32)
        training, record = lstm_training_set(benchmark, 32)
        self.assertTrue(set(LSTM_TRAINING_SEEDS).isdisjoint(BENCHMARK_SEEDS))
        self.assertEqual(record["seeds"], list(LSTM_TRAINING_SEEDS))
        benchmark_signatures = {_signature(r) for _, r in benchmark}
        self.assertTrue(training)
        for _, requests in training:
            self.assertNotIn(_signature(requests), benchmark_signatures)

    def test_a_training_trace_inside_the_benchmark_is_refused(self):
        training, _ = lstm_training_set([], 32)
        with self.assertRaises(ValueError):
            lstm_training_set([("leak", training[0][1])], 32)

    def test_resolve_lstm_trains_on_the_disjoint_set_and_records_it(self):
        main = _load_main()
        benchmark = default_datasets(32)
        seen = []

        def fake_train(path, traces=None, epochs=60, **_):
            seen.append(traces)
            Path(path).write_bytes(b"artifact")
            return {"examples": 10, "classes": 3}

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(main, "LSTM_ARTIFACT", Path(directory) / "l.pt"), \
                mock.patch.object(main, "LSTM_TRAINING_RECORD",
                                  Path(directory) / "l.training.json"), \
                mock.patch("adaptive_prefetch.lstm.train_lstm", fake_train), \
                mock.patch("adaptive_prefetch.lstm.load_predictor", lambda p: None), \
                contextlib.redirect_stdout(io.StringIO()):
            path, note = main.resolve_lstm(benchmark, False, False, 2, 32)
            self.assertEqual(len(seen), 1)
            self.assertIn("disjoint from the benchmark", note)
            benchmark_signatures = {_signature(r) for _, r in benchmark}
            for requests in seen[0]:
                self.assertNotIn(_signature(requests), benchmark_signatures)
            record = json.loads(main.LSTM_TRAINING_RECORD.read_text())
            self.assertEqual(record["seeds"], list(LSTM_TRAINING_SEEDS))

            # A matching record lets the artifact be reused...
            _, note = main.resolve_lstm(benchmark, False, False, 2, 32)
            self.assertEqual(len(seen), 1)
            self.assertIn("reused", note)
            # ...an artifact of unknown provenance is retrained.
            main.LSTM_TRAINING_RECORD.unlink()
            main.resolve_lstm(benchmark, False, False, 2, 32)
            self.assertEqual(len(seen), 2)


if __name__ == "__main__":
    unittest.main()

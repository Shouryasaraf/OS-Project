"""The results contract: the exact columns of every results file the paper reads.

Producers (WP6, WP8, WP9, WP10, WP11, WP12) write through :func:`write_rows`,
which refuses a row with an unknown or a missing column, or a value of the
wrong type, before anything is written. The paper generators (WP13) build
their fixtures from the same tuples, and ``tests/test_common.py`` checks every
file under ``tests/fixtures/results/`` against them, so the two sides cannot
drift apart while they are written in parallel.

Schemas are keyed by their path under ``results/``. A key with ``*`` is a
pattern (``realio/device_*.json``) and a concrete path is matched against it.

Three formats:

* ``csv`` -- a header equal to :func:`columns` in that order. ``None`` is
  written as an empty field, ``bool`` as ``True``/``False``, a ``json``
  column as compact JSON text, a float with ``repr`` (round-trips exactly,
  ``inf``/``nan`` included). An optional column (``required=False``) may be
  left out of the header altogether, so a column can be added to a file a
  running experiment is already writing without invalidating its output;
  when present it stays in schema order.
* ``jsonl`` -- one object per line with exactly the schema's keys. Appending
  is supported, because the disk sweep is resumable. Non-finite floats are
  refused (strict JSON).
* ``json`` -- one object; required keys must be present and optional keys
  may be. Values may nest.

Some files are written by code that does not go through :func:`write_rows`
(pandas ``to_csv``, ``json.dumps``); their schemas are derived from those
writers, and :func:`validate_tree` checks them after the run.
``results/pivot_original/`` is the pivot branch's output copied verbatim
(never edited, never a schema'd study), so :func:`validate_tree` checks only
its ``provenance.json``.

Dtypes: ``str``, ``int``, ``float``, ``bool``, ``json`` (anything JSON can
hold). A trailing ``?`` means ``None`` is allowed. Skipped rows
(``status="skipped: <reason>"``) carry ``None`` metrics, never substitutes, so
every metric column of a file that can hold skipped rows is nullable.
All ratios and deltas are fractions (0.0123), never percentage points.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import json
import numbers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .provenance import OPTIONAL_KEYS as _PROV_OPTIONAL
from .provenance import PROVENANCE_KEYS as _PROV_KEYS
from .utility import UTILITY_CS, utility_column

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
FORMATS = ("csv", "jsonl", "json")
DTYPES = ("str", "int", "float", "bool", "json")
#: Directories under ``results/`` copied verbatim from elsewhere; only their
#: ``provenance.json`` is held to a schema.
VERBATIM_DIRS = ("pivot_original",)


class SchemaError(ValueError):
    """A row or file does not match its schema."""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    note: str = ""
    required: bool = True  # json: key may be absent; csv: column may be absent

    @property
    def base(self) -> str:
        return self.dtype.rstrip("?")

    @property
    def nullable(self) -> bool:
        return self.dtype.endswith("?")


@dataclass(frozen=True)
class Schema:
    name: str
    fmt: str
    producer: str
    description: str
    columns: tuple[Column, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.required)


def _c(name: str, dtype: str, note: str = "", required: bool = True) -> Column:
    return Column(name, dtype, note, required)


def _ci(prefix: str, dtype: str, note: str) -> tuple[Column, ...]:
    """An estimate and its 95% interval: ``prefix``, ``prefix_lo``, ``prefix_hi``."""
    return (_c(prefix, dtype, note), _c(f"{prefix}_lo", dtype, "95% CI low"),
            _c(f"{prefix}_hi", dtype, "95% CI high"))


_UTILITY = tuple(_c(utility_column(c), "float", f"U(c) at c={c:.6g}; prefetch_common.utility")
                 for c in UTILITY_CS)

# --------------------------------------------------------------------- disk

#: The columns that identify one sweep job; a resumed run skips a row whose
#: key is already in ``sweep.jsonl``.
SWEEP_KEY = ("config", "trace", "skip_requests", "max_requests", "merge",
             "max_merge_sectors", "capacity", "mode", "depth", "max_blocks",
             "latency")

_SWEEP = (
    _c("config", "str", "C1..C4, or 'smoke'"),
    _c("trace", "str", "file stem without .oracleGeneral.zst, e.g. msr_hm_0"),
    _c("corpus", "str", "msr|fiu|alibaba|tencent (corpora.CORPUS_OF)"),
    _c("skip_requests", "int", "canonical requests skipped first; 0 = first slice"),
    _c("max_requests", "int?", "cap in canonical requests; None = rest of file"),
    _c("merge", "str", "auto (canonical) | none | same_second"),
    _c("max_merge_sectors", "int?", "None = uncapped"),
    _c("size_unit", "str", "bytes|sectors as resolved, never auto"),
    _c("capacity", "int", "cache size in 512-B blocks"),
    _c("mode", "str", "a benchmark.MODES name"),
    _c("depth", "int", "read_ahead_depth passed to replay"),
    _c("max_blocks", "int?", "linux_ra window cap; None for other modes"),
    _c("latency", "str", "assumed (5/100/50 us) | break_even (1/100/100 us)"),
    _c("hit_us", "float"), _c("miss_us", "float"), _c("prefetch_us", "float"),
    _c("break_even", "float", "LatencyModel.break_even_precision() = utility.cost_to_c"),
    _c("status", "str", "ok | skipped: <reason>"),
    _c("requests", "int?", "canonical requests in the slice; None only on a "
       "skipped row that never loaded it"),
    _c("blocks", "int?", "512-B blocks requested in the slice; None as requests"),
    _c("hours_covered", "float?", "trace time spanned by the slice; None as requests"),
    _c("ops_known", "bool", "False for oracle traces: writes replay as reads"),
    _c("ts_resolution_s", "float", "1 for oracle traces"),
    _c("read_blocks", "int?", "Metrics.read_blocks"),
    _c("hit_ratio", "float?", "block hit ratio"),
    _c("precision", "float?"),
    _c("total_prefetches", "int?"), _c("useful_prefetches", "int?"), _c("unused", "int?"),
    _c("mean_us", "float?", "modelled cost per block under this row's latency"),
    _c("policy_switches", "int?"),
    _c("observed_precision", "float?"), _c("precision_margin", "float?"),
    _c("observed_stride", "int?"),
    _c("demand_ios", "int?"), _c("prefetch_ios", "int?"),
    _c("prefetch_ios_mergeable", "int?"),
    _c("device_ios", "int?", "demand_ios + prefetch_ios - prefetch_ios_mergeable"),
    _c("blocks_moved", "int?", "demand-missed + prefetched blocks"),
    _c("belady_hit_ratio", "float?",
       "demand-only Belady/MIN at this capacity, from the block stream; a "
       "reference, not a bound on prefetching; None when not computed"),
    _c("age_table_bound", "int?", "WindowContext bound (adaptive); None otherwise"),
    _c("age_table_evictions", "int?"),
    _c("model_seed", "int?", "make_model seed for adaptive; None otherwise"),
    _c("git_sha", "str?"),
    _c("load_start", "float?", "1-min load average at job start"),
    _c("load_end", "float?"),
    _c("wall_s", "float"),
)

_DISK_TABLE = (
    _c("config", "str"), _c("merge", "str"), _c("capacity", "int"),
    _c("latency", "str", "assumed|break_even; only gated modes differ"),
    _c("corpus", "str", "'all' in table_main.csv; one corpus per row in table_by_corpus.csv"),
    _c("policy", "str", "mode label, e.g. deep@8, linux_ra@256"),
    _c("n_traces", "int"),
    _c("mean_hit_ratio", "float?", "unweighted mean over traces"),
    _c("weighted_hit_ratio", "float?", "read-block-weighted (M14: both weightings)"),
    *_ci("delta_vs_none", "float?", "paired bootstrap over traces"),
    *_ci("delta_vs_sequential", "float?", "paired bootstrap over traces"),
    _c("mean_precision", "float?", "over traces that issued a prefetch"),
    _c("mean_wasted_share", "float?", "unused / issued, same traces"),
    _c("device_io_ratio", "float?", "mean over traces of device_ios / none's"),
    _c("blocks_moved_ratio", "float?", "mean over traces of blocks_moved / none's"),
    _c("wins_vs_none", "int?"), _c("losses_vs_none", "int?"),
)

_DISK_DELTAS = (
    _c("config", "str"), _c("merge", "str"), _c("capacity", "int"),
    _c("latency", "str"), _c("corpus", "str", "'all' or one corpus"),
    _c("metric", "str", "hit_ratio|device_ios|blocks_moved|precision|wasted_share"),
    _c("statistic", "str", "difference|ratio (stats.paired_bootstrap)"),
    _c("policy", "str"), _c("baseline", "str"),
    _c("n", "int", "paired traces"),
    _c("estimate", "float"), _c("ci_lo", "float"), _c("ci_hi", "float"),
    _c("cluster_estimate", "float?", "stats.cluster_bootstrap over corpora then traces"),
    _c("cluster_ci_lo", "float?"), _c("cluster_ci_hi", "float?"),
    _c("twins", "str", "down_weighted | collapsed | none"),
    _c("wilcoxon_p", "float?", "None when scipy is missing (stats.WILCOXON_UNAVAILABLE)"),
    _c("wins", "int"), _c("losses", "int"), _c("ties", "int"),
    _c("reps", "int"), _c("seed", "int"),
)

_DISK_WINNERS = (
    _c("config", "str"), _c("merge", "str"), _c("capacity", "int"),
    _c("latency", "str", "assumed only: every competing policy under one cost model"),
    _c("corpus", "str", "'all' or one corpus"),
    _c("metric", "str", "hit_ratio (highest wins) | device_ios (fewest wins)"),
    _c("policy", "str"),
    _c("traces_won", "int", "traces where this policy is best; a tie counts for every tied policy"),
    _c("n_traces", "int", "traces this policy ran on"),
    _c("mean_value", "float?", "mean hit ratio, or mean device_ios ratio to none"),
    _c("rank_by_mean", "int?", "1 = best mean"),
    _c("note", "str?"),
)

_DISK_KAPPA = (
    _c("config", "str"), _c("merge", "str"), _c("capacity", "int"),
    _c("trace", "str"), _c("corpus", "str"), _c("policy", "str"),
    _c("ios_policy", "int"), _c("ios_none", "int"),
    _c("blocks_policy", "int"), _c("blocks_none", "int"),
    _c("delta_ios", "int"), _c("delta_blocks", "int"),
    _c("kappa_star", "float?",
       "IOCostModel.kappa_star: t_pos/t_xfer above which the policy lowers "
       "modelled device time; may be inf"),
)

_DISK_CORPUS = (
    _c("trace", "str"), _c("corpus", "str"),
    _c("file", "str", "repo-relative path"),
    _c("size_unit", "str"), _c("merge", "str"), _c("max_merge_sectors", "int?"),
    _c("records", "int", "raw oracleGeneral records"),
    _c("requests", "int", "canonical requests after reconstruction"),
    _c("blocks", "int"),
    _c("footprint_blocks", "int?", "distinct blocks; None when not computed"),
    _c("mean_request_blocks", "float"), _c("median_request_blocks", "float"),
    _c("contiguous_fraction", "float", "next request starts at the previous end"),
    _c("hours_covered", "float"),
    _c("tagged", "bool", "volume tag in obj_id bits >= 40"),
    _c("twin_group", "str?", "twin family label; None when not a twin"),
    _c("paper_subset", "bool", "in corpora.PAPER_SUBSET"),
)

_DISK_LEARNED_DEPTH = (
    _c("trace", "str"), _c("corpus", "str"), _c("seed", "int"),
    _c("status", "str", "ok | skipped: <reason>"),
    _c("action_cap", "int", "largest read-ahead segment end: 64 or 256 (both reported)"),
    _c("capacity", "int"),
    _c("n_requests", "int", "canonical requests in the prefix actually used"),
    _c("train_start", "int", "request index; ranges are half-open"),
    _c("train_end", "int"), _c("val_start", "int"), _c("val_end", "int"),
    _c("test_start", "int"), _c("test_end", "int"),
    _c("thresholds", "json?", "validation-chosen thresholds per segment"),
    _c("test_hit_ratio", "float?"), _c("test_issued", "int?"), _c("test_precision", "float?"),
    _c("vcf_depth", "int?", "validation-chosen fixed depth"),
    _c("vcf_hit_ratio", "float?", "on the same test segment"), _c("vcf_issued", "int?"),
    _c("relsize_k", "float?", "validation-chosen k of depth = k * size_blocks"),
    _c("relsize_hit_ratio", "float?"),
    _c("deep8_hit_ratio", "float?"), _c("none_hit_ratio", "float?"),
    _c("oracle_same_actions_hit_ratio", "float?",
       "best segment choice in hindsight, same action space"),
    _c("device", "str", "cuda|cpu"),
    _c("train_s", "float?"), _c("infer_us_per_decision", "float?"),
    _c("gpu_contention", "str?", "nvidia-smi note; GPU timings are never quoted as costs without it"),
    _c("linux_ra_hit_ratio", "float?",
       "linux_ra on the same test segment (the baseline the paper must show); "
       "optional: E2 as first run does not write it, and disk_depth_baselines.csv "
       "(segment=test, mode=linux_ra) holds the same replay", required=False),
)

_DISK_LSTM = (
    _c("trace", "str"), _c("corpus", "str"), _c("seed", "int"),
    _c("status", "str", "ok | skipped: <reason>; a skipped row has None metrics"),
    _c("n_requests", "int?", "canonical requests in the prefix"),
    _c("train_start", "int?", "request index; ranges are half-open"),
    _c("train_end", "int?"), _c("val_start", "int?"), _c("val_end", "int?"),
    _c("test_start", "int?"), _c("test_end", "int?"),
    _c("epochs_run", "int?"), _c("best_epoch", "int?"),
    _c("val_loss", "json?", "per-epoch validation loss list"),
    _c("test_bucket_top1", "float?",
       "next-delta bucket accuracy on the test slice; LSTM predicts deltas, not classes"),
    _c("test_majority_top1", "float?", "majority-bucket reference"),
    _c("test_repeat_top1", "float?", "repeat-last-delta reference"),
    _c("lstm_hit_ratio", "float?"), _c("lstm_precision", "float?"), _c("lstm_issued", "int?"),
    _c("none_hit_ratio", "float?"), _c("sequential_hit_ratio", "float?"),
    _c("deep8_hit_ratio", "float?"), _c("linux_ra_hit_ratio", "float?"),
    _c("infer_us_per_request", "float?", "replay inference time per request"),
    _c("device", "str?", "cuda|cpu"), _c("train_s", "float?"),
    _c("gpu_contention", "str?", "nvidia-smi note"),
)

_DISK_DEPTH_BASELINES = (
    _c("trace", "str"), _c("corpus", "str"),
    _c("segment", "str", "val | test"),
    _c("start", "int", "request index; half-open"), _c("end", "int"),
    _c("action_cap", "int?", "None for the cap-independent test references"),
    _c("mode", "str", "none|sequential|deep|linux_ra|vcf|relsize|oracle_same_actions"),
    _c("param", "float?", "depth (deep, vcf) or k (relsize); None otherwise"),
    _c("chosen", "bool?", "the validation choice; None for references"),
    _c("hit_ratio", "float"), _c("precision", "float"),
    _c("issued", "int"), _c("useful", "int"), _c("device_ios", "int"),
)

_GPU_ENV = (
    _c("device_requested", "str"), _c("torch", "str?"), _c("cuda_device", "str?"),
    _c("jobs", "json", "one entry per GPU job: device, train_s, OOM retries, nvidia-smi samples"),
    _c("note", "str"), _c("missing_traces", "json"),
    _c("config", "json", "repro/gpu_disk.Config without its paths"),
)

# ------------------------------------------------------------------- realio

_REALIO = (
    _c("device", "str"), _c("backend", "str", "odirect (B) | pagecache (A)"),
    _c("mode", "str"), _c("think", "str", "closed | fixed:US | spread:SCALE:CAP_US"),
    _c("repeat", "int"), _c("trace", "str"), _c("requests", "int"),
    _c("sim_hit_ratio", "float?", "None for kernel_ra/kernel_seq (no simulator plan)"),
    _c("sim_precision", "float?"), _c("sim_device_ios", "int?"),
    _c("real_hit_ratio", "float", "measured; Backend A is a measured divergence"),
    _c("full_hit_request_frac", "float"),
    _c("ready_hits", "int"), _c("late_hits", "int"), _c("late_wait_us", "float"),
    _c("lat_mean_us", "float"), _c("lat_p50_us", "float"), _c("lat_p95_us", "float"),
    _c("lat_p99_us", "float"), _c("lat_max_us", "float"),
    _c("device_MiB", "float"), _c("prefetch_MiB", "float"),
    _c("think_total_s", "float"), _c("think_error_us_p99", "float?"), _c("wall_s", "float"),
    _c("cg_file_MiB", "float?", "Backend A only"), _c("cg_events_max", "int?", "Backend A only"),
    _c("foreign_io_frac", "float?", "/proc/diskstats; flagged above 1%"),
    _c("wrapped", "int", "RegionMapper wraps; 0 unless --allow-alias"),
    _c("pred_mean_us", "float?", "prediction calibrated from the same run's none mode"),
    _c("pred_err", "float?"), _c("k_eff", "float?", "effective prefetch_multiple"),
    _c("load_avg", "float", "1-min load average at run start"),
)

_REALIO_SUMMARY = (
    _c("device", "str"), _c("backend", "str"), _c("mode", "str"), _c("think", "str"),
    _c("trace", "str"), _c("requests", "int"), _c("reps", "int"),
    _c("sim_hit_ratio", "float?"), _c("sim_device_ios", "int?"),
    _c("sim_device_time_us", "float?", "IOCostModel.device_time_us"),
    _c("real_hit_ratio", "float", "mean over repeats"),
    _c("full_hit_request_frac", "float", "mean over repeats"),
    *_ci("lat_mean_us", "float", "mean over repeats, CI over repeats"),
    _c("lat_p50_us", "float", "mean over repeats"), _c("lat_p99_us", "float", "mean over repeats"),
    _c("late_hits", "float", "mean over repeats"), _c("device_MiB", "float", "mean over repeats"),
    _c("pred_mean_us", "float?"), _c("pred_err", "float?"), _c("k_eff", "float?"),
    *_ci("delta_vs_none_us", "float?",
         "lat_mean_us minus the same repeat's none run, mean and 95% t interval over "
         "repeats paired by repeat; None for none and for modes without a none run"),
    _c("spearman_device_time", "float?",
       "rank correlation across modes of sim_device_time_us vs lat_mean_us; one "
       "value per (device, backend, think, trace), repeated on its rows"),
    _c("spearman_hit_ratio", "float?", "same, sim_hit_ratio vs lat_mean_us"),
    _c("foreign_io_frac_max", "float?"), _c("load_avg_max", "float"),
)

_REALIO_DEVICE = (
    _c("device", "str", "e.g. sdb, ssd"), _c("path", "str", "backing file"),
    _c("model", "str?", "drive model from sysfs"), _c("rotational", "bool?"),
    _c("read_ahead_kb", "int?"), _c("max_sectors_kb", "int?"),
    _c("rand4k_p50_us", "float", "QD1 O_DIRECT random 4 KiB"), _c("rand4k_mean_us", "float"),
    _c("seq4k_p50_us", "float"), _c("seq64k_p50_us", "float"), _c("seq1m_MBps", "float"),
    _c("t_pos_us", "float", "IOCostModel positioning time"),
    _c("t_xfer_us_per_block", "float", "IOCostModel transfer time per 512-B block"),
    _c("kappa", "float", "t_pos_us / t_xfer_us_per_block"),
    _c("samples", "int"), _c("foreign_io_frac", "float?"),
    _c("load_avg", "json?"), _c("created_utc", "str"),
    _c("extra", "json", "anything else the calibration recorded", required=False),
)

# ---------------------------------------------------------------------- web

_WEB_POLICY = (
    _c("split", "str", "validation | test"), _c("scorer", "str", "scorers.py registry name"),
    _c("cache_scope", "str", "session | visitor"),
    _c("capacity", "int"), _c("policy", "str"), _c("views", "int"),
    _c("hit_ratio", "float"), _c("prefetch_hits_per_view", "float"),
    _c("issued_per_view", "float"), _c("precision", "float"),
    _c("wasted_per_view", "float"), _c("demand_evicted_by_prefetch_per_view", "float"),
    _c("c_star", "float?", "utility.c_star vs fixed0, same split/scorer/capacity; None for fixed0"),
    *_UTILITY,
)

_WEB_COMPARISONS = (
    _c("split", "str"), _c("selection", "str", "per_capacity | capacity_transfer"),
    _c("scorer", "str"), _c("capacity", "int"), _c("cost", "float", "assumed c per prefetch"),
    _c("adaptive_policy", "str", "validation-selected"),
    _c("best_fixed_policy", "str", "best fixed depth at this capacity on validation"),
    _c("adaptive_utility", "float"), _c("best_fixed_utility", "float"),
    _c("fixed0_utility", "float"), _c("transition_top1_utility", "float?"),
    *_ci("vs_best_fixed_same_cap", "float", "paired session bootstrap"),
    *_ci("vs_fixed0", "float", "paired session bootstrap"),
    *_ci("vs_transition_top1", "float?", "paired session bootstrap"),
    _c("adaptive_hit_ratio", "float"), _c("best_fixed_hit_ratio", "float"),
    _c("adaptive_issued_per_view", "float"), _c("best_fixed_issued_per_view", "float"),
    _c("label", "str", "e.g. 'adaptive wins', 'loses to no prefetch' (P11)"),
    _c("reps", "int"), _c("seed", "int"),
)

_WEB_DECOMPOSITION = (
    _c("lever", "str", "ranker|freshness|depth|controller|feedback|cache_scope|sequence_model"),
    _c("policy", "str"), _c("baseline", "str"),
    _c("scorer", "str"), _c("baseline_scorer", "str"),
    _c("split", "str"), _c("capacity", "int?", "None for an infinite cache"),
    _c("cache_scope", "str"),
    _c("metric", "str", "utility_per_view | hit_ratio"),
    _c("cost", "float", "c used in the utility; 0 for hit_ratio"),
    _c("estimate", "float"), _c("ci_lo", "float"), _c("ci_hi", "float"),
    _c("sessions", "int"), _c("reps", "int"), _c("seed", "int"),
    _c("note", "str?"),
)

_WEB_FRESHNESS = (
    _c("split", "str"), _c("index", "str", "split0 | split012 | online"),
    _c("k", "int?", "None = every candidate"),
    _c("coverage", "float", "next item in the top k"),
    _c("eligible_pairs", "int"), _c("sources_with_transitions", "int?"),
)

_WEB_SEQ = (
    _c("model", "str", "gru | sasrec"), _c("protocol", "str", "validation | final"),
    _c("seed", "int"), _c("status", "str", "ok | skipped: <reason>"),
    _c("split", "str", "evaluated split"),
    _c("train_range", "str", "what the artifact was trained on"),
    _c("epochs", "int?"),
    _c("hit_at_1", "float?"), _c("mrr_at_8", "float?"),
    _c("novel_hit_at_1", "float?"), _c("novel_recall_at_8", "float?"),
    _c("capacity", "int"), _c("policy", "str"), _c("cost", "float"),
    _c("hit_ratio", "float?"), _c("issued_per_view", "float?"), _c("utility", "float?"),
    _c("train_s", "float?"), _c("infer_us_per_request", "float?"),
    _c("device", "str"), _c("gpu_contention", "str?"),
)

_WEB_CALIBRATION = (
    _c("split", "str", "validation | test"), _c("scorer", "str"),
    _c("bin", "int", "width-0.1 probability bin, 0..9"),
    _c("p_low", "float"), _c("p_high", "float"), _c("count", "int", "ranked candidates"),
    _c("mean_probability", "float?", "None for an empty bin"),
    _c("observed_frequency", "float?"), _c("gap", "float?"),
    _c("reported", "bool", "bin is in the reliability table (lower edge >= 0.1, non-empty)"),
    _c("brier", "float?", "per (split, scorer), repeated on its rows"),
    _c("brier_reference", "float?", "Brier score of always forecasting the base rate"),
    _c("brier_skill_score", "float?"), _c("ece", "float?", "secondary only"),
)

_WEB_LEARNING_CURVE = (
    _c("training_decisions", "int"), _c("candidate_rows", "int"),
    _c("positive_candidates", "int"), _c("selected_C", "float"),
    _c("validation_coverage_at_8", "float"), _c("validation_hit_at_1", "float"),
    _c("validation_mrr_at_8", "float"), _c("validation_views", "int"),
    _c("capacity", "int"), _c("policy", "str"), _c("cost", "float"),
    _c("validation_hit_ratio", "float"), _c("validation_issued_per_view", "float"),
    _c("validation_utility", "float"), _c("train_s", "float"),
)

#: ``data.build_views``'s profile (absent when ``profile.json`` is missing),
#: then the two keys the benchmark adds.
_WEB_PROFILE_KEYS = (
    ("source", "str"), ("dataset_version", "int"), ("source_verified", "bool"),
    ("session_gap_minutes", "int"), ("raw_sha256", "str"), ("raw_rows", "int"),
    ("duplicate_rows_removed", "int"), ("event_counts_after_dedup", "json"),
    ("views", "int"), ("visitors", "int"), ("sessions", "int"),
    ("sessions_with_2plus_views", "int"), ("sessions_with_3plus_views", "int"),
    ("sessions_with_5plus_views", "int"), ("eligible_next_view_pairs", "int"),
    ("session_length_quantiles", "json"), ("split_cutoffs_ms", "json"),
    ("views_by_split", "json"), ("eligible_pairs_by_split", "json"),
    ("prepared_path", "str"))
_WEB_DATASET_PROFILE = tuple(_c(name, dtype, required=False)
                             for name, dtype in _WEB_PROFILE_KEYS) + (
    _c("replayed_views", "json", "views replayed per split label"),
    _c("pair_kinds", "json", "eligible next views by kind: self_repeat, earlier_in_session, novel"),
)

_WEB_THINK_TIME = (
    _c("validation", "json", "eligible pairs, gap quantiles (s), share below thresholds"),
    _c("test", "json"),
)

_WEB_SESSION_PARITY = (
    _c("format", "str", "webprefetch.session_parity/2"),
    _c("selection_rule", "str"),
    _c("benchmark_json_sha256", "str", "the run whose validation selection the cases use; "
       "byte-identical only at a fixed --workers"),
    _c("capacity", "int"), _c("costs", "json"), _c("policy", "str"),
    _c("cases", "json", "per (session, cost): batch counters and match"),
    _c("mismatches", "int", "must be 0"),
)

# --------------------------------------------------------------------- live

_LIVE_METRICS = (
    _c("experiment", "str", "headline | fifo | hint | sensitivity"),
    _c("store", "str", "ssd | hdd"), _c("config", "str", "short label of the axis values"),
    _c("link", "str", "priority | fifo"),
    _c("rtt_ms", "float", "ASSUMED"), _c("mbps", "float", "ASSUMED"),
    _c("size_median_kib", "float", "ASSUMED"), _c("gap_cap_s", "float"),
    _c("ramp_s", "float"), _c("rep", "int"),
    _c("policy", "str"), _c("baseline", "str"),
    _c("sessions", "int"), _c("views", "int"), _c("hit_ratio", "float"),
    _c("ready", "int"), _c("inflight", "int"), _c("miss", "int"),
    _c("mean_wait_ms", "float"), _c("model_ideal_wait_ms", "float"),
    _c("excess_ms", "float"), _c("p95_wait_ms", "float"),
    *_ci("delta_vs_base_ms", "float", "paired session bootstrap vs baseline"),
    _c("ideal_delta_vs_base_ms", "float"),
    _c("kb_per_view", "float"), _c("wasted_kb_per_view", "float"), _c("bytes_vs_base", "float"),
    _c("demand_disk_us_mean", "float"),
    _c("server_lag_p99_us", "float"), _c("client_lag_p99_us", "float"),
    _c("req_lag_p99_us", "float"),
    _c("conn_errors", "int"), _c("parity_mismatches", "int"),
    _c("valid", "bool", "validity gate; an invalid row is never compared"),
    _c("foreign_io_frac", "float?"), _c("load_avg", "float"),
)

_LIVE_PARITY = (
    _c("experiment", "str"), _c("store", "str"), _c("config", "str"),
    _c("rep", "int"), _c("policy", "str"), _c("sessions", "int"),
    _c("sim_hits", "int"), _c("live_hits", "int"),
    _c("sim_issued", "int"), _c("live_issued", "int"),
    _c("sim_useful", "int"), _c("live_useful", "int"),
    _c("mismatched_sessions", "int", "must be 0"),
)

_LIVE_CONFIG = (
    _c("format", "str", "webprefetch.live.config/1"),
    _c("runs", "json", "run tag -> command, invocations, plan, assumptions (tagged "
       "ASSUMED/MEASURED), passes, superseded and aborted passes"),
)

_LIVE_DISK_PROBE = (
    _c("label", "str", "store label, e.g. ssd, hdd"), _c("store", "str", "object store file"),
    _c("device", "json", "objects.device_of: dev, name, disk, rotational, queue limits"),
    _c("objects", "int", "objects sampled"), _c("load_avg", "json"),
    _c("tag", "str", "MEASURED"),
    _c("resident_after_drop", "float"), _c("resident_after_read", "float"),
    _c("cold", "json", "mean/p50/p99 us and n"), _c("warm", "json"),
    _c("hinted", "json", "lead ms -> resident fraction and latency summary"),
    _c("load_avg_end", "json"),
)

# ------------------------------------------------- msrc light sample (E7)

_MSRC_TRACES = (
    _c("trace", "str", "file stem"), _c("family", "str", "MSRC host: stem before the last _"),
    _c("path", "str"), _c("requests", "int"), _c("read_requests", "int"),
    _c("seq_flag_requests", "int", "requests flagged seq by the trace"),
    _c("seq_flag_read_requests", "int"),
    _c("windows", "int", "complete 32-request windows"),
    _c("seq_windows_majority", "int", ">= 17 of 32 seq-flagged"),
    _c("seq_windows_quarter", "int", ">= 8 of 32 seq-flagged"),
    _c("small_n_majority", "bool", "< 100 windows or < 10 of a class"),
    _c("mean_service_ms", "float?", "mean t1"),
    _c("age_table_evictions", "int"),
    _c("flag_matches_contiguity", "float?", "share of requests whose seq flag equals "
       "lba == previous lba + previous size"),
)

_MSRC_METRICS = (
    _c("protocol", "str", "shipped_synthetic, sklearn_synthetic, loto_real, lofo_real, "
       "rule, baseline"),
    _c("classifier", "str"), _c("label_rule", "str", "majority (primary) or quarter"),
    _c("trace", "str", "file stem or pooled"), _c("status", "str"),
    _c("train_on", "str"), _c("n_windows", "int"), _c("n_seq", "int"),
    _c("small_n", "bool"),
    *_ci("accuracy", "float?", "window accuracy, binary seq vs not_seq"),
    *_ci("balanced_accuracy", "float?", "None when truth holds one class"),
    *_ci("macro_f1", "float?", "None when truth holds one class"),
    _c("seq_precision", "float?"), _c("seq_recall", "float?"),
    _c("predicted_seq", "int?"), _c("predicted_classes", "json?", "raw prediction counts"),
    _c("above_chance", "bool?", "balanced_accuracy_lo > 0.5; None when small_n"),
    _c("reps", "int", "window-bootstrap replicates"), _c("seed", "int"),
)

_MSRC_DELTAS = (
    _c("protocol", "str"), _c("classifier", "str"),
    _c("baseline", "str", "shipped_synthetic/gnb or rule/route_window"),
    _c("label_rule", "str"), _c("trace", "str"), _c("metric", "str"),
    _c("n_windows", "int"), _c("small_n", "bool"),
    *_ci("delta", "float?", "classifier minus baseline, paired window bootstrap"),
    _c("verdict", "str", "better, worse, no difference detected, insufficient n"),
    _c("reps", "int"), _c("seed", "int"),
)

_MSRC_REPLAY = (
    _c("trace", "str"), _c("view", "str", "native or writes_as_reads"),
    _c("mode", "str"), _c("latency_model", "str", "measured (mean t1) or assumed 5/100/50"),
    _c("status", "str"), _c("requests", "int"), _c("read_requests", "int"),
    _c("read_blocks", "int"), _c("read_hits", "int"),
    _c("hit_ratio", "float?", "simulated block hit ratio over read blocks"),
    _c("hit_ratio_delta_vs_none", "float?"),
    _c("prefetches", "int"), _c("useful_prefetches", "int"), _c("precision", "float?"),
    _c("device_ios", "int"), _c("device_ios_ratio_vs_none", "float?"),
    _c("blocks_moved", "int"), _c("blocks_moved_ratio_vs_none", "float?"),
    _c("mean_latency_us", "float?", "cost model, not a measurement"),
    _c("hit_us", "float"), _c("demand_miss_us", "float"), _c("prefetch_us", "float"),
    _c("break_even_precision", "float?"), _c("capacity", "int"), _c("seed", "int"),
)

# ------------------------------------------------------------------ claims

#: results/claims/paper_claims.csv: the claims ledger (paper/claims.py). One
#: row per number macro the paper or the slides use, per generated table and
#: per traced table cell; regenerated by ``reproduce.py paper`` from the
#: committed results only.
_CLAIMS = (
    _c("claim_id", "str", "macro name, tables/<t>.tex, or tables/<t>.tex#<cell>"),
    _c("kind", "str", "value | derived | decision | decision_count | constant | missing | "
                      "table | table_cell | table_derived"),
    _c("used_in", "str", "';'-joined paper (or slide) files that use it"),
    _c("printed", "str", "the LaTeX text the paper prints"),
    _c("source_file", "str", "results-relative file(s), ';'-joined; a module for a constant"),
    _c("selector", "json?", "key=value of the identifying columns of the one row read"),
    _c("rows", "int?", "rows the value was derived from (1 for a single row)"),
    _c("columns", "str?", "';'-joined column(s) read"),
    _c("raw", "json", "the unformatted value(s) printed"),
    _c("formatter", "json", "[[name, args, kwargs, printed], ...] of genlib formatters"),
    _c("rule", "str?", "pre-registered rule id, for a decision"),
    _c("verdict", "str?", "the verdict the rule gives"),
    _c("preregistered", "str?", "the PREREGISTERED.md the rule is from"),
    _c("note", "str?", "derivation, scope and intervals"),
)

# --------------------------------------------------------------- classifier
# Track CLF: the classifier zoo (Part 1, synthetic windows) and the real-trace
# router benchmark (Part 2). results/classifier/PREREGISTERED.md fixes the
# comparisons; adaptive_prefetch.clfzoo produces every file.

_CLF_TUNING = (
    _c("setting", "str", "regime, or 'transfer' (block-size shift, trained at size 1)"),
    _c("family", "str", "clfzoo.zoo.FAMILIES"),
    _c("candidate", "int", "random-search index; 0 is the library default"),
    _c("params", "json"), _c("folds", "int", "train-seed folds (validate on the next train seed)"),
    _c("val_accuracy", "float", "mean over folds; the selection criterion"),
    _c("val_macro_f1", "float"), _c("selected", "bool"),
    _c("error", "str?", "why the candidate could not be fitted (scored 0)"),
    _c("wall_s", "float", "contended"), _c("load_avg", "float?"),
)

_CLF_EVAL = (
    _c("setting", "str"), _c("condition", "str", "'in_regime' or '1->N' (transfer)"),
    _c("family", "str"), _c("params", "json", "selected on training seeds only"),
    _c("val_accuracy", "float", "the selected candidate's validation accuracy"),
    _c("n_seed_pairs", "int"), _c("n_test_per_pair", "int"),
    *_ci("accuracy", "float", "test accuracy, mean over seed pairs; bootstrap over pairs"),
    _c("accuracy_sd", "float", "population sd over seed pairs"),
    *_ci("macro_f1", "float", "mean over seed pairs; bootstrap over pairs"),
    _c("ece", "float", "expected calibration error, 15 bins, mean over pairs"),
    _c("brier", "float", "multiclass Brier score, mean over pairs"),
    _c("confusion", "json", "summed over seed pairs; rows true, cols predicted"),
    _c("predict_us", "float", "median single-window predict(); CONTENDED host"),
    _c("train_ms", "float", "median fit time; CONTENDED host"),
    _c("load_avg", "float", "mean 1-min load during evaluation"),
    *_ci("delta_accuracy", "float?", "paired bootstrap over seed pairs vs gnb_online"),
    *_ci("delta_macro_f1", "float?", "paired bootstrap over seed pairs vs gnb_online"),
    _c("verdict_vs_gnb", "str", "better | worse | no difference detected | baseline"),
    _c("wins", "int?"), _c("losses", "int?"), _c("ties", "int?"),
)

_CLF_ROUTER_RUNS = (
    _c("trace", "str"), _c("corpus", "str"), _c("capacity", "int"),
    _c("policy", "str", "a clfzoo family routing 'adaptive', or a baseline mode label"),
    _c("source", "str", "replayed_labels | e1_sweep_reused | multicorpus_run_job"),
    _c("label_key", "str?", "clfzoo.router.label_key of the routing sequence"),
    _c("gnb_label_key", "str?"), _c("requests", "int"), _c("windows", "int?"),
    _c("hit_ratio", "float", "SIMULATED block hit ratio"), _c("precision", "float"),
    _c("total_prefetches", "int"), _c("useful_prefetches", "int"),
    _c("device_ios", "int"), _c("blocks_moved", "int"), _c("policy_switches", "int"),
    _c("share_sequential", "float?"), _c("share_strided", "float?"),
    _c("share_random", "float?"), _c("share_mixed", "float?"),
    _c("agreement_with_gnb", "float?", "fraction of windows routed as GNB routes them"),
    _c("age_table_evictions", "int?"), _c("wall_s", "float?", "contended"),
    _c("load_avg", "float?"),
)

_CLF_ROUTER_COMPARISONS = (
    _c("capacity", "int"), _c("metric", "str", "hit_ratio | device_ios | blocks_moved"),
    _c("statistic", "str", "difference | ratio"), _c("policy", "str"), _c("baseline", "str"),
    _c("n", "int", "paired traces"),
    _c("estimate", "float"), _c("ci_lo", "float"), _c("ci_hi", "float"),
    _c("cluster_estimate", "float"), _c("cluster_ci_lo", "float"), _c("cluster_ci_hi", "float"),
    _c("verdict", "str", "PREREGISTERED.md section 4: both intervals must agree"),
    _c("wins", "int"), _c("losses", "int"), _c("ties", "int"),
    _c("reps", "int"), _c("seed", "int"),
)

_CLF_ROUTER_SUMMARY = (
    _c("capacity", "int"), _c("policy", "str"), _c("n_traces", "int"),
    _c("mean_hit_ratio", "float", "SIMULATED"), _c("mean_precision", "float?"),
    _c("mean_wasted_share", "float?"),
    _c("device_io_ratio", "float", "mean over traces of device_ios / none's"),
    _c("blocks_moved_ratio", "float", "mean over traces of blocks_moved / none's"),
    _c("mean_agreement_with_gnb", "float?"), _c("min_agreement_with_gnb", "float?"),
    _c("traces_identical_to_gnb", "int?", "traces whose routing sequence equals GNB's"),
    _c("share_sequential", "float?"), _c("share_strided", "float?"),
    _c("share_random", "float?"), _c("share_mixed", "float?"),
)

# --------------------------------------------------------------- provenance

# ---------------------------------------------------------------- hindsight

#: Track HSR: the hindsight-labelled router (``adaptive_prefetch.hindsight``).
_HSR_ROWS = (
    _c("trace", "str"), _c("corpus", "str"),
    _c("skip_requests", "int", "0 = slice 1 (training), max_requests = slice 2 (test)"),
    _c("max_requests", "int"), _c("capacity", "int", "512-B blocks"),
    _c("policy", "str", "fixed action, vcf_/gvcf_<obj>, routed_<obj>[_gnb|_loco], "
       "oracle_<obj> (reference, not a policy), adaptive, adaptive_evidence, linux_ra@256"),
    _c("family", "str?", "router classifier family (routed_* rows)"),
    _c("objective", "str?", "hit | u0.5 | dev"),
    _c("action", "str?", "the fixed action a fixed or vcf row replays"),
    _c("status", "str"),
    _c("source", "str", "record | replay | sweep (reused after an exact reproduction)"),
    _c("read_blocks", "int?"), _c("read_hits", "int?"), _c("hit_ratio", "float?"),
    _c("precision", "float?"), _c("total_prefetches", "int?"),
    _c("useful_prefetches", "int?"),
    _c("wasted_share", "float?", "unused / issued; None when nothing was issued"),
    _c("u05_per_block", "float?", "(hits - 0.5 * prefetches) / read_blocks"),
    _c("demand_ios", "int?"), _c("prefetch_ios", "int?"),
    _c("prefetch_ios_mergeable", "int?"), _c("device_ios", "int?"),
    _c("blocks_moved", "int?"), _c("policy_switches", "int?"),
    _c("action_shares", "json?", "share of windows per action (routed/oracle rows)"),
    _c("training_range", "str?", "what the router or VCF choice was trained/selected on"),
    _c("sweep_match", "bool?", "equals results/disk/sweep.jsonl on every counter; "
       "None when no sweep row exists"),
    _c("load_start", "float?"), _c("load_end", "float?"), _c("wall_s", "float?"),
)

_HSR_SELECTION = (
    _c("capacity", "int"), _c("objective", "str"), _c("family", "str"),
    _c("config", "int", "random-search draw index"), _c("params", "json"),
    _c("cv_score", "float?", "mean held-out share of hindsight headroom over none"),
    _c("cv_accuracy", "float?", "label accuracy, diagnostic only"),
    _c("cv_fold_scores", "json"), _c("cv_traces_scored", "int"),
    _c("family_best", "bool"), _c("selected", "bool"),
    _c("n_train_windows", "int"), _c("wall_s", "float"), _c("load", "float?"),
)

_HSR_FAMILIES = (
    _c("capacity", "int"), _c("objective", "str"),
    _c("which", "str", "selected | a family name (its CV-best configuration)"),
    _c("family", "str"), _c("params", "json"), _c("cv_score", "float?"),
    _c("n_traces", "int"),
    *_ci("proxy_headroom", "float?", "slice-2 headroom captured, trajectory proxy; "
         "paired bootstrap over traces"),
    _c("label_accuracy", "float?", "slice-2 hindsight label accuracy, diagnostic"),
    _c("action_shares", "json"),
)

_HSR_SUMMARY = (
    _c("capacity", "int"), _c("policy", "str"), _c("n_traces", "int"),
    _c("mean_hit_ratio", "float"), _c("mean_u05_per_block", "float"),
    _c("mean_precision", "float?", "over traces that issued a prefetch"),
    _c("mean_wasted_share", "float?", "same traces"),
    *_ci("device_io_ratio", "float", "mean per-trace device_ios / none's, paired bootstrap"),
    *_ci("blocks_moved_ratio", "float", "mean per-trace blocks_moved / none's"),
    _c("headroom_captured", "float?", "(routed - vcf) / (oracle - vcf) of the per-trace "
       "means; device I/Os relative to none"),
    _c("action_shares", "json?"),
)

_HSR_COMPARISONS = (
    _c("capacity", "int"), _c("comparison", "str", "H1-H7, S1-S2, R, D (PREREGISTERED.md)"),
    _c("metric", "str", "hit_ratio | u0.5 | device_ios | blocks_moved"),
    _c("policy", "str"), _c("baseline", "str"), _c("primary", "bool"),
    _c("statistic", "str", "difference | ratio"), _c("n", "int"),
    _c("estimate", "float"), _c("ci_lo", "float"), _c("ci_hi", "float"),
    _c("cluster_estimate", "float"), _c("cluster_ci_lo", "float"),
    _c("cluster_ci_hi", "float"),
    _c("wins", "int"), _c("losses", "int"), _c("ties", "int"),
    _c("verdict", "str", "better | worse | no difference detected [...]"),
)

_PROVENANCE = tuple(_c(key, "json?" if key not in ("format", "study") else "str")
                    for key in _PROV_KEYS) + tuple(
    _c(key, "json", required=False) for key in _PROV_OPTIONAL)


def _schema(name, fmt, producer, description, columns) -> Schema:
    return Schema(name, fmt, producer, description, tuple(columns))


SCHEMAS: dict[str, Schema] = {s.name: s for s in (
    _schema("disk/sweep.jsonl", "jsonl", "WP10 (E1)",
            "one row per sweep job (SWEEP_KEY)", _SWEEP),
    _schema("disk/table_main.csv", "csv", "WP10 analysis (E1)",
            "T2: pooled per-policy table, corpus='all'", _DISK_TABLE),
    _schema("disk/table_by_corpus.csv", "csv", "WP10 analysis (E1)",
            "T2 per-corpus sub-columns; same columns as table_main", _DISK_TABLE),
    _schema("disk/deltas.csv", "csv", "WP10 analysis (E1)",
            "every headline delta with paired and cluster CIs", _DISK_DELTAS),
    _schema("disk/winners.csv", "csv", "WP10 analysis (E1)",
            "per-trace winner counts by hit ratio and device I/Os", _DISK_WINNERS),
    _schema("disk/kappa.csv", "csv", "WP10 analysis (E1)",
            "kappa* per (trace, policy); F4", _DISK_KAPPA),
    _schema("disk/corpus_profile.csv", "csv", "WP10 (E1)",
            "datasets table, one row per trace", _DISK_CORPUS),
    _schema("gpu/disk_learned_depth.csv", "csv", "WP9 (E2)",
            "learned_depth vs VCF per (trace, seed, action_cap)", _DISK_LEARNED_DEPTH),
    _schema("gpu/disk_lstm.csv", "csv", "WP9 (E2; repro/gpu_disk.LSTM_COLUMNS)",
            "honest-split LSTM per trace: delta accuracy and cache metrics", _DISK_LSTM),
    _schema("gpu/disk_depth_baselines.csv", "csv", "WP9 (E2; repro/gpu_disk.BASELINE_COLUMNS)",
            "every seed-free baseline replay: validation grids and test references",
            _DISK_DEPTH_BASELINES),
    _schema("gpu/gpu_env.json", "json", "WP9 (E2)",
            "torch/CUDA versions, per-job GPU contention samples, run config", _GPU_ENV),
    _schema("realio/realio.csv", "csv", "WP8 (E3)",
            "one row per (device, backend, mode, think, repeat)", _REALIO),
    _schema("realio/realio_summary.csv", "csv", "WP8 (E3)",
            "mean and CI over repeats; T3/F5", _REALIO_SUMMARY),
    _schema("realio/device_*.json", "json", "WP8 (E0)",
            "device calibration for IOCostModel", _REALIO_DEVICE),
    _schema("web/policy_metrics.csv", "csv", "WP6 (E4)",
            "one row per (split, scorer, cache_scope, capacity, policy)", _WEB_POLICY),
    _schema("web/comparisons.csv", "csv", "WP6 (E4)",
            "T6: per-capacity validated comparisons", _WEB_COMPARISONS),
    _schema("web/decomposition.csv", "csv", "WP6 (E4, E5)",
            "T5: what moves the needle", _WEB_DECOMPOSITION),
    _schema("web/freshness.csv", "csv", "WP6 (E4)",
            "F6: coverage@k vs index freshness", _WEB_FRESHNESS),
    _schema("web/seq_metrics.csv", "csv", "WP11 (E5)",
            "GPU sequence model per (model, protocol, seed, capacity)", _WEB_SEQ),
    _schema("web/calibration.csv", "csv", "WP6 (E4; pandas, diagnostics.CALIBRATION_COLUMNS)",
            "candidate-level reliability bins and Brier skill per (split, scorer)",
            _WEB_CALIBRATION),
    _schema("web/learning_curve.csv", "csv",
            "WP6 (E4; pandas, diagnostics.LEARNING_CURVE_COLUMNS)",
            "P13: ranker retrained at larger training samples, validation only",
            _WEB_LEARNING_CURVE),
    _schema("web/dataset_profile.json", "json", "WP6 (E4)",
            "prepare's profile plus replayed views and pair kinds; T4", _WEB_DATASET_PROFILE),
    _schema("web/think_time.json", "json", "WP6 (E4)",
            "time to the session's next view per split", _WEB_THINK_TIME),
    _schema("web/session_parity.json", "json", "WP6 (E4; demo_server.run_parity)",
            "demo step loop vs batch replay on held-out sessions", _WEB_SESSION_PARITY),
    _schema("live/live_metrics.csv", "csv", "WP12 (E6)",
            "T7/F7: one row per pass and policy", _LIVE_METRICS),
    _schema("live/live_parity.csv", "csv", "WP12 (E6)",
            "live vs simulator counters per pass and policy", _LIVE_PARITY),
    _schema("live/live_config.json", "json", "WP12 (E6)",
            "every run's command, plan, assumptions and passes", _LIVE_CONFIG),
    _schema("live/disk_probe_*.json", "json", "WP12 (E6)",
            "MEASURED cold/warm/hinted page-cache reads of one object store",
            _LIVE_DISK_PROBE),
    _schema("msrc_samples/traces.csv", "csv", "WPMSRC (E7)",
            "one row per MSRC sample trace: windows and flag counts", _MSRC_TRACES),
    _schema("msrc_samples/classifier_metrics.csv", "csv", "WPMSRC (E7)",
            "binary seq-flag agreement per (protocol, classifier, label rule, trace)",
            _MSRC_METRICS),
    _schema("msrc_samples/classifier_deltas.csv", "csv", "WPMSRC (E7)",
            "paired balanced-accuracy differences vs shipped GNB and route_window",
            _MSRC_DELTAS),
    _schema("msrc_samples/replay.csv", "csv", "WPMSRC (E7)",
            "sanity replay per (trace, view, cost model, mode)", _MSRC_REPLAY),
    _schema("claims/paper_claims.csv", "csv", "paper/claims.py (reproduce.py paper)",
            "the claims ledger: every number the paper uses, traced to results/", _CLAIMS),
    _schema("classifier/synthetic_tuning.csv", "csv", "track CLF (Part 1)",
            "every random-search candidate's validation score", _CLF_TUNING),
    _schema("classifier/synthetic_eval.csv", "csv", "track CLF (Part 1)",
            "selected classifiers on held-out seeds, paired vs gnb_online", _CLF_EVAL),
    _schema("classifier/router_runs.csv", "csv", "track CLF (Part 2)",
            "one row per (trace, capacity, policy) on C3 slice 2", _CLF_ROUTER_RUNS),
    _schema("classifier/router_comparisons.csv", "csv", "track CLF (Part 2)",
            "paired and cluster comparisons over traces", _CLF_ROUTER_COMPARISONS),
    _schema("classifier/router_summary.csv", "csv", "track CLF (Part 2)",
            "per (capacity, policy) means and routing agreement", _CLF_ROUTER_SUMMARY),
    _schema("hindsight/rows.csv", "csv", "HSR",
            "every slice-1 and slice-2 replay of the hindsight study", _HSR_ROWS),
    _schema("hindsight/selection.csv", "csv", "HSR",
            "grouped-CV score of every random-search configuration", _HSR_SELECTION),
    _schema("hindsight/families.csv", "csv", "HSR",
            "every family's best router on slice 2 under the trajectory proxy",
            _HSR_FAMILIES),
    _schema("hindsight/summary.csv", "csv", "HSR",
            "per (capacity, policy) means and device work against none", _HSR_SUMMARY),
    _schema("hindsight/comparisons.csv", "csv", "HSR",
            "pre-registered comparisons with paired and cluster intervals",
            _HSR_COMPARISONS),
    _schema("*/provenance.json", "json", "prefetch_common.provenance",
            "provenance.write_provenance output", _PROVENANCE),
)}


# ------------------------------------------------------------------- lookup

def schema_for(name: str | Path) -> Schema:
    """The schema for a key or a path under ``results/`` (patterns match)."""
    key = Path(name).as_posix() if isinstance(name, Path) else str(name)
    if key in SCHEMAS:
        return SCHEMAS[key]
    for pattern, schema in SCHEMAS.items():
        if "*" in pattern and fnmatch.fnmatchcase(key, pattern):
            return schema
    raise SchemaError(f"no results schema for {key!r}; known: {sorted(SCHEMAS)}")


def columns(name: str) -> tuple[str, ...]:
    """Column names of a results file, in file order."""
    return schema_for(name).names


# --------------------------------------------------------------- validation

def _is_numpy_bool(value) -> bool:
    dtype = getattr(value, "dtype", None)
    return getattr(dtype, "kind", None) == "b" and getattr(value, "ndim", 1) == 0


def _type_ok(column: Column, value) -> bool:
    if value is None:
        return column.nullable
    base = column.base
    if base == "json":
        return True  # checked when serialised
    if base == "str":
        return isinstance(value, str)
    if base == "bool":
        return isinstance(value, bool) or _is_numpy_bool(value)
    if isinstance(value, bool) or _is_numpy_bool(value):
        return False  # bool is an int; never accept it as a number
    if base == "int":
        return isinstance(value, numbers.Integral)
    return isinstance(value, numbers.Real)  # float


def validate_row(name: str, row: Mapping) -> None:
    """Raise :class:`SchemaError` unless ``row`` matches the schema."""
    schema = schema_for(name)
    _validate(schema, row)


def _validate(schema: Schema, row: Mapping, where: str = "row") -> None:
    if not isinstance(row, Mapping):
        raise SchemaError(f"{schema.name}: {where} is {type(row).__name__}, not a mapping")
    known = set(schema.names)
    unknown = [k for k in row if k not in known]
    missing = [k for k in schema.required if k not in row]
    if unknown or missing:
        raise SchemaError(f"{schema.name}: {where} has unknown columns {unknown} "
                          f"and missing columns {missing}")
    for column in schema.columns:
        if column.name in row and not _type_ok(column, row[column.name]):
            raise SchemaError(f"{schema.name}: {where} column {column.name!r} expects "
                              f"{column.dtype}, got {row[column.name]!r}")


def validate_rows(name: str, rows: Iterable[Mapping]) -> list[Mapping]:
    """Validate every row; returns them as a list. ``json`` takes one row."""
    schema = schema_for(name)
    rows = list(rows)
    if schema.fmt == "json" and len(rows) != 1:
        raise SchemaError(f"{schema.name}: a json results file holds exactly one object, "
                          f"got {len(rows)}")
    for index, row in enumerate(rows):
        _validate(schema, row, f"row {index}")
    return rows


# ------------------------------------------------------------ serialisation

def _plain(value):
    """numpy scalars -> Python scalars, recursively; leaves the rest alone."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if _is_numpy_bool(value):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def _csv_cell(column: Column, value) -> str:
    if value is None:
        return ""
    if value == "" and column.base == "str" and column.nullable:
        # An empty field reads back as None in a nullable column.
        raise SchemaError(f"column {column.name!r}: write None, not '', in a nullable "
                          "str column (csv cannot tell them apart)")
    value = _plain(value)
    if column.base == "json":
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if column.base == "float":
        return repr(float(value))
    if column.base == "bool":
        return "True" if value else "False"
    return str(value)


def _json_text(row: Mapping, schema: Schema) -> str:
    ordered = {c.name: _plain(row[c.name]) for c in schema.columns if c.name in row}
    try:
        return json.dumps(ordered, allow_nan=False)
    except ValueError as error:
        raise SchemaError(f"{schema.name}: non-finite float in a JSON row ({error})") from None


def _csv_columns(schema: Schema, rows: list[Mapping]) -> list[Column]:
    """Columns a csv file is written with: every required one, plus each
    optional one that at least one row carries (the rest of the rows get
    ``None`` there, so an optional csv column must be nullable)."""
    present = [c for c in schema.columns
               if c.required or any(c.name in row for row in rows)]
    for column in present:
        if not column.required and not column.nullable:
            raise SchemaError(f"{schema.name}: optional csv column {column.name!r} "
                              "must be nullable")
    return present


def default_path(name: str) -> Path:
    if "*" in name:
        raise SchemaError(f"{name!r} is a pattern; pass a concrete path")
    return RESULTS_ROOT / name


def write_rows(name: str, rows: Iterable[Mapping], path: str | Path | None = None, *,
               append: bool = False) -> Path:
    """Validate ``rows`` against ``name``'s schema, then write them.

    ``name`` is a schema key or a concrete path under ``results/`` that
    matches one (``realio/device_sdb.json``). ``path`` defaults to
    ``results/<name>``. Nothing is written if any row fails. ``csv`` and
    ``json`` replace the file atomically; ``append=True`` is allowed only for
    ``jsonl`` (the resumable sweep) and adds lines to an existing file.
    """
    schema = schema_for(name)
    rows = validate_rows(name, rows)
    target = Path(path) if path is not None else default_path(name)
    if append and schema.fmt != "jsonl":
        raise SchemaError(f"{schema.name}: append is only supported for jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    if schema.fmt == "jsonl":
        text = "".join(_json_text(row, schema) + "\n" for row in rows)
        if append:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
            return target
    elif schema.fmt == "json":
        ordered = {c.name: _plain(rows[0][c.name]) for c in schema.columns if c.name in rows[0]}
        try:
            text = json.dumps(ordered, indent=2, allow_nan=False) + "\n"
        except ValueError as error:
            raise SchemaError(f"{schema.name}: non-finite float ({error})") from None
    else:
        present = _csv_columns(schema, rows)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(c.name for c in present)
        for row in rows:
            writer.writerow(_csv_cell(c, row.get(c.name)) for c in present)
        text = buffer.getvalue()
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    return target


def _parse_cell(column: Column, text: str, where: str):
    if text == "" and (column.nullable or column.base != "str"):
        if not column.nullable:
            raise SchemaError(f"{where}: empty value in non-nullable column {column.name!r}")
        return None
    base = column.base
    try:
        if base == "int":
            return int(text)
        if base == "float":
            return float(text)
        if base == "bool":
            lowered = text.lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
            raise ValueError(text)
        if base == "json":
            return json.loads(text)
    except ValueError:
        raise SchemaError(f"{where}: column {column.name!r} expects {column.dtype}, "
                          f"got {text!r}") from None
    return text


def read_rows(name: str, path: str | Path | None = None) -> list[dict]:
    """Read and validate a results file; values come back typed."""
    schema = schema_for(name)
    source = Path(path) if path is not None else default_path(name)
    text = source.read_text(encoding="utf-8")
    if schema.fmt == "csv":
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if header is None:
            raise SchemaError(f"{source}: empty file, expected a header")
        by_name = {c.name: c for c in schema.columns}
        expected = [c.name for c in schema.columns if c.required or c.name in header]
        if header != expected:
            unknown = [h for h in header if h not in by_name]
            missing = [n for n in expected if n not in header]
            raise SchemaError(f"{source}: header does not match {schema.name} "
                              f"(unknown {unknown}, missing {missing}, or out of order)")
        present = [by_name[name] for name in header]
        rows = []
        for line, record in enumerate(reader, start=2):
            if len(record) != len(header):
                raise SchemaError(f"{source}:{line}: {len(record)} fields, expected {len(header)}")
            rows.append({c.name: _parse_cell(c, cell, f"{source}:{line}")
                         for c, cell in zip(present, record)})
    elif schema.fmt == "jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = [json.loads(text)]
    for index, row in enumerate(rows):
        _validate(schema, row, f"{source} row {index}")
    return rows


def validate_file(path: str | Path, name: str | None = None) -> list[dict]:
    """:func:`read_rows` with the schema chosen from ``path`` when ``name`` is
    omitted (``path`` must then end in a key, e.g. ``.../disk/kappa.csv``)."""
    path = Path(path)
    if name is None:
        name = _key_from_path(path)
    return read_rows(name, path)


def _key_from_path(path: Path) -> str:
    # Every key is <study directory>/<file name>.
    return "/".join(path.parts[-2:])


def validate_tree(root: str | Path) -> dict[str, str]:
    """Check every ``.csv``/``.json``/``.jsonl`` under ``root`` (laid out like
    ``results/``). Returns ``{relative path: problem}``; empty means all pass.
    Other files (a README, ``PREREGISTERED.md``) are ignored, and so is
    everything under a :data:`VERBATIM_DIRS` directory except its
    ``provenance.json``."""
    root = Path(root)
    problems = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in (".csv", ".json", ".jsonl"):
            continue
        relative = path.relative_to(root).as_posix()
        if relative.split("/")[0] in VERBATIM_DIRS and path.name != "provenance.json":
            continue
        try:
            read_rows(relative, path)
        except (SchemaError, ValueError, OSError) as error:
            problems[relative] = str(error)
    return problems


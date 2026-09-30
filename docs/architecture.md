# Architecture

User-space, simulation-first prototype for *ML-Based Adaptive Disk I/O
Prefetching Using Workload Pattern Classification*. It does **not** modify the
operating system or perform physical disk reads; it replays recorded I/O
requests against a modelled cache and reports the cache-level outcomes of each
prefetch policy.

## Design principles

1. **Simulation over OS hooks.** Everything runs on recorded request streams;
   no kernel module, device firmware, or real latency measurements.
2. **Fair comparison.** Every policy sees the identical request stream, cache
   capacity, first-window warm-up, and LRU rules. Differences in outcome are
   attributable to the policy, not the harness.
3. **Causal replay.** A window is classified only after all its requests have
   occurred, and its policy affects *later* requests only. The first window
   never prefetches.
4. **Stdlib-only core.** The entire pipeline except the optional LSTM runs on
   the Python standard library. PyTorch is an optional extra
   (`pip install -e ".[lstm]"`); without it, the LSTM row is *skipped*, never
   silently replaced.
5. **Honest labels.** Ground truth exists only for synthetic workloads. On real
   traces the classifier is frozen (or, opt-in, self-trained with a documented
   weak label); updates require a trustworthy label and are never assumed.

## High-level data flow

```text
 trace file (normalized / msr / iotta8 / alibaba / revised)
 or synthetic generator
        |
        v
 Request records (timestamp_ms, lba, size_blocks, operation, stream_id)
        |
        +-- gathered into fixed-size windows (window_size, default 32)
        |                      |
        |                      v
        |            feature extraction (12 features)
        |                      |
        |                      v
        |          OnlineGaussianNB (frozen or incremental)
        |                      |
        |                      v
        |           class -> policy routing (seq/strided/random/mixed)
        |                      |
        `----> candidate generation (per request)
                          |
                          v
                 LRU cache replay + latency model
                          |
                          v
              Metrics -> benchmark table -> result interpretation
```

Two independent candidate sources feed the same cache: fixed heuristics and
the ML classifier (`simulator.candidates`) on one side, and the learned
per-request predictors (`baselines.py`, `lstm.py`, exposing
`next_candidates(request)`) on the other.

## Module map

| Module | Responsibility | Entry points |
| --- | --- | --- |
| `trace.py` | Canonical `Request` record, CSV/trace import for five schemas, synthetic labelled workload generation | `load_csv`, `write_csv`, `synthetic_window`, `synthetic_dataset`, `transition_trace` |
| `features.py` | Fixed-size window to 8 numeric features | `extract`, `dominant_stride` |
| `model.py` | `OnlineGaussianNB` (diagonal, default) and `OnlineQDA` (full covariance + shrinkage), both with exponential decay | `make_classifier`, `CLASSIFIERS`, `DEFAULT_CLASSIFIER` |
| `baselines.py` | Causal per-request stride and Markov prefetchers | `StridePrefetcher`, `MarkovPrefetcher` |
| `lstm.py` | Optional offline LSTM predicting the next address *delta* as (sign, magnitude-bucket) with explicit abstention | `train_lstm`, `load_predictor`, `LSTMPrefetcher` |
| `simulator.py` | Causal block-level LRU replay, latency model, class->policy routing, perfect-predictor bound | `replay`, `LRUCache`, `LatencyModel`, `oracle_reference`, `confusion_matrix` |
| `guard.py` | Feedback-gated policies: observed prefetch precision vs `break_even_precision()`, stride confirmation | `PrecisionMonitor`, `CorrelateDetector`, `GateController`, `read_ahead`, `depth_for_margin` |
| `benchmark.py` | Twelve-policy comparison, drift diagnostics, aggregation, result interpretation | `benchmark_dataset`, `drift_report`, `aggregate_results`, `analyze_results`, `markdown_table` |
| `training.py` | Conservative weak-label adaptation for the unlabelled MSR sample | `adapt_msr_sample` |
| `artifacts.py` | Versioned JSON persistence tagged by classifier kind (no pickle) | `save_model`, `load_model` |
| `eda.py` | Trace profiling, locality/reuse analysis, k-means elbow, domain-shift and label-support measurement | `profile_trace`, `domain_shift`, `label_support`, `locality_profile` |
| `pipeline.py` | Preprocessing, policy postprocessing, configurable regimes, model selection | `StandardScaler`, `PolicySmoother`, `evaluate_classifier`, `select_classifier` |
| `report.py` | Findings-vs-raw-data split for the console and the `outputs/` file map | `dataset_overview`, `classifier_section`, `benchmark_section`, `drift_section` |
| `cli.py` | argparse front-end: `demo`, `replay`, `benchmark`, `normalize`, `train-lstm`, `train-msr-sample`, `export-demo`, `evaluate`, `eda`, `shift` | `main` |
| `main.py` | End-to-end run: EDA, classifier selection, domain shift, benchmark, drift. Console shows findings; `outputs/` holds every raw row | `main` |

## Trace layer (`trace.py`)

`Request` is an immutable dataclass validated in `__post_init__`:
`timestamp_ms >= 0`, `lba >= 0`, `size_blocks > 0`, `operation in {R, W}`.

`load_csv(path, format)` dispatches on an explicit profile or auto-detects by
column names:

| Profile | Schema | Unit conversion |
| --- | --- | --- |
| `normalized` | `timestamp_ms,lba,size_blocks,operation[,stream_id]` | none (canonical) |
| `msr` | `Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime` | timestamps→elapsed ms, byte offset/size→512-B blocks, Read/Write→R/W |
| `iotta8` | `device,sector,size,op,offset,timestamp,lifetime,count` | µs→ms; sector index/count kept |
| `alibaba` | `device_id,opcode,offset,length,timestamp` | µs→ms, byte offset/length→blocks |
| `revised` | whitespace-separated `time_s op lba size seq\|rand t1 t2 t3` (MSRC-trace-003 final-trace) | s→ms elapsed; RS/WS→R/W; sector lba/size kept |

All real-trace loaders validate ascending timestamps and reject malformed
rows with a line number. The synthetic generators produce labelled windows
(`sequential`, `strided`, `random`, `mixed`) with configurable noise; the
label describes the construction, never a real trace.

## Feature layer (`features.py`)

For each complete window of `n >= 8` requests, 8 size-aware features in
`FEATURE_NAMES`:

1. `contiguous_ratio` - adjacent LBAs continuing the previous request's size
2. `dominant_stride_ratio` - most frequent non-contiguous, non-zero delta
3. `random_jump_ratio` - deltas with `abs > 64`
4. `short_run_ratio` - deltas with `abs <= 16`
5. `unique_ratio` - distinct LBAs / window length
6. `gap_cv` - coefficient of variation of inter-request timing (capped at 10)
7. `fast_gap_ratio` - fraction of gaps below 0.15 ms
8. `read_ratio` - fraction of read requests

`dominant_stride(window)` is reused by the simulator as a separate heuristic:
the most common delta between 1 and 1024 blocks, required at least
`max(2, len/4)` times.

## Classifier (`model.py`)

Two interchangeable classifiers share one `update`/`predict` contract. The
default is `OnlineGaussianNB`; `OnlineQDA` keeps a full per-class scatter
matrix with diagonal shrinkage and a cached Cholesky factor, and is selected
with `--classifier qda`. The choice was made on measurement, not preference:
see [DECISIONS.md](DECISIONS.md) D1 for the regime table and the
block-size transfer test that demoted QDA back to opt-in.

`OnlineGaussianNB` is a Gaussian Naive Bayes classifier with:

- per-class running means and M2 variances updated incrementally
- **exponential decay** (`decay_factor`, default 0.995) applied to historical
  counts and variances on every update, so old observations fade and the
  model can follow concept drift
- a `variance_floor` (0.0025) preventing log(0) and over-confident fits
- softmax-like confidence: `1 / sum(exp(score_i - score_max))`

`update(values, label)` requires a trustworthy label (synthetic only). On real
traces the model is frozen by default; `training.py` demonstrates an opt-in
weak-label adaptation for the bundled MSR sample with an explicit proxy rule,
and `--online-real` adds a confidence-gated self-training row that is
documented as *not* evidence of real-trace accuracy.

Persistence is JSON (`artifacts.py`): means, M2s, counts, hyperparameters,
and metadata, with format/feature/class compatibility checks on load.

## Simulator (`simulator.py`)

`replay(requests, ...)` is the single causal evaluation loop:

1. Builds an oracle `Counter` of future re-accesses for recall scoring only.
2. Creates one `LRUCache` (capacity, default 128) and one latency model.
3. For each request, in order:
   - touches every block (`R` → `cache.read`, `W` → `cache.write`)
   - asks the active candidate source for prefetch blocks
   - prefetches only after the first `window_size` requests (warm-up)
4. After each completed window: refreshes `dominant_stride`, and in adaptive
   mode classifies the window, possibly switching policy
   (`policy_switches`), optionally updating the model with a label or a
   confident pseudo-label.

Class→policy routing (fixed mapping):

| Class | Candidate policy |
| --- | --- |
| `sequential` | two-block read-ahead |
| `strided` | learned window stride, one block |
| `random` | none |
| `mixed` | one-block read-ahead |

`Metrics` accumulate read blocks/hits, prefetches, useful prefetches,
prefetchable misses, policy switches, pseudo updates, modelled latency, and
timed inference. Derived properties: hit ratio, prefetch precision, oracle
re-access recall, unused prefetches, mean access latency, inference µs/call.

`LatencyModel` defaults: 5 µs hit, 100 µs demand miss, 50 µs prefetch. It is
a *configured cost model*, not a measurement.

## Baselines (`baselines.py`, `lstm.py`)

All baselines implement `next_candidates(request) -> list[int]` and are
causal: they consume the current request and propose blocks for later reads.

- `StridePrefetcher` - detects a repeated delta, then prefetches up to
  `degree` blocks at that stride.
- `MarkovPrefetcher` - 1st/2nd-order transitions between address deltas;
  proposes the top-k most probable next deltas above a confidence threshold.
- `LSTMPrefetcher` - offline LSTM (1 input unit, 64 hidden, softmax over a
  256-delta vocabulary) predicting the next delta from the last 16 deltas;
  prefetches up to two positive in-range candidates from the top-10 scores.
  It predicts *deltas*, so compare cache outcomes and compute cost, never
  four-class accuracy.

## Benchmark (`benchmark.py`)

`benchmark_dataset` replays one trace under all `MODES = (none, sequential,
strided, stride, markov, lstm, adaptive, adaptive_evidence, guard,
depth_adaptive, correlate, deep)` with identical parameters, computes
`speedup_vs_none = baseline_mean_latency / mode_mean_latency`, and returns one
row per mode. Missing LSTM → row with `status="skipped: ..."`, not a fake
result. `aggregate_results` merges per-trace rows with request-weighted
counters. `drift_report` compares frozen vs genuinely labelled online GNB on
a synthetic transition trace, reporting per-phase switch lag and phase
accuracy. `analyze_results` interprets the table: winning policy per dataset
with speedup and hit-ratio gains, flags degenerate ties, calls out a
different best-hit-ratio mode, and adds relative adaptive-vs-baseline gains.

## Feedback-gated policies (`guard.py`)

`guard`, `depth_adaptive` and `correlate` use **no classifier**. `LRUCache`
already counts `useful_prefetches`, so the realised precision of the system's
own speculative reads is observable at read time — causally, with no future
knowledge. `GateController` compares that rolling precision against
`LatencyModel.break_even_precision()` and sets read-ahead depth from the
margin: depth 0 when precision is below break-even, otherwise depth scaled
from 1 to 8.

Two mechanisms keep the controller from getting stuck:

* **Exploration** — `EXPLORATION_PREFETCHES` (256) speculative reads are issued
  before the estimate is trusted. Without it the controller deadlocks: no
  measurement means no prefetch, and no prefetch means no measurement.
* **Re-probe** — after `REPROBE_AFTER` (8) consecutive closed decisions the
  gate spends one interval prefetching again. Without it a gate that closes
  never reopens, because closing stops the evidence that would reopen it.

`correlate` additionally requires a stride confirmed over a longer history
than one window (`CorrelateDetector`, bounded table, least-seen eviction).
It is the only policy that scores on a stride-512 scan.

`replay()` and `replay_stream()` both drive one `GateController` rather than
each reimplementing the decision, and `_publish_gate()` is the single place
gate state reaches `Metrics`. Duplicating that logic previously left the
streamed path without these modes entirely, which crashed the sweep.

## CLI and interactive entry points

`cli.py` exposes subcommands for targeted runs (`demo`, `replay`,
`benchmark`, `normalize`, `train-lstm`, `train-msr-sample`, `export-demo`);
all real-trace commands accept `--format` and the LSTM-related commands a
`--lstm-model`/`--save` path.

`main.py` is the guided full run: it asks for demo/cache/window/dataset
selection (synthetic, MSR sample, MSRC-trace-003 final traces, custom file),
optional LSTM inclusion with a retrain-or-cached prompt, latency model, and
output saving; then prints the cost-model block with its break-even, the
classifier demo, the benchmark table, the result interpretation, and the drift
report, and saves `outputs/results.md` + `outputs/results.csv`. With no
arguments it runs non-interactively. Note that it does **not** apply a
per-trace request limit when training the LSTM -- it trains on the whole
selected sample and prints a duration warning.

`--prefetch-multiple` (default 1.0) sets what a speculative read costs
relative to a demand read. It is a first-class flag on both `main.py` and
`sweep_modes.py` because the setting decides whether the modelled-cost column
is meaningful at all; see the cost model note below.

`sweep_modes.py` is the collection-wide runner: it streams every `.revised`
trace in bounded chunks under every mode in `sweep.RANKING_MODES`, so the
32-file, 11.86 GB collection is swept without materialising it.

## Mode map

| Family | Modes | What decides the policy |
| --- | --- | --- |
| Fixed | `none`, `sequential`, `deep`, `strided` | Nothing. `deep` varies only depth (`DEFAULT_READ_AHEAD_DEPTH`, 8 blocks). |
| Online per-request | `stride`, `markov`, `lstm` | Address deltas. Expose `next_candidates(request)`. |
| Classifier-routed | `adaptive` | A 4-class window label from `OnlineGaussianNB`, one decision per window. |
| Evidence-routed | `adaptive_evidence` | `route_window()` on the window's own measured features. No model. |
| Feedback-gated | `guard`, `depth_adaptive`, `correlate` | `GateController` comparing *realised* prefetch precision against break-even. No model, no prediction. |


## Evaluation metrics and their limits

- **Hit ratio** - read hits / requested read blocks (cache-level).
- **Prefetch precision** - useful prefetches / prefetches issued.
- **Oracle re-access recall** - uses future-read information *only* for
  scoring, never for candidate generation; it can be high even with low hit
  ratio, so read it alongside hit ratio and wasted work.
- **Unused prefetches** - prefetches never used during replay (including
  still-resident at stream end): a proxy for cache pollution.
- **Modelled latency/speedup** - under configured hit/miss/prefetch costs;
  queueing, bandwidth, asynchronous completion, and device scheduling are not
  modelled. **This metric is degenerate at the default
  `prefetch_multiple=1.0`**: break-even precision is 1.0101, so no prefetcher
  can pay for itself and "no prefetch" wins the column by arithmetic. Read
  `LatencyModel.break_even_precision()` before interpreting any cost ranking.
  The **hit** cost remains an approximation (1% of mean service) because the
  traces record device time only, never cache time.
- **Observed precision / margin / stride** (`observed_precision`,
  `precision_margin`, `observed_stride`) - what the feedback gate actually
  measured, as opposed to predicted. `observed_precision` is `None` when the
  gate never issued a prefetch, so "never measured" stays distinguishable from
  "measured as zero". These are written to `outputs/msrc_sweep.csv` per row.

## Non-goals

- OS/kernel integration, real device benchmarks
- Real-trace four-class ground truth (unavailable; classifier kept frozen by
  default on real traces)
- Redistribution of MSR Cambridge or SNIA IOTTA traces (licensing)
- Performance claims beyond the modelled cache simulation
- Adapting read-ahead *depth*. Depth is the dimension that decides the
  hit-ratio outcome (D13, D17: +4.10 points from depth alone versus a 0.55
  point spread across every other mode), and nothing in this codebase varies
  it per window. That is the main open problem, and it is a regression on
  observed reuse distance rather than a 4-class label.
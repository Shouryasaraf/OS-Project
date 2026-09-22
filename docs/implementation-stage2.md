# Implementation Plan — Stage 2 (delivers "Operating Systems Lab Submission 2" spec)

Source of truth: [`Operating Systems Lab Submission 2.pdf`](reference/Operating%20Systems%20Lab%20Submission%202.pdf) (Project Review 1).
Goal of Stage 2: **actually build and prove what the PDF promises** — classify
access pattern in real time, switch prefetch policy per class, learn online, and
benchmark against **stride, Markov, and offline-LSTM baselines**.

---

## 1. PDF requirements → current state → gap

| PDF requirement (slide) | Review 2 state today | Stage 2 gap |
| --- | --- | --- |
| Real-time classification: seq/random/strided/mixed (8) | ✅ `OnlineGaussianNB`, 4 classes, 8 features | None — keep, harden |
| Online learning, no offline retrain (8, 10) | ⚠️ Online updates only on *labelled synthetic*; real traces frozen | Add drift-adaptation benchmark + optional confidence-gated pseudo-label updates |
| Policy selected per detected class (10) | ✅ class→policy routing in `simulator.replay` | None — keep |
| **Baseline: Stride prefetcher** (11) | ❌ only fixed-`strided` mode (stride re-learned per window, 1 candidate) | Implement classic stride prefetcher (multi-block, on-line delta detect) |
| **Baseline: Markov-chain prefetcher** (11) | ❌ absent | Implement delta-state Markov prefetcher (stdlib) |
| **Baseline: offline LSTM model** (11) | ❌ absent (stdlib-only project) | Implement offline LSTM delta predictor behind optional torch extra |
| Metrics: accuracy, precision/recall, hit ratio, **latency** (11) | ⚠️ accuracy ✓ precision ✓ hit ✓ ; recall ✗ latency ✗ | Add prefetch recall (oracle-lookahead) + configurable latency model |
| Datasets: **MSR Cambridge + SNIA IOTTA** (11) | ⚠️ MSR loader + 1000-req sample only | IOTTA loader + trace normalization command |
| Expected outcomes (11): hit > heuristics on mixed real traces; ≈LSTM accuracy at fraction of compute; faster drift adaptation | — | Benchmark harness that produces evidence for all three |

## 2. Architecture (unchanged spine, pluggable policy layer)

Same 5-stage pipeline (see [`PIPELINE.md`](PIPELINE.md)). Change: **Stage 4 policy selection
becomes a pluggable predictor registry** so every baseline + the adaptive
classifier feed the same `candidates()`-style signature and are compared under
identical causality, stream, cache, and LRU rules.

```
request stream → windowing → 8 features → GNB classifier ──┐
                                                             ├─→ prefetch candidate generator → LRU cache → metrics
stride/markov/LSTM predictors ──(per-request, on-line)→─────┘          (predictors pluggable, same interface)
```

Labeling rule (kept): candidates are generated **after** the window closes;
first window observes only. Online model updates happen only from trustworthy
labels (synthetic) or confidence-gated pseudo-labels on real traces (opt-in).

## 3. New/modified modules

### 3.1 New: `src/adaptive_prefetch/baselines.py` (stdlib only)

| Class | Design (literature-anchored) |
| --- | --- |
| `StridePrefetcher` | Classic delta-detecting stride prefetcher. Tracks recent LBA deltas; when a delta recurs ≥ 2× consecutively, prefetch ahead along the stride: candidates `lba + k·stride` for k=1..`degree` (default degree=2). Per-request state, no windows. |
| `MarkovPrefetcher` | Delta-state Markov chain (Joseph & Grunwald style). State = last delta (bucketed); transition counts → probabilities. On each request, emits top-2 most likely next-delta candidates if their joint probability ≥ confidence threshold (default 0.3). Optional 2nd-order key = (delta₋₂, delta₋₁). |
| `CandidatePredictor` protocol | `next_candidates(request, state) -> list[int]` shared by all predictors, so `replay()` stays unchanged in spirit. |

### 3.2 New: `src/adaptive_prefetch/lstm.py` (optional dependency: torch)

- Offline, CPU-friendly: 1-layer LSTM, hidden=64, input = window of last 16
  LBA deltas (+size/op if cheap), output = distribution over top-64 bucketed
  deltas; prefetch top-2 candidates.
- `train-lstm` CLI: trains on generated + normalized real traces; saves
  artifact to `models/lstm_delta.pt` (gitignored).
- At replay: `mode="lstm"` loads artifact, runs per-request inference; if torch
  or artifact missing → clear error and benchmark skips the row.
- `pyproject.toml`: add `[project.optional-dependencies] lstm = ["torch"]`.
  Core paths stay stdlib-only (README invariant preserved).

### 3.3 Modify: `simulator.py`

- Rework `candidates()` into the registry above; add modes `markov`, `lstm`
  (and `stride` = classic) alongside existing `none|sequential|strided|adaptive`.
- `Metrics`: add `prefetch_recall` — **oracle-lookahead definition**:
  `recall = useful_prefetches / (useful_prefetches + prefetchable_misses)`
  where `prefetchable_misses` = read blocks that missed and are re-accessed
  later in the trace (measurement-only lookahead; documented as oracle-based).
- Add `LatencyModel` (params: `t_hit=5µs, t_demand_miss=100µs, t_prefetch=50µs`
  configurable): charges each access → mean access latency + `latency_speedup`
  vs `none` mode. Explicitly a simulation knob — no real-device claim.

### 3.4 Modify: `trace.py` — IOTTA loader

- `load_csv` auto-detects SNIA IOTTA format (`device,sector,size,op,offset,
  timestamp,lifetime,count` or the 8-column variant) and normalizes to
  `Request` (sectors→512-byte blocks, µs→ms). MSR path already works.

### 3.5 New: `cli.py` subcommands

- `benchmark [--trace ...] [--cache-blocks N] [--datasets msr|iotta|synthetic]`
  → runs the full comparison matrix, emits a Markdown/CSV result table.
- `train-lstm --save models/lstm_delta.pt`
- `normalize --input x.csv --format msr|iotta|auto --output y.csv` (so any
  downloaded public trace becomes replayable).

## 4. Baselines (what "benchmark" compares — 7 rows)

| Mode | Implementation | Nature |
| --- | --- | --- |
| `none` | existing | no prefetch |
| `sequential` | existing | fixed read-ahead ×2 |
| `strided` | existing (window re-learned stride) | heuristics |
| `stride` | §3.1 classic, degree=2 | **PDF baseline 1** |
| `markov` | §3.1 delta Markov, top-2 | **PDF baseline 2** |
| `lstm` | §3.2 offline delta LSTM, top-2 | **PDF baseline 3** |
| `adaptive` | existing GNB class→policy | ours |

All rows: same requests, same cache capacity (default 128 blocks), same LRU,
same causal timing. Fixed baselines use a **frozen** model; `adaptive` also
runs an update variant in synthetic labelled mode for the drift benchmark only.

## 5. Benchmarks (maps 1:1 to PDF slide 11)

### B1 — Classification accuracy
Held-out synthetic (seed-split, like today) → accuracy + confusion matrix.
Real traces are unlabeled: report per-window rule-label agreement clearly
marked as *noisy reference*, not ground truth.

### B2 — Policy comparison matrix (the headline table)
For each dataset × mode: `hit_ratio, prefetch_precision, prefetch_recall,
unused_prefetches, mean_access_latency (µs), latency_speedup_vs_none`.

Datasets:
1. Synthetic transition traces, multi-seed (have).
2. `data/samples/msr-cambridge1-sample.csv` (1000 reqs — real but tiny; mark as sample).
3. Full MSR + IOTTA: `normalize` script + README download pointers; runner
   benchmarks whatever normalized traces are present.

### B3 — Compute cost (PDF "fraction of the compute")
Per-window/request inference time for GNB vs stride vs Markov vs LSTM (CPU),
measured in the benchmark and printed per row.

### B4 — Drift adaptation (PDF objective 3 + expected outcome 3)
On labelled synthetic transition traces: report per-phase switch lag (windows
until policy matches class) for **frozen** vs **online-updated** GNB;
compare against LSTM-with-sliding-retrain as the "slow" reference.

### B5 — Optional pseudo-label online learning on real traces
Confidence-gated updates (only windows with confidence > 0.95 update the
model; never from prefetch success). Flagged `--online-real`. Compared head-
to-head with frozen on hit ratio per window — **no claim of correctness,
only measured effect**, consistent with §labeling rule.

## 6. Expected outcomes → how we will show them

| PDF claim | Evidence produced by |
| --- | --- |
| Higher hit ratio than heuristic baselines on mixed real traces | B2 table, adaptive vs stride/markov rows |
| Accuracy comparable to LSTM at fraction of compute | B2 + B3 (hit/precision vs inference µs) |
| Faster adaptation than single-model streaming | B4 switch-lag comparison |

## 7. Tests & verification

Add to `tests/test_project.py`:
- `StridePrefetcher` detects and steadies on strided trace; `MarkovPrefetcher`
  learns transitions and emits top-2; recall & latency model arithmetic;
  IOTTA loader normalization; `benchmark` runs all available modes and returns
  a table; causality re-check: first window never prefetches for new modes.

Verification gate: unit tests green (`python -m unittest discover -s tests -v`
with `PYTHONPATH=src`) + `benchmark` prints the
7-row matrix on the MSR sample with no missing rows (LSTM row skipped
gracefully when torch absent).

## 8. Milestones (small → prove later)

1. **M1 — Baselines**: `baselines.py` (stride, Markov) + replay modes +
   tests. Stdlib only.
2. **M2 — Metrics + latency**: recall, `LatencyModel`, tests.
3. **M3 — Benchmark harness**: `benchmark` + `normalize` (IOTTA) + doc tables.
4. **M4 — LSTM**: optional torch path, `train-lstm`, artifact loading, skip-
   gracefully behavior + README note.
5. **M5 — Drift + pseudo-label**: B4/B5 benchmarks, `--online-real` flag.
6. **M6 — Docs**: update `docs/PIPELINE.md` (7-mode matrix) + `README.md`
   (benchmark/normalize/train-lstm commands, optional lstm extra).

## 9. Explicit non-goals (scope control)

- No kernel/OS integration, no real device I/O (same as Review 2 boundary).
- No calibrated probabilities or production-label claims on real traces.
- No distributed/GPU training; LSTM is CPU-only and offline by spec.

# ML Pipeline: Adaptive Disk I/O Prefetching

This document explains the machine learning pipeline in this repository,
stage by stage, and what every output means and how to use it.

Project: **ML-Based Adaptive Disk I/O Prefetching Using Workload Pattern
Classification** — a user-space, simulation-first prototype. It classifies
short windows of I/O requests into an access pattern, selects a prefetch
policy for that pattern, and replays the requests through an LRU cache to
measure the benefit. It does **not** touch the OS or issue real disk reads.

```
                    ┌────────────────────────────────────────────┐
                    │                 STAGE 0 — INPUT            │
                    │  synthetic generator │ CSV loader │ MSR    │
                    │  (trace.py)         │ (trace.py)  adapter │
                    └───────────────┬────────────────────────────┘
                                    ↓
                    ┌────────────────────────────────────────────┐
                    │            STAGE 1 — WINDOWING             │
                    │  fixed windows of N requests (N=32, min 8) │
                    │  a window is complete only after its last  │
                    │  request arrives  (causal, no future info) │
                    └───────────────┬────────────────────────────┘
                                    ↓
                    ┌────────────────────────────────────────────┐
                    │          STAGE 2 — FEATURE EXTRACTION      │
                    │  features.py :: extract() → 8 floats       │
                    │  features.py :: dominant_stride() → stride │
                    └───────────────┬────────────────────────────┘
                                    ↓
                    ┌────────────────────────────────────────────┐
                    │        STAGE 3 — CLASSIFICATION            │
                    │  model.py :: OnlineGaussianNB              │
                    │  predicts sequential|strided|random|mixed  │
                    │  + confidence; optional online update      │
                    └───────────────┬────────────────────────────┘
                                    ↓
                    ┌────────────────────────────────────────────┐
                    │        STAGE 4 — POLICY SELECTION          │
                    │  class → prefetch candidate generator      │
                    │  simulator.py :: candidates()              │
                    └───────────────┬────────────────────────────┘
                                    ↓
                    ┌────────────────────────────────────────────┐
                    │        STAGE 5 — LRU CACHE REPLAY          │
                    │  simulator.py :: LRUCache / replay()       │
                    │  block-level simulation + metrics          │
                    └───────────────┬────────────────────────────┘
                                    ↓
                              OUTPUTS (Stage 6)
```

---

## Stage 0 — Input: where requests come from

All request records are `Request` dataclasses (`src/adaptive_prefetch/trace.py`):

```python
Request(timestamp_ms: float, lba: int, size_blocks: int = 1,
        operation: str = "R", stream_id: str = "default")
```

- `timestamp_ms` — non-negative, ascending; milliseconds.
- `lba` — **block index**, not byte offset.
- `size_blocks` — positive number of contiguous blocks in the request.
- `operation` — `R` or `W`.
- `stream_id` — optional; kept for future per-process/per-volume analysis.
  The current model classifies the **aggregate** request stream.

Three input sources:

| Source | Function | Notes |
| --- | --- | --- |
| Synthetic | `synthetic_window()`, `synthetic_dataset()`, `transition_trace()` | Labeled, reproducible (seeded `random.Random`). Four classes: `sequential`, `strided`, `random`, `mixed`. |
| Normalized CSV | `load_csv()` | Requires columns `timestamp_ms,lba,size_blocks,operation` (+ optional `stream_id`), sorted by timestamp. |
| MSR Cambridge trace | `load_csv()` auto-detects | Headers `Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime`. Converts byte offsets/sizes → 512-byte blocks, Windows filetime → elapsed ms, `Read`/`Write` → `R`/`W`. The bundled `msr-cambridge1-sample.csv` (1000 requests) is this format. |

Writes are kept in the stream (cache uses a simplified write-allocate model)
but **never trigger prefetch**.

## Stage 1 — Windowing

`replay()` in `src/adaptive_prefetch/simulator.py` slides a fixed window of
requests (`window_size=32`, minimum 8).

- The window is classified **only after all its requests have occurred**
  (the pipeline is causal). The prediction affects the **next** requests,
  never earlier ones.
- The **first window is observation only** — its policy cannot apply to
  itself, so the first window produces no prefetches.
- After each completed window, the model's prediction becomes the active
  policy; if it differs from the previous policy, `policy_switches` is
  incremented.

## Stage 2 — Feature extraction

`extract()` in `src/adaptive_prefetch/features.py` turns one window into
eight scale-independent features (`FEATURE_NAMES`):

| # | Feature | Meaning |
| --- | --- | --- |
| 1 | `contiguous_ratio` | Fraction of deltas equal to the *previous request's* size (size-aware sequentiality). |
| 2 | `dominant_stride_ratio` | Fraction of deltas matching the most common non-contiguous delta. |
| 3 | `random_jump_ratio` | Fraction of deltas with `|delta| > 64` (distant jumps). |
| 4 | `short_run_ratio` | Fraction of deltas with `|delta| ≤ 16` (localized movement). |
| 5 | `unique_ratio` | Distinct LBAs / requests (novelty / reuse). |
| 6 | `gap_cv` | Coefficient of variation of inter-arrival gaps, capped at 10. |
| 7 | `fast_gap_ratio` | Fraction of gaps < 0.15 ms (burstiness). |
| 8 | `read_ratio` | Fraction of reads in the window. |

`dominant_stride()` returns the most frequent positive delta ≤ 1024 recurring
in ≥ 25% of deltas, or `None`. This feeds the *strided* policy with the
concrete stride value.

## Stage 3 — Classification

`OnlineGaussianNB` in `src/adaptive_prefetch/model.py`:

- A **Gaussian Naive Bayes** classifier with **incremental (online) updates**
  via running counts, means, and `M2` sums (Welford-style). No batch
  retraining; one labeled example at a time.
- `predict(values)` returns `(class, confidence)` where class ∈
  `sequential | strided | random | mixed`. It computes log-prior +
  per-feature Gaussian log-likelihood per class (variance floored at
  `0.0025` for numerical stability) and normalizes scores into a
  softmax-like confidence.
- **Supervised**: every `update()` requires a trustworthy label.

Training in `cli.py :: train(seed=42, examples_per_class=100)`:
400 labeled synthetic windows (100 per class) from `synthetic_dataset()`.
The held-out test uses a different seed (`seed + 100000`) and independent
generation so training and test windows never overlap.

### Online updates — the label question

- **Synthetic demo**: labels are known because the workload was generated.
  The demo re-trains a model, then replays the transition trace with
  `online_updates=True` to demonstrate one-example-at-a-time learning.
- **Real CSV traces**: labels are **unknown**. Replay with a frozen model;
  updates are disabled. Successful prefetches are **not** treated as proof a
  workload label was correct (that would be circular).

## Stage 4 — Policy selection

`candidates()` in `src/adaptive_prefetch/simulator.py` maps predicted class
→ prefetch candidates for each read request:

| Predicted class | Policy | Prefetch candidates |
| --- | --- | --- |
| `sequential` | sequential | next 2 blocks after the request (`lba + size_blocks + 0..1`) |
| `strided` | strided (uses `dominant_stride` from recent history) | `lba + stride` |
| `mixed` | mixed | next 1 block after the request |
| `random` | none | no prefetch |

In `replay()`, the policy starts as `"random"` for adaptive mode (safe default
until the first window is classified). Writes never generate candidates.

## Stage 5 — LRU cache simulation

`replay()` in `src/adaptive_prefetch/simulator.py`:

1. For each request, touch every block (`lba .. lba+size_blocks-1`) in the
   `LRUCache` (`OrderedDict` keyed by block; `prefetched` flag per block).
   - Read: counts a hit if present; a hit on a prefetched block also counts
     a *useful prefetch* and clears the flag.
   - Write: write-allocate; never counted as a successful prefetch.
2. Issue the policy's prefetch candidates for the request.
3. Append the request to the window history; when full, compute stride,
   classify, switch policy, (optionally) online-update the model, reset.
4. Cache evicts LRU when over capacity (`capacity=128` by default).

Every policy (including baselines) sees the **same request stream, same cache
capacity, same LRU rules** — a fair controlled comparison.

## Stage 6 — Outputs and what to do with them

### What you get

From `python -m adaptive_prefetch demo`:

1. **Training / test sizes and held-out accuracy** — e.g.
   `Accuracy: 1.000 (160/160)` on synthetic data.
2. **Confusion matrix** — true class × predicted class. Every off-diagonal
   entry is a classifier mistake and tells you which patterns are being
   confused (e.g., `strided` predicted as `sequential`).
3. **Per-window prediction latency** — `0.21 ms/window` on the demo machine.
   A throughput figure of the classifier on *this* machine, **not** real
   disk latency or speedup.
4. **Per-mode cache metrics** over the transition trace
   (`none / sequential / strided / adaptive`):
   - `hit` — read-block cache hits ÷ read blocks requested.
   - `prefetch_precision` — prefetched blocks later read ÷ prefetches issued.
   - `prefetches` / `unused` — volume of prefetch work and how much was wasted.
   - `switches` — policy changes in adaptive mode.
5. **Per-window predictions with confidence** for adaptive mode, showing
   transitions `sequential → strided → random → mixed`.
6. **Online update count** — labeled windows applied incrementally.

From `python -m adaptive_prefetch replay trace.csv`: items 1–4 (minus the
transition display); labels are unknown, so no updates and no confusion
matrix. From `export-demo demo.csv`: normalized trace CSV you can replay
or inspect.

### How to read the numbers (worked example from a real run)

| Mode | hit | precision | prefetches | unused |
| --- | --- | --- | --- | --- |
| `none` | 0.000 | 0.000 | 0 | 0 |
| `sequential` (fixed) | 0.258 | 0.271 | 862 | 628 |
| `strided` (fixed) | 0.078 | 0.628 | 113 | 42 |
| `adaptive` | 0.251 | 0.567 | 402 | 174 |

Key insight: adaptive **almost matches** fixed sequential hit ratio (0.251
vs 0.258) while issuing **fewer than half the prefetches** (402 vs 862) with
more than twice the precision (0.567 vs 0.271) — far less wasted I/O. On
other mixes fixed read-ahead can win on hit ratio *by brute force* (many
extra prefetches); always compare precision and unused prefetches too, not
hit ratio alone.

### How to use the outputs

- **Compare policies, never just one number.** Hit ratio, precision, and
  unused prefetches together describe the hit-vs-waste tradeoff.
- **Use the confusion matrix to judge the classifier on your data.** A clean
  synthetic matrix says nothing about production workloads — rerun with the
  same model against your trace and inspect the actual window predictions.
- **Use window predictions for debugging.** The per-window `true -> predicted
  (confidence)` list shows where the classifier switches (or lags) at
  workload transitions.
- **Report the demo result as a simulation, not a benchmark.** The simulator
  omits queueing, device latency, prefetch completion time, and bandwidth. It
  cannot produce real latency numbers or speedup claims. It *can* fairly
  compare policies under identical conditions.
- **Exported CSVs are interchangeable inputs.** `export-demo` → `replay`
  round-trips cleanly; you can hand-edit or script CSV generation and feed
  any conforming trace through the same pipeline.

### Known limits to keep in mind

- Synthetic labels are produced by the generator; accuracy on synthetic data
  describes the generator, not production traffic.
- No online learning on real traces (labels unavailable).
- Aggregate-stream classification only; `stream_id` is not yet used.
- Prefetch memory cost (cache residency of unused blocks) is visible only
  through `unused` / precision, not as real I/O cost.

### Next experiments

1. Normalize a full public trace (MSR Cambridge, SNIA IOTTA) to the required
   schema and replay it; inspect window predictions against known phases.
2. Add an independent, non-circular labelling method so real traces can
   support online updates.
3. Vary `cache-blocks` and `window-size` across workloads; plot hit/precision
   tradeoff curves per policy.
4. Model per-`stream_id` workloads instead of the aggregate stream.
5. Add realistic I/O costs (prefetch completion time, bandwidth) to move from
   hit-ratio simulation toward latency estimation.

---

## Module map

| File | Responsibility |
| --- | --- |
| `src/adaptive_prefetch/trace.py` | `Request` schema, CSV load (normalized + MSR adapter), CSV export, synthetic labeled generators. |
| `src/adaptive_prefetch/features.py` | 8 window features (`extract`) + `dominant_stride` detection. |
| `src/adaptive_prefetch/model.py` | `OnlineGaussianNB` incremental classifier. |
| `src/adaptive_prefetch/simulator.py` | Causal replay, LRU cache, policy candidates, metrics, confusion matrix. |
| `src/adaptive_prefetch/cli.py` | `demo`, `replay`, `export-demo` commands; training + orchestration. |
| `tests/test_project.py` | Unit tests: trace round-trip, MSR conversion, feature edge cases, classifier accuracy ≥ 0.9, causality, online-update guard. |

Run everything from the repo root:

```powershell
.\run_review2.ps1            # demo
.\run_review2.ps1 -Test      # unit tests
.\run_review2.ps1 -Trace .\msr-cambridge1-sample.csv   # real-trace replay
```

or with Python directly (see README). See `docs/REVIEW2.md` for the review
rubric mapping and discussion answers; `presentation/` holds the slide deck.
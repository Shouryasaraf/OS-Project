# Classifier, LSTM, and pipeline — findings

All numbers below were produced by the commands in this repository, with
disjoint train/test seeds. Regenerate anything here before presenting it.

```
$env:PYTHONPATH='src'
python -m unittest discover -s tests            # 60 tests
python -m adaptive_prefetch evaluate            # classifier x regime table
python -m adaptive_prefetch eda <trace>         # trace profile
python -m adaptive_prefetch shift <trace>       # domain shift vs synthetic
python -m adaptive_prefetch benchmark --lstm-model models/lstm_delta.pt
```

---

## 1. The classifier: the algorithm was not the problem

The request was to replace Naive Bayes because it is "old and has low
accuracy". Measured over 12 disjoint seed pairs (train seed `1000+i`, test
seed `900000+i`), that is not what the data shows on the shipped generator:

| regime | gnb | qda |
| --- | ---: | ---: |
| baseline | **100.00 ± 0.00** | **100.00 ± 0.00** |
| noisy (30% random jumps) | 99.95 ± 0.17 | 99.95 ± 0.17 |
| overlap (strides 1-3) | 100.00 ± 0.00 | 100.00 ± 0.00 |
| realistic (1e8-1e11 LBA, 8-block) | 100.00 ± 0.00 | 100.00 ± 0.00 |
| transition (50% phase blend) | 51.25 ± 4.85 | 50.26 ± 3.36 |
| transition_heavy (75% blend) | 64.79 ± 5.74 | **86.77 ± 1.89** |

Naive Bayes was already at 100.00% with zero variance. **The 100% headline in
`docs/STAGE2_RESULTS.md` is a property of the generator, not of the model.**

### What was actually broken: two absolute thresholds in the features

`features.py` claimed to be "address-scale independent" while two of its eight
features used absolute block counts:

```python
sum(abs(d) > 64 for d in deltas) / len(deltas),   # random_jump_ratio
sum(abs(d) <= 16 for d in deltas) / len(deltas),  # short_run_ratio
```

A 128-block request was therefore never "short" and a 4 KiB stride was never
"distant". The same access pattern classified differently on devices with
different block sizes. Both are now measured relative to the request size
(`DISTANT_JUMP_SCALE = 8`, `SHORT_RUN_SCALE = 0.5`), and `fast_gap_ratio`
compares gaps against the window's own mean interval instead of a fixed
0.15 ms.

Block-size transfer, trained at 1-block requests and tested on the same
patterns at other sizes (6 seed pairs):

| train → test | gnb | qda |
| --- | ---: | ---: |
| 1 → 1 | 100.00% | 99.90% |
| 1 → 8 | 97.81% | 87.81% |
| 1 → 32 | **100.00%** | 79.17% |
| 1 → 128 | **100.00%** | 79.17% |

This is the decisive result. QDA wins the transition regime by 22 points but
**loses 21 points under the covariate shift that this project actually faces**
(synthetic training windows → real traces). A full covariance memorises joint
structure that also moves; a diagonal model does not.

### Why the default stayed Naive Bayes

`DEFAULT_CLASSIFIER = "gnb"`. Parsimony wins on the risk that actually occurs.
QDA is real and available via `--classifier qda`, and is the right choice if
your workload has correlated features and mid-window phase changes and *no*
train/deploy distribution shift. Standardisation (`--standardize`) was tested
and made **both** models worse, because the features are already ratios in
[0, 1] and z-scoring destroys that structure; it is off by default.

Cost, per window at `window_size=32`: 24.7 µs predict / 2.8 ms train for
Naive Bayes, 22.7 µs / 8.3 ms for QDA. Both are negligible against the
simulator's own 5 µs hit and 100 µs demand-miss cost model.

---

## 2. The LSTM: "the code that fixes itself" was the bug

`lstm.py` had a comment saying it took `topk(10)` instead of `topk(2)`
*specifically to bypass unknown predictions*. That workaround was the defect:
it converted a calibrated "I don't know" into a **guaranteed wrong prefetch on
every single read**. Measured emission rate on the old decoder: **100.0% of
reads in every workload phase**, including pure random, where the top-1
prediction was the unknown class 97-100% of the time.

Three further problems compounded it:

1. **The shipped artifact was untrained and mismatched.** `main.py` trained
   the delta model on *real* traces and then `benchmark.py` scored it against
   *synthetic* ones in the same table. The committed `models/lstm_delta.pt`
   was reproduced bit-for-bit as 575 examples × 2 epochs = **8 Adam steps**,
   a network that returned near-uniform output for every input. Its vocabulary
   was 163/170 multiples of 8 (real 4 KiB-aligned I/O) while the synthetic
   benchmark needs deltas of 1, 2 and 3 — 0.5% coverage on the sequential
   phase.
2. **The exact-delta vocabulary capped addressability at ~58%.** A fixed set
   of delta values can never emit a stride it did not memorise. The head now
   predicts (sign, magnitude-bucket) and resolves the bucket at decode time.
3. **Linear `delta / 4096` input scaling** squashed the only predictable
   deltas (1-64) into ~1% of the input range. Now signed `log1p`.

### Result on the same benchmark table

| dataset | old hit % | new hit % | old precision % | new precision % |
| --- | ---: | ---: | ---: | ---: |
| synthetic_seed_42 | 6.98 | **28.53** | 7.22 | **56.60** |
| synthetic_seed_43 | 18.71 | **33.16** | 21.87 | **68.05** |
| synthetic_seed_44 | 9.13 | **29.00** | 13.38 | **57.77** |
| msr_format_sample | 1.72 | 1.72 | 15.22 | 15.22 |

It now beats every fixed heuristic and beats the adaptive classifier on two
of three synthetic seeds. The MSR sample is unchanged, as expected: that
trace has a 1.48% cacheable ceiling, so no predictor can move it much.

Also fixed: `cli.py` passed `--seed` into the `vocab_size` positional slot, so
`train-lstm --seed 7` silently trained with `vocab_size=7`.

---

## 3. Bugs fixed

**Crashes**
- `load_csv` raised a raw `AttributeError` on a short CSV row, defeating the
  `invalid trace row N` message it was written to produce.
- `load_model` raised a raw `KeyError` on a well-formed partial artifact,
  bypassing the schema check in front of it.
- `load_model` accepted `decay_factor = 2.0`, turning the decay into an
  amplifier and invalidating the `log(n/total)` prior.
- `load_model` accepted an artifact with all-zero counts and only failed on
  the first window of a replay, with an error naming the model.
- `drift_report` divided by zero, reachable from the CLI via
  `benchmark --datasets msr --windows-per-class 0`.

**Wrong results in published output**
- `analyze_results` looked up the adaptive row with
  `next(row for row in completed if row["mode"] in ("adaptive", "adaptive_pseudo"))`.
  Because the pseudo row is appended *after* `MODES`, the frozen row always
  won and the entire `--online-real` experiment was computed and then
  discarded.
- The "Cache Pollution Reduction" line had no sign handling and printed
  **"-89900.0% fewer wasted prefetches"** — a negative reduction described as
  a benefit. It also compared raw unused counts across modes with wildly
  different prefetch volumes, so a policy that simply prefetched less always
  "won". Now reported as a volume-normalised wasted fraction with an explicit
  `less`/`MORE` word.
- `inference_us` is per *call*, and adaptive classifies once per 32-request
  window while the predictors run once per request. The published claim of
  "7.3x faster inference" was off by ~32x. There is now an
  `inference_us_per_request` column; the real gap is **0.88 µs/request vs
  564 µs/request, about 640x**.
- `OnlineGaussianNB` computed `m2 / (n - 1)` where `n` is the saturated
  effective count `1/(1-γ)`, biasing the variance ~4% low. Corrected by
  rescaling `m2` by `1/γ`.
- `MarkovPrefetcher.deltas` and `LSTMPrefetcher.deltas` retained one `int` per
  read for the whole trace while only ever reading the last 1-2 / 16 entries.
  Now trimmed — relevant because `main.py` dataset option 3 defaults to all 32
  MSRC traces, up to 2.6 GB each.
- `speedup_vs_none` reported `0.00x` for an infinitely fast mode instead of
  an undefined value.
- The two failing tests were the visible tail of a half-landed rename:
  `Total_Prefetches` → `total_prefetches` in `benchmark.py` without updating
  the tests.

---

## 4. The pipeline

New modules, all stdlib:

- **`eda.py`** — trace profiling, locality/reuse analysis, k-means elbow,
  domain-shift measurement, label-support checks. Reports a
  **cacheable ceiling**: the fraction of read blocks that repeat within the
  cache's reach, which is the honest upper bound for any prefetcher.
- **`pipeline.py`** — `StandardScaler`, `PolicySmoother`, configurable regime
  generators, and `evaluate_classifier` / `select_classifier` with disjoint
  train/test seeds.
- CLI: `evaluate`, `eda`, `shift`; plus `--classifier` and `--smoothing`.

`PolicySmoother` is opt-in and adds consecutive-run hysteresis so one
misclassified window cannot flip the policy for a whole window. Its run length
is counted at the tail of the history, not as a total within it — counting
totals lets a strictly alternating stream still produce a 3:2 majority once
the buffer trims, which is exactly the churn it exists to remove.

---

## 5. What the data can and cannot support

`python -m adaptive_prefetch shift data/samples/msr-cambridge1-sample.csv --format msr`:

- mean shift across features: **11.97 reference sd**, worst feature 76.8 sd
- only **58.5%** of target windows fall inside the reference per-feature range
- `gap_cv` is +6.8 sd with 3% coverage; `read_ratio` is -7.1 sd with 48%

On `msr-cambridge1-sample.csv` the cacheable ceiling is **1.48%** and the
adaptive policy reaches 2.33% hit ratio against a 1.54% no-prefetch
baseline — a +0.79 pp difference, which is inside the noise of a 1,000-request
trace. All 31 windows have `contiguous_ratio ≤ 0.16`, so there is no
sequential or strided population in the real data at all.

**The four-class framing is a property of `synthetic_window()`, not of any
trace in `data/`.** Every classification-accuracy number in this project
describes the generator. The MSR CSVs carry no access-pattern field, and the
`.revised` traces carry only a binary `seq`/`rand` column that the loader
discards and that has no window-level all-`seq` population. This is
consistent with what the project already documented
(`true_real_labels_available: False`) — the new tooling just quantifies how far
outside the training region the real windows sit.

### Resolved: real device latencies are now captured

`.revised` columns 5-8 were previously parsed and discarded. `Request` now
carries `service_ms` and `pattern`, both defaulting to `None` so every other
profile is unaffected. See `DECISIONS.md` D8.

Correction to an earlier version of this document: it claimed
`t1 = t2 + t3` exactly. That is wrong -- the residual reaches 9e-2 ms. `t1` is
the total service time in milliseconds, `t3` is zero exactly when `t1` is
tiny (the device-queueing component), and `t2` is a small fixed host overhead.

Measured mean service time is 260-5491 us, so the assumed 100 us demand-miss
cost understated a miss by 3-50x. `LatencyModel.measured()` calibrates from the
**mean**, not the median: 65-87% of requests complete in under a microsecond,
so a median-based model collapses to a near-zero cost.

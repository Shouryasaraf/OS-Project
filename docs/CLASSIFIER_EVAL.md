# Classifier, LSTM, and pipeline — findings

All numbers below were produced by the commands in this repository, with
disjoint train/test seeds. Regenerate anything here before presenting it.

```
$env:PYTHONPATH='src'
python -m unittest discover -s tests            # 131 tests
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
| baseline | 99.58 +/- 0.59 | **99.90 +/- 0.23** |
| noisy (30% random jumps) | 99.95 +/- 0.17 | 99.95 +/- 0.17 |
| overlap (strides 1-3) | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| overlap2 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| realistic (1e8-1e11 LBA, 8-block) | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| transition (50% phase blend) | **51.25 +/- 4.85** | 50.26 +/- 3.36 |
| transition_heavy (75% blend) | 64.79 +/- 5.74 | **86.77 +/- 1.89** |

Held-out accuracy for the model `main.py` actually ships is **0.9688
(155/160 windows)**, mean confidence on correct predictions 0.9980. The
confusion matrix has 2 errors in 160, both `strided` windows predicted
`sequential`.

That accuracy is high and largely irrelevant, because the decision it feeds
is worth very little. See section 5.

*Corrected:* an earlier version of this table read 100.00 +/- 0.00 across the
board. That was a property of the **old degenerate generator**, which had no
within-class variance at all. The corrected generator produces realistic
variance, and 0.9688 is the intended outcome, not a regression. The 100%
headline in `docs/STAGE2_RESULTS.md` is likewise a property of the generator,
and that document is now marked stale.

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

Cost, median over regimes: **17.69 us** predict / 1.86 ms train for Naive
Bayes, 14.63 us / 5.72 ms for QDA. Both are negligible next to the cost model
they act in, whose *measured* demand-miss cost is 260-5491 us. A prediction
costs about 0.3% of the single cache miss it might avoid, and about 5% of the
read-ahead it competes with.


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

---

## 6. Why the classifier is not the bottleneck (added D13/D17)

Everything above measures the classifier working as designed. The reason that
work does not show up in the cache numbers is that **the decision it makes is
worth at most half a point**, while the dimension nobody was tuning is worth
five.

Full MSRC collection, 32 traces x 250,000 requests, 12 modes, 384 rows, 31
traces scored. Unweighted mean hit ratio per trace, so one large trace cannot
dominate the average:

| | mean hit | vs `none` |
| --- | ---: | ---: |
| `deep` (fixed depth-8 read-ahead) | 49.87% | **+4.94 pp** |
| `sequential` (depth-2 read-ahead) | 45.45% | +0.52 pp |
| `lstm` | 45.13% | +0.20 pp |
| `adaptive_evidence` | 45.10% | +0.17 pp |
| `correlate` | 45.10% | +0.16 pp |
| `markov` | 45.09% | +0.15 pp |
| `guard` | 44.99% | +0.06 pp |
| `depth_adaptive` | 44.99% | +0.06 pp |
| `strided` | 44.96% | +0.02 pp |
| `none` | 44.93% | - |
| `stride` | 44.93% | -0.01 pp |
| `adaptive` (this project's classifier) | 44.89% | **-0.04 pp** |

**All eleven non-deep modes fall within 0.55 points of one another.** The
trained classifier is the *worst* of the twelve, below doing nothing. `deep`
sits 4.42 points above the best of them -- eight times the entire spread of
policy choice. Per-trace winners: `deep` 29, `none` 2, `adaptive` 1.

Isolating depth from policy choice, on 8 real traces capped at 40,000
requests, 2048-block cache:

| depth | 1 | 2 (every earlier mode) | 4 | 8 (default) | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean hit ratio | 33.12% | 33.77% | 35.08% | **37.87%** | 38.43% |
| gain over depth 2 | -0.65 | - | +1.31 | **+4.10** | +4.66 |

### This retires the "no routing decision can help" result

D9 concluded, from a perfect-window-router oracle experiment, that policy
selection had only +0.4 points of headroom and the traces were "not
recoverable from request history". That experiment was correct and the
conclusion was still wrong, because every candidate policy it routed between
prefetched exactly 2 blocks. The oracle was choosing between eight ways of
prefetching the same amount.

The correct statement is narrower: **choosing between prefetching *strategies*
is worth at most half a point; choosing how *deep* to prefetch is worth five.**
The "not recoverable from request history" diagnosis still holds for the
classifier specifically -- a 96.7%-accurate four-class window label is not a
next-address predictor, and the traces are ~49% backward seeks with reuse
distances far exceeding the cache.

### What would make the classifier relevant

Not a better classifier. A decision worth making. Concretely, the gap to close
is between a fixed depth and a *workload-dependent* depth, which is a
regression on observed reuse distance rather than a 4-way class label. The
feedback gate in `guard.py` is a first, crude attempt at exactly that -- it
throttles on measured precision -- and it is the mode that wins on the one
trace where prefetching actively destroys the cache (`web_3`: 48.18% for
`correlate`, 44.69% for `guard`, against `sequential`'s 25.70%, with 87-95%
less wasted I/O). It beats `sequential` on only 3 of 31 traces overall.

So the honest summary of this project: the classifier is competent and
irrelevant, depth is decisive and unguarded, and the interesting open problem
is per-window depth selection.

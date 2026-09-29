# Decisions and iterations

A log of what was tried, what the measurement said, and what was kept. Every
claim here is reproducible with the commands in `README.md`.

The short version: **two of the three changes that were requested on the
grounds that "Naive Bayes is old and the LSTM is broken" turned out to rest on
a premise that did not survive measurement.** The real defects were elsewhere
and were fixed instead. QDA was built, measured, wired in, and then
deliberately demoted back to opt-in.

---

## D1. Classifier: build QDA, measure it, then keep Naive Bayes

**Requested:** replace Gaussian Naive Bayes with a better, still-cheap model
because NB is "old and has low accuracy".

**Hypothesis formed:** the diagonal-covariance assumption is the defect. It
treats `contiguous_ratio`, `dominant_stride_ratio` and `short_run_ratio` as
independent, but all three are functions of the same delta sequence and are
strongly negatively correlated within a class.

**What was built:** `OnlineQDA` in `model.py` — shrinkage-regularised online
QDA, same `update`/`predict` contract, ~110 lines, stdlib only. Two
optimisations were needed to make it practical:

- Flat `d*d` scatter matrices so the per-update decay is a list
  comprehension rather than a nested Python loop.
- A cached Cholesky per class. Because `S -> g*S` implies `L -> sqrt(g)*L`,
  `logdet -> logdet + d*log(g)` and the Mahalanobis distance `-> maha/g`, a
  class that has not been updated costs nothing to score. Without this a
  1:1 predict/update stream refactorises all four classes every call
  (150 us/window -> 51 us/window).

**Measurement, 12 disjoint seed pairs per cell** (train seed `1000+i`, test
seed `900000+i`, never overlapping):

| regime | gnb | qda |
| --- | ---: | ---: |
| baseline | 100.00 ± 0.00 | 100.00 ± 0.00 |
| noisy | 99.90 ± 0.18 | 99.90 ± 0.18 |
| realistic (1e8–1e11 LBA, 8-block) | 100.00 ± 0.00 | 100.00 ± 0.00 |
| transition (50% phase blend) | 51.25 ± 4.85 | 50.26 ± 3.36 |
| transition_heavy (75% blend) | 60.52 ± 4.12 | **86.56 ± 1.94** |

QDA is decisively better on windows that straddle a phase change. That is
real and it is why QDA exists.

**Then the transfer test reversed the decision.** Train at 1-block requests,
test the same patterns at other block sizes:

| train → test | gnb | qda |
| --- | ---: | ---: |
| 1 → 1 | 100.00% | 100.00% |
| 1 → 8 | 96.67% | 94.72% |
| 1 → 32 | 100.00% | 91.67% |
| 1 → 128 | 100.00% | 91.67% |

**Decision:** `DEFAULT_CLASSIFIER = "gnb"`, QDA available via
`--classifier qda`.

**Reasoning:** this project's dominant risk is covariate shift — the
classifier is fitted on synthetic windows and applied to real traces, and
`shift` measures that at ~12 sd mean displacement. A full covariance
memorises joint structure that also moves; a diagonal model does not. QDA
buys +26 points in a regime that may not occur and loses 8 points in one
that does. Parsimony wins on the risk that actually exists.

**Also tested and rejected:** standardising the features (`--standardize`).
It made *both* models worse (e.g. 1→128 transfer dropped from 100% to 75%)
because the features are already ratios in [0, 1] and z-scoring destroys
that structure. Off by default.

**What this means for the original request:** Naive Bayes was never the
accuracy bottleneck. It was at 100.00% ± 0.00 on the shipped generator
before any of this work. See D2.

---

## D2. The accuracy bug was in the features, not the model

`features.py` was documented as producing "address-scale independent
features". Two of eight did not:

```python
sum(abs(d) > 64 for d in deltas) / len(deltas),   # random_jump_ratio
sum(abs(d) <= 16 for d in deltas) / len(deltas),  # short_run_ratio
```

A 128-block request was never "short". A 4 KiB stride was never "distant".
The same access pattern therefore classified differently depending on the
device's block size — the one thing the docstring promised it would not do.

**Fix:** both thresholds are now relative to the request size
(`DISTANT_JUMP_SCALE = 8`, `SHORT_RUN_SCALE = 0.5`), mirroring what
`contiguous_ratio` already did with `d == size_blocks`. `fast_gap_ratio`
was a third offender — a fixed 0.15 ms threshold hard-codes an assumed
device arrival rate — and now compares gaps against the window's own mean
interval.

**Effect:** block-size transfer goes from broken to perfect for the default
model, and `short_run_ratio` domain shift drops from −1.18 sd to +0.03 sd.

**Cost:** zero. No new parameters, no inference overhead.

---

## D3. LSTM: the "code that fixes itself" was the defect

**Requested:** fix the LSTM's poor accuracy.

`lstm.py` contained a comment explaining that it took `topk(10)` rather
than `topk(2)` *specifically to bypass unknown-delta predictions*. That
workaround was the bug: it converted a calibrated "I don't know" into a
**guaranteed wrong prefetch on 100.0% of reads**, in every workload phase
including pure random, where the top-1 prediction was the unknown class
96–100% of the time.

Three further problems compounded it:

| # | problem | fix |
| --- | --- | --- |
| 1 | `main.py` trained the delta model on **real** traces, then `benchmark.py` scored it against **synthetic** ones in the same table. The committed artifact was 575 examples × 2 epochs = **8 Adam steps** and returned near-uniform output for every input. | `main.py` now trains on the traces the run will actually score. |
| 2 | The exact-delta vocabulary capped addressability at ~58%: a fixed set of delta values can never emit a stride it did not memorise. 163/170 of the shipped vocabulary was multiples of 8 (real 4 KiB-aligned I/O) while the benchmark needs deltas of 1, 2, 3. | Head now predicts (sign, magnitude-bucket); the bucket is resolved to concrete deltas at decode time, with the learned stride tried first. |
| 3 | Linear `delta / 4096` scaling squashed the only predictable deltas (1–64) into ~1% of the input range. | Signed `log1p`. |

Plus a real argument bug: `cli.py` passed `--seed` into the `vocab_size`
positional slot, so `train-lstm --seed 7` silently trained with
`vocab_size=7`.

**Result, same benchmark table:**

| dataset | old hit % | new hit % | old precision % | new precision % |
| --- | ---: | ---: | ---: | ---: |
| synthetic_seed_42 | 6.98 | **28.75** | 7.22 | **62.37** |
| synthetic_seed_43 | 18.71 | **33.16** | 21.87 | **68.05** |
| synthetic_seed_44 | 9.13 | **29.66** | 13.38 | **63.81** |
| demo_normalized | — | **33.13** | — | **68.58** |

It now beats the adaptive classifier outright on two of three synthetic
seeds and on `demo.csv`.

---

## D4. Output redesign: findings on screen, raw data in files

**Requested:** the console output was messy, and the run asked for too much
input up front.

**Before:** six blocking prompts (run demo? cache size? window size?
datasets? LSTM? save?), then a 28-row 12-column markdown table dumped to the
terminal, followed by a paragraph describing what the output *was*.

**After:** `python main.py` with no arguments runs the whole pipeline.
`--interactive` restores the guided picker; `--quick` skips LSTM training
and the regime sweep. Console shows conclusions; `outputs/` holds every raw
row.

| file | contents |
| --- | --- |
| `results.csv` | every benchmark row, all datasets × all modes |
| `results.md` | narrative report with the full 7-mode table and interpretation |
| `eda.md` | per-trace request stream, locality, feature distributions, cluster elbow |
| `classifier_eval.csv` | classifier × regime accuracy, spread, predict and train cost |
| `domain_shift.csv` | per-feature shift and coverage vs the synthetic region |
| `drift.csv` | frozen vs online switch lag per phase |
| `label_support.md` | which classes the labelled population actually covers |

Runtime: 9.4 s quick, ~26 s full.

---

## D5. The ceiling metric was wrong, and hiding it would have been worse

The first version of the dataset table reported a "cacheable ceiling": the
share of read blocks that repeat within the cache's reach. On the synthetic
traces it read **0.00%**, while the adaptive policy achieved **28.75%** hit
ratio on the same traces.

That is not a contradiction — the metric was simply the wrong one. It
measures *demand-cache reuse*, and a synthetic scan reads each block exactly
once, so there is no reuse to find. Prefetching works by converting a
**first** touch into a hit, which the reuse ceiling does not count at all.

**Replaced with an oracle bound** (`simulator.oracle_reference`): a perfect
one-request-lookahead predictor, replayed through the identical cache. It
isolates "can the mechanism turn an access into a hit" from "can a predictor
know the address", and it is directly comparable to the policy rows.

| dataset | oracle | best policy | caught |
| --- | ---: | ---: | ---: |
| synthetic_seed_42 | 96.94% | 28.75% | 30% |
| msr-cambridge1-sample | 100.00% | 3.05% | 3% |
| msrc_wdev_3 | 100.00% | 0.16% | 0% |

This is a more useful finding than the one it replaced: on real traces the
policies capture 0–3% of what a perfect predictor would get, so the gap is
**address predictability**, not cache or policy design.

---

## D6. Bugs found and fixed while doing the above

Correctness of reported results:

- `analyze_results` looked up the adaptive row with
  `next(row for row in completed if row["mode"] in ("adaptive", "adaptive_pseudo"))`.
  The pseudo row is appended *after* `MODES`, so the frozen row always won
  and the entire `--online-real` experiment was computed then discarded.
- "Cache Pollution Reduction" had no sign handling and printed
  **"-89900.0% fewer wasted prefetches"** — a negative reduction described as
  a benefit. It also compared raw unused counts across modes with very
  different prefetch volumes, so a policy that simply prefetched less always
  "won". Now a volume-normalised wasted fraction with an explicit
  `less`/`MORE` word.
- `inference_us` is per *call*, and adaptive classifies once per 32-request
  window while the predictors run per request. The published "7.3x faster
  inference" was off by ~32x. Added `inference_us_per_request`; the real gap
  is **0.88 vs 564 us/request, about 640x**.
- `prefetchable_misses` counted a block again every time it was evicted and
  re-read, inflating the oracle-recall denominator whenever the cache was
  smaller than the working set. Now counted once per block.
- `aggregate_results` dropped an entire mode from `ALL_TRACES` if it was
  skipped on *any* dataset, discarding real measurements. Now aggregates
  over the datasets that ran and reports coverage in the status. The
  no-prefetch baseline is also restricted to the same datasets, otherwise the
  speedup compared different request sets.
- The LSTM summary averaged write-only traces, which score 0/0 by
  construction, dragging the mean down with no meaning.

Crashes and bad input:

- `load_csv` raised a raw `AttributeError` on a short CSV row, defeating the
  `invalid trace row N` message it exists to produce.
- `load_model` raised a raw `KeyError` on a well-formed partial artifact,
  bypassing the schema check in front of it; accepted `decay_factor = 2.0`
  (turning the decay into an amplifier and invalidating the `log(n/total)`
  prior); and accepted an all-zero-count artifact, which then failed on the
  first window of a replay with an error naming the model rather than the file.
- `drift_report` divided by zero, reachable from the CLI via
  `benchmark --datasets msr --windows-per-class 0`.
- `rankable` fix: a write-only trace makes every speedup `None`, and
  `max()` cannot order `None`.

Memory:

- `MarkovPrefetcher.deltas` and `LSTMPrefetcher.deltas` retained one `int`
  per read for the whole trace while only ever reading the last 1–2 / 16
  entries. Relevant because the MSRC directory holds 32 files up to 2.6 GB.

Statistics:

- `OnlineGaussianNB` computed `m2 / (n - 1)` where `n` is the saturated
  effective count `1/(1-γ)`, biasing the variance ~4% low. Now rescaled.



---

---

## D7. Deliberately not done

`.revised` columns 5-8 were parsed and discarded. Column 5 is `seq`/`rand`,
a real but binary label; columns 6-8 are real device latencies. **Column 6-8
were captured in D8**; the `seq`/`rand` flag is now retained as `pattern` for
ground-truth scoring only, but no policy is allowed to consume it, because
routing on the ground-truth label is not a result.

`.revised` columns 5–8 are parsed and then discarded. Column 5 is
`seq`/`rand`, a real but binary label with no window-level all-`seq`
population; columns 6–8 are real device latencies in milliseconds
(`t1 = t2 + t3` exactly), and are the only ground-truth timing anywhere in
`data/`. They would let the 5/100/50 µs cost model be *validated* rather
than assumed.

Capturing them means adding fields to the frozen `Request` dataclass and
changing the CSV schema, so it is recorded here rather than done silently.

Also unchanged: the four-class framing. It is a property of
`synthetic_window()`. No trace in `data/` carries a four-class label, and
`shift` measures the real windows at ~12 sd from the training region. Every
accuracy number in this project describes the generator.

~~The default cache capacity is still expressed in **blocks** (128)~~ --
**superseded by D10.** Once the generator was corrected to issue realistic
8-128-block requests, a 128-block cache held only ~16 whole requests and
starved every policy, so `DEFAULT_CAPACITY` is now 2048 blocks (1 MiB). This
restates every cache number reported before D10.
---

## D8. Real device latencies replace the assumed cost model

**Requested:** use the real latencies in `.revised` columns 6-8 instead of the
hard-coded 5/100/50 us model.

**What those columns actually are.** `t1` is the total service time in
milliseconds. Measured across the collection it is strongly right-skewed:

| trace | mean service | median | p99 | max | < 1 us share |
| --- | ---: | ---: | ---: | ---: | ---: |
| hm_1 | 2058 us | 14.2 us | 11050 us | 14401 us | 84.6% |
| mds_0 | 481 us | 26.0 us | 5007 us | 28710 us | 64.7% |
| stg_0 | 329 us | 13.6 us | 5000 us | 26190 us | 67.7% |
| proj_3 | 260 us | 11.5 us | 300 us | 14870 us | 86.7% |

`t3` is zero exactly when `t1` is tiny, so it is the device-queueing
component; `t2` is a small fixed host overhead.

**Correction to an earlier claim in this repo:** `docs/CLASSIFIER_EVAL.md`
stated that `t1 = t2 + t3` exactly. That is wrong. The residual reaches
9e-2 ms and the mean relative error is 0.7-7.4. No such identity is assumed in
the implementation.

**How the cost model is derived.** `LatencyModel.measured()` uses the
**mean**, not the median. This matters: the median is ~0 for most of these
traces because 65-87% of requests complete in under a microsecond, so a
median-based model collapses to a near-zero cost and makes every policy look
free. An earlier implementation used the median and produced a nonsense
result where all five policies cost 0.02 us/read; there is now a regression
test (`test_measured_cost_model_uses_mean_not_median`) for it.

| | assumed | measured |
| --- | ---: | ---: |
| hit | 5 us | 1% of mean service (approx) |
| demand miss | 100 us | 260-5491 us |
| prefetch | 50 us | 1.1x mean service |

The assumed model understated a miss by **3-50x**. A speculative prefetch that
misses therefore costs far more than a hit saves, which is why the economics
of prefetching are worse than the original model implied.

**The hit cost remains an approximation and is labelled as one.** The traces
record device service time only, never cache service time, so a hit is set to
1% of mean service. That number is not measured.

**What it changes:** the *relative* ranking of policies is unchanged. Under
both the assumed and the measured model, no prefetcher beats doing nothing on
these real traces. That is now a robust conclusion rather than an artefact of
assumed latencies.

**Schema change.** `Request` gained two optional fields, `service_ms` and
`pattern`, both defaulting to `None` so every existing call site and CSV
profile is unaffected. `pattern` holds the trace's own `seq`/`rand` flag and is
used for ground-truth scoring only; it must never reach candidate generation,
which would be lookahead.

---

## D9. The evidence router: a measured negative result

**Requested:** make the adaptive policy work on real workloads.

**What was built.** `simulator.route_window` and the `adaptive_evidence`
mode. Instead of routing through a four-class classifier trained on synthetic
windows -- which sit ~12 sd away from real ones, so any prediction is
extrapolation -- it routes on three features measured from the window itself:
contiguity, repeated stride, and jumpiness. It needs no training data and
degrades gracefully on any distribution.

Thresholds were calibrated against the *measured* real-trace distribution
rather than invented. Real windows have `contiguous_ratio` median 0.032 and
p90 0.097, so `MIN_CONTIGUITY = 0.10` fires on roughly the top quartile. A
first attempt used 0.30, which fired on 5% of windows and under-performed.

**Result, 40k-request heads, 128-block cache:**

| | vs classifier | vs fixed read-ahead |
| --- | --- | --- |
| hit ratio | better on all 7 | worse on all 7 |
| prefetches issued | 3-4x fewer | 2-4x fewer |
| precision | better | better |
| modelled cost | better | **still worse than doing nothing** |

**Three hypotheses were tested and rejected before this one:**

1. *Route on the trace's own `seq`/`rand` ground truth.* The signal is real --
   `contiguous_ratio` correlates r = +0.95 to +0.97 with the flag across four
   traces. But an **oracle** router with perfect knowledge of the label still
   loses to fixed read-ahead on 3 of 4 traces. It only helps where the reuse
   distance fits the cache.

2. *Learn a per-request delta n-gram predictor.* A 3rd/4th-order n-gram beats
   fixed read-ahead on `hm_1` (+0.41pp with 20x fewer prefetches, 66% vs 9%
   precision) and beats no-prefetch on `proj_3` (+0.64pp), but loses on the
   other four. Trace-dependent, not a general win.

3. *The cache is too small.* This one is a genuine defect: the default is 128
   **blocks** while these traces issue 8-128-block requests, so the cache
   holds 1-16 whole requests. On `proj_3`, 128 -> 2048 blocks moves no-prefetch
   hit ratio from 0.32% to 67.37%. It is left as a documented default rather
   than changed, because changing it would alter every previously reported
   number.

**Why the negative result is real.** These traces are ~49% backward seeks and
~43% long forward jumps, and 82% of re-reads in `hm_1` are at a reuse distance
above 1024 read blocks -- 8x the cache capacity. An address-only predictor
issuing a bounded number of speculative fetches cannot turn a reuse distance
of thousands of blocks into a hit. The bottleneck is address
**predictability**, not classifier quality, feature design, or cache policy.

**What is legitimately claimed.** The evidence router issues **3-4x fewer
prefetches** than fixed read-ahead while retaining comparable hit ratio. That
is a real reduction in wasted I/O and in device load, and it is the honest
positive result available from this data. It is not a hit-ratio win.

**Explicitly not done:** threshold tuning until adaptive won. With 7 traces
and sub-1.5pp differences, tuning would find a configuration that wins on
these traces and fails on any new workload -- overfitting to the evaluation
set while appearing to succeed.

---

---

## D10. Feature engineering pass: the generator was the root cause

**Requested:** implement (1) a realistic generator, (2) cross-window features,
(3) a reuse-distance feature, (4) a scale-relative `dominant_stride`.

### (1) The generator -- by far the largest effect

The old generator drew addresses from `randrange(1000, 100000)`, request sizes
from `randint(1, 3)`, read ratio fixed at 0.9, and never revisited an LBA. Its
within-class feature standard deviation was **0.0000-0.056** against
**0.03-1.28** measured on real traces. The model was fitted to a degenerate
distribution.

New generator draws, per window: pattern strength (0.55-1.0), block size
(1-128, median 8), timing dispersion, a bimodal read/write ratio, a varying
address span (1e6-4e8 blocks), and occasional re-access. `random` windows also
get weak local clusters -- previously that class had **exactly zero** feature
variance.

| measurement | before | after |
| --- | ---: | ---: |
| domain shift, mean abs | 11.97 sd | **0.86 sd** |
| worst feature | 76.82 sd | **1.63 sd** |
| coverage of target in reference range | 58.5% | **97.2%** |
| model prediction on real windows | 124/124 "mixed" @ confidence 1.000 | 22 random / 9 mixed @ 0.978 |
| held-out synthetic accuracy | 1.0000 | 0.9688 |

The model used to be *confidently wrong* on every real window. It now reports
"random" for most of them, which is correct: real traces are overwhelmingly
random-looking. The drop from 1.0000 to 0.9688 is the intended effect -- the
task is no longer trivially separable.

**Second defect found while fixing this:** strides were drawn as
`block * randint(1, 4)`, so a stride of exactly one request length is
geometrically identical to a sequential read. That collapsed two classes and
22/40 sequential windows were labelled strided. Strides are now 2-8 request
lengths.

### (2) and (3) Contextual features

Four features were added, bringing the vector from 8 to 12:

- `reuse_ratio` -- share of the window touching a previously-seen block.
- `reuse_distance_log` -- mean log2 reuse distance (scale-free across a range
  that spans 1 to ~1e6).
- `contiguous_delta` -- contiguity versus the previous window.
- `stride_persist` -- whether the stride or contiguity is continuing.

These use `features.WindowContext`, which maintains a block-age table and the
previous window's spatial features. That information is what a real system
gets from block-age tracking, so it is available at decision time and is not
lookahead. `extract()` still returns the 8 stream features alone
(`STREAM_FEATURES = 8`), so every existing caller keeps working.

**Train/deploy consistency matters here.** `benchmark.make_model` now lays
each class out as one continuous stream so `WindowContext` accumulates values
exactly as replay does. Training on shuffled isolated windows would have
taught the model that all four contextual features are always zero.

Measured at replay: `reuse_ratio` mean 0.155 (15 distinct values),
`reuse_distance_log` 0.033 (17 distinct), `contiguous_delta` -0.007 (16
distinct), `stride_persist` 0.438. None is degenerate.

### (4) `dominant_stride`

The old candidate filter was `d > size_blocks and d <= 1024` -- an absolute
ceiling. It returned `None` on **100%** of real windows measured (0/31, 0/62,
0/1562), making the window-level stride detector structurally blind on real
data.

Both an absolute cap and a relative one (tried at 64x request size, which
still rejected a legitimate 4096-block stride) conflate "jump" with "stride".
A stride is defined by **repetition, not magnitude**: 4096 seen 15 times is a
stride, 4096 seen once is a seek. The cap is removed entirely; only the
repetition requirement remains, expressed as a fraction of the window.

### (5) Cache default -- a consequence of (1), not a separate change

The default capacity was 128 **blocks**, calibrated for the old generator's
1-3 block requests. Once requests became a realistic 8-128 blocks, 128 blocks
held ~16 whole requests and starved every policy. Measured saturation point:

| capacity | ~requests | synthetic `none` | synthetic `adaptive` |
| ---: | ---: | ---: | ---: |
| 128 | 16 | 3.14% | 5.15% |
| 2048 | 256 | 5.68% | 7.86% |
| 32768 | 4096 | 5.68% | 7.86% |

Hit ratios saturate by 2048 blocks (1 MiB at 512 B). `DEFAULT_CAPACITY` is now
2048. This restates every previously reported cache number, which is why
`DECISIONS.md` D7 and D9 previously described 128 blocks as a known
mis-specification to be left alone; fixing the generator made leaving it
incorrect.

---

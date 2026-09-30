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
*Superseded: D10 later fixed the generator. The corrected figures are
99.58 +/- 0.59 on baseline with 0.9688 held-out -- see
`CLASSIFIER_EVAL.md` section 1. The decision recorded here is unaffected;
the numbers quoted were not real.*
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

## D11. The prefetch cost model made the cost column unwinnable

`LatencyModel.measured()` charged a speculative read `1.1 x mean_service`,
justified in a comment as "a full device read plus queueing pressure". That
put break-even precision at **1.111** -- above 100%, so no prefetcher could
ever pay for itself under any policy on any trace. The sweep's
"lowest modelled cost" column was therefore decided by arithmetic, not by
measurement.

Changed the default to `prefetch_multiple = 1.0` (a prefetch is one device
read) and added `LatencyModel.break_even_precision()` plus a
`--prefetch-multiple` flag on both entry points.

**This did not fix the problem, and the first draft of the code comment
claimed it had.** Break-even is `prefetch_us / (demand_miss_us - hit_us)`,
which at `hit = 0.01 x mean` and `prefetch = mean` is `1/0.99 = 1.0101`.
Still above 1. The cost column is degenerate at *any* `prefetch_multiple >= 1`.

That is not a modelling artefact, it is the arithmetic of this hardware: a
speculative read performs the same device read as the demand read it
replaces, and saves strictly less than a whole demand read because the hit
still costs a cache lookup. To be worth it, a prefetcher must exceed 100%
precision. No prefetcher can.

Consequences, stated plainly:

* The modelled-cost column of every benchmark is degenerate at the default
  setting. `none` wins it by construction. `report.cost_model_section()` and
  both CLI entry points now print the break-even and say so *before* any
  reader sees a cost table.
* Values **below 1.0** model overlapped or coalesced prefetch: identical
  device work, but the requester is not blocked by it. That is an assumption
  these traces cannot support -- they record demand-read service time only,
  never queue depth or overlap. It is a flag, not a default, and
  `--prefetch-multiple 0.5` is the setting under which prefetching is even
  arithmetically possible.
* Cost ranking is invariant to a **uniform rescale** of the calibration --
  every mode's cost is linear in the mean service time, so multiplying all
  three costs by the same factor cannot move `argmin`. That is pinned by
  `test_cost_ranking_is_invariant_to_the_calibration_scale`.
  It is **not** invariant to `prefetch_multiple`, which re-weights only the
  prefetch term. That distinction was originally written up wrongly here, and
  an audit caught it: at `prefetch_multiple = 1.0` `none` is cheapest on a
  contiguous scan (100.00 vs `deep` 101.01 us/read), and at 0.9 the argmin
  flips to `deep` (91.06 vs 100.00). The flip is pinned by
  `test_prefetch_multiple_does_change_the_cost_ranking`. So the setting does
  change which policy is cheapest -- it just cannot change the *hit ratio*,
  which is what the cache actually produces.

Reported metrics for the whole collection are therefore re-stated: hit ratio,
precision and wasted I/O are the informative columns; modelled cost is
degenerate and is labelled as such.

## D12. Invert the target: predict *harm*, not the best policy

The oracle experiment (D9 follow-up, temp script `trueoracle.py`) settled the
"can the classifier be better?" question: a perfect window router beats fixed
read-ahead by only +0.42 / +0.26 / +0.02 / +0.00 points of hit ratio on
hm_1 / src2_2 / ts_0 / proj_0. Four-way routing had almost no headroom,
because `none`, `sequential` and `strided` perform near-identically.

The one large real win was the opposite decision. On `web_3`, `none` reaches
50.61% and `sequential` reaches 25.70% -- read-ahead **halves** the hit ratio
by evicting useful blocks. A perfect router recovers +24.83 points there, all
of it from knowing when *not* to prefetch.

So the target was inverted: the only decision worth making is "will
prefetching hurt?".

**That decision does not need predicting.** `LRUCache` already counts
`useful_prefetches`, so the realised precision of our own speculative reads is
observable at read time, causally, with no future knowledge and no trained
model. `guard.py` adds three feedback-gated modes that compare realised
precision against `break_even_precision()`:

* `guard` -- prefetch only while precision clears break-even
* `depth_adaptive` -- depth scaled by the margin above break-even
* `correlate` -- additionally require a stride confirmed over a longer
  history than one window

`policy_interval` lets the gate re-decide faster than the classifier's window
(default: one decision per `window_size` requests).

## D13. Policy space: read-ahead *depth* was the missing dimension

The +0.4-point oracle ceiling in D9 was not evidence that prefetching had
little headroom. It was evidence that the policy space was degenerate: every
mode issued at most 2 blocks from one heuristic, so `sequential` and
`strided` produced identical hit ratios on 4 of 5 traces
(`dominant_stride` returns `None` on real data, so `strided` degenerates to
`none`).

Isolating depth from policy choice -- first 40,000 requests of 8 real traces,
2048-block cache, window 32:

| trace | d=1 | d=2 (baseline) | d=4 | d=8 | d=16 | d8-d2 | d16-d2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wdev_3 | 10.21% | 10.28% | 10.44% | 10.76% | 11.39% | +0.47 | +1.11 |
| rsrch_1 | 49.46% | 49.64% | 50.00% | 50.72% | 51.80% | +1.08 | +2.16 |
| rsrch_2 | 54.80% | 56.34% | 59.41% | **65.89%** | 58.69% | **+9.54** | +2.35 |
| web_3 | 25.52% | 25.70% | 26.07% | 26.77% | 27.25% | +1.07 | +1.55 |
| src2_2 | 29.98% | 31.67% | 35.24% | 42.72% | **46.77%** | **+11.05** | +15.10 |
| ts_0 | 32.50% | 33.50% | 35.52% | 39.63% | **42.35%** | +6.13 | +8.85 |
| proj_0 | 31.00% | 31.63% | 33.00% | 35.76% | **38.48%** | +4.13 | +6.85 |
| hm_1 | **31.52%** | 31.39% | 30.97% | 30.71% | 30.72% | -0.68 | -0.67 |
| **mean** | 33.12% | 33.77% | 35.08% | 37.87% | 38.43% | **+4.10** | **+4.66** |

**Depth 8 is worth +4.10 points of mean hit ratio** over the depth-2 baseline
that every earlier mode used -- an order of magnitude more headroom than the
entire policy-selection question. Depth 16 scores +4.66 on the mean but is
*not* the default: it issues 4x the prefetches and loses 7.20 points on
rsrch_2, where a deeper prefetch evicts blocks the workload still wants.
Depth 8 is best-or-near-best on all 8 traces; only `hm_1` prefers shallower
prefetch at all.

So `deep` was added: fixed-depth read-ahead, `DEFAULT_READ_AHEAD_DEPTH = 8`.
It is not a clever policy -- it is the baseline the earlier mode set was
missing, and the gate has to beat it.

On a contiguous synthetic scan the effect is stark: depth 2 gives 24.86% hit
ratio, depth 8 gives 99.45%, at 99.98% prefetch precision.

*Correction (second audit):* the first version of this entry quoted "+4.66"
and "+15.10 on src2_2" as the depth-8 result. Those are the depth-**16**
deltas. The depth-8 numbers are +4.10 and +11.05. Both are in the table
above.

## D14. Where the feedback gate actually wins, measured

First 40,000 requests of each real trace, 2048-block cache, window 32, default
cost model (break-even 1.0101, so the gate is shut for most of the run).

| trace | `none` | `sequential` | `deep` | `guard` | best |
| --- | ---: | ---: | ---: | ---: | --- |
| wdev_3 | 10.13% | 10.28% | **10.76%** | 10.28% | deep |
| rsrch_1 | 49.19% | 49.64% | **50.72%** | 49.64% | deep |
| rsrch_2 | 53.21% | 56.34% | **65.89%** | 53.69% | deep |
| web_3 | **50.61%** | 25.70% | 26.77% | 44.69% | none |
| src2_2 | 28.31% | 31.67% | **42.72%** | 28.67% | deep |
| ts_0 | 31.46% | 33.50% | **39.63%** | 33.18% | deep |
| proj_0 | 30.34% | 31.63% | **35.76%** | 30.63% | deep |
| hm_1 | **31.76%** | 31.39% | 30.71% | 31.62% | none |

Wasted prefetches -- issued minus subsequently read:

| trace | `sequential` | `deep` | `guard` | gate as % of sequential |
| --- | ---: | ---: | ---: | ---: |
| wdev_3 | 14 | 56 | 14 | 100.0% |
| rsrch_1 | 31 | 127 | 31 | 100.0% |
| rsrch_2 | 14,671 | 58,608 | 1,837 | 12.5% |
| web_3 | 9,718 | 41,073 | **1,275** | **13.1%** |
| src2_2 | 18,970 | 77,566 | 2,459 | 13.0% |
| ts_0 | 6,914 | 28,608 | 5,776 | 83.5% |
| proj_0 | 14,190 | 58,503 | 2,630 | 18.5% |
| hm_1 | 26,522 | 115,235 | 3,974 | 15.0% |

Read honestly:

* **`deep` wins hit ratio on 6 of 8** and is the strongest single result in
  the project. It is also by far the most wasteful -- 77,566 wasted
  prefetches on src2_2 against `sequential`'s 18,970. It buys hit ratio with
  speculative I/O, and the cost model says that trade does not pay.
* **The gate wins exactly where the hypothesis said it would** -- web_3, the
  one trace where read-ahead wrecks the cache. +18.99 points of hit ratio
  over `sequential` *and* 86.9% less wasted I/O. It does not beat `none`
  there (50.61%); it recovers most of the 24.91-point damage `sequential`
  does.
* **On the other seven traces the gate loses to `sequential`**, by 0.00 to
  3.00 points. What it buys is less waste: 12-19% of `sequential`'s on five
  traces, but only 16.5% less on ts_0 and none at all on the two traces short
  enough (wdev_3, rsrch_1) that the gate never finishes exploring.
* **`none` still wins on `web_3` and `hm_1`.** Unchanged from D9.

The gate is not a general win. It is a targeted win on cache-polluting
workloads, and it buys that by spending less, not by predicting better.

The gate's own final decision, for the record (default cost model, so every
margin is against a break-even of 1.0101):

| trace | observed precision | margin | policy switches |
| --- | ---: | ---: | ---: |
| wdev_3 | 0.125 | -0.401 | 1 |
| rsrch_1 | 0.139 | -0.387 | 1 |
| rsrch_2 | 0.312 | -0.215 | 232 |
| web_3 | 0.371 | -0.156 | 196 |
| src2_2 | 0.232 | -0.295 | 278 |
| ts_0 | 0.594 | **+0.068** | 53 |
| proj_0 | 0.440 | -0.086 | 196 |
| hm_1 | 0.177 | -0.349 | 276 |

Realised precision ranges from 0.125 to 0.594, never near the 1.0101
break-even, which is why the gate spends most intervals closed. ts_0 is the
one trace where it opens on its own evidence (margin +0.068) -- and there it
still scores 0.32 points *below* `sequential`, so a positive margin is not a
guarantee of a win. That is a reminder that break-even precision is a
necessary condition for a prefetcher to pay, not a sufficient one.

*Correction (second audit):* the first version of this entry reported
`src2_2 none 16.94%`, `ts_0 22.60%`, `proj_0 28.49%`, `hm_1 37.86%` and
`rsrch_2 53.63%`. Those came from a script whose `head()` helper kept
overwriting `out` on each chunk and broke on the second pass, so it returned
requests 40,001-80,000 for any trace longer than 80k. The table above takes
the *first* 40,000 requests and the numbers are reproducible from it. The
`web_3` conclusion (+18.99 over `sequential`, 86.9% less waste) was and
remains correct -- web_3 is shorter than 40k requests, so it was never
affected by the bug.

### D14.1 `depth_adaptive` is identical to `guard` at the default cost model

Measured, and worth stating rather than hiding: at `prefetch_multiple = 1.0`
the break-even is 1.0101 while realised precision is clamped to at most 1.0,
so `margin <= -0.0101 < 0` on **every** non-probing decision.
`depth_for_margin()` returns 0 whenever the margin is not positive, so the
mode is driven entirely by the exploration budget and the 1-in-9 re-probe,
both of which use the same `min_depth = 2` that `guard` uses. On all 8 traces
above the two modes agree on hit ratio, prefetch count, wasted I/O and policy
switches, to the digit.

It is kept because it is a genuinely different policy in the only regime
where prefetching is arithmetically possible (`--prefetch-multiple 0.5`),
where it does diverge -- though even there it trails `guard` on the traces
where they differ. It earns a row in the benchmark matrix so that divergence
is measurable rather than assumed, and `observed_precision`,
`precision_margin` and `observed_stride` are written into the sweep CSV so a
reader can see *why* a gated mode issued what it did.

## D15. Bugs the new modes introduced, and the audit that found them

A two-angle audit (one agent for bugs/dead code, one to re-verify claims with
numbers) found nine defects. All are fixed with regression tests.

**Crash.** `replay_stream` had no feedback-gated branch at all, so
`sweep_modes.py` died with `ValueError: unknown mode` on the 9th mode of the
first trace. Root cause was duplicated loop logic: `replay` and
`replay_stream` each carried their own copy. Fixed by extracting
`GateController` so both drive one state machine, plus `_publish_gate()` so
the two cannot report different things for the same run. A test asserts
`replay` == `replay_stream` on every gated mode across a chunk boundary.

**Wrong result: cumulative usefulness recorded as an interval delta.**
`GateController.decide()` recorded `self._useful_total` -- the *cumulative*
useful count -- as if it were the interval's own. Every interval re-counted
all previous usefulness, the rolling ratio pinned at 1.0, and the gate sat at
maximum depth permanently. This produced an apparent **+12.3-point "win" on
rsrch_2 and +30.9 on src2_2** that was entirely an artefact. Realised rolling
precision is 0.34-0.42, and the true numbers are in D14. The clamp at 1.0 in
`PrecisionMonitor.precision` had been hiding the symptom. Two tests now pin
the accounting invariant (`monitor.useful` must equal the true total) and
that precision must decay when usefulness stops arriving.

**Wrong result: `read_ahead` double-added `lba`.** A nested generator
expression produced `2*lba + size + k`, so every read-ahead prefetch went to
the wrong half of the disk and realised precision was exactly 0.0.

**Wrong result: the gate latched shut.** Once closed it never reopened: it
stopped issuing, so it never gathered new evidence, so it stayed closed even
if the workload changed completely. `REPROBE_AFTER = 8` forces a one-interval
probe after eight consecutive closed decisions. Pinned by a test that changes
workload mid-stream and requires the hit ratio to recover.

**Wrong result: `correlate` confirmed a stride and then ignored it.**
`metrics.observed_stride` was set but `read_ahead` was called without it, so
`correlate` was a byte-identical duplicate of `guard` -- a dead mode. Fixed
by threading the stride through `GateController.candidates()`. It is now the
only policy that scores on a stride-512 scan: 93.3% precision and 11.20% hit
against 0.00% for both `guard` and `sequential`.

**Wrong result: usefulness discarded in idle intervals.** `record()` returned
early when an interval issued nothing, but usefulness from a prefetch issued
several intervals earlier routinely lands in exactly such an interval.

**Wrong result: precision could exceed 1.0.** A bounded window pairs
usefulness from older issuances with fewer recent ones. Clamped, with the
reason recorded.

**Wrong result: `break_even == 0` treated as a veto.** A free prefetch was
refused. Only an infinite break-even (a hit that saves nothing) is a veto now.

**Wrong result: `observed_precision` reported 0.0 for "never measured".** Now
`None`, so unmeasured is distinguishable from measured-as-zero.

**Resource leak.** `CorrelateDetector.table` was an unbounded `Counter` of
offset tuples -- ~150 bytes per request on a narrow-address workload, a third
of the cost of the `Request` objects the streaming design exists to avoid,
and O(len(table)) per decision. Now capped with least-seen eviction (a stride
worth acting on is seen many times, so eviction cannot discard it) and
memoised.

Also removed: a leftover `oracle_horizon` parameter on `replay_stream` that
did nothing, a redundant `pending` counter duplicating `interval`, an unused
import, and a no-op `reset_interval()`.

## D16. Second audit round: five more wrong-result bugs, three wrong claims

The first round was validated by a second, independent audit that both
re-derived every published number and mutated the code to check the
regression tests actually bite. It found five more wrong-result bugs, three
incorrect claims in this file, and two metrics that reached no output at all.
All fixed.

**Wrong result: `read_ahead`'s stride branch under- and over-prefetched.**
It proposed `depth` blocks *spaced* `stride` apart -- the first block of each
of the next `depth` requests, not `depth` blocks -- and then dropped any
candidate inside the request's own extent. Two consequences: at
`stride < size_blocks` it returned **nothing at all**, and otherwise it
proposed one block per request instead of a full extent. On a contiguous scan
(a request at lba 1000 of size 8) `correlate` scored **half** of `guard`
(12.80% vs 24.79%), on the single most common workload shape there is; dense
stride-1 gave 88.28% against `guard`'s 99.88%. The existing tests only used
stride 512 with size 8, which is the one case the broken code handled
correctly. It now walks the predicted stream for `depth` blocks.

**Wrong result: `CorrelateDetector`'s memo was keyed on `len(table)`.**
Incrementing an existing key's count does not change `len(table)`, so after a
workload phase change the detector reported the **old** stride indefinitely:
300 reads at stride 512 followed by 300 at stride 1000 still reported 512.
A stride worth acting on is by definition a repeated one, which is exactly
the case this memo broke. The memo is gone; it was O(len) over a capped table
and called once per interval, so it was not buying anything.

**Wrong result: `replay_stream` skipped `replay`'s validation.** It accepted
`window_size = 4` (below the documented floor of 8) and
`read_ahead_depth = -5` (which silently produced zero prefetches). Both paths
now call one `_validate_replay()`.

**Wrong result: `adaptive_evidence` policy switches were counted in `replay`
and not in `replay_stream`** -- 230 versus 0 on web_3, while every cache
metric agreed. This falsified the blanket "the two paths are identical"
claim, so it is now restated as a test over the whole mode set
(`test_every_supported_mode_agrees_across_both_paths`).

**Wrong result: an infinite break-even was documented as a hard veto but was
not one.** Exploration and re-probe bypassed `margin()` entirely, so a cost
model where a prefetch saves nothing (`hit_us == demand_miss_us`) still
issued 1,088 provably wasted prefetches. Now vetoed before exploration.

**Wrong claim: cost ranking is invariant to `prefetch_multiple`.** It is not.
That claim was mine, in D11, backed by a test that rescaled all three costs
uniformly -- a different thing. Re-deriving it: `argmin` flips from `none` to
`deep` at `prefetch_multiple = 0.9`. Corrected in D11 and pinned by
`test_prefetch_multiple_does_change_the_cost_ranking`.

**Wrong claim: the D13 and D14 numbers.** "+4.66" and "+15.10 on src2_2" were
depth-16 deltas presented as depth-8 results. Worse, the D14 per-trace table
came from a script whose chunk helper returned requests 40,001-80,000 instead
of the first 40,000, so four of the eight `none` rows were simply the wrong
slice. Both entries are corrected above, and the sweep CSV now carries the
gate diagnostics so the numbers are checkable without rerunning anything.

**Dead output: `observed_precision`, `precision_margin` and `observed_stride`
reached no file.** They were written to `Metrics` and read only by tests, so
the central claim of D12 -- that a policy can observe its own prefetch value
and act on it -- produced a number no reader could obtain. All three are now
columns in `outputs/msrc_sweep.csv`.

**Cosmetic:** `PrecisionMonitor(window=4)` silently floored to 16, so three
tests were not exercising the window they named. The floor is now a named
`MIN_WINDOW` and the tests use it explicitly and prove the window rolls.

## D17. Collection-wide confirmation: depth is the whole result

The full MSRC collection, all 32 traces at 250,000 requests each (the default
cap), 2048-block cache, 12 modes, 384 rows, 3362 s. 31 traces score; `wdev_1`
has no read requests.

Request-weighted means, as reported by `sweep_modes.py`:

| mode | mean hit | mean precision | mean us/read | wasted / issued | traces won |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deep` | **51.48%** | 45.26% | 4952.02 | 5,862,075 / 11,180,287 | **29/31** |
| `sequential` | 46.91% | 45.40% | 4367.04 | 1,437,248 / 2,756,740 | 0/31 |
| `lstm` | 46.59% | 41.60% | 4166.49 | 305,111 / 620,325 | 0/31 |
| `adaptive_evidence` | 46.56% | 50.17% | 4228.30 | 605,842 / 1,362,927 | 0/31 |
| `correlate` | 46.55% | 40.74% | 4163.99 | 188,334 / 326,028 | 0/31 |
| `guard` | 46.44% | 43.57% | 4172.56 | 182,298 / 338,519 | 0/31 |
| `depth_adaptive` | 46.44% | 43.57% | 4172.56 | 182,298 / 338,519 | 0/31 |
| `markov` | 46.54% | 49.65% | 4134.76 | 132,401 / 358,838 | 0/31 |
| `strided` | 46.41% | 46.83% | 4106.31 | 9,436 / 32,898 | 0/31 |
| `none` | 46.38% | 0.00% | **4104.80** | - | 1/31 |
| `stride` | 46.38% | 61.23% | 4119.06 | 33,412 / 121,992 | 0/31 |
| `adaptive` | 46.34% | 53.79% | 4179.61 | 313,072 / 874,017 | 1/31 |

Unweighted per-trace means (each trace counts once, so one huge trace cannot
dominate), which is the fairer comparison:

| | mean hit | vs `none` |
| --- | ---: | ---: |
| `deep` | 49.87% | **+4.94 pp** |
| `sequential` | 45.45% | +0.52 pp |
| `lstm` | 45.13% | +0.20 pp |
| `adaptive_evidence` | 45.10% | +0.17 pp |
| `correlate` | 45.10% | +0.16 pp |
| `markov` | 45.09% | +0.15 pp |
| `guard` | 44.99% | +0.06 pp |
| `depth_adaptive` | 44.99% | +0.06 pp |
| `strided` | 44.96% | +0.02 pp |
| `none` | 44.93% | - |
| `stride` | 44.93% | -0.01 pp |
| `adaptive` | 44.89% | **-0.04 pp** |

**The decisive number is the contrast between two spreads.** All eleven
non-deep modes fall within **0.55 points** of each other, and the trained
classifier is the *worst* of them, below doing nothing. `deep` sits **4.42
points above the best of them** -- eight times the entire spread of policy
choice.

That is the D13 thesis at collection scale, and it retires the question this
project spent most of its effort on. Choosing *which* prefetching heuristic
to apply is worth at most half a point. Choosing how *deep* to prefetch is
worth five. Every earlier conclusion here -- including the "a perfect router
has only +0.4 points of headroom" result that motivated the whole inversion
of the target -- was measured correctly but against a degenerate baseline
that prefetched 2 blocks.

Per-trace winners: `deep` 29, `none` 2, `adaptive` 1.

### Where the feedback gate actually helps

`guard` beats `sequential` on **3 of 31** traces:

| trace | `guard` | `sequential` | `none` | gate vs seq |
| --- | ---: | ---: | ---: | ---: |
| web_3 | 44.69% | 25.70% | 50.61% | **+18.99** |
| proj_3 | 67.27% | 66.63% | 67.37% | +0.63 |
| hm_1 | 34.57% | 34.34% | 34.69% | +0.23 |

On the other 28 it loses. What it consistently buys is far less wasted I/O:
11.1% to 11.4% of `sequential`'s waste on the five largest offenders
(prn_1 9,219 vs 83,170; src2_2 6,352 vs 56,914; src1_1 12,684 vs 112,671).
A consistent ~89% reduction in speculative I/O for a small hit-ratio cost.

`correlate` is the better gate on web_3: **48.18%** against `guard`'s 44.69%,
on 272 wasted prefetches against 802. Requiring a confirmed stride *and* a
positive margin makes it the most conservative variant, and on the one trace
where conservatism is what pays, it is the one that pays most. It still does
not reach `none` (50.61%).

### What this does not say

`deep` buys its +4.94 points with 11.2 million prefetches of which **5.86
million are never read** -- 52% waste, and 4x the volume `sequential`
issues. At the default cost model the modelled us/read is 4952 against
`none`'s 4105, so this is a losing trade in the only currency the traces can
actually be measured in. The hit-ratio win is real; the claim that it is
worth having is not supported, and the two must be reported together.

### On the strength of the regression tests

The audit reintroduced each of the seven original bugs one at a time and
confirmed the suite fails every time, so the tests are load-bearing rather
than decorative. It also re-derived the D13 depth table from scratch and got
it exactly (33.12 / 33.77 / 35.08 / 37.87 / 38.43), and checked that
`CorrelateDetector` eviction preserves a genuine stride under heavy noise
over 500k requests at both capacity 4096 and 256 -- 19 of 19 checkpoints.

Test count: 91 -> 131.

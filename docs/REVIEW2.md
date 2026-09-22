# Project Review 2 guide

## Review rubric coverage

| Criterion | Evidence to show |
| --- | --- |
| System design and architecture (1) | Pipeline below, trace schema, and causal decision timing. |
| Module identification (1) | Module table below and file boundaries. |
| Module completion (4) | Live `demo`, seven-row `benchmark`, `replay`, and unit tests. |
| Presentation and discussion (4) | Slide outline, scripted demonstration, and viva answers below. |

## System design and data flow

```text
CSV or generated I/O requests
        ↓
validated records + fixed request windows
        ↓
size-aware spatial/timing features
        ↓
incremental supervised classifier
        ↓
sequential / strided / random / mixed
        ↓
policy applied to subsequent requests
        ↓
LRU cache and prefetch statistics
```

The first complete window is observation only. Its prediction changes the
policy for the following requests. This avoids using future information.
For controlled synthetic streams, a known window label can update the online
model *after* prediction. No online update is made on unlabelled real traces.

## Modules and status

| Module | Input | Output | Status |
| --- | --- | --- | --- |
| Trace generator / CSV parser | Seed or normalized CSV | Validated requests | Implemented, tested |
| Window feature extractor | Recent requests | Eight numerical features | Implemented, tested |
| Online classifier | Features and labelled training windows | Predicted class and confidence | Implemented, tested on synthetic data |
| Policy selector | Predicted class, recent stride | Future block candidates | Implemented, tested through replay |
| LRU cache simulator | Requests and prefetches | Hits, precision, wasted prefetches | Implemented, tested |
| Baselines | Same trace and cache settings | No/fixed prefetch comparison | Implemented |
| Real-trace adapter | MSR-format CSV fields | Normalized requests | Implemented and tested on the included sample |
| Classic stride + Markov baselines | Request address deltas | Prefetch candidates | Implemented and tested |
| Optional offline LSTM | Training traces + PyTorch | Saved next-delta model | Code implemented; runtime unverified here because PyTorch is unavailable |
| IOTTA-related format profiles | Confirmed per-trace CSV units | Normalized requests | Parsers tested on fixtures; no full public trace benchmarked |
| Benchmark + cost model | Same trace/cache for each mode | Hit, waste, oracle recall, modelled cost, compute timing | Implemented and tested |
| Kernel I/O integration | Live OS requests | Live prefetch actions | Outside current user-space prototype |

## Prepared live demonstration

1. Explain the five-block pipeline above and the four workload classes.
2. Run `.\run_review2.ps1`.
3. Identify separate training and held-out generated test windows.
4. Read one confusion-matrix row and explain what a mistake would mean.
5. Point out sequential → strided → random → mixed window predictions.
6. Run `.\run_review2.ps1 -Benchmark` and compare no prefetch, fixed
   read-ahead, window stride, classic stride, Markov, and adaptive policies.
   The optional LSTM row is explicitly skipped until its dependency and
   trained artifact are available.
7. Explain that a higher hit ratio is not guaranteed on every synthetic mix.
   Also compare precision and unnecessary prefetches.
8. Run `.\run_review2.ps1 -Test`.
9. Run `.\run_review2.ps1 -Trace msr-cambridge1-sample.csv` and state that its
   workload labels and source provenance are unverified.
10. State limitations: synthetic labels, modelled rather than measured device
    latency, unverified sample provenance, and no full IOTTA result.

## Suggested slide sequence

1. Title and project objective.
2. Problem: one fixed prefetcher does not suit every access pattern.
3. Review 1 objective and Review 2 implementation boundary.
4. Architecture and request-to-decision timing.
5. Input schema and four generated workload types.
6. Features and incremental classifier.
7. Class-to-policy mapping and LRU cache model.
8. Implemented modules and live demonstration.
9. Held-out confusion matrix and cache comparison from the actual run.
10. Limitations, next experiments, and conclusion.

## Discussion preparation

- **Why classify a window?** One request alone has no access pattern. Multiple
  recent requests reveal deltas, locality, and timing.
- **Why is the model ML rather than a rule table?** It estimates class-conditional
  feature distributions from labelled examples, predicts unseen feature vectors,
  and updates its statistics one labelled window at a time.
- **Where do online labels come from?** Generated workloads provide them. Real
  traces need an independent labelling method; cache success is not a label.
- **What does the confidence mean?** It is the model's normalized score under
  its assumptions, not a calibrated real-world probability.
- **Why use simulation?** It tests the classify-then-prefetch logic and exposes
  cache tradeoffs without kernel modification or privileged I/O hooks.
- **Why can adaptive lose to fixed read-ahead on a test?** The classifier needs
  an observation window, policy switches lag changes, and fixed read-ahead can
  get hits by issuing many extra prefetches. Compare wasted work too.
- **What remains?** Obtain a metadata-verified full public trace, run the
  optional LSTM in a PyTorch environment, test more cache capacities, and
  validate the latency model against actual device observations.
- **Does this prove the slide's expected wins?** No. The benchmark makes those
  hypotheses testable, but a win on every workload or faster adaptation is
  not guaranteed. The LSTM predicts the next address delta, not one of the
  four workload classes, so compare end-to-end cache metrics and compute
  cost, not their classification accuracies.

## Verification record

Run the commands in README immediately before the review and paste the actual
results into slides. Do not use the expected outcomes from Review 1 as if they
were measurements. The existing Review 2 deck was created before Stage 2 and
its numeric results should be refreshed from the current benchmark before
presentation.

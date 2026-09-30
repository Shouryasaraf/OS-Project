# Project Review 2 guide

Read `docs/DECISIONS.md` D11-D17 before the review. Those are the entries
that explain why the headline result is what it is, and a reviewer who reads
one of them and asks a follow-up is much easier to defend against than one
who only saw a table.

## Review rubric coverage

| Criterion | Evidence to show |
| --- | --- |
| System design and architecture (1) | Pipeline below, trace schema, and causal decision timing. |
| Module identification (1) | Module table below and file boundaries. |
| Module completion (4) | Live `demo`, **12-mode** `benchmark`, `replay`, `sweep_modes`, and 131 unit tests. |
| Presentation and discussion (4) | Slide outline, scripted demonstration, and viva answers below. |

## System design and data flow

```text
CSV or generated I/O requests
        ↓
validated records + fixed request windows
        ↓
12 numerical features (8 stream + 4 contextual)
        ↓
incremental supervised classifier
        ↓
sequential / strided / random / mixed
        ↓
policy applied to subsequent requests
        ↓
LRU cache and prefetch statistics
```

Three policy families now share that cache, and it matters which one you
present:

- **Classifier-routed** (`adaptive`) -- a class label chooses a policy.
- **Evidence-routed** (`adaptive_evidence`) -- the window's own measured
  features choose a policy, no trained model.
- **Feedback-gated** (`guard`, `depth_adaptive`, `correlate`) -- no classifier
  at all. Read-ahead depth comes from the *realised* precision of the
  system's own prefetches, which `LRUCache` already counts, so the decision is
  causal and needs no future knowledge.
- **Fixed** (`none`, `sequential`, `deep`, `strided`, `stride`, `markov`,
  `lstm`) -- no adaptivity of any kind.

The first complete window is observation only. Its prediction changes the
policy for the following requests. This avoids using future information.
For controlled synthetic streams, a known window label can update the online
model *after* prediction. No online update is made on unlabelled real traces.

## Modules and status

| Module | Input | Output | Status |
| --- | --- | --- | --- |
| Trace generator / CSV parser | Seed or normalized CSV | Validated requests | Implemented, tested |
| Window feature extractor | Recent requests | **12** numerical features | Implemented, tested |
| Online classifier | Features and labelled training windows | Predicted class and confidence | Implemented; **96.88%** held-out on synthetic windows only |
| Policy selector | Predicted class, recent stride | Future block candidates | Implemented, tested through replay |
| Feedback gate (`guard.py`) | Observed prefetch precision | Read-ahead depth 0-8 | Implemented, tested; **no model required** |
| LRU cache simulator | Requests and prefetches | Hits, precision, wasted prefetches | Implemented, tested |
| Baselines | Same trace and cache settings | No/fixed prefetch comparison | Implemented |
| Real-trace adapter | MSR-format CSV, `.revised` | Normalized requests | Implemented; MSR collection, 32 traces, 11.86 GB |
| Classic stride + Markov baselines | Request address deltas | Prefetch candidates | Implemented and tested |
| Optional offline LSTM | Training traces + PyTorch | Saved next-delta model | Implemented; PyTorch 2.14 CPU **is** installed, but the artifact is gitignored and must be trained |
| Streaming sweep (`sweep_modes.py`) | Whole `.revised` files in bounded chunks | 384 rows over 32 traces | Implemented; memory-bounded and bit-identical to `replay` |
| IOTTA-related format profiles | Confirmed per-trace CSV units | Normalized requests | Parsers tested on fixtures; no full public trace benchmarked |
| Benchmark + cost model | Same trace/cache for each mode | Hit, waste, oracle recall, modelled cost, compute timing | Implemented; **cost column degenerate by default** |
| Kernel I/O integration | Live OS requests | Live prefetch actions | Outside current user-space prototype |

## The result, stated honestly before you are asked for it

Say this first; do not let a reviewer discover it in a table.

> Across the full MSRC collection, a **fixed depth-8 read-ahead wins on 29 of
> 31 traces**. It is not adaptive -- it never looks at the workload. All
> eleven non-adaptive-or-weakly-adaptive modes fall within **0.55 points** of
> each other, and the trained classifier is the *worst* of the twelve, below
> doing nothing. Choosing which prefetching heuristic to apply is worth at
> most half a point; choosing how deep to prefetch is worth five.

Then give the two things that make it honest:

1. It costs a great deal to get. `deep` issues 11.2 million prefetches across
   the sweep, **5.86 million of which are never read**, at 4952 us/read
   against `none`'s 4105. The hit-ratio win is real; that it is *worth having*
   is not supported by the traces.
2. An earlier conclusion of this project -- that "no routing decision can
   help" -- has been **retracted**. It was an artefact of every mode
   prefetching 2 blocks (D13, D16).

## Prepared live demonstration

1. State the result above, including the retraction. Two minutes, and it
   frames everything that follows.
2. Run `python main.py` (accept defaults, ~6 min) and point at the
   `COST MODEL` block: **break-even precision 1.0101**, printed *before* any
   cost table, and the reason the cost column is degenerate.
3. Point at `WHAT WON, AND WHAT IT MEANS` in the same run -- the report
   retracts the old claim in its own text, which is the honest thing for it
   to do.
4. Run `python sweep_modes.py --per-trace 3000` (~2 min) for the 12-mode,
   32-trace collection view, and show that `deep` wins on hit ratio while
   `none` wins on modelled cost. Report both, never one alone.
5. Identify separate training and held-out generated test windows. Read one
   confusion-matrix row: 155/160 correct, 2 errors, both `strided` predicted
   `sequential`.
6. Run `python -m unittest discover -s tests -v` (131 tests, ~15 s) and point
   at `test_every_supported_mode_agrees_across_both_paths` and
   `test_gate_records_usefulness_deltas_not_cumulative_totals`. Explain that
   the second one exists because the cumulative-delta bug produced a
   **fake +12.3-point "win"** that had to be retracted. Bug regressions are
   evidence of method, not a distraction.
7. Run `python -m adaptive_prefetch replay
   data/samples/msr-cambridge1-sample.csv` and state that its provenance is
   unverified.
8. If asked to show the gate reacting: `--prefetch-multiple 0.5` opens it.
   At the default the break-even is above 100%, so a correct gate *stays
   shut*; that is the model working, not the model failing.

## Suggested slide sequence

1. Title and objective.
2. **Result first**: fixed depth-8 read-ahead wins 29/31; the classifier is
   the worst mode; here is the retraction.
3. Problem: one fixed prefetcher does not suit every access pattern.
4. Architecture and request-to-decision timing.
5. Input schema and four generated workload types.
6. Features and incremental classifier, with the 96.88% held-out number and
   its scope stated on the slide.
7. Class-to-policy mapping, the feedback gate, and the LRU cache model.
8. The depth experiment: +4.10 points from depth alone versus 0.55 points
   across all eleven other modes.
9. Cost model and why the cost column is degenerate (break-even 1.0101).
10. Implemented modules, the 131-test suite, and live demonstration.
11. Limitations, the honest caveats, and what would come next (per-window
    depth selection, not a better classifier).

## Discussion preparation

- **Why classify a window?** One request alone has no access pattern. Multiple
  recent requests reveal deltas, locality, and timing.
- **Why is the model ML rather than a rule table?** It estimates class-conditional
  feature distributions from labelled examples, predicts unseen feature vectors,
  and updates its statistics one labelled window at a time.
- **Why not QDA, which scores 86.77% on `transition_heavy`?** Because it loses
  21 points under the covariate shift this project actually faces
  (synthetic training windows to real traces, block sizes 1 to 128). Parsimony
  wins on the risk that occurs. See `docs/CLASSIFIER_EVAL.md` section 1.
- **Where do online labels come from?** Generated workloads provide them. Real
  traces need an independent labelling method; cache success is not a label.
- **Your classifier is 96.88% accurate and the worst mode. Why is that not a
  contradiction?** Accuracy on a four-class window label is not the same task as
  predicting the next address. These traces are ~49% backward seeks with reuse
  distances far exceeding the cache, so the next address genuinely is not
  recoverable from request history. The classifier is doing its job; the job
  is not worth much. The honest conclusion is that the bottleneck is the task,
  not the model.
- **So was the ML wrong?** The classification result stands. What was wrong was
  the *comparison*: every baseline it lost to prefetched 2 blocks.
- **Why can a fixed heuristic beat an adaptive one?** It buys hit ratio with
  speculative I/O. `deep` issues 4x the prefetches of `sequential` and wastes
  52% of them. The cost model says that trade does not pay.
- **What does the cost column mean?** At the default `prefetch_multiple=1.0` a
  speculative read does the same device work as the demand read it replaces and
  saves strictly less, so break-even precision is 1.0101 and no prefetcher can
  reach it. `none` wins that column by arithmetic. Only `--prefetch-multiple`
  below 1.0 makes prefetching feasible, and that models overlapped prefetch --
  an assumption these traces cannot support, since they record demand-read
  service time only.
- **What is the feedback gate for, if it only wins on one trace?** That is the
  point. It wins on `web_3`, the one trace where read-ahead actively destroys
  the cache: +18.99 points of hit ratio over `sequential` with 86.9% less
  wasted I/O. It loses on the other 28. A targeted result, presented as one.
- **Is break-even precision sufficient for prefetching to pay?** No. On `ts_0`
  the gate opens on its own evidence (margin +0.068) and still scores 0.32
  points *below* `sequential`. Necessary, not sufficient.
- **Why use simulation?** It tests the classify-then-prefetch logic and exposes
  cache tradeoffs without kernel modification or privileged I/O hooks.
- **Does this prove the expected wins?** No, and the deck should not imply it.
  The benchmark makes the hypotheses testable; a win on every workload is not
  guaranteed and was not achieved. The LSTM predicts the next address delta,
  not a workload class, so compare end-to-end cache metrics and compute cost,
  not classification accuracies.
- **What remains?** Per-window read-ahead *depth* selection -- the dimension
  that turned out to matter, and which nothing here adapts yet. Then: a
  metadata-verified full public trace, training the LSTM artifact, more cache
  capacities, and validating the latency model against real device
  observations.

## Verification record

Run the commands in README immediately before the review and paste the actual
results into slides. Do not use the expected outcomes from Review 1 as if they
were measurements. Both existing decks predate the 12-mode benchmark, the
measured cost model, the 2048-block cache and the depth result; they are
marked stale in `presentation/README.md` and must be rebuilt from `outputs/`
rather than hand-edited.

# Implementation plan

> **Status on `master`.** The `pivot` branch's executed implementation
> plan, kept as written. Current web decisions: `docs/web/WEB_DECISIONS.md`.

<!-- doc-numbers: off (the pivot branch's executed plan) -->

## Objective and evidence standard

Use actual RetailRocket product-view sequences to learn when to prefetch likely next product pages. The adaptive policy must compete with the best fixed depth on a time-held-out test interval. A positive result is not assumed. The dataset has no page sizes or network timing, so outcomes are cache-object hits and speculative-object costs, not measured page-load latency.

## Stage gates

1. **Acquire and validate.** Download Kaggle dataset `retailrocket/ecommerce-dataset`, version 2, file `events.csv`. Verify and record the extracted CSV SHA-256. Validate exact schema, integer IDs/timestamps, known event types, empty transaction IDs only where expected, ordering, duplicate count, and documented row count. Fail on malformed rows; never replace the dataset with generated events.
2. **Preprocess.** Sort by visitor and time, preserve original order for ties, create sessions after 30 minutes of inactivity, then retain `view` events as demand requests. Create each view's previous item, next item within session, gap from previous view, and session position. Rows whose next view falls across a split boundary are excluded from prediction scoring.
3. **EDA.** Report event and view counts, date range, session-length quantiles, repeated-item rate, transition concentration, and usable next-view pairs. Save a machine-readable summary and a readable report. If too few multi-view sessions survive, reconsider the dataset before training.
4. **Chronological partitions.** Partition by time into four consecutive periods: transition fitting (40%), ranking-model training (20%), validation and tuning (20%), and final test (20%). Keep timestamps and IDs; never random-split individual events. Build the candidate dictionary only from the first period. This makes ranking-model examples independent of the counts used to propose them.
5. **Candidate generation and features.** For each current item, rank historically observed next items and add a small popular-item fallback. Generate causal features: transition count and probability, global item frequency, candidate rank, recent-session repeat indicator, session position, and preceding time gap. Only current and earlier events can affect a decision.
6. **Train and tune the ranking model.** Fit scaled logistic classifiers with `C ∈ {0.1, 1, 10}` on real next-view positives and candidate negatives from the second period. Include session-ending views as all-negative decisions, so abstention is learnable. Select `C` by validation candidate log loss. Cap training and model-validation decisions with a documented, evenly spaced chronological sample for laptop-scale runs. Measure candidate coverage separately from ranking quality; the ranker cannot retrieve an item missing from its candidate pool.
7. **Tune the prefetch controller.** On validation data, compare no prefetch, fixed `K=1,2,4,8`, a confidence-threshold rule, and a probability-based adaptive policy. The adaptive policy selects `K` from `{0,1,2,4,8}` to maximize its estimated hit probability less an internal penalty per issued object. Search fixed depths, gate thresholds, and adaptive penalties on validation. Select separately for four predeclared **evaluation cost scenarios** (0, 0.01, 0.05, 0.25 hit-equivalents per issued object). These costs are assumptions, not observed economics. Measure immediate-next-item candidate coverage as a feasibility ceiling, not a deployable oracle policy.
8. **Final cache replay.** Freeze settings. Replay final-period views through an independent cache for each visitor session: browser caches are not shared across customers. Use the same capacity, speculative-first eviction rule, and candidate ranking for every policy. Record demand hits, useful prefetches, issued and unused prefetches, precision, request-normalized utility, controller depth distribution, and hit/waste tradeoffs. Use session-level paired bootstrap intervals for differences. Report capacities 2, 4, 8, 16, and 32 without selecting a best test result, plus a bounded alternate-session-gap analysis.
9. **Demonstration.** Produce `outputs/report.md`, validation/test/comparison CSVs, model files, an EDA summary, and a step-by-step walk-through of the first held-out session with at least five views in `demo.json`. Provide one command to reproduce the entire workflow. By default, replay the full validation and test periods. Historical 50,000/100,000 chronological prefixes remain a comparison diagnostic; explicit shorter view limits retain only complete session segments.

## Acceptance checks

- A downloaded real `events.csv` is the only demand stream.
- Each score uses only information available before its target view.
- Fixed policies share candidates, cache size, request order, and prefetch accounting with adaptive policies.
- Tuning never reads the final test period.
- A win is claimed only for a named cost scenario and capacity if test utility improves over the validation-selected fixed depth, with uncertainty stated. Otherwise report the failed comparison.

## Audit extensions

The audit required ranker-free top-1 transitions, context ablation, histogram gradient boosting as a diagnostic, calibration reliability/ECE, full-split replay, item cold-start/time cohorts, cache and session-gap sensitivity, and a validation break-even threshold for cost per unused prefetch. The deployed model and policy choices remain validation-only. The simulated object-cache result is never labeled a measured page-load improvement. See `audit/GRILL.md` for evidence and status.

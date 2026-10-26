# Real-data product-page prefetch benchmark

RetailRocket Kaggle events, version 2; see the dataset link in the README.
Raw events: 2,756,101; view requests: 2,664,218; eligible next-view pairs: 908,355.
Raw CSV SHA-256: `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7`.
Ranker trained on 30,000 real decisions; candidate coverage: 16.134%; selected logistic C: 10.0.
Validation next-view ranking: coverage@8 9.339%, hit@1 3.889%, MRR@8 0.0558.
Controller tuning: 532,843 validation views; final replay: 532,844 untouched test views.
Each session has an independent browser object cache; capacities are 2, 4, 8, 16, and 32 objects.

## Frozen-policy comparisons

The `cost` values are hypothetical useful-hit equivalents per issued object; the trace contains neither page bytes nor latency. A positive delta means the adaptive policy beats the validation-selected fixed policy under that assumption.

| Cache | Cost | Fixed | Adaptive | Hit-rate delta | Coverage@8 ceiling | Prefetches/view (fixed to adaptive) | Utility delta | 95% paired CI |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 2 | 0 | fixed4 | adaptive_0.002 | +0.6189% | 9.202% | 3.969 to 3.242 | +0.00619 | [+0.00590, +0.00651] |
| 2 | 0.01 | fixed1 | adaptive_0.01 | +0.0302% | 9.202% | 0.891 to 0.812 | +0.00109 | [+0.00066, +0.00154] |
| 2 | 0.05 | fixed0 | adaptive_0.05 | +0.0210% | 9.202% | 0.000 to 0.026 | -0.00111 | [-0.00126, -0.00097] |
| 2 | 0.25 | fixed0 | adaptive_0.2 | +0.0218% | 9.202% | 0.000 to 0.001 | +0.00002 | [-0.00002, +0.00008] |
| 4 | 0 | fixed4 | adaptive_0.002 | +0.0092% | 9.202% | 3.949 to 3.215 | +0.00009 | [-0.00019, +0.00036] |
| 4 | 0.01 | fixed1 | adaptive_0.01 | +0.0227% | 9.202% | 0.856 to 0.779 | +0.00099 | [+0.00074, +0.00124] |
| 4 | 0.05 | fixed0 | adaptive_0.05 | +0.1391% | 9.202% | 0.000 to 0.024 | +0.00017 | [+0.00003, +0.00031] |
| 4 | 0.25 | fixed0 | adaptive_0.2 | +0.0236% | 9.202% | 0.000 to 0.001 | +0.00005 | [+0.00001, +0.00010] |
| 8 | 0 | fixed4 | adaptive_0.002 | -0.0918% | 9.202% | 3.472 to 3.134 | -0.00092 | [-0.00121, -0.00065] |
| 8 | 0.01 | fixed1 | adaptive_0.01 | +0.2552% | 9.202% | 0.837 to 0.734 | +0.00358 | [+0.00331, +0.00385] |
| 8 | 0.05 | fixed0 | adaptive_0.05 | +0.1653% | 9.202% | 0.000 to 0.023 | +0.00049 | [+0.00036, +0.00063] |
| 8 | 0.25 | fixed0 | adaptive_0.2 | +0.0238% | 9.202% | 0.000 to 0.001 | +0.00006 | [+0.00001, +0.00011] |
| 16 | 0 | fixed4 | adaptive_0.002 | +0.4183% | 9.202% | 3.312 to 2.825 | +0.00418 | [+0.00395, +0.00440] |
| 16 | 0.01 | fixed1 | adaptive_0.01 | +0.4011% | 9.202% | 0.828 to 0.702 | +0.00526 | [+0.00500, +0.00551] |
| 16 | 0.05 | fixed0 | adaptive_0.05 | +0.1713% | 9.202% | 0.000 to 0.023 | +0.00056 | [+0.00042, +0.00070] |
| 16 | 0.25 | fixed0 | adaptive_0.2 | +0.0240% | 9.202% | 0.000 to 0.001 | +0.00006 | [+0.00002, +0.00011] |
| 32 | 0 | fixed4 | adaptive_0.002 | +0.5529% | 9.202% | 3.260 to 2.751 | +0.00553 | [+0.00530, +0.00573] |
| 32 | 0.01 | fixed1 | adaptive_0.01 | +0.4532% | 9.202% | 0.823 to 0.695 | +0.00581 | [+0.00555, +0.00606] |
| 32 | 0.05 | fixed0 | adaptive_0.05 | +0.1734% | 9.202% | 0.000 to 0.023 | +0.00058 | [+0.00044, +0.00073] |
| 32 | 0.25 | fixed0 | adaptive_0.2 | +0.0240% | 9.202% | 0.000 to 0.001 | +0.00006 | [+0.00002, +0.00011] |

## Interpretation

These are object-hit and request-count outcomes, not measured speedups. Candidate coverage bounds immediate next-item retrieval among eligible pairs; it does not bound ordinary demand-cache revisits.
No policy was selected using test data. See `validation_metrics.csv`, `test_metrics.csv`, `comparisons.csv`, and `demo.json` for the complete evidence.

## Limitations

The log lacks page size, response timing, cache headers, browser behavior, and visitor-side idle time. Equal-size objects and immediate prefetch completion are simulation assumptions. Session boundaries are heuristic. One retailer dataset does not establish cross-site generalisation or measured page-load speedups.


## Diagnostics and sensitivity

Candidate coverage@8 on full test: 9.202%; the ceiling appears beside each result in `comparisons.csv`.
Candidate-level ECE: validation 0.00071, test 0.00091; reliability bins are in `calibration.csv`. Candidate scores do not form a normalized next-item distribution. See `reliability.svg` for the plotted curve.
History-only top-1 transition cache baseline: 12.450% hits, 0.849 issued/view at capacity 8; details in `transition_baseline.csv`. Context ablation and a gradient-boosting diagnostic are in `models/training_summary.json`. Boosting is diagnostic and did not select the controller.
At capacity 8, the history-only transition baseline can outperform the logistic fixed-1 policy in raw hit rate; compare `transition_baseline.csv` with `test_metrics.csv`. This is a diagnostic, not a test-selected replacement.
`subset_vs_full.csv` compares old prefix budgets with full-period frozen policies. `break_even.csv` and `break_even_curve.csv` use cost per unused prefetch on validation. `cohorts.csv` examines cold-start and chronological retrieval coverage.

## Positive, negative, and uncertain results

- Cache 2, cost 0: adaptive wins; utility delta +0.00619, 95% CI [+0.00590, +0.00651]; versus no prefetch -0.01714.
- Cache 2, cost 0.01: adaptive wins; utility delta +0.00109, 95% CI [+0.00066, +0.00154]; versus no prefetch -0.02056.
- Cache 2, cost 0.05: adaptive loses; utility delta -0.00111, 95% CI [-0.00126, -0.00097]; versus no prefetch -0.00111.
- Cache 2, cost 0.25: adaptive is within paired-bootstrap noise; utility delta +0.00002, 95% CI [-0.00002, +0.00008]; versus no prefetch +0.00002.
- Cache 4, cost 0: adaptive is within paired-bootstrap noise; utility delta +0.00009, 95% CI [-0.00019, +0.00036]; versus no prefetch +0.00542.
- Cache 4, cost 0.01: adaptive wins; utility delta +0.00099, 95% CI [+0.00074, +0.00124]; versus no prefetch -0.00071.
- Cache 4, cost 0.05: adaptive wins; utility delta +0.00017, 95% CI [+0.00003, +0.00031]; versus no prefetch +0.00017.
- Cache 4, cost 0.25: adaptive wins; utility delta +0.00005, 95% CI [+0.00001, +0.00010]; versus no prefetch +0.00005.
- Cache 8, cost 0: adaptive loses; utility delta -0.00092, 95% CI [-0.00121, -0.00065]; versus no prefetch +0.01628.
- Cache 8, cost 0.01: adaptive wins; utility delta +0.00358, 95% CI [+0.00331, +0.00385]; versus no prefetch +0.00444.
- Cache 8, cost 0.05: adaptive wins; utility delta +0.00049, 95% CI [+0.00036, +0.00063]; versus no prefetch +0.00049.
- Cache 8, cost 0.25: adaptive wins; utility delta +0.00006, 95% CI [+0.00001, +0.00011]; versus no prefetch +0.00006.
- Cache 16, cost 0: adaptive wins; utility delta +0.00418, 95% CI [+0.00395, +0.00440]; versus no prefetch +0.02524.
- Cache 16, cost 0.01: adaptive wins; utility delta +0.00526, 95% CI [+0.00500, +0.00551]; versus no prefetch +0.00689.
- Cache 16, cost 0.05: adaptive wins; utility delta +0.00056, 95% CI [+0.00042, +0.00070]; versus no prefetch +0.00056.
- Cache 16, cost 0.25: adaptive wins; utility delta +0.00006, 95% CI [+0.00002, +0.00011]; versus no prefetch +0.00006.
- Cache 32, cost 0: adaptive wins; utility delta +0.00553, 95% CI [+0.00530, +0.00573]; versus no prefetch +0.02804.
- Cache 32, cost 0.01: adaptive wins; utility delta +0.00581, 95% CI [+0.00555, +0.00606]; versus no prefetch +0.00768.
- Cache 32, cost 0.05: adaptive wins; utility delta +0.00058, 95% CI [+0.00044, +0.00073]; versus no prefetch +0.00058.
- Cache 32, cost 0.25: adaptive wins; utility delta +0.00006, 95% CI [+0.00002, +0.00011]; versus no prefetch +0.00006.
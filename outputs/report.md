# Real-data product-page prefetch benchmark

RetailRocket Kaggle events, version 2; see the dataset link in the README.
Raw events: 2,756,101; view requests: 2,664,218; eligible next-view pairs: 908,355.
Raw CSV SHA-256: `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7`.
Ranker trained on 30,000 real decisions; candidate coverage: 16.134%; selected logistic C: 10.0.
Validation next-view ranking: coverage@8 9.339%, hit@1 3.889%, MRR@8 0.0558.
Controller tuning: first 50,000 validation views; final replay: first 100,000 untouched test views.
Each session has an independent browser object cache; capacities are 4, 8, and 16 objects.

## Frozen-policy comparisons

The `cost` values are hypothetical useful-hit equivalents per issued object; the trace contains neither page bytes nor latency. A positive delta means the adaptive policy beats the validation-selected fixed policy under that assumption.

| Cache | Cost | Fixed | Adaptive | Hit-rate delta | Prefetches/view (fixed to adaptive) | Utility delta | 95% paired CI |
|---:|---:|---|---|---:|---:|---:|---:|
| 4 | 0 | fixed4 | adaptive_0.002 | -0.0190% | 3.945 to 3.356 | -0.00019 | [-0.00084, +0.00042] |
| 4 | 0.01 | fixed1 | adaptive_0.01 | +0.0380% | 0.852 to 0.806 | +0.00084 | [+0.00025, +0.00143] |
| 4 | 0.05 | fixed0 | adaptive_0.05 | +0.1480% | 0.000 to 0.025 | +0.00025 | [-0.00008, +0.00065] |
| 4 | 0.25 | fixed0 | adaptive_0.2 | +0.0280% | 0.000 to 0.001 | +0.00010 | [-0.00001, +0.00023] |
| 8 | 0 | fixed4 | adaptive_0.002 | -0.1170% | 3.465 to 3.268 | -0.00117 | [-0.00181, -0.00053] |
| 8 | 0.01 | fixed1 | adaptive_0.01 | +0.2700% | 0.835 to 0.757 | +0.00348 | [+0.00290, +0.00404] |
| 8 | 0.05 | fixed0 | adaptive_0.05 | +0.1840% | 0.000 to 0.023 | +0.00067 | [+0.00033, +0.00106] |
| 8 | 0.25 | fixed0 | adaptive_0.2 | +0.0280% | 0.000 to 0.001 | +0.00010 | [-0.00001, +0.00024] |
| 16 | 0 | fixed4 | adaptive_0.002 | +0.4320% | 3.316 to 2.928 | +0.00432 | [+0.00379, +0.00482] |
| 16 | 0.01 | fixed1 | adaptive_0.01 | +0.4440% | 0.827 to 0.723 | +0.00548 | [+0.00490, +0.00612] |
| 16 | 0.05 | fixed0 | adaptive_0.05 | +0.1880% | 0.000 to 0.023 | +0.00072 | [+0.00039, +0.00111] |
| 16 | 0.25 | fixed0 | adaptive_0.2 | +0.0280% | 0.000 to 0.001 | +0.00010 | [-0.00001, +0.00024] |

## Interpretation

These are object-hit and request-count outcomes, not measured speedups. Candidate coverage is a strict upper bound on next-page prefetch accuracy; ineligible requests still participate in cache replay.
No policy was selected using test data. See `validation_metrics.csv`, `test_metrics.csv`, `comparisons.csv`, and `demo.json` for the complete evidence.

## Limitations

The log lacks page size, response timing, cache headers, browser behavior, and visitor-side idle time. Equal-size objects and immediate prefetch completion are simulation assumptions. Session boundaries are heuristic, and only the first N time-ordered requests of each holdout split are replayed by default.

# Web ranker alternatives (track WRK)

What the track asked: does another model family, tuned on validation, rank
the same next-page candidates better than the shipped logistic ranker, and
does that move the simulated cache metrics? Design, search spaces and
decision rules: `results/web_rankers/PREREGISTERED.md` (written after E1-E6,
committed before the run). Methods and citations:
`results/web_rankers/README.md`. Code: `src/webprefetch/rankers.py`.
Reproduce: `python reproduce.py web_rankers`.

<!-- source: results/web_rankers/summary.json -->
Every hit ratio and utility below is **simulated**; the prefetch cost `c`
and object sizes are **assumed**; timings are wall times on a **contended**
host and are never costs. All intervals are paired session-bootstrap percentile intervals
(10,000 resamples, seed 42) over the test split's sessions.

<!-- source: none -->

## Protocol in one paragraph

Five families (`logistic_tuned`, `hgb`, `rf`, `et`, `mlp`) are trained on
the same evenly spaced split-1 sample and see the shipped candidates and
nine features. Each family's hyperparameters come from a random search
scored by MRR@8 on a split-2 sample; each scorer's prefetch policy is then
chosen on the whole of split 2 per capacity, exactly as in E4; split 3 is
replayed once. Before any comparison the regenerated shipped ranker
reproduced E4's committed test row (`logistic`, cap 8, `fixed1`) exactly;
`summary.json` records the check (`e4_check.match`).

## Results

### Primary comparison (S1)

<!-- source: results/web_rankers/comparisons.csv comparison=S1 -->
The validation-selected scorer was `mlp` at both capacities. Against the
shipped logistic ranker, each with its own validated adaptive policy:

| capacity | metric | estimate | interval | verdict |
|---|---|---|---|---|
| 8 | U(0.01) per view | +0.00154 | [+0.00133, +0.00175] | better |
| 8 | hit ratio | -0.00163 | [-0.00184, -0.00143] | worse |
| 32 | U(0.01) per view | +0.00147 | [+0.00122, +0.00171] | better |
| 32 | hit ratio | +0.00569 | [+0.00544, +0.00594] | better |

At capacity 8 the gain in utility comes with a lower hit ratio: the
validated `mlp` controller (`adaptive_0.02`) issues fewer prefetches than
the logistic one (`adaptive_0.01`). By decision rule 2 the paper may say
the validated MLP ranker improves the controller's U(0.01) at both
capacities; it may not say it raises the hit ratio at capacity 8.

### Ranking quality (K1, secondary)

<!-- source: results/web_rankers/comparisons.csv comparison=K1 -->
Test next views, family minus shipped logistic:

| family | hit@1 | interval | MRR@8 | interval |
|---|---|---|---|---|
| `logistic_tuned` | +0.00001 | [-0.00019, +0.00020] | +0.00009 | [-0.00003, +0.00020] |
| `hgb` | +0.00134 | [+0.00084, +0.00184] | +0.00200 | [+0.00166, +0.00233] |
| `rf` | +0.00148 | [+0.00096, +0.00200] | +0.00214 | [+0.00178, +0.00248] |
| `et` | +0.00154 | [+0.00101, +0.00206] | +0.00212 | [+0.00178, +0.00248] |
| `mlp` | +0.00163 | [+0.00114, +0.00211] | +0.00234 | [+0.00202, +0.00267] |

By decision rule 1, `hgb`, `rf`, `et` and `mlp` are better rankers than
the shipped one (hit@1 and MRR@8 both "better"); `logistic_tuned` shows no
difference detected on either. The online-count reference row is far
larger: `trans_online` minus logistic is +0.04145 on MRR@8, interval
[+0.04023, +0.04269].

### Downstream replay (K2-K5, secondary)

<!-- source: results/web_rankers/comparisons.csv comparison=K3 metric=utility_per_view -->
Each family's validated adaptive policy minus the shipped logistic's, U(0.01):

| family | cap 8 | interval | cap 32 | interval |
|---|---|---|---|---|
| `logistic_tuned` | -0.00081 | [-0.00107, -0.00055] | -0.00312 | [-0.00339, -0.00287] |
| `hgb` | +0.00056 | [+0.00032, +0.00081] | +0.00157 | [+0.00132, +0.00181] |
| `rf` | +0.00099 | [+0.00076, +0.00123] | +0.00128 | [+0.00105, +0.00152] |
| `et` | +0.00122 | [+0.00098, +0.00147] | +0.00153 | [+0.00130, +0.00176] |
| `mlp` | +0.00154 | [+0.00133, +0.00175] | +0.00147 | [+0.00122, +0.00171] |

`hgb`, `rf`, `et` and `mlp` are "better" for prefetching at both
capacities. `logistic_tuned` is "worse": its search chose class-balanced
weights (highest tuning MRR@8, see `tuning.csv`), whose probabilities are
not calibrated, and the adaptive controller reads probabilities.

<!-- source: results/web_rankers/comparisons.csv comparison=K2 metric=hit_ratio -->
At fixed depth 1 every family's hit ratio is "better" than the shipped
ranker's at both capacities (K2), by +0.00054 to +0.00067 for the four
non-linear families.

<!-- source: results/web_rankers/comparisons.csv comparison=K4 -->
None of them beats the history-only transition top-1 at depth 1 (K4):
every hit-ratio row is "worse" or "no difference detected", for example
`mlp` at capacity 8, -0.00020, interval [-0.00037, -0.00003]. The trained
rankers learn from labels that include self-repeats and back-navigation,
which a session cache already holds; on novel next views the transition
order is the better top-1 (`ranking_metrics.csv`).

<!-- source: results/web_rankers/comparisons.csv comparison=K5 metric=hit_ratio -->
Every static ranker stays far below causal online counts at depth 4 (K5):
the hit-ratio gap is between -0.01812 and -0.01406 for every scorer at
both capacities, so a better split-0 ranker does not close the freshness
gap (decision rule 3).

### Controller rule (K6)

<!-- source: results/web_rankers/comparisons.csv comparison=K6 -->
For every scorer and capacity the validated adaptive policy beats both its
validated best fixed depth and no prefetching on U(0.01), so no row is
labelled "loses to no prefetch".

### Cost (contended, never a cost claim)

<!-- source: results/web_rankers/ranking_metrics.csv split=test -->
One-request scoring wall time on the test split (features, predict, sort):
shipped logistic `FastScorer` 27.3 us, `mlp` 222.6 us, `hgb` 6211.4 us,
`rf` 6932.3 us, `et` 7175.9 us, at a 1-minute load average between 40.57
and 47.32. Single-thread refit of the selected configuration: `mlp`
90.3 s, `hgb` 105.8 s, `rf` 477.1 s, `et` 839.2 s.

<!-- source: none -->
## What this does and does not show

- The shipped logistic ranker was not the best available ranker over its
  own features: four tuned non-linear families rank better and give the
  adaptive controller a higher simulated `U(0.01)` at the two capacities replayed.
- The gains are small next to the freshness lever (online counts) and do
  not beat transition top-1 at depth 1. One candidate pool, one feature
  set, two capacities and one cost; nothing generalises beyond them.
- No multiple-comparison correction is applied; S1 is the only primary
  comparison.
- The fully grown forest configurations were excluded from the search for
  time (pre-registration, search spaces), and LambdaMART-style listwise
  ranking was not run (no LightGBM or XGBoost here).

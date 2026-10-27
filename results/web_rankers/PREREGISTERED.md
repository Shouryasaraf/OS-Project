# Web ranker alternatives (track WRK): pre-registration

Written on 2026-10-27, **after** E1-E6 were run and their results known
(in particular E4's W3: history-only transition top-1 beats the shipped
logistic ranker at depth 1, and W4: online counts beat it by far more). It is
committed before the full run of this track, so the comparisons, search
spaces and decision rules below cannot be chosen after its results are seen.

What was seen before writing it, in bounded pilots that write only
`outputs/` and are never cited: single configurations of each family scored
on the split-2 tuning sample (orientation only; hgb, mlp, rf and et
configurations came out slightly above the shipped ranker's MRR@8 there,
logistic variants level with it), fit times (fully grown forests took over
ten minutes per fit, so their depth and leaf size are bounded below), and
a one-trial-per-family run of the whole pipeline to size the replays. No
test-split figure of any new family was looked at before this file was
committed.

## Question

Does any other model family, tuned on validation, rank the same next-page
candidates better than the shipped logistic ranker, and does a better
ranking move the replayed browser-cache metrics? The ranker is the
"predictor" lever of E4; the disk study's analogue is the GNB router.

## Data and protocol (GRILL rules, unchanged from E4)

- RetailRocket views, SHA-pinned, 30-minute sessions, four chronological
  splits; E4's `prepare`/`train` outputs regenerated in this worktree
  (`data/processed/views.csv.gz`, `models/web/`). Before any comparison the
  shipped logistic ranker must reproduce E4's test row `logistic`, cap 8,
  `fixed1`: 65,944 hits and 445,796 issued prefetches over 532,844 views.
  If it does not, the run stops and this is reported. (Checked before this
  file was committed with a one-policy replay: 65,944 hits, 445,796 issued,
  532,844 views, an exact match; the full run checks it again.)
- Candidates: the split-0 `CandidateIndex` (up to 20 transitions plus 8
  popular items), the same for every family. Features: the shipped nine
  (`webprefetch.model.FEATURE_NAMES`). Label: candidate is the next view.
- **Training data (split 1):** 100,000 evenly spaced split-1 decisions
  (`np.linspace`, as `train_ranker`), the same rows for every new family.
  The shipped ranker keeps its own 30,000-decision sample (E4's learning
  curve shows logistic flat from 30k to 300k); `logistic_tuned` is the
  same-data control.
- **Hyperparameter selection (split 2 sample):** 100,000 evenly spaced
  split-2 decisions. Per family, random search (Bergstra and Bengio 2012)
  with `sklearn.model_selection.ParameterSampler`, seed 42. The trial with
  the highest **MRR@8** over eligible next views wins; ties go to lower
  candidate log loss, then the earlier trial. The winning trial is refit
  with the same seed on one thread, as it was searched, so the replayed
  model is the model the search scored.
- **Policy selection (whole split 2):** for every scorer, E4's rule
  (`benchmark.select_policies`): per capacity and cost, per family (fixed
  depth, adaptive, adaptive_ca, gate), highest validation `U(c)`, ties to
  fewer issued prefetches, then the name. The feedback gate is not
  replayed (it is not a ranker comparison and needs a global-order replay
  per policy). Capacities 8 and 32 (E4's W1/W2 capacities).
- **Test (split 3):** replayed once with every choice frozen.

## Families and search spaces

| family | estimator (scikit-learn 1.9.1) | space | trials |
|---|---|---|---|
| `logistic` (baseline) | shipped `ranker.joblib`, StandardScaler + L2 logistic, C=10 | E4's C grid | - |
| `logistic_tuned` | StandardScaler + LogisticRegression | C log-uniform [1e-3, 1e2]; l1_ratio {0 (L2), 1 (L1)}; class_weight {None, balanced} | 16 |
| `hgb` | HistGradientBoostingClassifier, no early stopping | learning_rate log-uniform [0.01, 0.3]; max_iter {100, 200, 400}; max_leaf_nodes {7, 15, 31, 63}; min_samples_leaf {20, 50, 100, 200}; l2_regularization log-uniform [1e-4, 10] | 24 |
| `rf` | RandomForestClassifier | n_estimators {100, 200}; max_depth {8, 12, 16, 20}; min_samples_leaf {5, 20, 50, 100}; max_features {sqrt, 0.5}; max_samples {0.25, 0.5} | 12 |
| `et` | ExtraTreesClassifier | n_estimators {100, 200}; max_depth {8, 12, 16, 20}; min_samples_leaf {5, 20, 50, 100}; max_features {sqrt, 0.5} | 12 |
| `mlp` | StandardScaler + MLPClassifier (Adam), early stopping on 10% of the split-1 rows, max_iter 60 | hidden {(32), (64), (64, 32), (128, 64)}; alpha log-uniform [1e-5, 1e-2]; learning_rate_init log-uniform [1e-4, 1e-2]; batch_size {256, 1024} | 12 |

Every estimator is seeded (`random_state=42`). History-only baselines:
`trans_s0` (transition top-1 on split-0 counts) and `trans_online` (causal
online counts), exactly as E4 builds them.

## Statistics

- Unit: the test session (364,364 sessions in E4). Paired session
  bootstrap, **10,000 resamples, seed 42**, ratio estimator
  `sum(delta) / sum(views)`, 95% percentile interval
  (`benchmark.PairedSessions`, the draws of
  `prefetch_common.stats.session_bootstrap`). E4 used 500 resamples; more
  resamples only make the percentile interval more precise. Ranking metrics
  use the same draws over the sessions with at least one eligible next view
  (`rankers.RatioBootstrap`), denominator = eligible (or novel eligible)
  next views.
- Verdict for A - B: **better** if the interval's lower end is above 0,
  **worse** if its upper end is below 0, otherwise **no difference
  detected**. No other words are used.
- No multiple-comparison correction. S1 is the primary comparison; the K
  rows are secondary and are quoted with that label. With five families,
  several capacities and two metrics, some secondary intervals excluding 0
  by chance are expected.

## Comparisons (all on test)

| id | A - B | where | metric |
|---|---|---|---|
| S1 (primary) | the validation-selected scorer's validated adaptive - the shipped logistic's validated adaptive | caps 8 and 32, c = 0.01 | U(0.01), hit ratio |
| K1 | family F ranking - logistic ranking, every test next view | - | hit@1, coverage@8, MRR@8, novel hit@1, novel coverage@8 |
| K2 | F `fixed1` - logistic `fixed1` | caps 8, 32 | hit ratio, U(0.01) |
| K3 | F validated adaptive - logistic validated adaptive | caps 8, 32, c = 0.01 | hit ratio, U(0.01) |
| K4 | F `fixed1` - transition top-1 (`trans_s0` `fixed1`) | caps 8, 32 | hit ratio, U(0.01) |
| K5 | F `fixed4` - online counts (`trans_online` `fixed4`) | caps 8, 32 | hit ratio, U(0.01) |
| K6 | F validated adaptive - F validated best fixed depth, and - F `fixed0` | caps 8, 32, c = 0.01 | U(0.01) |

The S1 scorer at each capacity is chosen on **validation**: among
`logistic` and the five families, the highest validation U(0.01) of the
scorer's validated adaptive policy (ties: fewer issued, then name). If it
is `logistic`, S1 is written as "no alternative selected" with no interval.
K rows for `logistic` itself (K4-K6) are references that reproduce E4's
W1-W4 direction with more resamples; they are not new findings.

## Decision rules

1. **Better ranker.** A family is called a better ranker than the shipped
   one only if its K1 hit@1 **and** MRR@8 rows are both "better". It is
   called better *for prefetching* at a capacity only if its K3 U(0.01) row
   is "better" there. Neither is inferred from the other.
2. **Primary claim.** The paper may say "a tuned alternative ranker
   improves the controller" at a capacity only if S1 there is "better".
3. **Against history-only baselines.** A family is said to beat
   transition top-1 only by its K4 rows and to close the freshness gap
   only if its K5 U(0.01) row is not "worse"; a static ranker that stays
   below online counts is reported as such.
4. **Controller rule (E4's rule 1).** A K6 row whose interval against
   `fixed0` lies below 0 is labelled "loses to no prefetch".
5. **No generalisation** from one capacity or one cost to another, or from
   this candidate pool and feature set to rankers in general. Hit ratios
   are simulated; costs and object sizes are assumed; nothing here is a
   latency.
6. **Costs.** Training and inference wall times are on a contended shared
   host, recorded with the 1-minute load average, and never quoted as
   costs.

## Output files

`tuning.csv`, `ranking_metrics.csv`, `policy_metrics.csv`,
`comparisons.csv`, `summary.json`, `provenance.json`, all under
`results/web_rankers/`, written by `python reproduce.py web_rankers`.

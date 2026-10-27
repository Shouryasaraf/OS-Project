# Hindsight router (track HSR): pre-registered comparisons and decision rules

**Written on 2026-10-27, after E1-E6 were run and their results known**
(in particular E1's finding that `adaptive`, the GNB-routed mode, is worse
than `sequential` at both capacities, `results/disk/`). It is therefore not
blind to the disk study's outcome. It is committed before slice 2 (the test
slice) of this experiment is replayed by any router and before any model is
trained or selected. Slice-1 fixed-action replays (training-label generation,
no model involved) were started a few minutes before this commit. It states
no expected outcome. Any later change is added below as a dated amendment;
the original text stays.

Code: `src/adaptive_prefetch/hindsight.py` (labels, routers),
`hindsight_models.py` (families, search, CV), `hindsight_experiment.py` and
`hindsight_pipeline.py` (driver), run by `python reproduce.py hindsight`.

## 1. Question

The shipped `adaptive` mode maps a synthetic-trained classifier's workload
class to candidate sets of at most two blocks, so even a perfect classifier
is capped near `sequential` (D11). Does a router that (a) is trained on
**real-trace hindsight labels** and (b) chooses among an **expanded action
space that includes read-ahead depth** beat the strongest fair fixed
baseline, a validation-chosen fixed action?

## 2. Data, slices, simulation

- Traces: the 21 `corpora.PAPER_SUBSET` traces, canonical requests
  (`merge="auto"`), libCacheSim oracleGeneral (writes replay as reads).
- **Slice 1 (training/selection):** the first 1,000,000 canonical requests
  (C1's slice). **Slice 2 (test):** skip 1,000,000, next 1,000,000 (C3's
  slice 2), scored once. A trace without a full slice 2 is an `n/a` row
  (E1 found two such traces), never a shorter slice.
- Capacities 2048 and 131072 blocks (1 MiB, 64 MiB). Window 32 requests.
- Cost model `LatencyModel(5, 100, 50)` for every row, as in the multicorpus
  sweep. Everything is **simulated**; hit ratios are LRU block hit ratios,
  `device_ios`/`blocks_moved` are modelled device work.
- No scale-down of requests is planned at 131072; if time forces one it is
  added here as an amendment before the run it affects.

## 3. Router, actions, labels

- **Actions** A = {`none`, `sequential` (the shipped 2-block policy),
  `deep@8`, `deep@32`, `deep@256`}, in that cost order.
- **Router (`routed` mode):** after each complete window k the router reads
  the window's 12 `FEATURE_NAMES` (`features.WindowContext`, as `adaptive`)
  and chooses the action for window k+1. The first window never prefetches.
  Inference is batch (the features are causal and cache-independent); a test
  pins that the in-loop router gives bit-identical counters.
- **Labels (trajectory approximation):** every fixed action is replayed over
  the slice; window k's label is the action with the best objective on window
  k+1 in its own fixed-action trajectory. Ties go to the cheaper action. This
  is an approximation of the true counterfactual (the cache state is the
  always-a state, not the router's), stated wherever labels are used.
- **Objectives** (per window, higher is better): `hit` = hits (U(c) at c=0);
  `u0.5` = hits - 0.5 * prefetches issued (U(c) at c=0.5); `dev` = -device
  I/Os. One router family is selected per (objective, capacity).

## 4. Models, search, selection

- Families (scikit-learn 1.9): Gaussian naive Bayes (`gnb`), logistic
  regression, CART decision tree, random forest, extra trees,
  histogram gradient boosting (`hgb`), MLP (Adam), k-nearest neighbours.
- **Search space** (`hindsight_models.sample_params`): random search, 8
  configurations per family, seed `"42:<family>"`. gnb var_smoothing
  log-U[1e-12, 1e-1]; logreg C log-U[1e-3, 1e2], class_weight {None,
  balanced}; tree max_depth {3,4,6,8,12,16,20,None}, min_samples_leaf
  {1,5,20,50,100,200}, criterion {gini, entropy}, class_weight; forest and
  extra trees n_estimators {100,200}, max_depth {8,12,16,None},
  min_samples_leaf {1,5,20,50}, max_features {sqrt,0.5,1.0}, class_weight;
  hgb learning_rate log-U[0.02,0.3], max_leaf_nodes {15,31,63,127},
  max_depth {None,4,8}, l2 {0,0.1,1}, max_iter {100,200,400}, class_weight;
  mlp hidden {(32),(64),(64,32),(128,64)}, alpha log-U[1e-5,1e-1],
  learning_rate_init log-U[1e-4,1e-2], early stopping; knn k
  {1,3,5,11,21,51,101}, weights {uniform, distance}. Families that accept
  sample weights also draw `weighting` in {none, gap} (gap = best minus worst
  action's objective on the target window, normalised per trace).
- Training set: a uniform random subsample of 4,000 decision points per
  trace (seed `"42:<trace>"`), so every trace weighs the same.
- **Selection data: slice 1 only.** Grouped 5-fold CV over traces, a twin
  family (`corpora.TWIN_GROUPS`) always in one fold. Score = mean over
  held-out traces of the share of hindsight headroom over `none` captured on
  all their windows, `(S(pred) - S(none)) / (S(oracle) - S(none))` under the
  trajectory approximation (traces without headroom are left out). Highest
  score wins; ties go to the earlier family, then config. Label accuracy is a
  diagnostic only.
- The selected configuration, and every family's best, is refitted on all 21
  slice-1 training sets and persisted with a training record (traces and
  request ranges). Scoring refuses any slice that overlaps it.
- **Leave-one-corpus-out (objective `hit`):** for each corpus, the selected
  configuration refitted on the other corpora's slice 1 and scored on this
  corpus's slice 2 (hyperparameters were chosen by CV that included this
  corpus's slice 1, which is stated with the result).

## 5. Policies scored on slice 2

- `routed_<obj>`: the selected router per objective and capacity.
- `routed_hit_gnb`: the best GNB-family router (objective `hit`), to answer
  "is the classifier family the problem".
- `routed_hit_loco`: the leave-one-corpus-out router.
- `oracle_<obj>`: the hindsight labels of slice 2 itself. A **reference
  ceiling under the trajectory approximation, never a policy**.
- Baselines (strongest fair): `none`, `sequential`, `deep@8`, `deep@32`,
  `deep@256`, `linux_ra@256`, `adaptive` (shipped GNB, seed 42),
  `adaptive_evidence`; **`vcf_<obj>`** = per trace, the fixed action with the
  best slice-1 objective on that trace; **`gvcf_<obj>`** = the single fixed
  action with the best mean slice-1 objective over the 21 traces.
- Fixed-action slice-2 rows come from this study's own replays; they must
  equal `results/disk/sweep.jsonl` C3 rows on every counter (checked for all
  of them). `linux_ra@256` and `adaptive` slice-2 rows are reused from
  `sweep.jsonl` only after one row of each is reproduced exactly here.
- Every family's best router is also scored on slice 2 under the trajectory
  proxy (no replay; descriptive only, labelled proxy).

## 6. Comparisons and decision rules

Per trace, per capacity, n = traces with a full slice 2. Metric per
objective: `hit` hit ratio; `u0.5` (hits - 0.5 * prefetches) per read block;
`dev` device I/Os (paired ratio, lower is better).

| # | policy | baseline | metric |
|---|---|---|---|
| H1 (primary) | routed_hit | vcf_hit | hit ratio |
| H2 | routed_hit | sequential | hit ratio |
| H3 | routed_u0.5 | vcf_u0.5 | u0.5 |
| H4 | routed_dev | vcf_dev | device I/Os (ratio) |
| H5 | routed_hit | adaptive | hit ratio |
| H6 | routed_hit_loco | vcf_hit | hit ratio |
| H7 | routed_<obj> | gvcf_<obj> | that objective's metric |
| S1 | routed_hit | deep@8, linux_ra@256, adaptive_evidence | hit ratio |
| S2 | routed_hit_gnb | routed_hit, vcf_hit | hit ratio |
| D | every policy | none | device I/Os and blocks moved (ratio) |

- Paired bootstrap over traces and cluster bootstrap (corpora, then traces,
  twin families down-weighted), `prefetch_common.stats`, 10,000 resamples,
  seed 42, 95% percentile intervals.
- "**better**" iff both intervals lie entirely on the better side; "**worse**"
  iff both lie on the worse side; otherwise "**no difference detected**".
  A paired-only exclusion is written "paired interval only; not robust to
  corpus clustering".
- Every policy's row reports device-I/O and blocks-moved ratios against
  `none`, precision and wasted share; a hit-ratio gain is never described as
  less device work without D.
- Headroom captured on slice 2, `(routed - vcf) / (oracle - vcf)`, is
  descriptive.

## 7. What will not be done

No test-slice information is used for any choice; no trace, capacity or
objective is dropped after seeing results; no number from a bounded or
smoke run is quoted; the oracle is never called a policy.

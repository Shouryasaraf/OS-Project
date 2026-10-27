# Classifier zoo and router benchmark (track CLF): pre-registered comparisons

**Written after experiments E1-E6 were known**, on 2026-10-27, before the
full run of this track (the bounded `--quick` smoke path had run, into
`outputs/classifier-quick/`, to test the code; no figure from it is used or
quoted). In particular E1's finding that `adaptive` (the shipped GNB router)
is **worse** than `sequential` at both capacities was known when this was
written, and that finding motivated the track. This file fixes the families,
search spaces, selection data, comparisons and decision rules before the
full run; any later change is a dated amendment at the end, and the original
text stays.

Code: `adaptive_prefetch.clfzoo` (`data`, `zoo`, `synthetic`, `router`,
`analysis`), run by `python reproduce.py classifier` (repro/classifier.py),
20 worker processes on the shared host.

## 1. Families (Part 1 and Part 2)

The shipped baselines, used exactly as shipped and never tuned:
`gnb_online` (`model.OnlineGaussianNB`, the deployed default) and
`qda_online` (`model.OnlineQDA`). Thirteen tuned scikit-learn families:
`gnb` (GaussianNB), `lda`, `qda`, `logreg` (multinomial), `knn`,
`svm_linear` and `svm_rbf` (SVC with Platt probabilities), `cart`, `rf`,
`extra_trees`, `hgb` (HistGradientBoosting), `mlp` (Adam), and `sgd_online`
(logistic SGD trained only by `partial_fit`, the online alternative). A
streaming Hoeffding tree is **not** run (no correct stdlib implementation in
the time available). Citations: `results/classifier/README.md`.

**Search spaces** (`clfzoo.zoo._sample`, `DEFAULTS`): 16 candidates per tuned
family, candidate 0 the library default, then seeded random-search draws
(seed `"42:<family>"`): GaussianNB `var_smoothing` log-uniform
[1e-12, 1]; LDA `shrinkage` in {none, auto, U[0,1]}; QDA `reg_param`
U[0,1]; logreg `C` log-uniform [1e-3, 1e3], standardisation on/off; kNN
`k` in 1..50, uniform/distance weights, standardisation on/off; linear SVM
`C` log [1e-3, 1e3], scaling on/off; RBF SVM `C` log [1e-2, 1e3], `gamma`
log [1e-4, 10], scaling on/off; CART depth in {2,3,4,6,8,12,16,none}, min
leaf 1..20; RF and extra trees 100/200/400 trees, depth {4,8,16,none},
`max_features` {sqrt, 0.5, 1.0}, min leaf 1..10; HGB learning rate log
[0.01, 0.3], 100/200/400 iterations, 7/15/31/63 leaves, L2 log [1e-4, 10];
MLP hidden {32, 64, 128, 64-32, 128-64}, `alpha` log [1e-6, 0.1], learning
rate log [1e-4, 1e-2]; SGD `alpha` log [1e-6, 0.1], 1/5/20 shuffled
streaming epochs. A candidate that cannot be fitted on some fold (for
example QDA with a singular class covariance) scores 0.

## 2. Part 1: synthetic windows

- **Vectors**: the 12 contextual `FEATURE_NAMES`, built through the
  `WindowContext` path the replay uses. Settings: `baseline` (exactly
  `benchmark.make_model`'s generator and layout), the CLASSIFIER_EVAL regimes
  `noisy`, `overlap`, `overlap2`, `realistic`, `transition`,
  `transition_heavy`, and `transfer` (`main.transfer_probe`'s block-size
  covariate-shift test: trained at 1-block requests, tested at 1, 8, 32 and
  128).
- **Seeds**: 12 pairs, train seed `1000+i`, test seed `900000+i`; 100 train
  / 40 test windows per class (transfer: 60 / 30).
- **Selection (training seeds only)**: for each setting and family, each
  candidate is scored on 12 folds; fold i fits on train seed `1000+i` and
  validates on train seed `1000+((i+1) mod 12)`. The candidate with the
  highest mean validation accuracy is selected; ties go to the lower
  candidate index. Transfer is selected at size 1 only.
- **Test (once)**: the selected candidate is refitted on train seed
  `1000+i` and scored on test seed `900000+i`. Reported per (setting,
  condition, family): accuracy and macro-F1 (mean over 12 pairs, bootstrap
  interval over pairs), summed confusion matrix, ECE (15 bins) and
  multiclass Brier, median single-window `predict` time and fit time
  (**contended host**, labelled so, never a cost claim).
- **Comparison**: each family vs `gnb_online`, accuracy difference, paired
  bootstrap over the 12 seed pairs (`stats.paired_bootstrap`, 10,000
  resamples, seed 42). Macro-F1 difference is reported beside it.
- These are **synthetic** accuracies: real traces have no workload labels.

## 3. Part 2: real-trace router benchmark

- **Deployed classifiers**: each family with its `baseline`-setting
  selection, fitted on `make_model(42)`'s own training vectors (seed 42, 100
  windows per class), the data E1's GNB was fitted on. `gnb_online` is
  exactly `make_model(42)`.
- **Replay**: the unchanged `adaptive` mode (`simulator.replay_stream`,
  32-request windows, policy from the predicted class through
  `simulator.candidates`), E1's C3 slice 2 (skip 1,000,000 canonical
  requests, next 1,000,000, merge `auto`), `LatencyModel(5, 100, 50)`,
  capacities 2048 and 131072 blocks, on the `PAPER_SUBSET` traces whose
  slice 2 is full in E1 (19 of 21; the two FIU traces E1 marks `n/a` are
  excluded the same way).
- **Speed-up, proved not assumed**: window vectors depend only on the
  request stream, so they are computed once per trace and every classifier
  is batch-scored; the replay is driven by the precomputed labels
  (`clfzoo.router.Replayed`), and classifiers with an identical label
  sequence share one replay. Validity gate 1: the shipped GNB through this
  path must equal E1's committed `adaptive` row on every counter on all 38
  cells (else nothing is reported). `tests/test_clfzoo.py` also pins it on a
  sample trace.
- **Baselines** per trace and capacity: `none`, `sequential`, `deep@8`,
  `linux_ra@256` reused from `results/disk/sweep.jsonl` only after validity
  gate 2: re-running all four on `msr_prxy_0` at 2048 through
  `multicorpus.run_job` reproduces the committed rows exactly; and
  `adaptive_evidence` (the rule-based router, not in C3), run here through
  `multicorpus.run_job`.
- **Comparisons**, each at each capacity, n = 19 traces: hit-ratio
  difference of every routed classifier, `adaptive_evidence`, `deep@8` and
  `linux_ra@256` vs `sequential`; hit-ratio difference of every routed
  classifier and `adaptive_evidence` vs `gnb_online` (shipped `adaptive`);
  `device_ios` and `blocks_moved` as mean per-trace ratios vs `none`.
  Routing agreement with GNB (fraction of windows given the same class) and
  class shares are reported for every classifier and trace.
- Every hit ratio is **simulated**; writes replay as reads (oracleGeneral).
  No hit-ratio claim is made without its device-I/O and blocks-moved ratios.

## 4. Decision rules

- **Paired bootstrap** over units (seed pairs in Part 1, traces in Part 2):
  10,000 resamples, seed 42, 95% percentile interval.
- **Cluster bootstrap** (Part 2 only): corpora, then traces within each
  drawn corpus, twin families down-weighted to total weight 1
  (`stats.cluster_bootstrap` with `stats.twin_weights(corpora.TWIN_GROUPS)`),
  10,000 resamples, seed 42.
- "**better**" iff the interval lies entirely on the better side (Part 2:
  both the paired and the cluster interval); "**worse**" iff entirely on the
  worse side (Part 2: both); otherwise "**no difference detected**", stated
  with the interval(s). For a ratio the null is 1 and lower device work is
  better.
- "A tuned classifier makes the router better" is written only if some
  family is **better** than `gnb_online` on hit ratio at a capacity **and**
  its device-I/O ratio interval is reported beside it; "the router beats
  read-ahead" only if it is **better** than `sequential`.
- Winner counts are never reported without precision, wasted share and the
  device-I/O ratio. No family, trace, capacity or setting is dropped after
  seeing results; there is no multiplicity correction, and the number of
  comparisons (15 families x settings in Part 1; 19 policies x 4 metrics x 2
  capacities in Part 2) is stated wherever verdicts are counted.

## 5. What will not be done

- No synthetic accuracy is presented as accuracy on real traces.
- No test seed or Part 2 trace is used to choose a hyperparameter.
- No number from `--quick` is quoted.

## Amendments

1. **2026-10-27, before any full-run result existed.** The first full run
   was stopped during Part 1 tuning because one linear-SVM candidate (a
   large `C` on unscaled vectors) did not finish a single fit in over two
   minutes; no tuning or test score had been written or read. Both SVM
   families now cap libsvm at `max_iter=200000` iterations
   (`zoo.SVC_MAX_ITER`); a capped fit is scored like any other candidate.
   Nothing else changed, and the run restarted from the beginning.

# results/classifier: classifier zoo and real-trace router benchmark (track CLF)

Produced by `python reproduce.py classifier` (`repro/classifier.py`,
`adaptive_prefetch.clfzoo`), 20 worker processes on the shared 128-core
host. The comparisons and decision rules are in `PREREGISTERED.md`, which was
written **after E1-E6 were known** and committed before the full run (one
dated amendment, also before any result). `provenance.json` hashes every file
here, records the inputs (the 19 traces and `results/disk/sweep.jsonl`), the
load average, and the two validity checks.

| File | One row per |
|---|---|
| `synthetic_tuning.csv` | random-search candidate x setting x family (validation score on training seeds) |
| `synthetic_eval.csv` | setting x test condition x family: test accuracy, macro-F1, ECE, Brier, confusion, cost, paired difference vs `gnb_online` and verdict |
| `router_runs.csv` | trace x capacity x policy on E1's C3 slice 2 (simulated) |
| `router_comparisons.csv` | capacity x metric x policy x baseline: paired and cluster intervals, verdict |
| `HEADLINES.md` | generated headline tables (from the CSVs above) |
| `router_summary.csv` | capacity x policy: mean hit ratio, device-I/O and blocks-moved ratios to `none`, precision, routing agreement with GNB, class shares |

## Methods and citations

BibTeX keys: `paper/refs.bib` (`pedregosa2011sklearn`) and
`paper/refs_CLF.bib` (every other key, to be merged into `refs.bib`).

| Family (`zoo.FAMILIES`) | Implementation | Tuned hyperparameters | Reference |
|---|---|---|---|
| `gnb_online` (shipped baseline, deployed default) | `model.OnlineGaussianNB` | none (as shipped) | Domingos & Pazzani 1997 `domingos1997optimality`; Zhang 2004 `zhang2004optimality` |
| `qda_online` (shipped) | `model.OnlineQDA` | none (as shipped) | Hastie, Tibshirani & Friedman 2009 `hastie2009elements` |
| `gnb` | sklearn `GaussianNB` | `var_smoothing` | `domingos1997optimality`, `zhang2004optimality` |
| `lda` | `LinearDiscriminantAnalysis` | shrinkage (none / Ledoit-Wolf / fixed) | Fisher 1936 `fisher1936use`; Ledoit & Wolf 2004 `ledoit2004well` |
| `qda` | `QuadraticDiscriminantAnalysis` | `reg_param` | `hastie2009elements` |
| `logreg` | multinomial `LogisticRegression` | `C`, standardisation | Cox 1958 `cox1958regression` |
| `knn` | `KNeighborsClassifier` | `k`, weights, standardisation | Cover & Hart 1967 `cover1967nearest` |
| `svm_linear` | `SVC(kernel="linear")`, Platt probabilities | `C`, standardisation | Cortes & Vapnik 1995 `cortes1995support`; Platt 1999 `platt1999probabilistic` |
| `svm_rbf` | `SVC(kernel="rbf")`, Platt probabilities | `C`, `gamma`, standardisation | `cortes1995support`, `platt1999probabilistic` |
| `cart` | `DecisionTreeClassifier` | depth, min leaf | Breiman et al. 1984 `breiman1984classification` |
| `rf` | `RandomForestClassifier` | trees, depth, `max_features`, min leaf | Breiman 2001 `breiman2001random` |
| `extra_trees` | `ExtraTreesClassifier` | as `rf` | Geurts, Ernst & Wehenkel 2006 `geurts2006extremely` |
| `hgb` | `HistGradientBoostingClassifier` (LightGBM-style histogram boosting) | learning rate, iterations, leaves, L2 | Friedman 2001 `friedman2001greedy`; Ke et al. 2017 `ke2017lightgbm` |
| `mlp` | `MLPClassifier`, Adam | hidden sizes, `alpha`, learning rate | Rumelhart, Hinton & Williams 1986 `rumelhart1986learning`; Kingma & Ba 2015 `kingma2015adam` |
| `sgd_online` | `SGDClassifier(loss="log_loss")`, `partial_fit` only | `alpha`, streaming epochs | Bottou 2010 `bottou2010large` |
| (not run) Hoeffding tree | - | - | Domingos & Hulten 2000 `domingos2000mining`: omitted, no correct stdlib implementation in the time available |

| Procedure | Reference |
|---|---|
| Hyperparameter search: seeded random search, 16 candidates, default first | Bergstra & Bengio 2012 `bergstra2012random` |
| Library | Pedregosa et al. 2011 `pedregosa2011sklearn` |
| Calibration: expected calibration error (15 bins) | Guo et al. 2017 `guo2017calibration` |
| Calibration: multiclass Brier score | Brier 1950 `brier1950verification` |
| Intervals: percentile bootstrap (paired; cluster over corpora) | Efron 1979 `efron1979bootstrap` |

## How to read the numbers

- **Part 1 accuracies are synthetic.** Real traces have no workload labels;
  no accuracy here is an accuracy on a real trace. Selection used training
  seeds only (12 folds, validate on the next training seed); the test seeds
  were scored once.
- **Part 1 assumes phases persist across windows.** In `baseline` and the
  CLASSIFIER_EVAL regimes each class is laid out as its own stream (the
  `make_model` layout), so a window's four contextual features
  (`reuse_ratio`, `reuse_distance_log`, `contiguous_delta`,
  `stride_persist`) are computed from preceding windows **of the same
  class**, in training and in test alike. A real trace changes phase, so
  accuracies that use the contextual features are optimistic, and part of a
  family's difference from `gnb_online` can be how well it exploits that
  same-class context. (A review probe after the run with the selected
  `cart`, not pre-registered and not a result, found that the large tree
  margin on `transition_heavy` is reproduced by the eight stream features
  alone and by a layout that shuffles every class into one stream, so it is
  not an artifact of this layout; on `transition` the same-class context
  did lift that tree's accuracy.) `transfer` builds one context in class order before
  shuffling, as `main.transfer_probe` does.
- **`transition` / `transition_heavy` labels.** `pipeline.blended_regime`
  labels a window by the class of its *first* part: at the 0.75 blend that
  is the first quarter of the window, a minority of its requests, not the
  dominant phase its docstring names.
- **SVM labels are Platt-probability argmaxes.** Both SVM families predict
  (Part 1 and Part 2 alike) the argmax of `predict_proba`, which can differ
  from libsvm's own decision-function `predict`; the same rule is used for
  every scikit-learn family (`zoo.Router`).
- **Part 2 hit ratios are simulated** LRU block hit ratios (oracleGeneral
  traces: writes replay as reads), each reported with its device-I/O and
  blocks-moved ratios to `none`.
- **Timings** (`predict_us`, `train_ms`, `wall_s`) were taken on a shared host
  whose 1-minute load average reached well above the 128 cores during the run
  (recorded per row); they are contended and are not cost claims.
- The routed policy only changes **which of four fixed candidate rules**
  (`simulator.candidates`: depth-2 read-ahead, one stride block, one block,
  nothing) runs in the next window. A classifier can only move results
  through that choice; `agreement_with_gnb` shows how often it actually
  chose differently.

## Headline outcomes

`HEADLINES.md` holds the headline tables, generated from the CSVs by the
same run (no figure in it is typed by hand): Part 1 test accuracy per family
and setting with its verdict against `gnb_online`, and Part 2 per capacity
the mean simulated hit ratio, device-I/O and blocks-moved ratios to `none`,
precision, routing agreement with GNB, and the paired and cluster intervals
with verdicts against `sequential` and against `gnb_online`.

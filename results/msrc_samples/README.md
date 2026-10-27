# MSRC light sample (E7): labelled-window transfer check and sanity replay

Regenerate with `python reproduce.py msrc` (5 min 20 s here on 8 workers, load
average 37 to 151 from other tenants; `provenance.json` records it). Rules:
`PREREGISTERED.md` (with amendments A1, A2). Code:
`src/adaptive_prefetch/msrc_eval.py`; tests: `tests/test_msrc_eval.py`.

**Read this first.** The seven files are the smallest of the 32 in
MSRC-trace-003, **selected by size**, not at random and not by workload. They
are not representative, and nothing here is a benchmark claim. Each figure
below has its window count (n); intervals are 95% window-bootstrap
percentiles, 2,000 replicates, seed 42. Windows of one trace are serially
correlated, so the intervals are too narrow, and the pooled rows are mostly
`web_1`, `rsrch_2` and `wdev_2`.

## Files

| File | Contents |
|---|---|
| `traces.csv` | per trace: requests, reads, seq-flagged requests, windows, seq windows under both label rules, `small_n`, mean `t1`, `flag_matches_contiguity` |
| `classifier_metrics.csv` | per (protocol, classifier, label rule, trace or `pooled`): accuracy, balanced accuracy, macro-F1 with intervals, predicted class counts |
| `classifier_deltas.csv` | paired balanced-accuracy differences against the shipped GNB and against `route_window`, with the pre-registered verdict |
| `replay.csv` | sanity replay: 7 traces x 2 views x 2 cost models x 6 modes |

## Finding 0: the "real label" is an address-contiguity flag

`flag_matches_contiguity` in `traces.csv` is the share of requests whose
`seq` flag equals "this request starts where the previous one ended". It is
0.984 to 1.000 on all seven traces. The flag is therefore almost exactly a
per-request contiguity rule, which the window features (`contiguous_ratio`
above all) already measure. Agreement with it measures how well a classifier
recovers address contiguity. It does **not** show that the classifier
recognises a workload, and it is no evidence that a classifier helps
prefetching.

The windows are also very unbalanced. Under the primary rule (at least 17 of
32 requests flagged), only `web_3` (340 of 980 windows) and `web_1` (74 of
5,027) have enough `seq` windows for a verdict. The other five traces have 0
or 1 such windows and are `small_n` (`wdev_3` has only 21 windows), so
balanced accuracy and macro-F1 are undefined there. Pooled: 415 seq windows
out of 18,641.

## Finding 1: synthetic-trained classifiers on the real flag (primary rule)

Balanced accuracy, `seq` vs `not_seq`, as `est [lo, hi]`:

| Classifier (trained on synthetic windows) | web_3 (n=980) | web_1 (n=5,027) | pooled (n=18,641) |
|---|---|---|---|
| shipped GNB (`make_model(42)`) | 0.805 [0.786, 0.825] | 0.809 [0.802, 0.816] | 0.916 [0.912, 0.919] |
| shipped QDA | 0.831 [0.813, 0.850] | 0.820 [0.803, 0.832] | 0.941 [0.935, 0.946] |
| sklearn GaussianNB | 0.802 [0.783, 0.822] | 0.822 [0.815, 0.829] | 0.939 [0.934, 0.943] |
| decision tree | 0.800 [0.781, 0.820] | 0.818 [0.811, 0.824] | 0.924 [0.919, 0.928] |
| extra trees | 0.813 [0.794, 0.833] | 0.821 [0.815, 0.828] | 0.912 [0.908, 0.916] |
| logistic | 0.792 [0.774, 0.811] | 0.789 [0.782, 0.795] | 0.925 [0.922, 0.927] |
| MLP | 0.811 [0.791, 0.830] | 0.778 [0.771, 0.784] | 0.915 [0.909, 0.920] |
| random forest | 0.802 [0.782, 0.822] | 0.795 [0.788, 0.802] | 0.899 [0.895, 0.903] |
| hist gradient boosting | 0.800 [0.778, 0.824] | 0.809 [0.792, 0.821] | 0.897 [0.884, 0.909] |
| QDA (sklearn, reg 0.01) | 0.791 [0.772, 0.811] | 0.763 [0.756, 0.770] | 0.877 [0.874, 0.881] |
| kNN (k=5) | 0.724 [0.705, 0.743] | 0.707 [0.700, 0.714] | 0.849 [0.844, 0.852] |
| LDA | 0.645 [0.628, 0.663] | 0.646 [0.640, 0.652] | 0.735 [0.731, 0.738] |
| RBF SVM | 0.505 [0.502, 0.509] | 0.504 [0.503, 0.505] | 0.516 [0.515, 0.517] |
| `route_window` (rule, no training) | 0.740 [0.720, 0.760] | 0.778 [0.772, 0.785] | 0.866 [0.863, 0.869] |
| always `not_seq` | 0.500 | 0.500 | 0.500 |

- **Transfer.** By the pre-registered rule (lower end above 0.5), all 13
  synthetic-trained classifiers transfer on `web_3` and `web_1`. The RBF
  SVM passes only barely (0.505 and 0.504), because it calls nearly every
  real window `sequential` (4,991 of 5,027 on `web_1`).
- **No family is clearly better than the shipped GNB.** Against GNB
  (`classifier_deltas.csv`), the shipped QDA is *better* on `web_3` (+0.026
  [+0.017, +0.035]) and pooled (+0.025 [+0.019, +0.029]), with no difference
  detected on `web_1` (+0.010 [-0.005, +0.019]). sklearn GaussianNB, logistic, the
  decision tree and extra trees are better on some scopes, by at most
  +0.023, and none of them is better on both traces and pooled. LDA, kNN,
  the SVM and sklearn's QDA are *worse* on all three scopes. The random
  forest is worse on `web_1` and pooled, with no difference detected on
  `web_3`. D1 found QDA losing to GNB under covariate shift; on this flag
  the shipped QDA does not lose.
- Every synthetic-trained classifier except the SVM, LDA and kNN has a
  higher point estimate than `route_window`'s 0.740 on `web_3`.
- **Accuracy is a poor guide.** Always predicting `not_seq` has pooled
  accuracy 0.978 against GNB's 0.838. GNB calls 3,426 windows `sequential`
  where the flag marks 415, so it over-predicts `seq`. Its macro-F1 on
  `web_1` is 0.418 [0.408, 0.428].

## Finding 2: within-real baselines (leave-one-out)

These are fitted on real windows of the other traces (LOTO, 7 folds) or of
the other two hosts (LOFO, 3 folds). They are a reference for what the
features can do, not deployable models, and no choice is made from them.

- **LOTO** on `web_3` (training includes its sibling `web_1`): HistGB 0.980
  [0.971, 0.987], random forest 0.974 [0.963, 0.984], logistic 0.969
  [0.957, 0.980], decision tree 0.963 [0.952, 0.973]. Against the shipped
  GNB these are *better* on `web_3` (HistGB +0.175 [+0.156, +0.194]).
  Pooled, only the random forest (+0.020 [+0.003, +0.036]) and HistGB
  (+0.018 [+0.001, +0.035]) are better; LDA and logistic show no
  difference detected. Fitted on real windows, GaussianNB
  collapses to the majority class (0.487 [0.481, 0.493]).
- **LOFO**, the stricter protocol that A1 says to quote. Against the shipped
  GNB, only LDA is *better*: on `web_3` 0.866 [0.849, 0.883] (+0.062
  [+0.048, +0.075]) and pooled 0.959 [0.945, 0.971] (+0.042 [+0.029,
  +0.055]). Logistic on `web_3` (0.834) shows no difference detected (+0.029
  [-0.003, +0.062]). Most other families predict the majority class. When
  `web` is held out, the `rsrch` and `wdev` training windows contain a
  single `seq` window, so QDA and HistGB are `skipped:` and most of the rest
  fall back to the majority class. Pooled LOFO: logistic 0.775 [0.749,
  0.798], the other families except LDA at most 0.556.
- Most of the headroom over the shipped GNB comes from real labelled
  training data from the same host family (LOTO), not from a better
  algorithm. Under LOFO only LDA keeps an advantage.
  Since the label is a contiguity flag (Finding 0), even that headroom
  measures how well contiguity is recovered.

## Finding 3: sanity replay (simulated, capacity 2048 blocks)

`replay.csv`; ratios are against `none` on the same trace, view and cost
model. In the `native` view, writes are written into the cache
(write-allocate), never trigger a prefetch and never count as hits. `wdev_1`
has no read requests, so its native rows are `n/a`. In the
`writes_as_reads` view every request replays as a read (the oracle-trace
convention). The hit ratios do not depend on the cost model for these modes.

- In the native view, on the four traces with real read traffic (`rsrch_2`,
  `web_1`, `web_3` with over 1M read blocks each, and `wdev_2` with 2,315),
  `linux_ra@256` has the highest hit ratio and the lowest device-I/O ratio
  of the six modes. Examples (measured cost): `web_1` hit ratio 0.904 vs
  0.748 for `none`, device-I/O ratio 0.780, blocks moved 1.082; `rsrch_2`
  0.697 vs 0.534, 0.667, 1.046. The other traces have at most 1,264 native
  read blocks, and there `deep@8` is sometimes highest.
- In the native view, `adaptive` (shipped GNB) gains at most +0.022 hit
  ratio over `none` (`wdev_2`). On `web_3`, every mode except `linux_ra`
  *loses* hit ratio: `sequential` 0.257 and `adaptive` 0.391 against `none` at 0.506,
  with blocks moved 1.525 and 1.236. At 1 MiB, prefetch pollution evicts
  blocks the trace re-reads.
- The `measured` cost model uses the mean `t1` service time. Its prefetch
  multiple is 1.0, so the break-even precision is above 1 and the
  `mean_latency_us` column cannot favour any prefetcher. It is a cost model,
  not a latency measurement.

## Answer to "GNB shows no real benefit: which classifier is best?"

On these seven size-selected traces, no classifier family trained on
synthetic windows is clearly better than the shipped GNB on the only real
label available. QDA is better by about 0.025 balanced accuracy on `web_3`
and pooled, with no difference detected on `web_1`. The flag itself is an
address-contiguity rule, and in replay the classifier-routed `adaptive` mode
is not where the hit ratio or the device-I/O savings come from: `linux_ra`
is. Changing the classifier does not address why GNB shows no benefit.

## Method -> reference

| Method here | Reference (BibTeX key; `paper/refs.bib` or `paper/refs_MSRC.bib`) |
|---|---|
| MSRC enterprise traces | Narayanan, Donnelly, Rowstron, FAST 2008 (`narayanan2008write`) |
| scikit-learn implementations | Pedregosa et al., JMLR 2011 (`pedregosa2011sklearn`) |
| Gaussian naive Bayes (shipped GNB, `gaussian_nb`) | Domingos & Pazzani, Machine Learning 1997 (`domingos1997optimality`) |
| LDA | Fisher, Annals of Eugenics 1936 (`fisher1936use`) |
| QDA with covariance regularisation | Friedman, JASA 1989 (`friedman1989regularized`) |
| logistic regression | Cox, JRSS B 1958 (`cox1958regression`) |
| k-nearest neighbours | Cover & Hart, IEEE Trans. Inf. Theory 1967 (`cover1967nearest`) |
| RBF support-vector machine | Cortes & Vapnik, Machine Learning 1995 (`cortes1995support`) |
| decision tree (CART) | Breiman, Friedman, Olshen, Stone 1984 (`breiman1984cart`) |
| random forest | Breiman, Machine Learning 2001 (`breiman2001random`) |
| extra trees | Geurts, Ernst, Wehenkel, Machine Learning 2006 (`geurts2006extremely`) |
| gradient boosting; histogram variant | Friedman, Ann. Stat. 2001 (`friedman2001greedy`); Ke et al., NIPS 2017 (`ke2017lightgbm`) |
| MLP trained by back-propagation | Rumelhart, Hinton, Williams, Nature 1986 (`rumelhart1986learning`) |
| bootstrap intervals | Efron, Ann. Stat. 1979 (`efron1979bootstrap`) |
| balanced accuracy | Brodersen et al., ICPR 2010 (`brodersen2010balanced`) |

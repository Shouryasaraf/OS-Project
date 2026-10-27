# Hindsight-labelled router (track HSR)

**Simulated.** LRU block-cache replay of libCacheSim oracleGeneral traces
(writes replay as reads) under `LatencyModel(5, 100, 50)`. Hit ratios are
block hit ratios; `device_ios` and `blocks_moved` are modelled device work.
Nothing here is a latency measurement. Pre-registration (written after
E1-E6 were known, with amendments A1-A3): `PREREGISTERED.md`. Reproduce:
`python reproduce.py hindsight` (`--quick` writes `outputs/hindsight-quick`).

## What was asked and what was built

The shipped `adaptive` mode routes each 32-request window by a GNB
classifier trained on **synthetic** workload classes, and every class maps
to a candidate set of at most two blocks. E1 found it worse than
`sequential`. This study replaces both limits:

- a new disk mode `routed` (appended last in `benchmark.MODES` and
  `sweep.RANKING_MODES`, registered in all four places, bit-identical in
  `replay` and `replay_stream`) whose per-window action is one of
  `none`, `sequential`, `deep@8`, `deep@32`, `deep@256`;
- labels from **real traces**: every fixed action is replayed over slice 1
  and each window is labelled with the action that scored best on the next
  window (trajectory hindsight labels: an approximation of the true
  counterfactual, because the cache state is the always-a state);
- eight classifier families tuned by random search with grouped CV over
  traces, three objectives (`hit`, `u0.5`, `dev`), two capacities.

## Files

| File | Rows |
|---|---|
| `rows.csv` | every replay: slice 1 fixed actions (training), slice 2 fixed actions, VCF/GVCF, routers, oracles, `adaptive_evidence`, reused `adaptive` / `linux_ra@256` sweep rows |
| `selection.csv` | every random-search configuration's grouped-CV score (selection data: slice 1 only) |
| `families.csv` | every family's CV-best router on slice 2 under the trajectory proxy (descriptive, no replay) |
| `summary.csv` | per capacity and policy: hit ratio, u0.5, precision, wasted share, device-I/O and blocks-moved ratios vs `none` with intervals, headroom captured |
| `comparisons.csv` | H1-H7, S1-S2, R, D with paired and cluster intervals and the pre-registered word |

## Methods and citations

| Method | Where | Citation (key in `paper/refs.bib` or `paper/refs_HSR.bib`) |
|---|---|---|
| Gaussian naive Bayes | family `gnb` | Domingos and Pazzani 1997 `domingos1997optimality`; Zhang 2004 `zhang2004optimality` |
| Logistic regression | family `logreg` | Cox 1958 `cox1958regression` |
| CART decision tree | family `tree` | Breiman, Friedman, Olshen, Stone 1984 `breiman1984cart` |
| Random forest | family `forest` | Breiman 2001 `breiman2001random` |
| Extremely randomized trees | family `extra_trees` | Geurts, Ernst, Wehenkel 2006 `geurts2006extremely` |
| Histogram gradient boosting | family `hgb` (scikit-learn's follows LightGBM) | Friedman 2001 `friedman2001greedy`; Ke et al. 2017 `ke2017lightgbm` |
| Multilayer perceptron, Adam | family `mlp` | Rumelhart, Hinton, Williams 1986 `rumelhart1986learning`; Kingma and Ba 2015 `kingma2015adam` |
| k-nearest neighbours | family `knn` | Cover and Hart 1967 `cover1967nearest` |
| Classifier background | all | Hastie, Tibshirani, Friedman 2009 `hastie2009elements` |
| Random search | `hindsight_models.search_space` | Bergstra and Bengio 2012 `bergstra2012random` |
| Implementation | all families | scikit-learn, Pedregosa et al. 2011 `pedregosa2011sklearn` |
| Hindsight (oracle) labels for a cache policy | the labelling idea | Jain and Lin 2016 `jain2016hawkeye` (Belady-labelled training); Shi et al. 2019 `shi2019glider` |
| Distribution shift of trajectory labels | the stated approximation | Ross, Gordon, Bagnell 2011 `ross2011reduction` |
| Belady's MIN | reference only | Belady 1966 `belady1966study` |

## Results

All figures below are **simulated**, slice 2 (requests [1M, 2M)), n = 19
traces (`fiu_madmax` and `fiu_online` have no full slice 2: `n/a`), paired
bootstrap and cluster bootstrap, 10,000 resamples, seed 42, 95% intervals;
the words follow `PREREGISTERED.md` section 6. Source: `comparisons.csv`,
`summary.csv`, `families.csv`, `selection.csv`.

**Checks.** All 400 fixed-action replays (slice 1 and 2, both capacities)
equal `results/disk/sweep.jsonl` on every counter; one `adaptive` and one
`linux_ra@256` slice-2 row were reproduced exactly before those sweep rows
were reused (`provenance.json`, `extra.checks`).

**Primary (H1): the router does not beat a validation-chosen fixed action.**
`routed_hit` vs `vcf_hit`, hit ratio: at 2048 blocks +0.0003, paired
[-0.0199, 0.0226], cluster [-0.0214, 0.0315], **no difference detected**
(5 wins, 12 losses, 2 ties). At 131072 the router picks `deep@256` on 99.9%
of windows and `vcf_hit` is `deep@256` on every trace; the difference is
-0.00003, paired [-0.00005, -0.00001], cluster [-0.00005, -0.00001], so it is
**worse** by a negligible amount (0 wins, 8 losses, 11 ties).

**Against the shipped and fixed baselines it is better on hit ratio, and the
gain is read-ahead depth, not classification.** `routed_hit` is **better**
than `sequential` (+0.2981 at 2048, +0.3914 at 131072), than `adaptive`
(the shipped GNB router; +0.3006, +0.3995), than `deep@8`, `linux_ra@256`
(+0.0877, +0.1555) and `adaptive_evidence`, with both intervals above zero
at both capacities. That gain costs device work: blocks moved relative to
`none` are 3.65x (2048) and 2.79x (131072), **worse**; device I/Os are 1.07x
at 2048 (no difference detected) and 1.38x at 131072 (**worse**). The same
hit ratio is available from the fixed `vcf_hit` choice (`deep@256` on 15 of
19 traces at 2048 and on all 19 at 131072).

**Other objectives.** `routed_u0.5` vs `vcf_u0.5` on (hits - 0.5 x
prefetches) per block: no difference detected at either capacity (H3).
`routed_dev` vs `vcf_dev` on device I/Os: 0.9911 at 2048 (paired interval
only; not robust to corpus clustering) and 0.9834 at 131072 (no difference
detected) (H4). `routed_dev` issues fewer device I/Os than `none` (0.9756
and 0.9562, **better**) but moves more blocks (1.12x, 1.35x, **worse**);
`linux_ra@256` does better on both (0.7132 and 0.5481 device I/Os, 1.13x and
1.07x blocks moved).

**Single global fixed action (H7).** `gvcf_<obj>` is the one action with
the best mean slice-1 objective (`deep@256` for `hit` at both capacities;
`none` for `dev` at both; `none` for `u0.5` at 2048 and `deep@8` at 131072).
`routed_hit` vs `gvcf_hit`: no difference detected at 2048 (+0.0319), worse
at 131072 (the same rows as H1, because `vcf_hit` is `deep@256` on every
trace there). `routed_u0.5` vs `gvcf_u0.5`: no difference detected at either
capacity. `routed_dev` vs `gvcf_dev` (= `none`): **better** at both (device-I/O
ratio 0.9756 and 0.9562, the D rows below).

**Leave-one-corpus-out (H6).** Trained without the test trace's corpus, the
router is **worse** than `vcf_hit` at both capacities (-0.0433 at 2048;
-0.0003 at 131072).

**Is the classifier family the problem? (S2, `selection.csv`).** On the
grouped-CV score (share of hindsight headroom over `none` captured on
held-out traces; descriptive, no interval), Gaussian naive Bayes is the
lowest-scoring family in all six (objective, capacity) cells; the selected families are HGB (hit at 2048, u0.5 at 131072), a CART tree (u0.5
at 2048), logistic regression (hit at 131072) and random forest (dev at
both). Replayed, the GNB-family hit router is below the selected router at
both capacities (paired interval only at 2048; **worse** at 131072, -0.0146)
and **worse** than `vcf_hit` at 131072. So GNB scores lowest here, but none
of the eight families produced a router that beats the strongest fixed
baseline; on these traces, capacities and labels the limit is what a
per-window choice adds over a per-trace fixed depth, not the classifier
family.

**Hindsight ceiling (R).** The trajectory oracle (slice 2's own labels,
a reference, never a policy) beats `vcf_hit` by +0.0315 at 2048
(**better**) but is *below* it at 131072 (-0.0038, paired interval only):
the trajectory labels are not a true counterfactual ceiling. The router
captured 0.009 of the oracle's headroom over `vcf_hit` at 2048
(`summary.csv`, `headroom_captured`, descriptive, no interval); at 131072
there is no headroom over `vcf_hit` to capture (the oracle is below it), so
the column is empty (amendment A3). For `dev`: 0.247 and 0.227.

**Runtime and load.** Stages ran resumably from `outputs/hindsight/`:
slice-1 records 264 s on 28 workers; slice-2 records 1,688 s on 10 workers
concurrently with the grouped-CV search (685 s on 18 workers, 384
configurations) and fits; slice-2 replays 962 s on 28 workers; then one
`reproduce.py hindsight` pass that reused every cached stage (its
`provenance.json` `wall_s`). The host's 1-minute load ranged 32-157 during
the run (other tenants); per-row `load_start`/`wall_s` are in `rows.csv`.
CPU times are contended and are not costs.

## Limitations

- Trajectory labels approximate the counterfactual; the oracle row shows the
  approximation can sit below a fixed policy (Ross et al. 2011 describe the
  general distribution-shift problem of such labels).
- One window size (32), one cost model, five actions, two capacities, 21
  traces from four corpora (19 scored); the selection metric is a mean of
  per-trace ratios and is dominated by traces with little headroom under
  `u0.5` (negative CV scores).
- Amendment A1 cut the CV subsample and the forest sizes for time.
- Not in the real-device grid (E3): no latency claim of any kind.

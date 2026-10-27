# Web ranker alternatives (track WRK)

Does a different, tuned model family rank RetailRocket's next-page
candidates better than the shipped logistic ranker, and does that move the
simulated browser-cache metrics? Pre-registered in
[`PREREGISTERED.md`](PREREGISTERED.md), written after E1-E6 and committed
before this run. Reproduce with `python reproduce.py web_rankers`.

Every number here is **simulated** (session-cache replay); prefetch costs
and object sizes are **assumed**; timings are wall times on a **contended**
shared host and are not costs.

## Methods and references

Every family sees the same split-0 candidates and the same nine causal
features (`webprefetch.model.FEATURE_NAMES`); only the scoring function
changes. All estimators are scikit-learn 1.9.1 (`pedregosa2011sklearn`).
BibTeX keys are in `paper/refs.bib` (existing) and `paper/refs_WRK.bib`
(new in this track).

| method | where | citation key(s) |
|---|---|---|
| logistic regression (shipped ranker, `logistic`; and `logistic_tuned`) | `model.train_ranker`, `rankers.make_estimator` | `cox1958regression` |
| L1 (lasso) penalty in `logistic_tuned` | `l1_ratio = 1` | `tibshirani1996lasso` |
| gradient boosting (`hgb`) | `HistGradientBoostingClassifier` | `friedman2001greedy` |
| histogram-binned boosting, the LightGBM design scikit-learn follows | `HistGradientBoostingClassifier` | `ke2017lightgbm` |
| random forest (`rf`) | `RandomForestClassifier` | `breiman2001random` |
| extremely randomised trees (`et`) | `ExtraTreesClassifier` | `geurts2006extremely` |
| CART decision trees, the base learner of `rf`, `et` | | `breiman1984cart` |
| multilayer perceptron by back-propagation (`mlp`) | `MLPClassifier` | `rumelhart1986learning` |
| Adam optimiser (`mlp`) | `solver="adam"` | `kingma2015adam` |
| random hyperparameter search | `ParameterSampler`, seed 42 | `bergstra2012random` |
| paired session bootstrap, percentile interval | `benchmark.PairedSessions`, `rankers.RatioBootstrap` | `efron1979bootstrap` |
| transition top-1 and online counts (history-only baselines) | `scorers.build_scorer("trans_s0" / "trans_online")` | the repository's E4 (`results/web/`) |
| RetailRocket data | `data/raw/events.csv` (SHA-pinned) | `retailrocket2017` |

LambdaMART-style listwise ranking (Burges 2010) needs LightGBM or XGBoost,
which are not installed here; it was not run.

## Files

| file | rows |
|---|---|
| `tuning.csv` | every random-search trial per family (and the shipped ranker as a reference row) on the split-2 tuning sample |
| `ranking_metrics.csv` | next-view hit@1, coverage@8, MRR@8 and their novel-view versions per (split, scorer); training and inference wall times on the test row |
| `policy_metrics.csv` | `web/policy_metrics.csv` columns for every scorer, policy and capacity (8, 32), validation and test |
| `comparisons.csv` | the pre-registered comparisons S1 and K1-K6 with 10,000-resample paired session-bootstrap intervals and verdicts |
| `summary.json` | selections, chosen hyperparameters, costs, loads, the E4 reproduction check |
| `provenance.json` | command, git state, host, loads, input and output hashes |

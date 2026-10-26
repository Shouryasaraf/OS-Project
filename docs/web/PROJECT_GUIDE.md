# Project guide: adaptive product-page prefetching

## Scope

This repository is an offline, reproducible machine-learning experiment on real product-view events. It predicts likely next product pages and evaluates how many pages to prefetch into a simulated browser cache. It is **not** a browser extension, server integration, or measured page-load acceleration system.

The only research demand stream is RetailRocket `events.csv`, version 2, obtained from [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). The pipeline verifies the CSV SHA-256 before processing. No generated events are used for training, validation, test, or demonstration. The small constructed fixtures in `tests/` only check implementation behavior.

## End-to-end data flow

```text
Kaggle events.csv
  -> schema, checksum, and row-count validation
  -> time-ordered visitor sessions and product-view requests
  -> four chronological splits
  -> history-only candidate retrieval
  -> logistic candidate scorer trained on later real views
  -> validation-only model and controller selection
  -> held-out per-session cache replay
  -> aggregate metrics and one real-session walkthrough
```

| Stage | Input | Action | Output |
| --- | --- | --- | --- |
| Acquisition | Kaggle version 2 archive | Download or reuse a matching local CSV; verify checksum | `data/raw/events.csv` |
| Preprocessing | All recorded events | Validate fields, remove exact duplicates, sort, form 30-minute visitor sessions, derive view context | `data/processed/views.csv.gz`, `profile.json` |
| EDA | Prepared views | Count events/sessions, date range, repeats, split sizes, and transition concentration | `outputs/eda.md` |
| Retrieval | Oldest 40% of views | Count within-session next-item transitions; retain up to 20 per source plus 8 popular fallbacks | `models/candidate_index.joblib` |
| Model training | Next 20% of views | Sample 30,000 time-spaced decisions; fit logistic candidate scores; include session-ending views as negatives | `models/ranker.joblib`, `training_summary.json` |
| Model tuning | Third 20% of views | Compare regularization `C=0.1, 1, 10` using 10,000 time-spaced decisions and candidate log loss | Frozen ranker |
| Policy tuning | Full validation split by default | Replay fixed, gated, and adaptive controllers in eight-object session caches; select per cost scenario | `outputs/validation_metrics.csv` and selection in `benchmark.json` |
| Final evaluation | Full newest 20% by default | Replay frozen selections at capacities 2, 4, 8, 16, and 32; paired session bootstrap | `outputs/test_metrics.csv`, `comparisons.csv`, `report.md` |
| Demonstration | First held-out session with at least five views | Show each view, predicted candidates, chosen depth, issued requests, and cache outcome | `outputs/demo.json` |

The default replay uses the full validation and test periods. Optional positive view limits choose approximate chronological prefixes containing only complete session segments; zero means full. The first 50,000/100,000-view budgets are retained as diagnostics in `subset_vs_full.csv`.

## Prediction and cache rules

At each view, the retriever proposes historical successors of the current item and a small popular-item fallback. The scorer uses transition count/probability, source support, candidate popularity and rank, whether the candidate was viewed earlier in the session, position, preceding time gap, and cart count. Current and past information alone determine a decision. The actual next item is used only later as a label or evaluation outcome.

The adaptive controller chooses `K` from `{0,1,2,4,8}` using ranked scores and a validation-tuned internal penalty. Comparators include no prefetch, fixed depths, and a confidence gate. Each visitor session owns a separate equal-capacity object cache. Prefetched objects are speculative eviction victims before demanded objects. A cache hit is a later view whose item is already resident; a prefetch is useful only if demanded before eviction. All policies receive the same request stream and cache rules.

The evaluation scores a policy as `(demand hits - assumed cost × issued prefetches) / views`. The cost scenarios are `0`, `0.01`, `0.05`, and `0.25` hit-equivalents per issued object. These are sensitivity parameters, **not measured bandwidth, energy, or latency**. Fixed and adaptive settings are selected separately for each scenario using validation data at capacity 8, then frozen for all three test capacities. The paired 95% intervals resample visitor sessions 500 times with a fixed seed.

## Reproduce

From the repository root with Python 3.10 or newer:

```powershell
python -m pip install -e .
webprefetch run
python -m unittest discover -s tests -v
webprefetch demo
```

The first run downloads the public archive and verifies the extracted CSV. The original prefix run measured 602.698 seconds and 1,004,056 KiB peak RSS in a Linux/Python 3.12 container; see the audit measurement. The full-split run measured 2,979.459 seconds and 4,186,944 KiB peak child RSS with `OPENBLAS_NUM_THREADS=1`; see `audit/phase2_measurement.txt`. `data/raw/`, `data/processed/`, and `models/` are not committed. To run one stage at a time, use `webprefetch download`, `prepare`, `eda`, `train`, and `benchmark` in that order. Example for a shorter smoke replay:

```powershell
webprefetch benchmark --validation-views 1000 --test-views 2000
```

That shorter command overwrites generated benchmark files. Re-run `webprefetch benchmark` without overrides to restore full-split evidence. If download fails, obtain the same version-2 `events.csv` from Kaggle and place it at `data/raw/events.csv`; `webprefetch prepare` still enforces the pinned checksum.

## Reading the results

The [default report](../../results/pivot_original/report.md) contains every tested capacity/cost comparison. The prior eight-object, cost-0.01 result was measured only on the 100,000-view chronological test prefix. Use the regenerated full-split `outputs/report.md` and `subset_vs_full.csv` for the revised result and uncertainty; the older prefix number must not be presented as a full-period estimate.

Candidate coverage is a hard retrieval ceiling: even a perfect ranker cannot prefetch a next item missing from its suggestions. The dataset also omits object size, response timing, cache headers, and whether a prefetch would finish before the next view. The replay assumes equal-size objects and immediate completion. Consequently, the report supports claims about **simulated object hits and request counts only**. A live-browser implementation and measured latency experiment are separate future work.

## Verification boundary

On the development machine, the pinned CSV was processed, model artifacts were trained, the default held-out benchmark and demo completed, and five original automated tests passed. The fresh download endpoint returned a ZIP response, but a full `webprefetch run` from an empty checkout must be checked separately. The generated report is evidence from this run, not a guarantee of identical network access or runtime elsewhere.

## Updated evidence files

`outputs/transition_baseline.csv` gives the ranker-free top-1 cache baseline; `models/training_summary.json` compares full logistic, context-disabled logistic, and histogram gradient boosting on validation. `outputs/calibration.csv` contains reliability bins and ECE. `outputs/break_even.csv` and `break_even_curve.csv` vary cost per unused prefetch on frozen validation-selected policies. `outputs/cohorts.csv` separates cold and historically seen items and chronological periods. `outputs/session_gap_sensitivity.csv` resegments the data with 15/30/60-minute gaps while keeping the 30-minute model and policies frozen; this supplemental replay uses a documented approximate 100,000-view test budget. These analyses do not establish generalisation beyond this dataset.

The adaptive internal penalty uses the number of available ranked candidates, but it does not inspect each policy cache for already resident candidates before choosing a depth. Issued-request utility is always calculated from actual cache insertions; the controller score is a heuristic approximation.

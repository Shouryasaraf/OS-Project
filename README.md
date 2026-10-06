# Adaptive product-page prefetching on real clickstreams

This branch replaces the earlier disk-I/O prototype with a reproducible web-prefetch experiment. It uses [RetailRocket's public Kaggle e-commerce events](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), version 2, under its stated CC BY-NC-SA 4.0 license. Training, validation, replay, and the demonstration use recorded visitor events—not generated traffic. Small constructed examples in `tests/` test code behavior only.

## What the system does

After a visitor views a product page, the system retrieves plausible next products from older clickstream transitions. A logistic model scores those candidates using only known history. The controller chooses how many, if any, to prefetch into that visitor session's browser cache. A replay compares this adaptive controller with no prefetch, fixed depths, and a confidence gate. A hit is a later product view already in that session's cache; issued but unused prefetches are counted as waste.

There is no measured network latency, page size, or response cacheability in this dataset. Results are object-cache simulation outcomes, **not page-load speedups**. Prefetch costs in the report are sensitivity assumptions, not measured bytes or energy.

## Run the complete pipeline

Python 3.10+ is required. In PowerShell, from the repository root:

```powershell
python -m pip install -e .
webprefetch run
webprefetch demo
python -m unittest discover -s tests -v
```

`run` downloads and verifies the pinned `events.csv` if absent, preprocesses it, writes EDA, trains and tunes the ranker, selects controllers on validation data, and evaluates frozen selections on the held-out test period. It takes several minutes and may need around 1 GB of working memory. The 94 MB raw CSV and derived data are ignored by Git. A previously downloaded matching CSV is reused after SHA-256 verification.

Stage-by-stage commands are `webprefetch download`, `prepare`, `eda`, `train`, `benchmark`, and `demo`. The default benchmark replays the first 50,000 validation views and first 100,000 test views, both in time order. To change the replay limits, pass `--validation-views N --test-views N` to `benchmark` or `run`; these must not exceed their split sizes. All split sizes and cutoffs are recorded in `data/processed/profile.json`.

## Pipeline and outputs

1. Download the pinned Kaggle file and check the exact CSV SHA-256 and row count.
2. Sort by visitor/time, remove exact duplicates, make 30-minute sessions, and derive page-view sequences.
3. Write EDA and four chronological partitions: transition history 40%, ranker training 20%, validation 20%, test 20%.
4. Learn top source-to-next-page candidates plus popular fallback from history only; create causal context features.
5. Tune logistic regularization on validation candidate log loss and freeze the ranker.
6. Tune fixed, gated, and adaptive prefetch choices on validation cache replay for four stated cost scenarios.
7. Replay the frozen policies on a later test period with independent per-session caches of 4, 8, and 16 objects; estimate paired session-bootstrap intervals.
8. Inspect the report and `demo.json`, which walks through the first held-out session with at least five views (selected without looking for a success).

The main deliverables are `outputs/eda.md`, `outputs/report.md`, `outputs/validation_metrics.csv`, `outputs/test_metrics.csv`, `outputs/comparisons.csv`, and `outputs/demo.json`. Trained artifacts are in `models/`. [The project guide](docs/PROJECT_GUIDE.md) explains the architecture, reproduction steps, results, and limitations; [the implementation plan](docs/IMPLEMENTATION_PLAN.md) records stage gates.

## Evidence and caveats

The ranker cannot predict an item missing from its candidate set. This makes candidate coverage an important ceiling, reported separately from cache hits. The benchmark compares identical request streams, retrieval candidates, cache rules, and capacities across policies. Controllers use current and prior information only; next-item labels are used for training or scoring after the event. Model and controller selection never use test data. We do not assume an adaptive win: the report shows both positive and negative deltas against the validation-selected fixed baseline, under each cost assumption.

# Adaptive product-page prefetching on RetailRocket clickstreams

<!-- doc-numbers: off (the pivot branch's README; figures from the pivot run and audit/ measurements) -->

> **Location note (master branch).** This is the `pivot` branch's README,
> moved here when the web study joined `master`. Paths in it are the pivot
> layout. On `master` the web CLI writes to `outputs/web/` and `models/web/`
> (not `outputs/` and `models/`, which belong to the disk study); the pivot
> run's committed outputs are kept verbatim in
> [`results/pivot_original/`](../../results/pivot_original/), and the tests are
> `tests/test_web_*.py`. Numbers quoted below are from that pivot run and have
> not been regenerated on `master`.

This project replays real RetailRocket product-view sessions and simulates a per-session object cache. It learns candidates from the oldest time partition, trains a logistic ranker on the next partition, selects controllers on validation, and evaluates frozen choices on the newest test period. Its outcomes are simulated cache hits and speculative requests. **No page-load speedup, byte cost, or network latency is measured.**

The source is RetailRocket's [e-commerce events dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), version 2, labeled CC BY-NC-SA 4.0. The pipeline verifies the exact `events.csv` SHA-256. [NOTICE.md](../../NOTICE.md) describes attribution and handling of derived artifacts.

## Run

Python 3.10+ is required. From the repository root:

| Action | Bash | PowerShell |
|---|---|---|
| Install | `python -m pip install -e .` | `python -m pip install -e .` |
| Full pipeline, including gap sensitivity | `webprefetch run` | `webprefetch run` |
| Tests | `python -m unittest discover -s tests -v` | `python -m unittest discover -s tests -v` |
| Serve interactive demo | `webprefetch serve` | `webprefetch serve` |
| Export bounded static demo | `webprefetch export-demo` | `webprefetch export-demo` |
| Print recorded walkthrough | `webprefetch demo` | `webprefetch demo` |

`webprefetch serve --host 127.0.0.1 --port 8765` opens the interactive local demo at
`http://127.0.0.1:8765/`. The server loads the generated model and prepared test views,
then provides `GET /api/sessions` (with `offset`, `limit`, or random `seed`),
`GET /api/session/{id}`, `POST /api/step`, and `GET /api/summary`. Session IDs are
held-out test segments; visitor IDs are displayed only as short one-way tokens. A step
response includes scored candidates, the chosen depth, cache state, the next observed
item, and cumulative hit and request accounting. Changing the cost replays the adaptive
decision rule with that hypothetical penalty; the benchmark table keeps its frozen
validation-selected policies.

`webprefetch export-demo` writes a bounded 20-session `demo/export.json`. To view it
without the Python API, run `python -m http.server 8000` from the repository root and
open `http://127.0.0.1:8000/demo/`. The static export contains three precomputed policies
at capacity 8 and cost 0.01; other capacities, costs, confidence gate, and custom paths
require `webprefetch serve`. The “try a sequence” mode uses real historical product IDs
and the saved ranker but is explicitly hypothetical; there is no observed next-view label
for a user-authored path. The demo ships with the source and generated artifacts, and
uses no external frontend dependencies.

Individual stages use the same syntax in Bash and PowerShell:

```text
webprefetch download
webprefetch prepare
webprefetch eda
webprefetch train
webprefetch benchmark
webprefetch sensitivity
```

`run` regenerates `data/processed/`, trained `models/`, and `outputs/`. It uses the complete validation and test splits by default, with complete visitor-session segments. For a **smoke run only**, either shell can use `webprefetch benchmark --validation-views 1000 --test-views 2000`; it overwrites the benchmark outputs, so run the full command again before citing results. Zero for either limit means the full split. `prepare --session-gap-minutes 15` changes the primary segmentation and requires retraining and rebenchmarking; `sensitivity` instead varies segmentation with the frozen 30-minute model for a controlled, bounded diagnostic.

If download is unavailable, obtain the same version-2 `events.csv` from the linked Kaggle dataset and put it in `data/raw/events.csv`. Its checksum is still enforced. Never substitute generated request streams for the research data. The source CSV and processed views are intentionally excluded from Git; the uploaded ZIP may include them for reproduction.

The pre-fix default pipeline in this container completed in **602.698 s** with peak child RSS **1,004,056 KiB** (`audit/phase1_measurement.txt`; Python 3.12, Linux). This measured the old 50,000/100,000-view default. The full-split pipeline measured **2,979.459 s** and **4,186,944 KiB peak child RSS** with `OPENBLAS_NUM_THREADS=1`; see `audit/phase2_measurement.txt`.
A separate fresh source checkout installed the CLI and completed the same full workflow
in **1,015.574 s**, peaking at **4,231,668 KiB**. Its comparison results matched to
floating point precision; see `audit/clean_checkout_measurement.txt`.

## Study design

Sessions break after 30 minutes of inactivity and also at chronological partition boundaries. Historical transitions and popularity come only from split 0; logistic examples come from split 1; regularization and controller settings are chosen in split 2. The split 3 test period cannot choose settings. Source-popularity ties are ordered by item ID, and bootstrap resamples whole sessions with paired policy outcomes and seed 42.

Each policy shares the same candidate list and cache rules. No prefetch, fixed depths, confidence gate, and adaptive depth are compared under assumed issued-request costs. The top-1 transition table is a ranker-free baseline. Context ablation and histogram gradient boosting are diagnostics, not replacement selections. Candidate coverage is a ceiling for immediate next-item retrieval among eligible pairs; ordinary cache revisits can still hit without candidate retrieval. Logistic probabilities are candidate-level outputs, not a normalized distribution over next items.

Read `outputs/report.md` with `comparisons.csv`, `subset_vs_full.csv`, `break_even.csv`, `calibration.csv`, `cohorts.csv`, `session_gap_sensitivity.csv`, and `models/training_summary.json`. The comparison table includes the coverage ceiling, paired intervals, positive, negative and inconclusive results. See [project guide](PROJECT_GUIDE.md), [implementation plan](IMPLEMENTATION_PLAN.md), and [audit](../../audit/GRILL.md) for assumptions and fixes.

`outputs/session_parity.json` is a batch replay reference for one held-out session.
`tests/test_demo_parity.py` checks that the interactive engine gives the same complete
session counts and policy depths; the HTTP routes were also exercised on loopback.

## Limits

RetailRocket contains anonymized events, without response size, cache directives, download completion, or real page-load latency. The simulated cache treats objects as equal size and prefetches as instantaneous. A single retailer's clickstream does not establish results for other websites. Break-even costs are hit-equivalent assumptions per unused prefetch, not observed financial or bandwidth costs. A live-browser experiment and another dataset are required to test generalisation and speedup claims.

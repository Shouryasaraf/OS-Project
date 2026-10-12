# Phase 1 grill: RetailRocket object-cache experiment

Scope: supplied ZIP, including the working tree (which already has uncommitted changes), model artifacts, outputs, configs, code and tests. Findings below refer to the pre-fix snapshot. Research outcomes are simulated cache-object hits and assumed request costs.

## Findings, ordered by severity

| ID | Severity | Finding | Evidence | Proposed fix | Verification method | Status |
|---|---|---|---|---|---|---|
| G01 | major | Chronological partitions cut through visitor sessions. Context and session IDs straddle boundaries; although next-item labels crossing a cutoff are masked, the evaluation unit is not a contained session. | `src/webprefetch/data.py:89-111` creates sessions before time-based splits; command `groupby(session_id).split.nunique()` on supplied prepared views counted **82** cross-partition sessions. | Split sessions at each partition boundary and recompute within-segment context and labels; retain causal cart count. | Constructed boundary fixture plus zero cross-partition sessions on regenerated data. | fixed: `test_partition_breaks_session_and_clears_cross_period_labels`; regenerated views have zero cross-partition sessions |
| G02 | major | Replay prefixes can truncate a session, but eligibility and candidate ceiling still include its next view outside the replay. This biases the ceiling and makes the unit dependent on an arbitrary row limit. | `src/webprefetch/simulator.py:116,127-143` truncates rows yet uses `row.eligible`/`row.next_item`; command on supplied views found the 50,000 validation and 100,000 test prefixes end inside a session. | Align replay limit to full sessions or explicitly mask truncated transitions. | Fixture where next view is just outside limit; compare prefix/full counts. | fixed: `test_prefix_excludes_partial_sessions_and_next_label` |
| G03 | major | Default controller selection and final evaluation cover chronological prefixes, so the report may not represent the held-out periods. | `src/webprefetch/benchmark.py:49-50,63-64,82-83`; `data/processed/profile.json` gives 532,843 validation and 532,844 test views versus default 50,000/100,000. | Run full split selection/evaluation and report prefix-versus-full comparison without using test to change selections. | Full benchmark and frozen subset comparison. | fixed: `outputs/subset_vs_full.csv`, `outputs/comparisons.csv`; full splits 532,843/532,844 views |
| G04 | major | Missing ranker isolation: no top-1 transition baseline, context ablation, or stronger-model diagnostic. The observed results cannot identify whether retrieval, ranking, or controller limits performance. | `src/webprefetch/benchmark.py:55-58` only constructs fixed/adaptive/gate policies with one scorer; `src/webprefetch/model.py:111-118` tunes logistic C only. | Add history top-1, context-disabled logistic, and gradient-boosting diagnostic trained on train and evaluated on validation/test without replacing the frozen controller. | Focused ranking test and validation/test diagnostic tables with coverage ceiling. | fixed: `test_transition_baseline_ignores_ranker`, `outputs/transition_baseline.csv`, `outputs/model_diagnostics.csv` |
| G05 | major | Probability calibration is unmeasured, despite the controller summing probabilities to choose depth. | `src/webprefetch/simulator.py:30-35`; `src/webprefetch/model.py:109-147` reports log loss/ranking but no reliability bins or ECE. | Save reliability data and ECE from held-out validation/test candidates; state that candidate-level probabilities are not a normalized next-item distribution. | Diagnostic file plus bin-count/weighted-ECE test. | fixed: `test_calibration_weighting_and_waste_threshold`, `outputs/calibration.csv` |
| G06 | major | No session-gap/cache sensitivity beyond the single 30-minute rule and three capacities; no break-even cost curve. | `src/webprefetch/data.py:16,90-92`; `src/webprefetch/benchmark.py:11,80`; `outputs/benchmark.json` enumerates only capacities 4/8/16 and four costs. | Run 15/30/60-minute preprocessing/retraining or explicitly constrain sensitivity and measure it; extend capacities and derive validation break-even threshold with paired session bootstrap. | Generated sensitivity/break-even CSV and arithmetic fixture. | fixed: `outputs/session_gap_sensitivity.csv` (bounded, frozen 100,000-view budgets), `outputs/test_metrics.csv` (2/32 capacities), `outputs/break_even.csv` |
| G07 | major | No analysis of cold-start items, drift, or seasonality across the test period. | `src/webprefetch/benchmark.py:52-120` aggregates by capacity/cost only; `outputs/report.md` has no cohort/time breakdown. | Report item-history status and chronological test slices, with eligible-pair denominators; avoid extrapolation. | Generated cohort/time table checked against full test counts. | deferred (annual seasonality cannot be estimated from the August/September test period); cold items and monthly drift fixed in `outputs/cohorts.csv` |
| G08 | major | A page demo requires a real step-level replay API and parity check, currently absent. | `src/webprefetch/cli.py:84` only has download/prepare/eda/train/benchmark/demo/run; `outputs/demo.json` is a single selected walkthrough; `tests/test_workflow.py` has five tests and no HTTP parity test. | Implement server and static export through shared replay/cache transitions. | Full session HTTP versus batch counters; browser exercise. | fixed: `demo/index.html`, `demo/export.json`, four HTTP routes, `tests/test_demo_parity.py`, `outputs/session_parity.json`; browser screenshot blocked by inaccessible loopback/browser binary |
| G09 | minor | Cache and accounting implementation handles existing objects and speculative evictions, but current tests do not cover duplicate prefetch, repeated demand, end-of-session waste, or demand eviction. | `src/webprefetch/simulator.py:75-106` skips resident objects and counts eviction/use; `tests/test_workflow.py:37-73` has one eviction fixture. | Add constructed cache accounting tests and explicit per-step waste. | Fixture asserting issued/useful/unused/hits/evictions. | fixed: `test_duplicate_current_eviction_and_unused`; `test_multi_capacity_replay_matches_single` |
| G10 | minor | Adaptive depth charges the nominal K even if fewer candidates exist and even though cached candidates are skipped. Its internal objective can diverge from issued cost. | `src/webprefetch/simulator.py:31-35,102-106,149-150`. | Base policy value on actual available/issuable candidates where practical, and state this controller approximation. | Sparse/resident-candidate fixture. | fixed: `test_adaptive_depth_uses_available_count`; resident-item penalty remains a heuristic, documented |
| G11 | minor | Determinism is mostly specified, but tie ordering in global popularity depends on pandas `value_counts` implementation; split cutoffs depend on full timestamp distribution. | `src/webprefetch/model.py:55-58` uses frequency order without an explicit item-ID tie rule; `src/webprefetch/data.py:106-110`; bootstrap fixed seed in `simulator.py:165-179`. | Explicit popularity tie break and document chronological quantile cutoff construction. | Repeated index fixture with tied items. | fixed: `test_popularity_ties_are_item_id_ordered` |
| G12 | minor | Documentation is PowerShell-centric and runtime/memory estimates differ from the measured 602.698 s / 1,004,056 KiB peak RSS in this environment. | `README.md:13-25` and `docs/PROJECT_GUIDE.md:45-60`; pipeline measurement below. | Add bash commands and observed wall time/RSS with environment and limit. | Execute shell commands and record measurement. | fixed: Bash/PowerShell README commands; `audit/phase1_measurement.txt` and `audit/phase2_measurement.txt` |
| G13 | minor | Dataset license is named but downstream artifacts have no clear NOTICE or sharing guidance. | `README.md:3` cites CC BY-NC-SA; `rg --files` shows no `LICENSE`/`NOTICE` in supplied project; `models/` and `outputs/` contain derived data. | Add source credit, license URL, change notice, and conservative sharing guidance for data-derived artifacts. | Inspect README/NOTICE and artifact handling. | fixed: `NOTICE.md` and README attribution section |
| G14 | minor | Single dataset and immediate equal-size object simulation cannot establish cross-site generalisation or actual page-load speedups. | `README.md:9`, `docs/PROJECT_GUIDE.md:68` already acknowledge simulation; only `events.csv` is read by `data.py:54-65`. | Make limitations prominent in report/demo; defer independent dataset and live latency study. | Report and banner inspection. | deferred: independent dataset and live latency study are outside this object-cache experiment; README/report limitations |

## Leakage and validity disposition

- **Future events and test use:** candidate transitions and popularity use split 0 (`model.py:45-58`); training rows use split 1 and model regularization selection uses split 2 (`model.py:73-118`); controllers are selected using split 2 before split 3 replay (`benchmark.py:52-83`). The feature vector uses current/previous context and history index (`model.py:61-70`), not `next_item`; the label appears in supervised targets (`model.py:90,102`) and in later scoring (`simulator.py:131-142`). No direct test-label feature or test-driven selection was found. Splits are defined with whole-dataset timestamp quantiles (`data.py:106`), which fixes partition sizes in retrospect, but does not fit the predictive table/scaler to test labels.
- **Boundary:** G01 and G02 qualify the chronological claim. The same session may have views in adjacent partitions, and a prefix may end before its labeled next view.
- **Normalization:** `StandardScaler` is fitted inside train-split model pipelines (`model.py:111-115`). The candidate index and fallback are split-0-only. No later-period fit found.
- **Hit/waste:** `Cache.demand` counts any resident object as a hit, distinguishing useful speculative hits; `prefetch` skips already cached/current items because demand is inserted first (`simulator.py:89-106,146-150`). Eviction counts unused speculative objects (`simulator.py:75-86`). `unused=issued-useful` (`simulator.py:55-59`) includes both evicted and still resident unused objects. More tests needed (G09); ordinary demand-cache reuse contributes to hits and should not be called prefetch benefit.
- **Bootstrap:** `paired_bootstrap` resamples sorted session IDs once and uses both policies' deltas on the same draw (`simulator.py:165-179`), with a fixed seed; genuinely paired at the replay's session unit. Session boundary/prefix issues still matter.

## Evidence gap classification

The following classification records the **pre-fix Phase 1 snapshot**; current dispositions are in the finding status column and the measured sections below.

| Question | Disposition | Evidence |
|---|---|---|
| Prefix-only replay | partially handled | Explicitly disclosed in README, but full-split comparison absent; G03. |
| Top-1, ablation, stronger model | missing | G04. |
| Session gap and broader cache sizes | missing | G06. |
| Four costs and break-even | partially handled | Four scenarios exist; continuous threshold missing; G06. |
| Cold start, drift, seasonality | missing | G07. |
| Calibration/reliability/ECE | missing | G05. |
| Single-dataset generalisation | partially handled | Limitation stated, no independent dataset; G14. |
| Replay/cache tests | partially handled | Five fixtures pass, accounting edge cases missing; G09. |
| Deterministic seeds/order | partially handled | Fixed bootstrap seed and stable sort; tied popularity unspecified; G11. |
| Bash commands | missing | G12. |
| License notice | partially handled | License named, handling absent; G13. |

## Phase 1 commands and measurements

- `python -m unittest discover -s tests -v` in the supplied tree: five tests passed, exit 0.
- Full unmodified pipeline in an isolated copy with `PYTHONPATH=src python -m webprefetch run`: exit **0**, elapsed **602.698 s**, peak child RSS **1,004,056 KiB**, measured with `time.monotonic()` and `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`; see `audit/phase1_measurement.txt` and `audit/phase1_pipeline.log`. The isolated copy prevented Phase 1 from changing source/outputs in this working tree.
- `groupby(session_id).split.nunique()` on supplied prepared data: 82 cross-partition sessions; the default 50,000 validation and 100,000 test chronological prefixes end inside sessions.

## Claims in the README/report that the evidence does not yet support

- The headline eight-object cost-0.01 gain (`docs/PROJECT_GUIDE.md:65` and `outputs/report.md:20`) is a **prefix-only** result, not a full-test estimate; G03.
- Candidate coverage in `outputs/report.md:32` is called a strict upper bound on next-page prefetch accuracy, but the reported coverage includes a target outside the truncated replay prefix; G02. It is also not an upper bound on all demand-cache hits, which include revisits.
- “Current and prior information alone” is true for the scorer, but complete-session partition isolation is not established; G01.
- “Several minutes / around 1 GB” should be replaced with the observed **602.698 s** and **1,004,056 KiB peak RSS** on this container; G12.
- No evidence supports universal cost superiority, cross-dataset generalisation, or measured page-load improvements. The README already disclaims the latter two.

## Phase 2 measured regeneration and output checks

- `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -u -m webprefetch run` completed with exit 0, **2,979.459 s**, peak child RSS **4,186,944 KiB** (`audit/phase2_measurement.txt`). The installed CLI and a fresh source tree are checked separately.
- `python -m unittest discover -s tests -v`: **14 tests passed**.
- Regenerated prepared views: zero cross-partition sessions; all `eligible` values match the presence of an in-segment next item. Full validation/test denominators are 532,843/532,844. All 20 comparison rows have intervals and a common 9.201686% candidate coverage@8 ceiling. Five capacities have the same test view denominator.
- Eight-object cost-0.01 utility delta is +0.003576 per view, paired interval [+0.003309, +0.003849], versus the old prefix +0.003475. Eight-object zero-cost delta remains negative; two-object cost-0.05 delta is negative. See full table and no-prefetch comparisons in `outputs/report.md`.
- The 15/30/60-minute sensitivity uses the frozen 30-minute model/policies on complete-session prefixes near 100,000 views. It is a controlled segmentation check, not a newly trained alternative.

## Gate 2 fresh-checkout reproduction

- A fresh source-only tree received only the validated raw `events.csv`; its own editable install ran `webprefetch run` to completion with **exit 0**, elapsed **1,015.574 s**, peak child RSS **4,231,668 KiB** (`audit/clean_checkout_measurement.txt`, `audit/clean_checkout_pipeline.log`). It regenerated data, models, all benchmark outputs, and gap sensitivity. The difference in elapsed time from the first full run reflects execution conditions; both used the same default full-split workflow.
- The 20 comparison rows and eight model diagnostic rows agree with the main outputs within at most **4.44e-16** and **7.12e-17** respectively; policy names match exactly. All 45 test and 16 validation metric rows, six cohort rows, and three gap sensitivity rows match exactly.
- The clean checkout was run from a snapshot taken before the Phase 3 demo code. The demo adds no changes to `run`, and its parity test passes against the regenerated main artifacts.

## Phase 3 demo verification

- `python -m unittest discover -s tests -v`: **16 tests passed**, including a real held-out 15-view session where the API replay matches the independently generated batch replay on all totals and selected depth counts (`outputs/session_parity.json`).
- `webprefetch export-demo` produced 20 bounded real sessions and a 172 KiB JSON payload. `node --check` accepted the frontend script.
- Loopback HTTP requests returned 200 for the HTML page, sessions listing, held-out session, summary, and POST step. The step response for session 51147 matched the batch reference: 15 views, 10 hits, 4 issued, 4 unused, 1 unused eviction.
- Browser screenshot and visual interaction remain unverified: there is no installed local browser binary, and the cloud browser returned `ERR_CONNECTION_REFUSED` for the local loopback server. The static HTML/JS and API are available for manual exercise.

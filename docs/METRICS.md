# Metrics: one set of definitions for both studies

Every number in `results/` and in the paper is one of the quantities below.
Each entry says what is counted, what the denominator is, where the code
lives, and what the number does **not** mean. Read this before quoting any
figure, and see `docs/REPRODUCE.md` for the files that hold them.

Notation: a *request* is one disk I/O request (disk study) or one product
view (web study). *Issued* counts speculative fetches actually sent; a
prefetch of something already resident is a no-op and is not issued.

## Cache-level metrics (both studies)

| Metric | Definition | Code |
|---|---|---|
| **Block hit ratio** (disk) | read hits / read blocks, both counted in 512-B blocks. A 64-block read that hits on 60 blocks counts 60 hits. | `simulator.Metrics.hit_ratio` |
| **Hit ratio** (web) | views served from the per-session browser cache / views | `webprefetch.cachecore.Counters` |
| **Precision** | useful prefetches / issued prefetches. A prefetch is useful when the object or block is demanded while still resident and still unused. | `Metrics.prefetch_precision`, `Counters.row` |
| **Wasted share** | 1 - precision: issued prefetches never used, including those still resident at the end of the replay | `Metrics.unused_prefetches` |
| **Pollution** (web) | demand objects evicted by a prefetch, per view (`demand_evicted_by_prefetch_per_view`) | `cachecore.Cache._insert` |
| **Oracle re-access recall** (disk, diagnostic) | useful / (useful + read misses on blocks read again later). Uses future knowledge for **scoring only**; it can be high while hit ratio is low, so it is never reported alone. Not computed on streamed runs. | `Metrics.prefetch_recall` |
| **Full-hit request fraction** (real device) | requests whose every block was a hit; the unit a requester's latency depends on | `realio.report` |

A block hit ratio is **not device work** and **not latency**. Every disk
hit-ratio claim is paired with the two device-work counters below.

## Device work (disk study, D18)

| Metric | Definition | Code |
|---|---|---|
| **`device_ios`** | demand I/Os + prefetch I/Os - mergeable prefetch I/Os. A demand I/O is one maximal run of contiguous missed blocks inside one request; a prefetch I/O is one maximal run of contiguous blocks newly prefetched after one request; a prefetch run that starts where a missed request ends would be merged into that read by the block layer and is counted once. | `Metrics.device_ios` |
| **`blocks_moved`** | blocks read from the device: demand-missed blocks plus prefetched blocks | `Metrics.blocks_moved` |
| **device-I/O ratio, blocks-moved ratio** | a policy's count divided by `none`'s on the same trace slice and capacity; the tables report the mean of per-trace ratios | `analysis.py` |
| **Device time** (model) | `T = device_ios * t_pos + blocks_moved * t_xfer`, with `t_pos` and `t_xfer` from a device calibration | `simulator.IOCostModel` |
| **kappa** (device) | `t_pos / t_xfer`: how many block transfers one positioning costs on a device, from `results/realio/device_*.json` | `IOCostModel.kappa` |
| **kappa\*** (policy) | `dBlocks / -dIOs` against `none`: the device kappa above which the policy's modelled device time drops below `none`'s. `inf` means it never does (it issues at least as many I/Os); 0 means it wins at every ratio. | `IOCostModel.kappa_star`, `results/disk/kappa.csv` |

`IOCostModel` is a device-work model: it has no queueing, no overlap of a
prefetch with demand reads and no completion timing. Those effects are what
the real-device testbed measures.

## Cost and utility: one objective for both studies

`prefetch_common.utility` defines

    U(c) = (hits - c * issued) / requests

where `c` is the cost of one speculative fetch in units of one hit. A policy
is worth running at cost `c` exactly when its `U(c)` beats no prefetching.

- **Disk.** The legacy `LatencyModel(hit, miss, prefetch)` charges a mean
  cost per 512-B block. Its modelled cost per block rearranges to
  `cost = miss - (miss - hit) * U(c)` with `c = prefetch / (miss - hit)`,
  so minimising modelled cost and maximising `U(c)` pick the same policy.
  That `c` is `LatencyModel.break_even_precision()`. Its values are written
  in `results/disk/PREREGISTERED.md`: 0.526 for the assumed 5/100/50 us
  model and 1.0101 for the measured shape (hit = 1% of a miss, prefetch =
  one miss). At `c > 1` a prefetcher must gain more than one hit per block it
  prefetches, which no prefetcher can. That follows from charging every
  block as a full device read; it is **not** a property of hardware (D18).
- **Web.** There was never a cost model; `c` is an **assumed** number of
  hit-equivalents. `results/web/policy_metrics.csv` carries `U` at every `c`
  in `utility.UTILITY_CS`, and the headline comparisons use `c` = 0.01.
- **c\*** (per policy): the `c` at which a policy ties `none` (disk) or
  `fixed0` (web), `(hit_ratio_p - hit_ratio_n) / (issued_rate_p - issued_rate_n)`.
  The policy beats the baseline for `c < c*`; a negative `c*` means it loses
  at every cost. `utility.c_star`.
- **Envelope**: at each `c`, the best `U(c)` over a set of policies
  (`utility.envelope`); paper figure F8 draws one per study.

## Ranking metrics (web study, diagnostic only)

| Metric | Definition |
|---|---|
| **coverage@K** | share of eligible next views (the session has a next view) whose item is in the scorer's top K; `results/web/freshness.csv` reports it per candidate index (split 0, splits 0-2, online) |
| **hit@1, MRR@8** | the usual ranking scores of the next view |
| **novel hit@1, novel recall@8** | the same, counted only on next views that are **not** an item seen earlier in the session (P14). A ranker can score well on hit@1 by predicting back-navigation, which the cache already holds; the novel versions measure what a prefetch can add. |

Ranking scores are not cache utility, so rankers are compared by the cache
outcome of the same policy (for the GRU against online transition counts,
`results/web/decomposition.csv`, row `sequence_model`). The GRU's history
mask never ranks an item already in the window, so its `hit_at_1` and
`mrr_at_8` are not comparable with an unmasked ranker's; compare sequence
models on the `novel_*` columns.

## Classifier metrics (classifier study, after E1-E6)

| Metric | Definition | Where |
|---|---|---|
| **synthetic accuracy, macro-F1** | share of synthetic test windows given their generating class, and the mean of the four per-class F1 scores; mean over independent (train, test) seed pairs. **Synthetic only**: real traces carry no workload label | `results/classifier/synthetic_eval.csv` |
| **ECE, Brier** | expected calibration error (15 equal-width confidence bins) and the multiclass Brier score of the predicted probabilities; descriptive | same |
| **predict_us, train_ms** | median single-window predict time and fit time on a contended shared host; never a cost | same |
| **agreement with GNB** | share of a trace's windows a classifier gives the same class as the shipped GNB; with the class shares it shows how differently a router routes, independent of the hit ratio | `results/classifier/router_runs.csv`, `router_summary.csv` |
| **CV score (hindsight)** | per held-out trace, the share of hindsight headroom over `none` a router's choices capture under the trajectory approximation, `(S(pred) - S(none)) / (S(oracle) - S(none))`, averaged over traces; selection only | `results/hindsight/selection.csv` |
| **headroom captured** | `(routed - vcf) / (oracle - vcf)` on slice 2; empty when the trajectory oracle does not beat VCF (amendment A3); descriptive | `results/hindsight/summary.csv` |
| **U(0.5) per block (`u05_per_block`)** | `(hits - 0.5 x prefetches issued) / read blocks`, the hindsight `u0.5` objective | `results/hindsight/rows.csv` |
| **balanced accuracy** | mean of the two class recalls on E7's `seq`/`not_seq` window label; undefined when the truth holds one class | `results/msrc_samples/classifier_metrics.csv` |

The hindsight `oracle_*` rows replay the slice's own hindsight labels: a
reference under the trajectory approximation, never a policy and not a true
ceiling (it does not beat VCF at 64 MiB). The web rankers (`results/web_rankers/`)
use the ranking metrics above, over the same eligible next views as E4.

## Measured latency (real systems)

| Metric | Definition | Where |
|---|---|---|
| **mean / p50 / p99 request latency** (disk) | wall time from issuing a demand read until all its blocks are available, per request, on a real device | `results/realio/realio.csv` |
| **delta vs none** (disk) | mode's mean latency minus `none`'s in the same repeat, mean over repeats, 95% Student-t interval (df = repeats - 1) | `realio_summary.csv` |
| **late hit** (disk) | a hit on a block whose prefetch had been issued but had not completed; the request waits for it | Backend B and the `cachestat`/`RWF_NOWAIT` probe of Backend A |
| **pred_err** (disk) | (hit-ratio-only latency prediction - measured) / measured, the prediction built from the run's full-hit fraction and `none`'s hit and miss latencies | `realio.report` |
| **k_eff** (disk) | the `prefetch_multiple` at which `LatencyModel`'s per-block cost reproduces the measured mean; below 1, a prefetched block cost the requester less than a demand-read block **in that run** | `realio.report` |
| **Spearman rho** (disk) | rank correlation across modes, per cell, of simulated hit ratio (and of simulated `IOCostModel` device time) with measured mean latency | `realio_summary.csv` |
| **mean wait per view** (web) | measured time from a view's scheduled start until its object is available, over real HTTP on loopback with an emulated network | `results/live/live_metrics.csv` |
| **model-ideal wait** (web) | 0 on a hit (ready or in flight), `rtt + size/bandwidth` on a miss: what an analytic simulator would predict | `live/analyze.py` |
| **bytes vs base** (web) | bytes fetched per view divided by `fixed0`'s | `live_metrics.csv` |

All latency figures are labelled **measured**, and each is stated with its
device, backend, think-time model, trace prefix and load. A simulated hit
ratio is never written as a latency gain.

## Intervals and decision words

- **Disk (E1):** paired bootstrap over traces and a cluster bootstrap over
  corpora with twin families down-weighted (`prefetch_common.stats`). The
  words "better", "worse" and "no difference detected" follow
  `results/disk/PREREGISTERED.md` section 5: better or worse only when the
  paired **and** the cluster interval lie on one side.
- **Disk real device (E3):** Student-t interval over paired repeats;
  `results/realio/PREREGISTERED.md` section 4.
- **Web (E4, E5) and live (E6):** paired session bootstrap of the ratio
  estimator `sum(delta) / sum(views)`; `results/web/PREREGISTERED.md` and
  `results/live/PREREGISTERED.md`.
- No interval is reported without its `n` (traces, sessions or repeats) and
  its seed or repeat count.

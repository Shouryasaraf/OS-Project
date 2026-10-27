# Architecture

This repository holds two prefetching studies that share one cache/prefetch
abstraction, one objective and one results contract:

- the **disk study** (`src/adaptive_prefetch`): block-level read-ahead on
  public block I/O traces, a causal LRU simulator, and a real Linux testbed
  that replays the simulator's decisions on a real device;
- the **web study** (`src/webprefetch`): product-page prefetching on the
  RetailRocket clickstream, a per-session browser-cache simulator, and a real
  HTTP replay that reproduces the simulator's counters exactly;
- shared code (`src/prefetch_common`): the utility `U(c)`, the bootstraps,
  provenance records and the results schemas;
- the paper pipeline (`paper/`) and the reproduce dispatcher (`reproduce.py`,
  `repro/`).

Metric definitions are in `docs/METRICS.md`; commands and runtimes in
`docs/REPRODUCE.md`; the decision history in `docs/DECISIONS.md` (disk) and
`docs/web/WEB_DECISIONS.md` (web).

## Design principles (both studies)

1. **Causal replay.** A decision at request (or view) *i* sees only requests
   up to *i*. Future knowledge is used only to score (oracle recall, the
   Belady/MIN reference), never to decide.
2. **Equal rules for every policy.** All policies of a study see the same
   stream, the same cache capacity and eviction rule, and the same warm-up.
   A difference in outcome is the policy's, not the harness's.
3. **Simulate first, then measure the gap.** The simulators assume a
   prefetch completes instantly and costs nothing. The real-device testbed
   and the live HTTP replay execute the simulators' own decisions with
   asserted parity, so what they add is exactly the timing the simulators
   leave out.
4. **Honest labels.** Synthetic data has ground truth; real traces do not. A
   trained model is frozen on real traces, scored only on data it was not
   trained or selected on, and compared against the strongest fair baseline.
5. **Results are a contract.** Every file under `results/` is written through
   `prefetch_common.schemas` (exact columns and types) and is hashed in its
   directory's `provenance.json`. The paper reads only `results/`.
6. **The disk core is stdlib-only.** Everything else is an extra
   (`pyproject.toml`): `[traces]`, `[web]`, `[gpu]`, `[paper]`.

## Data flow

```text
DISK                                              WEB
data/block-traces/*.oracleGeneral.zst             data/raw/events.csv (SHA-pinned)
  trace.iter_oracle_chunks (unit detection,         data.prepare: dedup, 30-min sessions,
  FIU page merge, volume-tag mask)                   4 chronological splits, causal context
        |                                                  |
  Request(timestamp_ms, lba, size_blocks, op)       view rows (session, position, item, next_item)
        |                                                  |
  simulator.replay / replay_stream                  scorers: logistic | trans_s0 | trans_prior
   - per-request predictors (stride, markov,          | trans_online (OnlineIndex) | seq (GRU)
     lstm, linux_ra, learned_depth)                        |
   - window router (adaptive, adaptive_evidence)    cachecore.Policy / FeedbackGate -> depth
   - feedback gate (guard, depth_adaptive,                 |
     correlate) -> read-ahead depth                 cachecore.Cache (per session, LRU,
   - LRUCache (cache_factory)                         speculative-first eviction)
        |                                                  |
  Metrics: hits, prefetches, device_ios,            Counters: hits, issued, useful, pollution
  blocks_moved                                             |
        |                                           benchmark: select on validation per capacity,
  multicorpus (C1-C4) -> sweep.jsonl                  freeze, replay test -> policy_metrics,
  analysis -> tables, deltas, winners, kappa          comparisons, decomposition, freshness
        |                                                  |
  realio: decisions.record_plan -> Backend A/B      live: plan.json + sim.json -> origin +
  on a real device -> realio.csv                      emulated browsers -> live_metrics.csv
        \__________________ results/<study>/ ______________/
                                   |
                  paper/numbers.py, tables/, figures/ -> paper/, presentation/
```

## Disk study (`src/adaptive_prefetch`)

### Module map

| Module | Responsibility | Main entry points |
|---|---|---|
| `trace.py` | `Request` record; loaders for `normalized`, `msr`, `iotta8`, `alibaba`, `revised` and `oracle` (`FORMATS`); `iter_oracle_chunks` and `iter_csv_chunks` streaming; synthetic labelled generators | `load_csv`, `iter_oracle_chunks`, `oracle_profile`, `transition_trace` |
| `corpora.py` | the four corpora, twin families (`TWIN_GROUPS`), `PAPER_SUBSET` (21 stems), trace discovery | `discover_block_traces` |
| `features.py` | 8 stream features per window, plus 4 contextual ones from `WindowContext` (bounded block-age table) | `extract`, `WindowContext`, `dominant_stride` |
| `model.py` | `OnlineGaussianNB` (default) and `OnlineQDA`, with exponential decay | `make_classifier` |
| `baselines.py` | `StridePrefetcher`, `MarkovPrefetcher` (`next_candidates`), and `LinuxReadahead` (cache-aware, `observe`) | |
| `lstm.py` | optional delta LSTM (bucketed deltas, abstention), honest split, CUDA | `train_lstm`, `load_predictor` |
| `depth_model.py` | `learned_depth`: per-segment read-ahead probabilities, thresholds chosen on validation, SHA-bound training sidecars | `train_and_select`, `load_prefetcher`, `DepthPrefetcher`, `PrecomputedDepth`, `RelativeDepth`, `scoring_conflict` |
| `guard.py` | feedback gate: observed precision against break-even sets depth | `GateController`, `PrecisionMonitor`, `CorrelateDetector`, `read_ahead` |
| `simulator.py` | the causal replay loops, `LRUCache`, `CacheView`, `Metrics`, `LatencyModel`, `IOCostModel`, routing | `replay`, `replay_stream`, `_validate_replay`, `_make_predictor`, `MODES_SUPPORTED` |
| `benchmark.py` | one-trace benchmark over `MODES`, pooled rows, drift report, interpretation | `benchmark_dataset`, `aggregate_results`, `analyze_results` |
| `sweep.py` | streamed sweep of a trace collection (`RANKING_MODES`) | `sweep_collection` |
| `multicorpus.py` | E1 driver: configs C1-C4, explicit latency model per row, resumable `sweep.jsonl` | `python -m adaptive_prefetch.multicorpus` |
| `analysis.py` | E1 tables, paired and cluster deltas, winners, `kappa*`, validity checks | `write_analysis`, `checks` |
| `realio/` | real Linux testbed (below) | `python -m adaptive_prefetch.realio` |
| `eda.py`, `pipeline.py`, `training.py`, `artifacts.py`, `report.py`, `datasets.py`, `cli.py` | trace profiling, classifier evaluation regimes, weak-label adaptation of the MSR sample, JSON model persistence, the `main.py` report, dataset lists, the CLI | |

### The replay loop

`replay(requests, capacity, window_size, mode, ...)` and its streamed twin
`replay_stream(chunks, ...)` are the only places a policy runs. For each
request, in order:

1. Demand: every block is touched in the cache (`LRUCache.read`/`write`);
   misses are counted in maximal contiguous runs (`demand_ios`).
2. The candidate source proposes blocks for later requests:
   - **per-request predictors** expose `next_candidates(request)` (`stride`,
     `markov`, `lstm`, `learned_depth`);
   - **cache-aware predictors** (`linux_ra`) are driven through
     `observe(request, missed_blocks, CacheView)`, because the read-ahead
     policy depends on what is resident;
   - **window modes** (`sequential`, `strided`, `deep`, `adaptive`,
     `adaptive_evidence`, the gated modes) use the current window policy and
     `guard.read_ahead`, which always returns exactly `depth` blocks.
3. Prefetch, only once `index >= window_size`: **the first window never
   prefetches**. Blocks already resident are not re-issued; new runs are
   counted as prefetch I/Os (and as mergeable when they continue a missed
   read).
4. When a window completes: refresh the stride, and in `adaptive` mode
   classify it and switch policy for *later* requests. The gated modes ask
   `GateController` for a depth every `policy_interval` requests, feeding it
   the **delta** of useful prefetches since its last decision.

`replay_stream` carries cache, metrics, partial-window state, block ages,
stride and gate state across chunk boundaries and is bit-identical to
`replay` on every counter (pinned for every mode). `_validate_replay` is the
only parameter validator, `_make_predictor` the only predictor factory, and
`_publish_gate` the only place gate state reaches `Metrics`.

### Modes

`benchmark.MODES` and `sweep.RANKING_MODES` (14 modes, this order):

| Family | Modes | What decides |
|---|---|---|
| fixed | `none`, `sequential` (depth 2), `strided`, `deep` (depth `DEFAULT_READ_AHEAD_DEPTH`, or `deep@N`) | nothing |
| per-request | `stride`, `markov`, `lstm` (optional artifact) | address deltas |
| classifier-routed | `adaptive` | a 4-class window label from `OnlineGaussianNB` |
| evidence-routed | `adaptive_evidence` | the window's own contiguity, stride and jumpiness; no model |
| feedback-gated | `guard`, `depth_adaptive`, `correlate` (`GATED_MODES`) | *observed* precision against `LatencyModel.break_even_precision()` |
| kernel policy | `linux_ra` (window ceiling `linux_ra@N`, default 256 blocks) | an emulation of mm/readahead.c's on-demand read-ahead policy: sync window on a sequential miss, async ramp on the marker block |
| learned depth | `learned_depth` (optional artifact) | a trained per-segment model; how far to read ahead |

`OPTIONAL_MODES = ("lstm", "learned_depth")`: without an artifact they give
a `skipped:` row with `None` metrics, never a substitute.

### Cost models

- `LatencyModel(hit_us, demand_miss_us, prefetch_us)`: the legacy per-block
  cost. `measured()` calibrates it from a trace's recorded service times (the
  MSRC `.revised` format only), using the mean. Its break-even is degenerate
  at `prefetch_multiple=1.0` (D11, corrected in D18).
- `IOCostModel(t_pos_us, t_xfer_us_per_block)`: device time from
  `device_ios` and `blocks_moved`, loaded from `results/realio/device_*.json`.
  `kappa_star` gives each policy's break-even device ratio.
- Oracle traces carry no service times, so every multi-corpus row names its
  latency model explicitly (`assumed` 5/100/50 us, and `break_even`
  1/100/100 us for the gated modes) and records its break-even.

### The real Linux testbed (`adaptive_prefetch/realio/`)

| Module | Role |
|---|---|
| `decisions.py` | `record_plan`: runs `replay_stream` once with `RecordingLRU` as `cache_factory` and turns the cache log into a per-request plan (hit/miss of every block, prefetch runs). `ShadowLRU` has `LRUCache`'s exact rules; `verify_plan` replays a plan with no I/O. |
| `layout.py` | `RegionMapper`: packs the trace's 1 MiB regions into a backing file in first-touch order (contiguity kept, empty space dropped; aliasing refused) |
| `backend_odirect.py` | **Backend B**: user-space LRU, `O_DIRECT` demand reads, prefetch on a 4-thread pool, preallocated aligned buffers; raises `CacheDivergence` if a hit/miss differs from the plan; late and ready hits timed |
| `backend_pagecache.py` | **Backend A**: the kernel page cache in a memory cgroup sized to the simulated cache; prefetch as `POSIX_FADV_WILLNEED` in 256 KiB chunks; `FADV_RANDOM` for policy modes, `FADV_NORMAL`/`FADV_SEQUENTIAL` for `kernel_ra`/`kernel_seq`; whole file dropped with `FADV_DONTNEED` before every run |
| `online.py` | policies run inline against Backend A with a shadow `LRUCache`; the gated modes are fed *measured* usefulness |
| `clock.py` | think-time models `closed`, `fixed:US`, `spread:SCALE:CAP_US`, and a precise pacer |
| `calibrate.py` | QD1 `O_DIRECT` probes: `t_pos`, `t_xfer`, `kappa` -> `device_<dev>.json` |
| `devstat.py` | which disk a file is on, foreign I/O share, drive-cache displacement |
| `syscalls.py` | ctypes `cachestat(2)`, `mincore(2)` fallback, timer slack |
| `runner.py`, `report.py` | fresh child process per run, shuffled order, load gate; `realio.csv` and the summary (delta vs `none`, Spearman, `pred_err`, `k_eff`) |

## Web study (`src/webprefetch`)

### Module map

| Module | Responsibility |
|---|---|
| `data.py` | SHA-pinned `events.csv`; dedup; 30-minute sessions broken at split boundaries; four chronological splits; causal view records |
| `model.py` | `CandidateIndex` and `fit_index(views, max_split=0)`; the logistic ranker (`train_ranker`) and `FastScorer` |
| `index.py` | `OnlineIndex`: a transition enters only after its later view's time is strictly before the decision time |
| `scorers.py` | the scorer registry (`SCORERS`: `logistic`, `logistic_novel`, `trans_s0`, `trans_prior`, `trans_online`, `seq`), each fixing its index and replay order; `RankTable` for precomputed rankings |
| `cachecore.py` | **the one browser-cache model**: `Policy` (`fixed`, `gate`, `adaptive`, `adaptive_ca`, `feedback`), `Counters`, `Cache` (per-scope LRU, speculative objects evicted first), `FeedbackGate`; stdlib only |
| `simulator.py` | `replay`, `replay_capacities`, `paired_bootstrap`, calibration and break-even helpers |
| `parallel.py` | session-sharded replay with exact counts; feedback and online-index state replayed in global time order |
| `benchmark.py` | validation search **per capacity**, frozen test replay, comparisons against best fixed / `fixed0` / transition top-1, decomposition, freshness, learning curve |
| `seqmodel.py` | GRU (optionally SASRec) next-view model, chronological protocols, no-pickle artifacts, `SequenceScorer.precompute` |
| `diagnostics.py` | model comparisons, calibration, pair kinds, think time |
| `demo_server.py`, `demo/` | local API and static demo; `adaptive` = the validated policy, `adaptive_cost` = penalty equals cost; `session_parity` |
| `live/` | real HTTP replay (below) |
| `cli.py`, `paths.py` | `python -m webprefetch {download, prepare, eda, train, benchmark, freshness, decompose, learning-curve, parity, serve, export-demo, ...}`; web outputs go to `outputs/web/` and `models/web/` |

### Policies and the replay

For each view of a session, in time order: the view is demanded from the
session's cache (hit or miss), the scorer ranks candidate next items, the
policy chooses how many to prefetch, and the chosen items not already
resident are fetched into the cache. A fixed policy prefetches the top *k*;
`adaptive` picks the depth that maximises predicted hits minus a penalty per
slot; `adaptive_ca` drops resident items first, so it pays only for real
fetches; `gate` prefetches 2 when the top score clears a threshold;
`feedback` issues a candidate only while the observed precision of its
(rank, score bin) bucket beats the cost, with exploration rate `epsilon`.
Every parameter is chosen on validation per capacity and frozen for test.

### The live HTTP replay (`webprefetch/live/`)

| Module | Role |
|---|---|
| `plan.py` | samples test sessions, records per-view decisions through the same `cachecore` objects, and the simulator reference `sim.json`; a plan that disagrees with the simulator is refused |
| `selection.py` | resolves `validated_<family>` names from `results/web/comparisons.csv`; no fallback |
| `objects.py` | the object store (one file, aligned objects, **assumed** lognormal sizes), page-cache helpers, disk probe |
| `origin.py` | asyncio HTTP/1.1 origin: real `pread`, emulated RTT and per-session link (FIFO or demand-first), `/hint` (server-side WILLNEED), `/stats` |
| `client.py` | one emulated browser per session; its cache **is** `cachecore.Cache`; a demand-priority connection pool; recorded inter-view gaps |
| `timing.py` | one clock, a microsecond timer for asyncio, event-loop lag monitor |
| `analyze.py`, `run.py` | parity, the validity gate, paired session bootstrap, the report; sequential passes with dropped pages and a load gate |

## Shared code (`src/prefetch_common`)

| Module | Role |
|---|---|
| `utility.py` | `U(c)`, `cost_to_c`, `c_star`, envelopes; the mapping from the disk cost model to `c` |
| `stats.py` | `paired_bootstrap`, `cluster_bootstrap` with `twin_weights`, `session_bootstrap`, optional `wilcoxon` |
| `provenance.py` | `write_provenance`, `verify_provenance`, `snapshot` |
| `schemas.py` | the results contract: exact columns and types per results file; `write_rows`, `validate_tree` |

## Paper and presentation

`reproduce.py paper` runs `paper/numbers.py` (one `\newcommand` per quoted
number), `paper/tables/make_tables.py` and `paper/figures/make_figures.py`,
which share `paper/genlib.py` and are loaded by path, then
`paper/check.py --strict`. `--fixtures` builds the same outputs from the
synthetic `tests/fixtures/results/` tree into `outputs/paper-fixture/`.
`presentation/slides.tex` inputs `paper/numbers.tex` and the same figure
PDFs. `tools/check_doc_numbers.py` keeps the Markdown docs honest in the same
way (`docs/METRICS.md` explains every metric the docs quote).

## Limits that apply everywhere

- Simulated latency is a cost model; only `results/realio/` and
  `results/live/` hold measured latency, each for one device or one emulated
  network, a few traces or sessions, and a shared host.
- The oracle traces have no read/write flag (writes replay as reads), no
  service time and whole-second timestamps.
- Classification accuracy exists only for synthetic windows.
- Web object sizes, RTT, bandwidth and the cost `c` are assumptions; the
  browser cache is per session; one retailer.

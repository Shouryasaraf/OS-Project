# AGENTS.md

Two prefetching studies and one IEEE paper in one repository, for an
Operating Systems course project (Ashvik Mishra, Shourya Saraf; VIT):

- **disk** (`src/adaptive_prefetch`): block read-ahead, a causal LRU
  simulator over 56 libCacheSim `oracleGeneral` block traces, and a real Linux
  testbed (`realio/`) that replays the simulator's decisions on a real device;
- **web** (`src/webprefetch`): product-page prefetching on RetailRocket, a
  per-session browser-cache simulator, and a real HTTP replay (`live/`) with
  exact parity to it;
- **shared** (`src/prefetch_common`): `U(c)`, bootstraps, provenance and the
  results schemas; the paper pipeline is `paper/`, the deck `presentation/`.

Read this before changing code, numbers, claims or docs. The binding work
plan is `docs/MASTER_PLAN.md` (its "Lead decisions" L1-L6 override the rest
of it). How components work: `docs/architecture.md`. Metric definitions:
`docs/METRICS.md`. Commands and runtimes: `docs/REPRODUCE.md`.

## Shared rules

- **Where things live.** The disk study owns `outputs/` and `models/`; the
  web study writes `outputs/web/` and `models/web/` (`webprefetch/paths.py`),
  so neither overwrites the other. Both are gitignored, except the tracked
  `models/msr_sample_gnb.json`. `results/` is **tracked** and holds the
  aggregates the paper reads, one directory per study (`disk`, `gpu`,
  `realio`, `web`, `live`), each with `PREREGISTERED.md` and
  `provenance.json`. `results/pivot_original/` is the web branch's original
  run kept verbatim: never edit it.
- **Never hand-edit a results file.** Every file is written through
  `prefetch_common.schemas.write_rows` (exact columns and types; 29 schemas)
  and hashed in its directory's `provenance.json`;
  `provenance.verify_provenance` detects an edit. A results file changes only
  when its full reproduce command re-runs.
- **Bounded runs never write `results/`.** `--quick`, smoke runs and tests
  write under `outputs/` (`outputs/disk-quick/`, `outputs/web/quick/`,
  `outputs/realio-quick/`, `outputs/web/live/`). A smoke number is never
  quoted.
- **One reproduce module per study.** `python reproduce.py <study>`
  dispatches to `repro/<study>.py` (`disk`, `realio`, `web`, `live`,
  `paper`; `--gpu` sends `disk` and `web` to `repro/gpu.py`, whose disk half
  is `repro/gpu_disk.py`). Each prints its runtime estimate first, returns
  `EXIT_UNAVAILABLE` rather than faking a result when it cannot run, and
  writes provenance last.
- **Data is gitignored by name**: `data/block-traces`, `data/raw`,
  `data/processed`, `data/MSRC-trace-003`. `data/samples/` is tracked. In a
  worktree, symlink the data directories; never copy them.
- **Environments.** The disk core is **stdlib-only**. The venv `.venv/` has
  every extra (`[traces]` zstandard, `[web]` numpy/pandas/scikit-learn/
  joblib, `[gpu]` torch with CUDA, `[paper]` matplotlib/scipy). System
  `python3` has none: web, GPU and device tests `skipUnless` their extra and
  skip, never error. The package is not installed: use `PYTHONPATH=src`
  (`main.py` and `reproduce.py` set the path themselves).
- **The unit suite is the only verification gate.** No CI, linter,
  formatter or type checker runs automatically, and there is no pytest
  config. The whole suite must stay green under both interpreters.
- **The host is shared** (128 cores, one GPU, other tenants). Keep worker
  counts at or below 64 unless the task says otherwise. Every timing is
  recorded with its load average; GPU timings carry `gpu_contention` and are
  never quoted as costs.
- **Pushing** happens only from the merge step of a wave (lead decision L1).
  Work packages commit on their own `wp/*` branch in their own worktree.

## Commands

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests   # 618 tests at this commit, about 105-140 s
PYTHONPATH=src python3 -m unittest discover -s tests            # stdlib only; web/GPU tests skip
PYTHONPATH=src .venv/bin/python -m unittest tests.test_project.SimulatorTests   # one class
python3 tools/check_doc_numbers.py        # every number in the docs is in results/
python3 paper/check.py --strict           # paper lint (after reproduce.py paper)
python reproduce.py disk --quick          # smoke; reproduce.py --help lists every study
python main.py --quick                    # the original disk entry point (outputs/ only)
python -m adaptive_prefetch.multicorpus --config C1 --dry-run   # E1 job list
python -m adaptive_prefetch.realio {mkfile,calibrate,run,grid,report}
python -m webprefetch {prepare,train,benchmark,freshness,decompose,parity,learning-curve,serve,...}
python -m webprefetch.seqmodel {train,experiment}
python -m webprefetch.live {plan,store,probe,origin,run}
```

`unittest` prints the current test count; update the figure above when you
add tests. `docs/REPRODUCE.md` has every reproduce command with its measured
runtime.

## Claim discipline (applies to code output, docs, README, paper, slides)

This is a graded submission and the paper must survive an examiner. These
rules come from plan section 5 and the pre-registrations:

- **No number is typed by hand.** The paper and slides quote
  `paper/numbers.tex` macros; Markdown docs quote only figures that
  `tools/check_doc_numbers.py` finds in the `results/` file their block
  cites (history regions are marked `<!-- doc-numbers: off (reason) -->`).
- **Pre-registered words only.** "Better", "worse" and "no difference
  detected" follow each study's `PREREGISTERED.md`; on disk both the paired
  and the cluster interval must agree. Report negative and null results
  plainly. A winner count is never reported without precision, wasted share
  and device-I/O ratio.
- **Simulated vs measured.** Every simulated number is labelled simulated.
  Latency is measured only in `results/realio/` (device, backend, think model,
  trace prefix and load stated) and `results/live/` (loopback, emulated
  network, assumed object sizes). A simulated hit ratio is never written as a
  latency gain.
- **Hit ratio is not device work.** Every disk hit-ratio claim is paired with
  `device_ios` and `blocks_moved` ratios.
- **`belady_min` is Belady's MIN for demand paging without prefetch**, a
  reference for replacement, not an upper bound for prefetching policies
  (`deep@256` exceeds it). Say so wherever it appears.
- **Classifier accuracy is synthetic only.** The held-out accuracy is on
  independently generated synthetic windows; real traces have no workload
  labels. The LSTM predicts address **deltas**: compare cache metrics and
  inference cost, never its "accuracy" as a classifier. Oracle re-access
  recall is reported only with hit ratio.
- **Strongest fair baseline.** `learned_depth` is compared with a
  validation-chosen fixed depth (VCF), the web controller with the best fixed
  depth *at the same capacity* and with `fixed0`, the ranker with transition
  top-1. Do not generalise from one trace, one capacity or one cost.
- **Trained artifacts** (LSTM, `learned_depth`, GRU) are scored only on
  segments or splits they were not trained or selected on, and each row names
  its training range.
- **Oracle-trace hit ratios include writes replayed as reads** and are never
  compared numerically with the MSRC-trace-003 figures of D13-D17.
- **The disk break-even above 1 is a property of the per-block, no-merge
  cost model**, not of hardware (D18).
- **Realio figures** (`pred_err`, `k_eff`, rank correlations) are per trace,
  think model and device, never properties of HDDs or SSDs. HDD `sdb` is the
  headline device; SSD rows are secondary and carry `foreign_io_frac`.
- **Invalid or contaminated runs are never cited**: a live pass failing the
  validity gate, a realio row on a busy host or with foreign I/O above 1% on
  `sdb`, the web-realexp ramp-2 s prototype runs, any smoke run.
- **Every quantitative claim has n, an interval and a seed or repeat count.**

## Disk study invariants

### The replay loops

- `simulator.replay()` and `simulator.replay_stream()` are the two causal
  loops shared by every mode, and they must stay **bit-identical** on every
  counter, including `device_ios`, `blocks_moved`, inference calls and gate
  state. `test_every_supported_mode_agrees_across_both_paths` (test_project)
  and `test_every_mode_agrees_across_both_paths_including_inference`
  (test_disk_core) pin it. Both drive `guard.GateController`;
  `_publish_gate()` is the only place gate state reaches `Metrics`.
  Duplicating that logic is what crashed `replay_stream` on the gated modes
  (D15).
- `simulator._validate_replay()` is the **only** parameter validator, called
  by both entry points. `MODES_SUPPORTED` is the single source of truth for
  the mode set; `_make_predictor` reads it and is the only predictor factory.
- `read_ahead()` must return exactly `depth` blocks in **both** branches
  (D16).
- Prefetches are gated on `index >= window_size`: **the first window never
  prefetches**, and a window's policy only affects later requests.
- `cache_factory(capacity, metrics)` builds the cache in both loops and
  defaults to `LRUCache`. A replacement must keep `LRUCache`'s semantics and
  attributes; it exists so `realio.decisions.RecordingLRU` can log every cache
  operation. `test_default_cache_factory_is_identity_for_every_mode` pins
  that the default changes nothing.
- **Device-I/O accounting** (D18): a demand I/O is a maximal contiguous run
  of missed blocks inside one request (a resident block splits it); a
  prefetch I/O is a maximal contiguous run newly prefetched after one
  request; a prefetch run starting where a missed request ends is counted as
  mergeable. `device_ios = demand_ios + prefetch_ios -
  prefetch_ios_mergeable`, `blocks_moved = misses + prefetches`. The legacy
  `LatencyModel` fields are untouched; `IOCostModel` sits beside them.

### Modes: the four-places rule

`benchmark.MODES` and `sweep.RANKING_MODES` hold **15** modes in this order:
`none, sequential, strided, stride, markov, lstm, adaptive,
adaptive_evidence, guard, depth_adaptive, correlate, deep, linux_ra,
learned_depth, routed` (`routed` appended by track HSR). `rows[0]` is `none` and `rows[5]` is `lstm`, which
`test_benchmark_includes_skipped_optional_lstm` asserts; it also checks
`len(rows) == len(MODES)`, so **appending is safe, reordering is not**.

A new mode must be added in **four places** or it is silently wrong
somewhere (`test_registered_in_all_four_places` checks `linux_ra`):

1. `benchmark.MODES`;
2. `sweep.RANKING_MODES`;
3. `simulator.MODES_SUPPORTED` and `_make_predictor`;
4. `simulator.GATED_MODES` if it is feedback-gated, and
   `benchmark.OPTIONAL_MODES` / `sweep.OPTIONAL_MODES`
   (`("lstm", "learned_depth", "routed")`) if it needs a trained artifact.

Two candidate interfaces feed the same cache:

- `next_candidates(request) -> list[int]` for per-request predictors
  (`stride`, `markov`, `lstm`, `learned_depth`); a new baseline implements it.
- `observe(request, missed_blocks, cache_view)` for **cache-aware**
  predictors (`cache_aware = True`): `linux_ra` reads residency through a
  read-only `CacheView` after the demand reads and before any prefetch. It
  cannot be a `next_candidates` predictor.

Mode parameters: `deep` takes `read_ahead_depth` (`deep@N` in sweeps,
default `DEFAULT_READ_AHEAD_DEPTH`); `linux_ra` takes `linux_ra_max_blocks`
(`linux_ra@N`, default `DEFAULT_LINUX_RA_MAX_BLOCKS`, the host's
`read_ahead_kb`); `learned_depth` takes `depth_model=` in `replay`,
`replay_stream` and `_make_predictor`: an artifact path, or a fresh object
with `next_candidates` (`depth_model.PrecomputedDepth`, `RelativeDepth`). A
predictor is stateful; each replay needs a fresh one. `routed` (track HSR,
`hindsight.py`) takes `router=`: a fresh `hindsight.WindowRouter` (decides
in the loop from the window's 12 features) or `ScheduledRouter` (a
batch-inferred or oracle per-window schedule); a window's action applies to
the next window, the first window never prefetches, and its trained routers
are scored only by `reproduce.py hindsight` (`results/hindsight/`). In E1
(`multicorpus`) its rows are `skipped: evaluated in the hindsight study
only`.

### Optional modes and skipped rows

- A missing or refused optional mode produces a row with
  `status="skipped: <reason>"` and `None` metrics, never a substituted or
  synthetic value, in the benchmark **and** the sweep.
- `learned_depth` is refused (`skipped:`) on any slice that overlaps its
  training or validation range (`depth_model.scoring_conflict`, the
  `.training.json` sidecar bound by SHA-256). E1 therefore marks `lstm` and
  `learned_depth` `skipped: evaluated in E2 only` in every row.
- `models/lstm_delta.pt` is versioned (`ARTIFACT_VERSION`); an older one is
  rejected with a "retrain" message. `train_lstm` holds out
  `val_fraction=0.2`, keeps the best epoch and can run on CUDA. `main.py`
  reuses a cached artifact only when its training record (seeds disjoint
  from the benchmark, `LSTM_TRAIN_KW` including `val_fraction`) and the
  artifact's SHA-256 both match; otherwise it retrains.
- Benchmark rows carry private `_read_blocks`, `_prefetchable_misses`,
  `_total_latency_us`, `_inference_calls`, `_total_inference_us` and
  `_requests` keys that `aggregate_results` depends on; `write_results_csv`
  strips them via `COLUMNS`.

### Feedback gate

- The gated modes (`guard`, `depth_adaptive`, `correlate`) decide from
  **observed** precision. `PrecisionMonitor.record()` must receive the
  **delta** in useful prefetches since the last decision; the cumulative
  total pins the ratio at 1 and fakes a win (D15). Two tests pin it.
- A closed gate must be able to reopen: `GateController.REPROBE_AFTER`
  forces a probe after consecutive closed decisions.
- `depth_adaptive` is identical to `guard` at the break-even cost model.
  This is validity gate 2 of E1 and held on every pair
  (`results/disk/provenance.json`, `extra.checks`).
- `CorrelateDetector.confirmed_stride()` is recomputed every call (D16).

### Cost models

- **Oracle traces need an explicit latency model.** They carry no service
  times; falling back to the assumed 5/100/50 model silently moves the gated
  modes' break-even (M19). The multicorpus driver passes `LatencyModel(5,
  100, 50)` (`assumed`) for every row and also `LatencyModel(1, 100, 100)`
  (`break_even`) for the gated modes, and every row records its break-even.
  `test_oracle_traces_need_an_explicit_latency_model` pins it.
- `LatencyModel.measured()` (MSRC `.revised` only) uses the **mean** service
  time, never the median; the hit cost is an approximation; the cost column
  is degenerate at `prefetch_multiple=1.0` and is labelled so; cost ranking
  is invariant to a uniform rescale but not to `prefetch_multiple` (pinned by
  two tests, D11, D16).
- `IOCostModel` constants come only from a device calibration file
  (`from_device_json`); missing keys raise rather than default.
- `Metrics.prefetch_recall` uses a future-read `Counter` and is
  **measurement only**; never feed it into candidate generation. It is not
  reported on streamed runs.

### Classifier and features

- `DEFAULT_CLASSIFIER` is `gnb`, measured, not assumed: QDA loses under the
  covariate shift this project faces (D1). Do not swap in a "better"
  classifier on intuition.
- `replay(..., online_updates=True)` raises unless `labels` covers every
  complete window; on real traces the model is frozen. `--online-real` /
  `adaptive_pseudo` self-trains and is not accuracy evidence.
- `adaptive_evidence` (`simulator.route_window`) routes on the window's own
  contiguity, repeated stride and jumpiness and needs **no trained model**.
- `Request.pattern` holds the `.revised` trace's own `seq`/`rand` flag. It is
  for ground-truth scoring **only**; routing on it is not a result.
- `FEATURE_NAMES` is 12: 8 stream features plus 4 contextual ones from
  `features.WindowContext`, whose block-age table is LRU-bounded
  (`DEFAULT_AGE_TABLE_BOUND`; `Metrics.age_table_evictions` says when the
  bound bit). `extract()` returns only the 8 stream features.
  `benchmark.make_model` lays training windows out as one continuous stream
  so train and replay build the same vector; `make_model(interleave=True)`
  is an opt-in sensitivity, not the default.
- Classifier persistence is **JSON, not pickle** (`artifacts.py`, format
  version 2; `shrink` persisted for QDA). Changing `FEATURE_NAMES` or
  `CLASSES` invalidates `models/msr_sample_gnb.json`; regenerate it with
  `python -m adaptive_prefetch train-msr-sample`. That model was adapted
  in-sample with a `random` proxy label: replaying the same sample is a load
  check, not an evaluation.

### Trace loading: the oracleGeneral format (corrected)

`trace.FORMATS` = `auto, normalized, msr, iotta8, alibaba, revised, oracle`
(the same tuple in `cli`, `datasets` and `trace`). The `oracleGeneral`
format is **not** one record per 512-B block:

- 24-byte records `<IQIq` = (timestamp in whole seconds, `obj_id`,
  `obj_size`, `next_access_vtime`);
- `obj_id` is a **byte offset**, a multiple of 512; Alibaba and Tencent carry
  a volume tag in bits 40 and up, masked with `& (2**40 - 1)`;
- `obj_size` is bytes for MSR, Alibaba, Tencent and `fiu_cheetah`, 512-B
  sectors for the other FIU files (`trace.oracle_profile` detects it);
- MSR, Alibaba, Tencent: one record is one request. FIU: one record is one
  4 KiB page; the canonical reconstruction (`merge="auto"`) merges
  same-second contiguous pages with a 1024-sector cap. `none` and uncapped
  `same_second` are labelled sensitivity views only;
- no read/write flag (writes replay as reads), no service time.

`iter_oracle_chunks` and `iter_csv_chunks` stream bounded chunks. Chunk by
**request count, never bytes**. `skip_requests`/`max_requests` count
canonical requests; a slice that cannot be filled is a `skipped: n/a` row,
never a short slice. `corpora.PAPER_SUBSET` has 21 stems;
`corpora.TWIN_GROUPS` lists the near-duplicate families the cluster bootstrap
down-weights.

The CSV loaders: `auto` detects a profile by column name only, so headerless
`iotta8` and `.revised` files need an explicit `--format`. Loaders reject
timestamps that go backwards and report the offending line; the MSR loader
first re-sorts by raw `Timestamp`, because merged samples concatenate hosts.
The committed `data/samples/` traces are small and their provenance is
unverified (`data/samples/README.md`).

### Defaults the tests pin

`DEFAULT_CAPACITY` = 2048 blocks (1 MiB) and `DEFAULT_READ_AHEAD_DEPTH` = 8
stay as they are because tests pin them. They are **not** optima: the hit
ratio does not saturate at 1 MiB on real traces and keeps rising with depth
(D20), which is why E1 reports two capacities (C1) and four (C2) and a depth
grid. `main.py --quick` caps MSRC traces at `QUICK_REAL_TRACE_LIMIT`
requests.

### Legacy entry points and the MSRC sweep

<!-- doc-numbers: off (measurements of the MSRC-trace-003 collection on branch main; not in results/) -->
- `main.py` is the original end-to-end run (EDA, classifier selection,
  domain shift, the benchmark, drift) and writes only `outputs/`. The console
  shows *conclusions*; raw rows go to `outputs/` via `report.py`. Do not dump
  wide tables back to the console. Its pooled rows are `ALL_SYNTHETIC` and
  `ALL_REAL`, never one pool over both. The drift report demonstrates nothing
  (saturated: every row lag 0, accuracy 1.000) and is excluded from the paper.
- `sweep_modes.py` streams the MSRC-trace-003 `.revised` collection (11.86
  GB, 32 files, about 430 bytes per resident `Request`; the default chunk of
  250k requests is about 108 MB; a 500 MB byte slice would be about 3.8 GB
  resident). Its warm-up matches `replay()` exactly (`seen >= window_size`).
  `outputs/msrc_sweep.csv` carries `observed_precision`, `precision_margin`
  and `observed_stride` per row; keep them, D12's claim depends on them.
- `data/MSRC-trace-003` is absent on this machine; nothing in `results/`
  depends on it.
<!-- doc-numbers: on -->

## Web study invariants

- **GRILL rules** (`audit/GRILL.md`, `docs/web/WEB_DECISIONS.md` W0):
  sessions break after 30 minutes and at every split boundary (zero
  cross-split sessions); a bounded replay takes complete sessions only
  (`simulator.complete_views`); split 0 builds the index, split 1 trains the
  ranker, split 2 is the only split any choice is made on, split 3 is
  replayed once with every choice frozen; popularity ties break by item id;
  bootstraps resample whole sessions, seed 42; `events.csv` is SHA-pinned and
  never replaced by generated data.
- **Selection is per capacity** (W1). `benchmark.select_policies` chooses on
  validation separately at every capacity; the pivot's cap-8-for-all rule is
  the separately labelled `capacity_transfer` comparison, never a headline.
  Every comparison is reported against the best fixed depth at the same
  capacity, against `fixed0` (a negative interval reads "loses to no
  prefetch") and against transition top-1.
- **The online index is causal** (W2). `index.OnlineIndex` adds `a -> b` only
  once `b`'s view time is strictly earlier than the decision time. Scorers on
  an online index, and every `feedback` policy (its gate is shared across
  sessions), replay in **global** time order; `parallel.py` gives each shard
  the whole split's pending transitions and runs feedback policies in a task
  of their own. Counts equal the serial replay exactly; float sums agree to
  rounding, so `benchmark.json` is byte-identical only at a fixed
  `--workers`.
- **`cachecore.Cache` is the single browser cache.** The replay simulator and
  the live client both wrap it (the demo server still has its own loop, and
  `session_parity` checks it against the batch replay). Do not copy its
  rules into a new class.
- **Demo semantics** (P5): `adaptive` is the validation-selected controller
  for the chosen capacity and cost, read from `benchmark.json`;
  `adaptive_cost` is the pivot's penalty = cost rule, labelled as such.
  `webprefetch parity` regenerates `session_parity.json` from the same run;
  never compare it across runs or worker counts.
- **Bounded runs never write `results/web`.** `--validation-views`,
  `--test-views` and `--quick` write `outputs/web/`;
  `test_parity_and_decompose_after_a_bounded_benchmark_stay_out_of_results`
  pins it.
- **Scorers.** An optional scorer whose artifact or module is missing returns
  `status="skipped: <reason>"`, never a substitute. The GRU (`seq`) needs
  `--seq-model models/web/seq/gru_final_s<seed>` (artifacts
  `models/web/seq/<model>_<protocol>_s<seed>/{weights.npz, vocab.npy,
  config.json}`, for example `gru_validation_s0` and `gru_final_s0`; no
  pickle). Compare `seq` with other rankers on the `novel_*` columns, because
  its history mask never ranks an item already in the window.
- **Costs are assumed.** `c` is a hit-equivalent assumption, reported at a
  fixed grid (`utility.UTILITY_CS`); object sizes are equal in the replay and
  prefetches complete instantly.

## Real-system rules (realio and live)

- **Devices (lead decision L2).** Only the SATA SSD (repository or scratch
  paths) and the HDD `sdb` under
  `/mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch/` are ever
  written. **Nothing is written to `sdd` or `sde`.**
  `webprefetch.live.run.check_store_dir` refuses any other path under `/mnt`.
- **Drop before every run.** realio drops the whole backing file with
  `POSIX_FADV_DONTNEED` before every run (a page charged to another cgroup is
  not re-charged, so the budget would not bind otherwise) and displaces the
  drive's cache with an `O_DIRECT` read of an unused region; live drops the
  store's pages before every pass.
- **Chunked WILLNEED.** One `WILLNEED` (or `readahead(2)`) call is truncated
  to the device's `max_sectors_kb`, so every hint is issued in chunks of at
  most that size; `RWF_NOWAIT` reads are chunked the same way.
- **Quiet-host gate.** A realio run or live pass starts only when the
  1-minute load average is at most 32 (`--max-load`); the grid waits
  (`--max-load-wait`) and records the load of every row. Never run E3 and E6
  together, or either with E1.
- **Fresh process, shuffled order.** Every realio run is a fresh child
  process; within a cell the (backend, mode) order is shuffled with a
  recorded seed; three repeats.
- **Parity.** Backend B asserts the simulator's hit/miss sequence on every
  request (`CacheDivergence`); a divergence is a defect and its rows are not
  reported. Backend A's hit ratio is a measured divergence (MGLRU, 4 KiB
  pages), never asserted. The live client's per-session counters must equal
  `sim.json`; `webprefetch.live run` exits 1 on any mismatch.
- **Live validity gate.** A pass is valid only if server lag, origin and
  client event-loop lag p99 are all at most 2 ms, with no parity mismatch and
  no connection error. An invalid pass is written with `valid=False` and its
  comparison columns empty; it is never compared. `run` exits 2 when the
  planned request rate exceeds what one event loop times accurately.
- **Validated policies only.** `--selection results/web/comparisons.csv`
  resolves `validated_<family>` names to E4's per-capacity validation choice
  (for the feedback gate, its `c` and `epsilon`); there is no built-in
  fallback.
- **No SSD headline numbers.** HDD `sdb` is the headline device in E3; the
  SSD carries other tenants' I/O, every SSD row reports `foreign_io_frac`, and
  a row above 1% is flagged and cited only with the flag. In E6 cite the HDD
  store rows or unflagged SSD rows.
- **Realio CLI.** `python -m adaptive_prefetch.realio grid --plan
  {full,hdd,ssd} --max-load-wait S`; `reproduce.py realio --device DIR
  [--grid auto|hdd|ssd|full] [--recalibrate]` (calibrates only when
  `results/realio/device_<dev>.json` is absent or `--recalibrate` is given).
  `--skip-requests N` rows are labelled `<trace>+skip<N>`.

## Paper, slides and docs

- `paper/` generators (`numbers.py`, `tables/make_tables.py`,
  `figures/make_figures.py`, sharing `genlib.py`) read only `results/` and are
  loaded by path, never by putting `paper/` on `sys.path` (`numbers.py` would
  shadow the stdlib module). A missing input gives a "MISSING" placeholder.
  `reproduce.py paper --fixtures` writes `outputs/paper-fixture/`, never
  `paper/`.
- `paper/check.py` is the local gate (no LaTeX here, L6): `--strict` after
  generation, `--final` for submission (no `\todo`). It fails on a
  hand-typed number in `sections/*.tex`, on adjacent `$$`, under `--strict`
  on an undefined number macro, and under `--final` on a macro that prints
  `\missingnumber`. The committed paper must pass `--final`
  (`tests/test_paper_check.py`).
- `paper/decisions.py` applies every `results/*/PREREGISTERED.md` rule to
  `results/` (disk P1-P6, lever ordering, C3/C4; realio R1-R7; web W1-W8 and
  rules 1-3; live H1-H7 on citable passes only, `genlib.live_citable`) and
  emits the `\dec...` macros and Table T9. The prose is written from those
  verdicts: the paper's title and every verdict sentence depend on them.
- `tests/test_paper_decisions.py` pins the committed `numbers.tex`, tables
  and figures byte-for-byte to a fresh regeneration from `results/`, and pins
  the verdicts the prose relies on. **Any change to `results/` needs
  `reproduce.py paper` and a re-read of the verdict sentences**; the test
  fails until both are done. `test_numbers_tex_fits_tex_memory` caps the
  size of `numbers.tex` (TeX memory on Overleaf), so trim
  `numbers.FAMILIES` rather than raising the cap.
- `presentation/slides.tex` uses `\pnum{...}` for systematic macros (a bold
  ?? if `numbers.py` stops emitting one). Check it after trimming
  `numbers.FAMILIES`.
- `presentation/slides.tex` inputs `../paper/numbers.tex` and the paper's
  figure PDFs; `tools/check_doc_numbers.py` fails on a hand-typed figure in
  it. Compile both on Overleaf.
- `tools/check_doc_numbers.py` checks README, AGENTS and `docs/` (except
  `MASTER_PLAN.md` and `REFERENCES.md`): every figure must be found in the
  results file, or the filtered rows, its block cites
  (`<!-- source: results/<file> key=value -->`). `tests/test_doc_numbers.py`
  runs it over the repository.

## Known current state

What the experiments found (details and figures: `docs/DECISIONS.md` D18-D21,
`docs/web/WEB_DECISIONS.md` W1-W4, the paper):

- **Disk, simulated (E1, `results/disk/`).** Read-ahead depth and window
  growth carry the hit-ratio effect, and hit ratio keeps rising with depth
  well past 8; but fixed-depth read-ahead issues *more* device I/Os than
  `none` at 1 MiB, and only `linux_ra` lowers device I/Os. The trained
  classifier router (`adaptive`) is **worse** than `sequential` at both
  capacities. By the pre-registered lever rule the predictor is smaller than
  depth and window growth but **not** the smallest lever: at 64 MiB the
  feedback gate's effect is smaller. All four E1 validity gates passed. The
  old claims "`deep` wins, depth is the whole result" (D13, D14, D17) were
  measured on MSRC-trace-003 and are not this study's result.
- **Disk, GPU (E2, `results/gpu/`).** `learned_depth` is below the
  validation-chosen fixed depth on most of 21 traces over 5 seeds; the LSTM
  is below `linux_ra` on all four traces it was scored on.
- **Disk, real device (E3, `results/realio/`).** The mode with the highest
  simulated hit ratio was never the fastest measured; `none` was often the
  fastest; simulated hit ratio did not rank policies by latency in any HDD
  cell. `learned_depth` is not in the real-device grid.
- **Web (E4, E5, `results/web/`).** The validated adaptive controller beats
  the best fixed depth at capacities 8 and 32 at `c` = 0.01; freshness
  (online counts) is the largest lever; the learned ranker's effect lies
  below both depth and freshness, but the GRU's overlaps depth, so by rule 2
  (W3 and W8 together) the predictor is **not** called the smallest lever;
  the GRU is worse than online counting; the feedback gate issues fewer
  prefetches without a detectable utility difference.
- **Live (E6, `results/live/`).** Every pass valid, zero parity mismatches;
  measured waits are on loopback with an emulated network.
- **Open items.** The paper's page count (estimated, not compiled) and the
  authors' e-mail placeholder in `paper/main.tex` await an Overleaf build.
  The C2 best depths are descriptive argmaxes (`deltas.csv` has no
  depth-vs-depth rows). The suite prints harmless `fork()` deprecation warnings. The
  `DEFAULT_READ_AHEAD_DEPTH` comment in `simulator.py` still quotes the D13
  MSRC figures.

## Where the detail lives

| Document | Contents |
|---|---|
| `docs/MASTER_PLAN.md` | the binding plan, work packages, honesty constraints, lead decisions |
| `docs/architecture.md` | module maps of all three packages, the replay loop, data flow |
| `docs/METRICS.md` | every metric, its denominator and its limits |
| `docs/REPRODUCE.md` | every command, its runtime here, inputs and outputs |
| `docs/DECISIONS.md` | disk decision log: D1-D17 history on `main`, D18-D21 corrections from E1-E3 |
| `docs/web/WEB_DECISIONS.md` | web decision log: W0 (GRILL), W1-W4 |
| `docs/CLASSIFIER_EVAL.md`, `docs/MSR_SAMPLE_TRAINING.md` | the synthetic classifier evaluation and the MSR sample adaptation (history; regenerate before quoting) |
| `docs/PIPELINE.md` | the disk pipeline and the 14 modes in one page |
| `docs/PRESENTATION.md`, `presentation/` | the talk, speaker notes, viva answers, Beamer deck |
| `docs/REFERENCES.md` | corrections to the Review 1 deck's citations |
| `results/*/PREREGISTERED.md` | the comparisons and decision rules each study committed to before running |

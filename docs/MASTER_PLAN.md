# PLAN — `master`: one functional repository, two prefetching studies, one IEEE paper

Authors of the work: Ashvik Mishra (24BAI0342), Shourya Saraf (24BDS0391), VIT.
Lead-architect plan. Built from 9 evaluation reports and the scratch measurements they produced.
The scratch root used throughout is
`/tmp/claude-1000/-home-antpc-Ashvik-OS-Project-OS-Project/30a4e740-9eeb-4ee7-b80b-95503b869ef2/scratchpad/eval/`
(written `$EVAL/` below). The repository is
`/home/antpc/Ashvik/OS-Project/OS-Project` (written `$REPO`).
The read-only worktrees are `$SCR/mainref` (branch `main` @ db05d74) and `$SCR/pivot`
(branch `pivot/retailrocket-adaptive-prefetch` @ 8569cdf), where `$SCR` is the scratchpad.

All line numbers below were checked against those worktrees in this session. Several line numbers
quoted in the pivot evaluations were wrong, for example `model.py:240,247` and `demo_server.py:221`.
The verified numbers appear here instead.

---

## 0. Ground truth this plan depends on (corrections to earlier assumptions)

1. **The oracleGeneral format is not "one record per 512-B block".** The workflow context and the saved
   memory note `project-paper-goal.md` both say it is, and both are wrong. Measured on all 1.45 B records
   (blocktrace-scout `stats_all.jsonl`, `cmp_sample.py`):
   - Each record is 24 bytes, `'<IQIq'` = (ts in seconds, obj_id, obj_size, next_access_vtime).
   - `obj_id` is a byte offset and is always a multiple of 512. Alibaba and Tencent carry a volume tag
     in bits 40 and up, equal to file# x 10, which must be masked with `& (2**40-1)`.
   - `obj_size` is in bytes for MSR, Alibaba, Tencent and fiu_cheetah, and in 512-B sectors (value 8)
     for the other 9 FIU files.
   - MSR, Alibaba and Tencent: one record is one original request. FIU: one record is one 4 KiB page,
     so same-second contiguous pages must be merged back into requests, with a 1024-sector cap.
   - There is no read/write flag and no service time. Timestamps are whole seconds.
   - Cross-check: the hm_0 oracle offsets match the committed `data/samples/msr-cambridge1-sample.csv`
     on 1000/1000 rows.
   The lead should fix the memory note after this phase. That is not done here, because this phase is
   read-only.
2. `master` already exists locally and on `origin`. It holds 2 commits on top of `db05d74`:
   `aecb0a2` (docs/MASTER_BRANCH.md) and `969acd8` (adds `data/` to `.gitignore`). Its tree is otherwise
   identical to `main`. The `data/` ignore rule is fine for tracked samples, but new fixtures under
   `data/samples/` need `git add -f`, or the rule should be narrowed (see WP0).
3. Host: 128 cores and 251 GB RAM, shared with other users. Load average ranged from about 7 to 290
   during evaluation. The GPU (RTX PRO 6000) is shared: 39 GB used and 97% utilisation by other tenants
   at plan time, and 5 CUDA OOMs during the evaluations. Every timing number must be recorded together
   with load and contention metadata.
4. Devices:
   - `/dev/sdc2`: root, Micron 5300 SATA SSD. It carries about 7.8 MB/s of foreign reads, so it is noisy.
   - `/dev/sdb`: `/mnt/b18edc65…`, idle Seagate ST10000NM017B. It already holds
     `os-project-prefetch/hdd_test.bin`, 24 GiB, written by the realio prototype.
   - `/dev/sdd`: `/mnt/9792e947…`, empty, identical HDD model.
   - `/dev/sde`: `/mnt/c609e3de…`, empty, identical HDD model.
   - All block devices: `read_ahead_kb=128`, `max_sectors_kb=256`. Each WILLNEED or readahead(2) call is
     truncated to 256 KiB, so hints must be issued in chunks. The kernel uses MGLRU. A user systemd
     manager delegates the `memory` controller (not `io`), so `systemd-run --user --scope -p MemoryMax=`
     works without sudo.
5. No LaTeX toolchain is installed, and downloading binaries is not permitted. The paper is written as
   IEEEtran `.tex` and is structurally linted locally. The students compile it on Overleaf.
   `python-pptx` is not installed in the venv or in system python3 (checked), so no `.pptx` can be
   generated either (see C14 in "Critic changes").
6. **(Critic) The 56-trace numbers quoted from `$EVAL/main-science/` pool *unmerged* FIU pages.**
   `run2.py:load()` "raw" variant does not coalesce FIU 4 KiB page records, and its "merged" variant
   coalesces same-second runs for **every** corpus with **no** 1024-sector cap. Neither is the canonical
   reconstruction defined in §2.1 (MSR/Alibaba/Tencent as-is, FIU pages merged with cap 1024). The
   scout showed this matters a lot: fiu_madmax `deep` 0.927 unmerged vs 0.232 merged, and main-science's
   own FIU deep@8 gain is +60.18 pp raw vs +1.93 pp merged. Every "raw" pooled figure below
   (+16.32 pp, +42.47 pp, +31.27 pp, 0.659x device I/Os, 51/56 wins) is therefore **orientation only,
   inflated by FIU page splitting**, and must not be cited until E1 re-derives it on canonical requests.

---

## 1. Evaluation of the two branches

### 1.1 Branch `main` — ML-based adaptive disk I/O prefetching (`src/adaptive_prefetch`, stdlib core)

**Verdict: KEEP AS THE DISK STUDY'S ENGINE.** The causal simulator is trustworthy. Its central finding
(policy choice is noise; depth is the signal) reproduces on 56 public traces from 4 corpora. Around that
core sit several crash bugs and wrong-output bugs. The cost model is the main scientific weakness. Nothing
in `main` does real I/O.

Measured health:
- Unit tests: 131 tests OK in 14.8 s with the venv and torch.
- `main.py --quick` runs in 12.0 s. Full `main.py` runs in 32.2 s with the LSTM trained.
- Causality was checked adversarially: replacing the future with random requests leaves prefix outcomes
  unchanged in all 11 non-LSTM modes.
- `replay_stream` with 37-request chunks equals `replay` on every cache field.
- On 56 traces x 250k requests: `adaptive` (GNB) vs `sequential` is -0.89 pp [-1.16,-0.64], worse on
  52/56. `deep@8` vs `none` is +16.32 pp [10.53,23.18]. `deep@256` vs `none` is +42.47 pp.
  `linux_ra@256` gives 0.659x the device I/Os of `none` [0.593,0.722]
  (`$EVAL/main-science/analysis.txt`, `io_analysis.txt`).
  **(Critic) These pooled figures include FIU as unmerged pages (§0.6); treat them as orientation
  only.** The classifier-vs-sequential direction is likely robust; the depth and linux_ra magnitudes
  are not.

**Kept as-is or extended:**
- From `simulator.py`: `replay` (`:321`), `replay_stream` (`:488`), `LRUCache` (`:157`), `Metrics`
  (`:108`), `LatencyModel` (`:18`), `route_window` (`:225`), `oracle_reference` (`:613`), and
  `_validate_replay` / `_publish_gate` (`:286`, `:308`).
- From `guard.py`: `GateController` (`:228`), `PrecisionMonitor` (`:36`), `read_ahead` (`:131`),
  `CorrelateDetector` (`:163`).
- `baselines.py`, `features.py`, the `trace.py` loaders, `sweep.py`, `benchmark.py`
  (`MODES` at `:15`, `aggregate_results`) and `artifacts.py`.
- The 131 tests.
- `docs/DECISIONS.md` (D1–D17), kept as history.
- `model.py` (GNB/QDA) and `lstm.py`, kept **as the measured negative result**.

**Bugs to fix (verified locations):**

| # | Location | Defect | Fix | WP |
|---|---|---|---|---|
| M1 | `cli.py:29` | `model.predict(extract(window))` passes 8 features to a 12-feature model, so `python -m adaptive_prefetch demo` always fails with "incorrect feature count". | Build the vector with `WindowContext().observe(window, dominant_stride(window))`, as `main.classifier_facts` does (`main.py:244-249`). | WP1 |
| M2 | `sweep.py:38` (`RANKING_MODES` includes `lstm`) and `simulator.py:598-607` | A sweep with default modes and no `models/lstm_delta.pt` raises in `_make_predictor`. The whole sweep aborts, and no `skipped:` row is written. | In `sweep_collection`, catch it for `lstm` (and later `learned_depth`) and emit `status="skipped: no trained artifact"` with None metrics. | WP3 |
| M3 | `sweep_modes.py:207-208` | `f"{'all' if args.full else args.per_trace:,}"` makes every `--full` sweep crash after `msrc_sweep.csv` is written. | Use `'all' if args.full else f'{args.per_trace:,}'`. | WP3 |
| M4 | `main.py:342-370` | The "MEASURED"/"measured" label is chosen from `measured_sources` and not from `args.cost_model`, so a run with `--cost-model assumed` is labelled measured. | Derive the label from the LatencyModel actually applied. | WP1 |
| M5 | `main.py:456,464,474` | `results.md` and `eda.md` use a hard-coded capacity of 128 and window of 32, while the console uses 2048. | Pass `args.cache_blocks` and the window size through `write_outputs`. | WP1 |
| M6 | `report.py:296,397-426` | The narrative is hard-coded (`+0.4`, `+4.10`, `web_3 +18.99`, `1.0101`, `287-5491 us`) and contradicts the run's own figures, e.g. break-even 0.526 versus the printed 1.0101. | Compute every figure from the rows and the LatencyModel, or drop the sentence. | WP1 |
| M7 | `main.py:198-225` (`resolve_lstm`) | The LSTM is trained on the synthetic benchmark traces and then scored on them, so it is evaluated in-sample. | Train on `transition_trace` seeds that are disjoint from the benchmark, or on a chronological prefix with a disjoint suffix for real traces. Add a test that the training and benchmark sets are disjoint. | WP1 |
| M8 | `eda.py:58-87,272-274` | `cacheable_ceiling` is called "the honest upper bound for any prefetcher", but `none` exceeds it on 5 of 6 datasets. | Rename it to "read-reuse within N accesses (not a bound)". Add a true LRU stack-distance demand hit ratio (Mattson) and a Belady MIN reference. | WP1 |
| M9 | `features.py:89-107` (`WindowContext.last_seen`) | The dict grows without bound. `adaptive` reaches 1,429 MB RSS versus 86 MB for `none` on 100k requests. | Use an LRU-bounded `OrderedDict` (default 4 M entries); an evicted entry means "never seen". Pin that output is bit-identical when the bound exceeds the footprint. | WP3 |
| M10 | `trace.py:68-69` | `iter_csv_chunks` re-parses the whole file twice per chunk for non-revised formats, which is O(n²). | Parse once, then slice. | WP2 |
| M11 | `artifacts.py:63-74` | `shrink` is not persisted for QDA, so a QDA model with a non-default shrink silently loads as 0.05. | Write `shrink` and reject a mismatch on load. | WP3 |
| M12 | `simulator.py:64-75` (`LatencyModel.measured`) | `prefetch_multiple <= 0` is validated after the "no measurements" early return. All-zero service times give a break-even of inf with no warning. | Validate first, and raise or warn on inf. | WP3 |
| M13 | `simulator.py:541-548,569-576` (`replay_stream`) | Inference calls and time are not counted for stride, markov, lstm or adaptive_evidence. | Count them as `replay` does (`:405-409`). Extend the parity test to cover them. | WP3 |
| M14 | `sweep_modes.py:177-178`, `sweep.py:168-187` | The parse-time estimate is wrong (0.5 min/GB; measured about 7.9 min/GB). `winners()` crowns `none` for cost and waste by construction. The aggregate is an unweighted mean, while `benchmark.aggregate_results` is request-weighted. | Fix the arithmetic. Label the cost and waste winners "degenerate when break-even > 1". Report both weightings. | WP3 |
| M15 | `lstm.py:117-246` | torch is left at 64 threads: 859 s under load versus 3.7 s with 4 threads. There is no device selection. | Use `torch.set_num_threads(min(4, os.cpu_count()))`, add `device="cuda"|"cpu"`, and record the training set. | WP9 |
| M16 | `benchmark.py:24-57` (`make_model`) | Class-by-class training under decay 0.995 skews priors to 0.10/0.17/0.28/0.46. Accuracy is unchanged (0.96875). | **(Critic: opt-in only.)** Add `make_model(..., interleave=False)`; the default stays as shipped so `sweep2.jsonl` `adaptive` cells, `models/msr_sample_gnb.json` and the CLASSIFIER_EVAL figures still reproduce. Test that `interleave=True` gives priors 0.25 ± 0.05 and held-out accuracy 0.96875. Report the switch as a sensitivity note, not a fix. | WP3 |
| M17 | `data/samples/demo.csv` | Comes from the pre-D10 generator, yet carries the largest headline gain (+45.57 pp) and is pooled into ALL_TRACES. | Regenerate it with `export-demo`, or remove it from `main.py`'s dataset list. Report the synthetic and real aggregates separately. | WP1 |
| M18 (science) | `simulator.py:401,420` (also `:537/:554`, `:635/:644`) | Unit mismatch: the mean per-request service time is charged per 512-B block for both demand and prefetch. This makes D11's "break-even 1.0101 is the arithmetic of this hardware" an artefact of costing per block with no merging. | Leave the legacy fields alone so the invariant tests still pass. Add device-I/O counters and an `IOCostModel` T = N_io·t_pos + N_blk·t_xfer (D18). | WP3 |
| M19 (science) | `simulator.py:72-73` | A trace without `service_ms` (every oracle trace) silently falls back to the 5/100/50 model, break-even 0.526. That changes gated modes: `depth_adaptive` differs from `guard` on 41/56 traces at 0.526 and on 0/56 at 1.0101. | The multi-corpus driver always passes an explicit LatencyModel and records `break_even` in every row. | WP10 |
| M20 (science) | D10.5 (`DEFAULT_CAPACITY=2048`, `simulator.py:276`) and D13 (depth search ≤16, `DEFAULT_READ_AHEAD_DEPTH=8`, `:267`) | Hit ratio does not saturate on real traces: msr_proj_0 `none` is 25.1% at 1 MiB and 67.2% at 512 MiB. Depth 256 beats depth 8 by +26.2 pp. | Keep the defaults, which the tests pin. The sweep reports a capacity x depth grid, and the docs correct D10.5 and D13. | WP10, WP15 |
| M21 (minor) | `lstm.py:35-51` | `bucket_of` maps deltas 0 and 1 to the same class, and class 26 is never a target. | Document it in D19. Do not change it (the artifact stays compatible). | WP15 |

**Gaps `main` leaves:**
- There is no real I/O.
- There is no oracleGeneral loader.
- There are no CLI, report or sweep tests, which is why M1–M6 survived a green suite.
- There are no confidence intervals.
- The oracle bound is vacuous (about 100%).
- There is no metric for device work.
- The drift report is saturated, with every row at lag 0 and accuracy 1.0.

### 1.2 Branch `pivot` — adaptive product-page prefetching on RetailRocket (`src/webprefetch`, pandas/sklearn)

**Verdict: KEEP AS THE WEB STUDY.** The protocol is leakage-audited (0 cross-split sessions, index from
split 0, ranker on split 1, selection on split 2, test on split 3). The results are deterministic and
reproduce exactly: 0 of 61 committed metric rows differ, and the headline is cache 8, cost 0.01,
adaptive_0.01 − fixed1 = +0.00358 [0.00333, 0.00382]. There are 16 tests, all passing in 4.85 s.

Its conclusions need re-baselining. Its "candidate-coverage ceiling" is mostly index staleness. Its learned
ranker adds nothing over counting. It does no real network or I/O.

**Kept:**
- `data.py` (SHA-pinned acquisition `:22-51`, `partition_sessions` `:54`, `prepare` `:73`).
- `model.py` (`CandidateIndex` `:29`, `fit_index` `:46`, `train_ranker` `:75`, `FastScorer` `:200`).
- `simulator.py` (`Policy` `:18-41`, `Counters` `:45`, `Cache` `:72-111`, `replay` `:134`,
  `replay_capacities` `:198`, `paired_bootstrap` `:278`).
- `benchmark.py`, `diagnostics.py`, `cli.py`, `demo_server.py`, `demo/`, `audit/GRILL.md`, `NOTICE.md`,
  and the tests.

**Bugs to fix (verified locations):**

| # | Location | Defect | Fix | WP |
|---|---|---|---|---|
| P1 | `benchmark.py:106` (validation `capacity=8`) and `:124-125` (frozen across caps 2–32) | Policies are selected only at cap 8 and reused at every capacity. At cap 4, cost 0, the reported win reverses (fixed2 0.12145 > adaptive 0.11949). At caps 16/32, cost 0, adaptive is compared against fixed4 although fixed8 ties it at 0.14367. | Select per capacity on split 2. Report against the best fixed depth at that capacity and against `fixed0`. | WP6 |
| P2 | `model.py:69,93` (label = `item == next_item`; feature `candidate == prev_item`) and `simulator.py:182-183` (prefetches `ranked[:depth]` without regard to cache residency) | The ranker learns back-navigation, which is already cached. As a depth-1 prefetcher it loses to transition top-1: −0.00061 [−0.00075, −0.00047]. | (a) A cache-aware controller that filters resident items before choosing depth: +0.00026 [0.00018, 0.00034]. (b) Optionally retrain with the label "next ≠ prev and next ≠ current". (c) Report transition top-1 as a first-class baseline. | WP5 (sim), WP6 (bench) |
| P3 | `model.py:47` (`history = views.loc[views.split.eq(0)]`) | The candidate index is frozen at the oldest 40% of history. Test coverage@8 is 9.08% with the split-0 index, 13.99% with an index over splits 0–2, and 17.53% with a causal online index. Online fixed4 minus logistic fixed4 is +0.01523 [0.01483, 0.01563], about 4x the controller win. | `fit_index(views, max_split=0)` plus an `OnlineIndex` that adds a pair only after its `next_timestamp` is earlier than the decision time. | WP5 |
| P4 | `simulator.py:30-40` (adaptive objective) | λ is charged on every nominal slot, including no-op prefetches of already-resident items (16% of fixed1 slots). | Fixed by the cache-aware variant in P2(a). Keep the old variant under its existing name so the committed rows still reproduce. | WP5 |
| P5 | `demo_server.py:68` (`Policy(name, "adaptive", cost)`) | The demo's adaptive uses penalty = cost, not the validation-selected penalty. At cost 0 it behaves as fixed8: 72,833 issued versus 33,721. | Map the UI cost to `benchmark.json["validation_selection"]`. Keep "adaptive (penalty = cost)" as a separately labelled option. Extend the parity test to costs 0, 0.05 and 0.25. | WP6 |
| P6 | `simulator.py:250-252` and the report text in `cli.py:35-110` | ECE of 0.0009 hides overconfidence: 99.8% of candidates fall in bin 0, while bin 3 predicts 0.343 and observes 0.192. | Report the reliability table for p ≥ 0.1 and a Brier skill score. ECE stays only as a secondary number. | WP6 |
| P7 | `simulator.py:255-275` (`break_even_waste`) | When adaptive dominates, a negative threshold is labelled `"wins_above"`. Zero-denominator bootstrap draws are dropped silently. | Add a `"dominates"` direction and count the dropped draws. | WP5 |
| P8 | `benchmark.py:139` | `_real_session_demo` takes `test.iloc[:test_views]`, a raw prefix that can truncate a session. | Use `complete_views` (`simulator.py:114`). | WP6 |
| P9 | `benchmark.py:87-247` (single process) and `:166` (redundant 100k "subset" replay) | Runs for more than 900 s single-threaded. Session-sharded replay is exact and took 93 s on 100 processes. | `replay_parallel(..., workers=N)`. Keep the subset replay behind a flag. | WP5, WP6 |
| P10 | `data.py:108,119`, `model.py:50,57` (`sort_values(..., inplace=True)`) | pandas 3.1 deprecation warnings. | Assign the result instead. | WP5 |
| P11 | `outputs/report.md` and the `cli.py` report text | Cap-2 rows are labelled "adaptive wins", although every prefetch policy loses to `fixed0` there (−0.017 to −0.021). | Add a "loses to no prefetch" label whenever adaptive − fixed0 is below 0 with a CI excluding 0. | WP6 |
| P12 | `simulator.py:151-154` (a new Cache per session) | Session-scoped caches understate browser reuse. With an infinite cache the hit ratio is 11.58% per session and 18.31% per visitor. | Add a `cache_scope="session"\|"visitor"` option. The default stays `session`. | WP5 |
| P13 | `model.py:76` (`max_decisions=30000`) | Only 30k of 532,844 decisions are used, without justification. | Add a learning-curve diagnostic at 30k, 100k and 300k. | WP6 |
| P14 | Ranking metrics `simulator.py:162-169`, `model.py:131-141` | They count revisits that are already cached. SASRec scores higher hit@1 but a lower cache hit ratio than the GRU. | Add `novel_hit_at_1` and `novel_coverage_at_8` (next item not among the session's earlier items). | WP5 |
| P15 | `outputs/session_parity.json` | Committed by hand. No script generates it. | Add a `webprefetch parity` command that writes it. | WP6 |

**Gaps `pivot` leaves:**
- There is no HTTP, no bytes, no latency and no OS involvement; cost is an assumed hit-equivalent.
- There is no fresh or online index and no sequence model.
- No metric is shared with the disk study.
- The scratch HTTP prototype (`$EVAL/web-realexp/`) works. It showed exact parity and gave valid timings
  only at ramp 180 s; the ramp-2 s runs are INVALID and must never be cited.

---

## 2. DISK study implementation plan (main's code carried into `master`)

### 2.1 oracleGeneral loader and request reconstruction (WP2)

Port the loader from `$EVAL/blocktrace-scout/oracle_loader.py`, which was verified against main's MSR loader
and against numpy counts, into `src/adaptive_prefetch/trace.py`:

```python
ORACLE_TAG_MASK = (1 << 40) - 1
def iter_oracle_chunks(path, chunk_requests=250_000, *, size_unit="auto", merge="auto",
                       max_merge_sectors=1024, max_requests=None, skip_requests=0,
                       stream_id=None) -> Iterator[list[Request]]
def oracle_profile(path, probe=4096) -> dict   # {size_unit, merge, corpus, tagged}
```

- Decompression: use `zstandard.ZstdDecompressor().stream_reader`, importing it inside the function so the
  core stays stdlib-only. If the module is missing, fall back to a `zstd -dc` subprocess
  (`/usr/bin/zstd` exists on this machine). If neither works, raise `RuntimeError`. A file without a
  `.zst` suffix is read raw. A truncated trailing record raises an error that gives the record index.
- Address: `lba = (obj_id & ORACLE_TAG_MASK) // 512`. An offset that is not 512-aligned raises.
- Size unit, `auto`: treat the unit as sectors when more than half of the first 4096 `obj_size` values are
  not multiples of 512 (true for 9 FIU files). Otherwise treat it as bytes (MSR, Alibaba, Tencent,
  fiu_cheetah). Then also compute the next-record contiguity rate under both interpretations, and raise if
  the chosen unit has a **strictly** lower rate (ties, e.g. both 0 on a very random 4096-record probe,
  warn but do not raise). The scout measured 0.000 contiguity under the wrong unit in every
  file, so this check is a sharp guard.
- Merge vocabulary (Critic, so that cross-checks are well defined): `merge="auto"` is the **canonical**
  reconstruction used for every headline number; `merge="none"` means raw records (= `$EVAL/main-science`
  variant `raw`); `merge="same_second"` with `max_merge_sectors=None` (uncapped) means main-science
  variant `merged`. The `same_second` default cap of 1024 is a separate sensitivity setting.
- Merge, `auto`: use `pages` when the file name starts with `fiu_`, otherwise `none`. `pages` coalesces
  consecutive records that have equal ts and start equal to the previous end, while the run stays at or
  below `max_merge_sectors`. Merging happens **before** chunking, so runs that cross a chunk boundary are
  still merged. `merge="same_second"` applies the same rule to any corpus; it is used for the sensitivity
  row (main-science "merged" view).
- Request fields: `operation="R"`, `timestamp_ms=(ts-ts0)*1000`, `service_ms=None`, `pattern=None`, and
  `stream_id` set to the file stem. `_validate_order` runs on every chunk **and** across chunk boundaries,
  by carrying the last ts. This also closes the cross-chunk gap noted for M10.
- `skip_requests` gives a second disjoint slice (records 1M–2M) for the robustness check.
- Wiring:
  - `iter_csv_chunks(path, "oracle", chunk_requests)` routes to `iter_oracle_chunks`.
  - `load_csv(path, "oracle")` is `list(chain(*iter_oracle_chunks(path)))`.
  - `oracle` is never auto-detected.
  - Add it to the CLI `--format` choices. WP1 owns `cli.py`; WP2 hands it the string.
- New module `src/adaptive_prefetch/corpora.py`, holding only data and pure functions:
  - `CORPUS_OF(stem)`.
  - `TWIN_GROUPS`: alibaba {13,16,18,20,22}, {11,15}, {1,5}, {8,19,24}; tencent {1162,1591,1911}.
  - `PAPER_SUBSET`, 21 traces chosen by the blocktrace-scout for sequentiality diversity:
    - msr: hm_0, prn_1, proj_0, web_2, prxy_0
    - fiu: casa, homes, online, madmax
    - alibaba: 23, 19, 14, 17, 13, and 3 as the drift case
    - tencent: 1791, 1757, 2145, 1162, 1894
  - `discover_block_traces(root="data/block-traces")`.
- Tests: new file `tests/test_disk_oracle.py`, built from tiny `struct.pack` files in a tempdir.
  - (a) Byte-unit records give one Request each.
  - (b) FIU sector records: same-second contiguous runs merge, a change of ts breaks a run, and a
    1032-sector run splits at 1024.
  - (c) The tag `(130<<40)+4096` gives lba 8.
  - (d) A truncated file raises.
  - (e) Decreasing ts raises, including across chunk boundaries.
  - (f) Streaming and `load_csv` give identical results.
  - (g) A wrong-unit override raises the contiguity guard.
  - (h) `skipUnless` the real file exists: the first 1000 lbas of `msr_hm_0` equal `load_csv(msr-cambridge1-sample.csv,'msr')` 1000/1000.
  - (i) `skipUnless` zstandard is available: a written `.zst` round-trips.
  - (j) `skipUnless` the real file exists: `fiu_cheetah` median request size is ≥ 8 blocks, and it is
    detected as bytes.
- Throughput target: at least 150k requests/s and at most 300 MB RSS at chunk 250k. The prototype measured
  210–260k/s and 66–240 MB.

### 2.2 Simulator extensions (WP3)

All additions are additive. Default outputs must stay bit-identical, and the existing 131 tests pin this.

1. **Cache injection hook.** Add `cache_factory=LRUCache` to `replay` (cache built at `:357`) and to
   `replay_stream`. Test that the default output is identical for every `MODES` entry.
2. **Device-I/O accounting (the D18 metric).** Add these fields to `Metrics`. The logic is ported from
   `$EVAL/main-science/run4.py` (`IOCache`), which had 0 hit-ratio mismatches against `sweep2.jsonl`.
   - `demand_ios`: the number of maximal runs of contiguous missing blocks within each read request.
   - `prefetch_ios`: the number of maximal runs of contiguous blocks newly prefetched after each request.
   - `prefetch_ios_mergeable`: prefetch runs that start at `request.end` while that request had a miss
     run ending there.
   - `device_ios = demand_ios + prefetch_ios - prefetch_ios_mergeable`.
   - `blocks_moved = demand-missed blocks + prefetched blocks`.
   `replay_stream` must keep these identical, so extend `test_every_supported_mode_agrees_across_both_paths`
   (`tests/test_project.py:1374`).
3. **`IOCostModel(t_pos_us, t_xfer_us_per_block)`** with `device_time_us(metrics)` and
   `kappa_star(policy, none)`, where kappa* = Δblocks/(−ΔIOs) is the critical ratio of positioning time to
   transfer time. It is constructed from `results/realio/device_<dev>.json` produced by WP8. It does **not**
   replace `LatencyModel`. The legacy per-block fields, the D11 break-even and the
   `test_cost_ranking_is_invariant_to_the_calibration_scale` and `test_prefetch_multiple_does_change_the_cost_ranking`
   tests are all unchanged.
4. **New mode `linux_ra`.** A Linux-ondemand-style ramping read-ahead emulation, ported from
   `$EVAL/main-science/run2.py`'s `linux_ra` and validated there: 99.83% hit with 628 prefetch I/Os on a
   sequential scan, and 0 prefetches on random input. **(Critic) It cannot be a plain
   `next_candidates(request)` predictor.** `run2.py:120-185` is its own loop that reads cache state:
   the pre-read miss list, `marker in c.blocks and c.blocks[marker]` (marker still resident and
   unused), and a backwards residency scan (`page_cache_prev_miss`). WP3 must therefore add an optional
   cache-aware predictor hook to **both** `replay` and `replay_stream`: `observe(request, missed_blocks,
   cache_view)`, where `cache_view` is read-only and is called before prefetch, after the demand reads,
   and only for `index >= window_size`. `LinuxReadahead` in `baselines.py` implements that hook.
   Acceptance: on non-FIU traces it reproduces the `sweep2.jsonl` `linuxra` rows exactly (same
   `variant=raw`, cap, max_blocks), and replay/replay_stream stay bit-identical. It models per-stream window state, the
   async marker (PG_readahead), window doubling via a `get_next_ra_size`-like rule, and the
   `page_cache_prev_miss`-style sequential check. It is labelled everywhere as an emulation of
   mm/readahead.c **policy**, not the kernel. Register it in the four places AGENTS.md lists:
   - `benchmark.MODES`, **appended** after `deep` (keeps `rows[5]=="lstm"`)
   - `sweep.RANKING_MODES`
   - `simulator.MODES_SUPPORTED`
   - `_make_predictor`
   It is not gated.
5. **Parameterised depth in the sweep.** `deep` accepts `read_ahead_depth`, which the sweep varies. No new
   mode names are added. (Critic: `replay`/`replay_stream` already take `read_ahead_depth`
   (`simulator.py:331,495`), but `sweep_collection` (`sweep.py:69-77`) does not. Add the parameter there,
   plus `max_blocks` for `linux_ra`, and record both in every row.)
6. The M2, M3, M9, M11, M12, M13, M14 and M16 fixes from §1.1.
7. `sweep.py`:
   - `discover_traces(directory, patterns=("*.revised","*.oracleGeneral.zst"))`
   - `capped_chunks(path, cap, chunk, format=…)`, so it no longer hardcodes `"revised"` (`:59`)
   - a `latency_model=` parameter, mandatory for oracle traces
   - meta columns per row: `corpus, merge, size_unit, requests, blocks, hours_covered, ops_known=False, ts_resolution_s=1, break_even`.

### 2.3 Multi-corpus sweep with caps and bootstrap CIs (WP10 builds it, E1 runs it)

The new module `src/adaptive_prefetch/multicorpus.py` and the entry `python -m adaptive_prefetch.multicorpus`
contain no edits to `sweep.py`. It calls `sweep`/`replay_stream` as a library.

- Job unit: (trace, slice, merge, capacity, mode, depth). It uses a `multiprocessing` Pool with
  `--workers` (default `min(100, cpu_count()-8)`), longest-first scheduling by the block budget from
  `caps.jsonl`, and a resumable `results/disk/sweep.jsonl`. A job whose key is already present is skipped.
  Each row records git sha, host load average at start and end, and wall time.
- Latency: always the explicit `LatencyModel(5,100,50)`, labelled `assumed`. The gated modes also run at
  the break-even-1.0101 model (`LatencyModel(1,100,100)`, as in `$EVAL/main-science/run3.py`). Rows carry
  `break_even`.
- **Configurations:**
  - **C1 collection table**: 56 traces, cap **1,000,000** canonical requests (`merge="auto"`: FIU pages
    merged with cap 1024, every other corpus as-is), capacities {2048 (1 MiB), 131072 (64 MiB)}.
    - Modes: the 12 existing modes, plus `linux_ra` (max 64/256/1024), plus `deep` at depths {32, 256}.
      **`lstm` and `learned_depth` are always `skipped: evaluated in E2 only` in C1.** (Critic: E2 trains
      them on the first 60% of each trace's 2M prefix, which overlaps C1's first 1M requests, so
      scoring them in C1 would be in-sample.)
    - Per trace, also report the Belady/MIN demand hit ratio (no prefetch) at each capacity, as the
      reference the oracle column fails to give (M8). Compute it from the reconstructed **block** stream
      with a two-pass next-use array and a max-heap. Do not use `next_access_vtime`: it is keyed per
      obj_id (request start offset), not per 512-B block. It is a demand-only reference, not a bound on
      prefetching policies.
    - Budget: 2.82 G blocks per mode at 0.42–1.18 M blocks/s gives about 1.1 CPU-h per mode-capacity, or
      about 40–60 CPU-h in total.
    - On 100 workers that is **about 1–1.5 h wall**. The largest single jobs are alibabaBlock_6,
      alibabaBlock_0 and tencent_1162 (128–143 M blocks), about 3–6 min each, longer for deep@256.
  - **C2 sensitivity grid**: 56 traces at cap 250k.
    - Capacities {2048, 16384, 131072, 1048576}, depths {0, 2, 8, 32, 64, 128, 256, 512} and linux_ra
      max {64, 256, 1024}, for the canonical rule (`auto`) and the `same_second` (uncapped) sensitivity
      view. `merge="none"` is run only for FIU, as the "why pages must be merged" row.
    - About 4.9 k jobs and 0.72 G blocks per configuration, roughly 25–35 CPU-h, **about 30–45 min wall**.
    - Reuse: `$EVAL/main-science/grid.jsonl` (123 cells) and `sweep2.jsonl` (1,624 rows, 250k) are the
      acceptance cross-check. Cells with identical keys must match exactly, under the mapping
      sweep2 `raw` = master `merge="none"` and sweep2 `merged` = `merge="same_second",
      max_merge_sectors=None`. For MSR/Alibaba/Tencent `raw` also equals canonical. sweep2 used the
      default 5/100/50 LatencyModel and `make_model(seed=42)`, so these must be the defaults too (M16 stays
      opt-in).
  - **C3 robustness**: the 21 `PAPER_SUBSET` traces on slice 2 (`skip_requests=1_000_000`, cap 1M; Critic:
    traces with fewer than 2M canonical requests, i.e. fiu_madmax, fiu_online and fiu_webresearch, have no
    full second slice and are reported as `n/a`, never as a short slice), and on
    the whole file where the trace is at most 4 M requests. Covered hours are reported per trace.
    About 20 min wall.
  - **C4 FIU merge-cap sensitivity**: caps 256 and 1024 sectors on 10 FIU traces. About 5 min.
- Memory: at most about 1.5 GB per job once M9 is fixed, so 100 workers fit in 251 GB. Run `adaptive` with
  the bounded WindowContext only. The 4 M-entry bound **does** change `adaptive` features on traces whose
  1 M-request footprint exceeds 4 M blocks (e.g. msr_proj_2, 123 M blocks), so it is a behaviour change.
  Record `age_table_bound` and `age_table_evictions` in each row. Never run whole files across the board: msr_prxy_1 is 4.79 G blocks,
  fiu_cheetah 3.54 G.
- **Analysis**: `src/adaptive_prefetch/analysis.py` uses `prefetch_common.stats`. It writes
  `results/disk/table_main.csv`, `table_by_corpus.csv`, `deltas.csv`, `winners.csv` and `kappa.csv`.
  - For every headline delta it reports:
    - a paired bootstrap 95% CI over traces (10,000 resamples, seed 42)
    - a two-level **cluster bootstrap** (resample corpora, then traces, with twins collapsed to one
      representative or down-weighted to 1/|group|)
    - a Wilcoxon signed-rank p (scipy optional; omitted with a note if it is missing)
    - the win count
  - The headline deltas:
    - `adaptive` and `guard` vs `sequential`
    - `deep@{8,32,256}` and `linux_ra` vs `none`
    - `device_ios` ratio vs `none`
    - `blocks_moved` ratio
    - precision, wasted share
    - kappa* per (trace, policy)
  - Pooled and per-corpus results appear side by side, at both 1 MiB and 64 MiB.
- Acceptance:
  - (1) msr_hm_0, first 250k, cap 2048, `deep@8` gives hit **0.357231** and prefetches **901487**,
    matching `sweep2.jsonl`.
  - (2) `guard == depth_adaptive` byte-identical at break-even 1.0101 on all traces (D14.1).
  - (3) A sequential 8-block scan under `linux_ra` gives hit ≥ 0.998 and fewer than 1000 prefetch I/Os.
  - (4) A killed and restarted run resumes without duplicate keys.
  - (5) A 3-trace x 3-mode x 20k smoke run finishes in under 3 min.

### 2.4 Linux readahead baseline — **justified**, in two forms

- **Emulated** (`linux_ra`, simulator mode, §2.2.4). It is the only policy that reduces device I/Os:
  0.659x of `none`, fewer on 55/56 traces, at 1.15x the blocks moved. It is the strongest non-oracle
  hit-ratio policy after unbounded fixed depth.
- **Real kernel** (`kernel_ra` and `kernel_seq` pseudo-modes in the real testbed, §2.6). These are buffered
  reads with `POSIX_FADV_NORMAL` or `POSIX_FADV_SEQUENTIAL` and no user prefetch, under the same memory
  cgroup. The prototype measured on hm_0, SSD, closed loop, 20k requests: full-hit fraction 0.5752 and
  p50 6.6 µs, against Backend-B `deep` at 0.5772 and p50 17.2 µs. This is the baseline any real
  prefetcher must beat. It also gives the paper a check of the emulation against the kernel, comparing
  hit-request fractions.

### 2.5 GPU LSTM training (WP9 builds it, E2 runs it)

Two models. Both use torch as optional, the existing artifact-version discipline and a JSON sidecar.

1. **The shipped `lstm` (bucketed delta), retrained honestly.** In `lstm.py`:
   - `train_lstm(..., device="auto", num_threads=4, val_fraction=0.2)`.
   - A chronological split per real trace: train on the first 60% of a 2M-record prefix, validate on the
     next 20%, test on the last 20% (300k requests).
   - The `resolve_lstm` leak fix (M7) is in WP1. WP9 adds `train-lstm --trace F --format oracle --device cuda`.
   - Expected result, already measured in `$EVAL/gpu-models/v1_full.jsonl`: bucket top-1 0.61–0.72 against a
     majority baseline of 0.44–0.56. Hit ratio is still below `deep@8` on 4/4 traces (−1.3 to −51.5 pp),
     because the decoder emits at most 2 blocks. This is reported as a negative result (D19).
2. **New `learned_depth` mode** (`src/adaptive_prefetch/depth_model.py`), ported from
   `$EVAL/gpu-models/disk_seq_v2.py`.
   - Features: 5 per request (signed-log start delta, signed-log gap from the previous end, log2 size,
     is_contiguous, log dt).
   - Targets: 4 read-ahead segments past the request end, [0,8), [8,16), [16,32), [32,64), each labelled
     "read within horizon H = capacity / mean request size".
   - Network: `DepthNet` = Linear(5,128) → LSTM(128,128,2) → seg(4) and gap(256) heads, 298,500 params.
   - Training: AdamW 2e-3, bf16 autocast, early stop on validation loss.
   - Threshold: grid-selected on **validation only** through `replay()`.
   - `DepthPrefetcher.next_candidates` re-decides every `decide_every=32` requests to amortise inference,
     which costs 477–1904 µs at batch 1 on CPU.
   - Register it in the 4 places (appended after `linux_ra`).
   - Required baseline **VCF** (validation-chosen fixed depth, grid {0,2,4,8,16,32,64,128,256}) and a
     request-relative depth `k·size_blocks`, k ∈ {0.5, 1, 2}.
   - Pilot result (single seed, 8 traces, `v2_full.jsonl` and `fixed_depth.jsonl`): learned_depth 40.50%
     vs VCF 37.88% (+2.62 pp mean; 3 clear wins, 4 ties, 1 loss) vs deep8 26.11%.
     **The paper's headline must be ML vs VCF, not vs deep8.**
   - (Critic) **Fair action space.** The pilot capped both the learned segments and the VCF grid at
     64 blocks, and VCF chose the cap on 7/8 traces. If VCF now goes up to 256, `learned_depth` must have
     segments up to 256 as well: add [64,128) and [128,256), 6 heads. Otherwise the comparison is rigged
     against the model. Report both the 64-capped and the 256-capped comparisons.
   - (Critic) **Prefix definition.** "2M" means 2,000,000 **canonical requests** (FIU merged). Traces
     shorter than that (fiu_madmax 219k, fiu_online 1.56M, fiu_webresearch 1.06M) use the whole trace
     with the same 60/20/20 split, and the row records the actual n.
   - (Critic) **Test-slice discipline everywhere.** A trained artifact may only be scored on its own
     trace's test segment (requests 1.6M–2.0M) or on a trace it never saw. This applies to C1 (where it
     is skipped), to E3 (realio replays the test segment via `skip_requests`, not the first 20k) and to
     any demo.
   - Tests:
     - `seg_labels` gives all 1 on a scan and a mean < 0.05 on random input
     - causality: perturbing a future request does not change the candidates at i
     - replay and replay_stream are bit-identical
     - the first window never prefetches
     - `read_ahead` returns exactly `depth` blocks
     - a stale `artifact_version` raises "retrain"
     - threshold selection never touches test indices
     - `len(rows)==len(MODES)` and `rows[5]` is still `lstm`
     The GPU-dependent tests use `skipUnless(torch.cuda.is_available())`, and every CPU path is tested
     on a tiny model.
- E2 runtime: 21 traces x 5 seeds x (label build 30–70 s CPU plus training 20–46 s GPU), or about
  **2–3 h**, mostly label construction, which runs in parallel on CPU. Run one GPU job at a time. Use
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and catch OOM with a retry, because the GPU is shared.
  Report mean ± 95% CI over the seeds, with a GPU contention note (an `nvidia-smi` snapshot is stored per
  run).

### 2.6 REAL Linux prefetch testbed (WP8 builds it, E3 runs it) — what makes the disk study "functional"

Package `src/adaptive_prefetch/realio/`, with entry point `python -m adaptive_prefetch.realio`. It is
stdlib-only and Linux-only, and it skips cleanly elsewhere. It is ported from `$EVAL/realio/realio_proto.py`,
`t1–t5`, `calib.py` and `mkfile.py`.

| file | contents |
|---|---|
| `decisions.py` | `RecordingLRU(LRUCache)` logs `(op, block, hit)`. `record_plan(chunks, mode, **kw) -> (Metrics, plan)` with per-request `(hit_flags, prefetch_runs)`. Uses WP3's `cache_factory`. |
| `layout.py` | `make_backing_file(path, gib)` writes page-stamped data with O_DIRECT at about 200 MB/s on the HDD. `RegionMapper` packs 1 MiB trace regions in first-touch order, counts wraps and refuses `wrapped>0` without `--allow-alias`. |
| `clock.py` | Think-time models: `closed`, `fixed:US`, `spread:SCALE:CAP_US` (spreads the n requests of one 1-s tick over 1/n s each). Sets `prctl(PR_SET_TIMERSLACK, 1000)`, sleeps then spins, and logs `think_error_us`. |
| `backend_odirect.py` | **Backend B**, faithful to the simulator: a user-space LRU with the exact `LRUCache` semantics. Demand misses are coalesced into O_DIRECT `preadv` reads. Prefetch runs go to `ThreadPoolExecutor(4)`, and their entries stay in-flight until done. Each demand **asserts** `hit == plan flag`, raising `CacheDivergence` otherwise. It records late hits, wait time, and a preallocated 1 MiB-slab aligned mmap pool. Bookkeeping time is reported in its own `cpu_bookkeeping_us` column. |
| `backend_pagecache.py` | **Backend A**, the real kernel page cache. Buffered reads, `POSIX_FADV_RANDOM` by default, prefetch as `WILLNEED` in **≤256 KiB chunks** (from `/sys/block/<dev>/queue/{read_ahead_kb,max_sectors_kb}`). Hit classification uses `cachestat(2)` (syscall 451, falling back to `mincore`) plus a `preadv2(RWF_NOWAIT)` probe, giving ready, late or miss. Pseudo-modes `kernel_ra` (FADV_NORMAL) and `kernel_seq` (FADV_SEQUENTIAL). **Mandatory** `FADV_DONTNEED` of the whole backing file before every run, because pages charged to another cgroup are not re-charged. Re-execs itself under `systemd-run --user --scope -q -p MemoryMax=<cap+256M> -p MemorySwapMax=0`, then writes `memory.max = memory.current + capacity + 7 MiB` (measured budget offset). Records `memory.stat file` and `memory.events max`. |
| `online.py` | **Online (closed-loop) policy driver for Backend A.** The policy object (`deep`, `linux_ra`, `stride`, `markov`, `learned_depth`, or `guard` through `GateController`) runs **inline**. Each real request is fed to `next_candidates`, and the WILLNEED is issued immediately. Gated modes are fed **measured** useful prefetches (pages demanded while resident), recorded as deltas per the D15 invariant. This is the "real Linux prefetcher" with no precomputed plan, so it answers the objection that the plan is computed offline. |
| `devstat.py` | Reads `/proc/diskstats` deltas (own vs foreign I/O; flags the run when foreign I/O exceeds 1%). Implements `flush_drive(path, mib=1024)` by reading an unused file region with O_DIRECT, because the drive cache cannot be flushed without root. |
| `calibrate.py` | Device calibration: QD1 O_DIRECT random 4 KiB, sequential 4 KiB and 64 KiB, 1 MiB throughput, giving `t_pos` and `t_xfer` in `results/realio/device_<dev>.json` for `IOCostModel`. The prototype measured HDD random 4K p50 5907 µs, seq 4K 31 µs, seq 64K 235.5 µs, 252.8 MB/s; SSD random 4K 122 µs. |
| `runner.py` / `report.py` / `__main__.py` | Sub-commands `mkfile`, `calibrate`, `run`, `grid` and `report`. One row per (device, backend, mode, think, repeat), with randomised mode order and repeats ≥ 3. Writes `results/realio/realio.csv` (columns in §6 WP8) and prints conclusions only (main's report convention). |

**Parity versus the simulator** (the core validity statement):
- Backend B: block-level hit/miss sequence and `useful_prefetches` are **identical** to `replay()`,
  asserted on every request.
- Backend A: hit ratio is a *measured divergence* (MGLRU, 4 KiB pages), reported as such, never matched.
- Both: (i) a prediction block calibrated from the same run's `none` mode, giving pred = f_hit·c_hit +
  (1−f_hit)·c_miss and its error against the measured mean; (ii) the effective `prefetch_multiple`
  k solved from measurements, which answers D11 empirically (prototype: about 2.9 SSD closed, 0.94 HDD
  closed, 0.40 HDD spread); (iii) Spearman rank correlation across policies between simulated
  `device_time_us` (IOCostModel) and measured mean latency, and between simulated hit ratio and measured
  mean latency.

**Devices and placement:**
- Primary: HDD `/dev/sdb` (`/mnt/b18edc65…/os-project-prefetch/`), idle with 0 foreign I/O.
- Replica: HDD `/dev/sde` (`/mnt/c609e3de…/os-project-prefetch/`), the same model, so the result can be
  confirmed on a second drive.
- Secondary: SSD (scratch or `$REPO/.realio/`, gitignored), noisy, reported with `foreign_io_frac`.
- The backing file is at most 32 GiB per device.

**Traces:**
- msr_hm_0, which also matches the committed sample
- msr_web_2 (64 KiB sequential)
- msr_prn_0 (more random). Critic: it is not in `PAPER_SUBSET`, which has prn_1, so E2 must also train
  it (22 traces) or E3 has no `learned_depth` row for it.
- alibabaBlock_17
- fiu_homes (merged)
Each runs the first 20k requests, for footprint and wrap reasons. (Critic) `learned_depth` runs on its
trace's E2 **test** segment instead (`--skip-requests 1600000 --requests 20000`), and so do the `none`,
`deep@64` and `linux_ra` rows it is compared against, so that the comparison uses the same requests.
**Cache capacity for E3 is 65,536 blocks (32 MiB)**, as in the prototype (`$EVAL/realio/grid.sh:6`
`CAP=65536`), because the cgroup page-cache budget is MemoryMax minus about 7 MiB (8M → 0 MiB resident),
so a 1 MiB cache cannot be emulated in Backend A. The simulator plan for every E3 row is computed at
the same 65,536 blocks, which differs from E1's 2,048. The paper must say so.
`linux_ra` in Backend B and in `online.py` keeps its window state against the **user-space shadow
LRU**, not the kernel page cache. The kernel's own policy is measured by `kernel_ra` / `kernel_seq`.

**Modes:**
- none
- sequential
- deep@8
- deep@64
- linux_ra
- guard
- adaptive_evidence
- learned_depth, if E2 produced artifacts
- kernel_ra (A)
- kernel_seq (A)
- online-guard (A)

**Think time** {closed, spread:1:5000}. **Prototype findings to re-test with repeats:**
- Closed loop, hm_0 HDD: deep raises hit ratio 0.2652 → 0.3212, yet mean latency rises 588 → 646 µs
  (3 repeats: deep 631.0/634.0/630.9 vs none 595.1/599.0/604.2 µs).
- Spread think time: HDD flat, SSD −25%.
- The hit-ratio-based prediction underestimates the closed-loop mean by 33% (HDD) and 49% (SSD).

**E3 runtime (Critic: recomputed; the earlier "4–6 h" was about 3x too low).** Prototype timings:
about 6–7 s per 10k-request HDD closed run and 29–36 s per spread run, so at 20k requests about 13 s
closed and about 65 s spread. There are 8 Backend-B modes and 11 Backend-A modes, so 19 runs per
(trace, think, rep). Each run also needs a 1 GiB drive flush (about 4–5 s) and a DONTNEED. Per HDD:
5 traces x 19 x 3 reps x (13 + 65 + 5 s) ≈ **6.6 h**. Two full HDDs would be about 13 h.
Revised grid:
- **sdb (primary)**: the full grid, about 6.6 h.
- **sde (replica)**: reduced to confirm direction. 2 traces (msr_hm_0, msr_web_2), modes none, deep@8,
  linux_ra, guard (B) and kernel_ra (A), both think models, 3 reps. About 1 h.
- **SSD**: closed loop only, hm_0 and web_2, the same 5 modes, 3 reps. About 15 min.
Total about **8 h**. Run it exclusively, with no E1, E4 or E6 at the same time, and record the load average.

---

## 3. WEB study implementation plan (pivot's code carried into `master`)

### 3.1 Bug fixes and core extensions (WP5 core, WP6 benchmark/report/demo)

WP5 (core, files `simulator.py`, `model.py`, `data.py`, new `cachecore.py`, `parallel.py`, `index.py`):
- **`cachecore.py`**: move `Policy`, `Counters` and `Cache` (`simulator.py:18-111`) here, numpy-free.
  `simulator` re-exports them so the existing imports and tests keep working. Add an optional
  `on_evict(item, speculative)` callback so the live client wraps the **same** class (removes the hand
  copy in `$EVAL/web-realexp/client.py:82-140`). Add `Policy` kinds:
  - `adaptive_ca`: cache-aware, filters resident items, then applies the same prefix-sum objective (P2a/P4)
  - `feedback`: the disk precision gate ported from `$EVAL/pivot-science/scripts/gate_replay.py`
    - buckets = (rank, score bin), edges [0,.005,.01,.02,.05,.1,.2,.4,1]
    - a bucket opens when its Beta(1,1) posterior mean exceeds c, optionally net of a pollution term
    - exploration ε is selected on validation
    - outcomes are credited at hit, eviction or session end
    - state is shared across sessions, so it requires the global-time replay order
- **`index.py`**:
  - `fit_index(views, max_split=0)`, whose default reproduces the committed output
  - `OnlineIndex`: adds the (a→b) pair once `next_timestamp < decision_ts`, using logic from
    `$EVAL/pivot-science/scripts/freshness.py` and `precompute.py`
  - `TransitionScorer` variants `trans_s0`, `trans_prior` and `trans_online`
  - unit test: a transition observed after the decision time is never used
- **`parallel.py`**: `replay_parallel(views, index, scorer, policies, capacities, workers)`. Shards by
  `session_id`, runs `replay_capacities` per shard in a Pool, and sums the Counters and session dicts.
  Exactness test: a fixture plus a 20k-view prefix give the same result in parallel and serial
  (`$EVAL/pivot-code/exp/percap.py` reproduced all 61 committed rows in 93 s).
  - `order="session"|"global"`. Global order replays interleaved by timestamp and is required for
    `feedback` and `OnlineIndex` state.
  - `cache_scope="session"|"visitor"` (P12).
- **Metrics** in `Counters.row`:
  - `prefetch_hits_per_view`
  - `novel_hit_at_1` and `novel_coverage_at_8` (P14)
  - `demand_evicted_by_prefetch` (already counted, now surfaced)
  - `c_star = (hit_ratio − hit_ratio_fixed0)/issued_per_view`, filled at report time
- Fix P7 and P10.
- Keep `replay` single-process semantics unchanged. The 16 existing tests and the committed numbers
  (fixed1 cap 8: 65,944 hits, 445,796 issued) must still reproduce.

WP6 (benchmark, report and demo; files `benchmark.py`, `cli.py`, `demo_server.py`, `diagnostics.py`, new `scorers.py`):
- **Per-capacity validation selection (P1).** Validation replay over all capacities; `selection[cap][cost]`.
  `comparisons.csv` gets the columns `vs_best_fixed_same_cap`, `vs_fixed0` and `vs_transition_top1`,
  each with a paired bootstrap CI. Keep a separately labelled `capacity_transfer` table, since that is
  the setting the committed pivot results describe. Acceptance: matches `$EVAL/pivot-code/exp/percap.json`
  (cap 2, cost 0 → fixed0 0.10529 and adaptive_0.1 0.10569; cap 4, cost 0 → fixed2 0.12145 > adaptive
  0.11949).
- **Scorer registry `scorers.py`**: `logistic` (FastScorer), `trans_s0`, `trans_prior`, `trans_online`, and
  `seq` (lazy import of `seqmodel.SequenceScorer`, `skipped:` if no artifact). `benchmark --scorers …`.
- **Decomposition table** ("what moves the needle"), with paired session-bootstrap CIs on test at cap 8:
  - ranker: trans_s0 K1 − logistic K1, expected +0.00074 [0.00060, 0.00087]
  - freshness: online − logistic at fixed4, +0.01523 [0.01483, 0.01563]
  - depth
  - controller at c=0.01, vs the best fixed at the same capacity: +0.00358 (cap 8), +0.00581 (cap 32)
  - feedback gate vs adaptive
  - cache scope: +6.73 pp with an infinite cache
  - sequence model vs trans_online
  Expected values come from `$EVAL/pivot-science/out/*`. The invalid `out/variants.csv` and
  `facts.json:coverage_at8_by_kind` must never be used.
- P5 demo parity, P6 calibration reporting, P8, P9 (benchmark calls `replay_parallel`; the subset replay
  sits behind `--subset-check`), P11 labels, P13 learning curve, P15 `webprefetch parity`.
- `webprefetch freshness` and `webprefetch decompose` sub-commands, so every paper number comes from repo
  code and not from scratch scripts.
- Unified metric output: `results/web/policy_metrics.csv`, one row per (split, scorer, capacity, policy).
  Columns: hit_ratio, prefetch_hits/view, issued/view, precision, wasted/view,
  demand_evicted_by_prefetch/view, c_star, and U(c) for c ∈ {0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05,
  0.1, 0.25, 0.5, 1.0101}, computed by `prefetch_common.utility`.

### 3.2 GPU sequence model (WP11 builds it, E5 runs it)

`src/webprefetch/seqmodel.py`, ported from `$EVAL/gpu-models/web_seq_gpu.py` and `$EVAL/pivot-science/scripts/gru_next.py`.
- `build_windows(views)`: the last K=10 items per row within the session, plus 3 context features. The
  vocabulary comes from training splits only (0 = pad, 1 = OOV).
- `GRURec(V, d=128, h=256)` with a full softmax plus an END class. A SASRec-style 2-layer causal
  transformer is optional.
- **History masking at inference**: exclude the current item and every item already in the session window
  (the cache-aware variant).
- Training protocol, chronological:
  - Validation run: train on split 0 plus the first 90% of split 1, early-stop on MRR@8 over the last 10%
    of split 1, select on split 2.
  - Final run: train on splits 0–2 and evaluate once on split 3.
  - 20 epochs, AdamW 1e-3, batch 2048, bf16.
- Precompute a `{(session_id, position): ranked}` table in one batched GPU pass, so that replay is
  CPU-only and exact.
- Persist a torch `state_dict` plus vocab `.npy` plus a JSON config. No pickle.
- Tests:
  - no item seen only in split ≥ 2 is in the vocabulary
  - the current item is never ranked
  - Σp ≤ 1
  - causality
  - a toy A→B→C corpus reaches hit@1 = 1.0
  - the precomputed table equals live scoring
  - the harness reproduces logistic hit@1 0.038885 / MRR@8 0.055767 / coverage@8 0.093386 on pivot's 10k
    sample
- **Honest expectation**:
  - Pilot validation: GRU-masked fixed4 0.1397 vs logistic 0.1360 vs no-prefetch 0.1162.
  - Pilot test (pivot-science): a GRU on splits 0–2 is *below* online counting (novel recall@8 13.1% vs
    17.8%).
  - The paper reports whatever E5 measures, with seeds 0–4 and CIs.
- E5 runtime: about 260 s per 20 epochs, x 5 seeds x 2 protocols, plus scoring, or about **1–1.5 h GPU**.
  Run it after E2, never concurrently with it.

### 3.3 Real HTTP prefetch experiment with parity (WP12 builds it, E6 runs it)

Package `src/webprefetch/live/`, entry point `python -m webprefetch.live`. It is ported from
`$EVAL/web-realexp/{store.py, prep_plan.py, origin.py (v2), client2.py, analyze.py, test_live_proto.py}`.

- `objects.py`:
  - `object_size(item)`: deterministic lognormal, median 64 KiB, σ 1.0, clipped 1 KiB–4 MiB, labelled
    **ASSUMED**
  - `build_store(...)`: shuffled layout, 4 KiB aligned, fsync
  - `drop_page_cache` (fsync plus DONTNEED)
  - `resident()` (chunked `RWF_NOWAIT`)
  - `willneed(chunk=256 KiB)`
- `plan.py`: `build_plan(...)` → `plan.json` plus `sim.json`, where `sim.json` is
  `simulator.replay(with_sessions=True)` on the **same** sessions. Defaults: min 2 views, seed 42.
- `origin.py`:
  - stdlib asyncio, with a `PreciseTimer` thread (the epoll selector rounds timeouts up to 1 ms)
  - a per-session link that emulates RTT and bandwidth, FIFO or demand-priority, with `busy=True` set
    before `ensure_future` (fixes the race)
  - `/item/<id>` does a real `pread` from the store
  - `/hint?ids=` issues chunked WILLNEED (server-side page-cache prefetch)
  - `/stats`
  - per-response timing headers
  - `--io-threads 8`
- `client.py`:
  - a demand-priority pool: 6 connections, at most 2 for prefetch
  - `SessionCache` wraps `cachecore.Cache`
  - replays recorded inter-view gaps, capped at 2 s by default
  - per-fetch absolute timestamps
  - session starts spread over `--ramp-s` (default **180**) or paced by `--target-rps`
- `analyze.py`:
  - paired session bootstrap
  - model-ideal wait (0 on a hit, rtt + size/bw on a miss) and the decomposition of the excess into disk,
    link queue, timer lag and time outside the server
  - the **validity gate**: per-request server lag p99 ≤ 2 ms, loop-lag p99 ≤ 2 ms, 0 parity mismatches,
    0 connection errors
  - an invalid pass is written with `valid=false` and never compared
- `run.py`:
  - orchestrates sequential passes only, with interleaved policy order and `--reps 3`
  - drops the page cache before each pass
  - prints the planned peak offered rate and refuses when it exceeds `--max-rps-per-shard 300`
    (measured: accurate below about 250–500 req/s, saturation at about 2.2k)
  - `--shards N` uses distinct `127.0.0.x` source addresses, to stay under 28,232 ephemeral ports per pair
- **Parity**: live hits, issued and useful must equal `sim.json` per session, for every policy and every
  pass. The prototype achieved 0 mismatches in every pass.
- **Store placement** (Critic: the "no SSD headline numbers" rule in WP15 applies to E3 only. E6's
  latency is dominated by the emulated network, so SSD-store rows may be headline rows, provided they
  carry `foreign_io_frac`): SSD (scratch), and the HDD `/mnt/9792e947…/os-project-web/` (2k-session store about
  1.43 GB; full test split about 13.6 GB). The HDD run supplies the OS bridge to the disk study:
  server-side WILLNEED and disk contention from speculative reads at about 1–2% precision.
- Outputs in `results/live/`: `live_config.json` (every assumption tagged ASSUMED or MEASURED),
  `live_metrics.csv`, `live_parity.csv`, `disk_probe_<dev>.json` and `live_report.md`. The per-view
  `.csv.gz` files stay in `outputs/` and are not committed.
- **Experiments E6:**
  - (a) Headline: 2000 sessions, browser client plus priority link, RTT 50 ms, 10 Mbps, cap 2 s, ramp 180 s,
    cold cache. Policies: fixed0, fixed1, fixed4, fixed8, adaptive_0.002, adaptive_0.01, gate_0.01,
    feedback_0.01, and the online-index variant of adaptive_0.01. 3 reps on SSD and 3 on HDD.
    About 360 s per pass, x 9 policies x 3 reps x 2 stores, about 5.4 h.
  - (b) FIFO/FIFO ablation for fixed0, fixed4 and fixed8, 1 rep: about 20 min.
  - (c) hint:fixed4 and hint:fixed8 on the HDD store: about 15 min.
  - (d) One-axis-at-a-time sensitivity: RTT {20, 150}, bandwidth {5, 100} Mbps, size median {16, 256} KiB,
    cap {10, 60} s. 8 configs x 4 policies x 1 rep, about 3.5 h. P2, optional.
  - Expected from the prototype (single rep, valid; for orientation, **not to be cited** until
    reproduced): fixed4 −5.55 ms/view [−6.57, −4.67] at 6.71x bytes; gate_0.01 −3.50 at 2.09x; fixed0
    baseline 105.25 ms; in-flight-at-view 0–3 of 6,685 views.
  - The host must be quiet: check `uptime` load below 32 before each pass and abort the pass otherwise.
    Never run concurrently with E1 or E3.

### 3.4 How the web outputs feed the paper

| results file | paper element |
|---|---|
| `results/web/dataset_profile.json` (from `prepare` and `kinds`) | Table "Datasets" (web row): 2,756,101 events, 2,664,218 views, 1,755,863 sessions, novel / self-repeat / revisit 63.4 / 25.2 / 11.4% |
| `results/web/freshness.csv` | Fig. "coverage@K vs index freshness" and Table "Web decomposition" (freshness row) |
| `results/web/policy_metrics.csv` | Fig. "U(c) envelopes" (web panel); Fig. "hit vs depth per capacity" |
| `results/web/comparisons.csv` (per-capacity selection) | Table "Web controller results", re-baselined |
| `results/web/decomposition.csv` | Table "What moves the needle" (web column) |
| `results/web/seq_metrics.csv` | GPU model rows; training and inference cost next to the disk LSTM |
| `results/web/calibration.csv` | Sentence and footnote on calibration (bins with p ≥ 0.1) |
| `results/live/live_metrics.csv`, `live_parity.csv` | Table "Real HTTP replay", Fig. "latency saved vs bytes", parity statement |
| `results/web/think_time.json` | Fig. "think-time CDF vs prefetch lead" (supports the instant-prefetch assumption) |

---

## 4. `master` integration

### 4.1 Final layout

```
OS-Project/ (branch master)
├── AGENTS.md                 unified: shared rules + §Disk invariants (from main) + §Web invariants (from GRILL) + §Real-system rules
├── README.md                 user-facing: what the two studies are, install, reproduce commands, results summary (generated numbers only)
├── NOTICE.md                 RetailRocket CC BY-NC-SA 4.0 attribution (from pivot) + trace-source attributions
├── pyproject.toml            ONE project, 3 packages, extras
├── reproduce.py              top-level: `reproduce.py disk|realio|web|live|paper|all [--quick]`
├── main.py, sweep_modes.py   kept (disk legacy entry points; fixed)
├── src/
│   ├── adaptive_prefetch/    disk study (stdlib core) + corpora.py, multicorpus.py, analysis.py, depth_model.py, realio/
│   ├── webprefetch/          web study + cachecore.py, index.py, parallel.py, scorers.py, seqmodel.py, live/
│   └── prefetch_common/      utility.py (U(c), c*, break-even mapping), stats.py (paired/cluster bootstrap, wilcoxon wrapper), provenance.py
├── tests/                    flat; test_project.py (disk, unchanged name), test_disk_*.py, test_web_*.py, test_common_*.py, test_repro_smoke.py
├── data/
│   ├── samples/              tracked (disk 1k-row samples) + web fixture (tiny synthetic clickstream for tests)
│   ├── block-traces/         ignored (56 *.oracleGeneral.zst + manifest.txt)
│   ├── raw/                  ignored (events.csv, SHA-pinned)
│   └── MSRC-trace-003/       ignored (absent on this machine)
├── models/                   ignored except msr_sample_gnb.json
├── results/                  TRACKED small aggregates that the paper reads; each dir has provenance.json
│   ├── disk/ realio/ web/ live/ gpu/
│   └── pivot_original/       the pivot branch's committed outputs, kept verbatim for the reproduction claim
├── outputs/                  ignored scratch (per-view logs, raw jsonl dumps, demo state)
├── paper/
│   ├── main.tex  sections/*.tex  refs.bib  IEEEtran note (README: compile on Overleaf)
│   ├── figures/make_figures.py → figures/*.pdf
│   ├── tables/make_tables.py  → tables/*.tex
│   ├── numbers.py             → numbers.tex (\newcommand macros for every number quoted in prose)
│   └── check.py               structural lint (cites resolve, inputs exist, braces, no hand numbers in prose)
├── demo/                     web demo (index.html, export.json)
├── audit/GRILL.md            web audit log (kept)
├── docs/
│   ├── DECISIONS.md          disk D1–D17 history + D18 (device-I/O cost), D19 (LSTM retired, learned_depth), D20 (oracle corpora), D21 (real testbed findings)
│   ├── WEB_DECISIONS.md      from GRILL + W1 (per-capacity selection), W2 (freshness), W3 (feedback gate), W4 (live HTTP)
│   ├── architecture.md       updated: both packages, data flow, metric limits
│   ├── REPRODUCE.md          every command, runtime, hardware, expected artifacts
│   ├── METRICS.md            unified definitions (block hit ratio, device_ios, precision, U(c), c*, kappa*)
│   ├── REFERENCES.md         deck-citation corrections (CAPSULE, Savva, Akgun, ECML LNCS)
│   ├── PRESENTATION.md       slide-by-slide outline + viva Q&A (replaces REVIEW2.md and pptx decks)
│   ├── CLASSIFIER_EVAL.md, MSR_SAMPLE_TRAINING.md, PIPELINE.md   kept (PIPELINE updated)
│   ├── web/PROJECT_GUIDE.md, web/IMPLEMENTATION_PLAN.md          from pivot docs/
│   ├── MASTER_BRANCH.md      updated
│   └── reference/Operating Systems Lab Submission 2.pdf          kept (historical Review 1)
└── presentation/README.md    points to docs/PRESENTATION.md and paper figures
```

**pyproject.toml**
- `name="os-prefetch-studies"`, `requires-python=">=3.10"`, `dependencies=[]`.
- Packages are found under `src`: `adaptive_prefetch`, `webprefetch`, `prefetch_common`.
- Extras:
  - `traces=["zstandard>=0.22"]`
  - `web=["numpy>=1.24","pandas>=2.0","scikit-learn>=1.3","joblib>=1.3"]`
  - `gpu=["torch>=2.2"]`
  - `paper=["matplotlib>=3.7","scipy>=1.10"]`
  - `all=[…all of them…]`
- Scripts: `adaptive-prefetch=adaptive_prefetch.cli:main` and `webprefetch=webprefetch.cli:main`.
- The venv already contains every extra, so no installation is needed. Use `PYTHONPATH=src`.

**Tests.**
- The venv runs everything: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`.
- Web, GPU and device tests use `skipUnless` (pandas, torch.cuda, Linux plus O_DIRECT), so system
  `python3` still runs the stdlib disk suite.
- Rename the pivot tests: `test_workflow.py`→`test_web_workflow.py`,
  `test_audit_regressions.py`→`test_web_audit_regressions.py`, `test_demo_parity.py`→`test_web_demo_parity.py`.
  Their path inserts use `parents[1]/"src"`, which still works.
- `tests/test_project.py` keeps its name because AGENTS.md references it.
- Target at the end: 131 + 16 + about 90 new tests, all green, in under 90 s, excluding tests marked `slow`
  (env `RUN_SLOW=1`).

**One reproduce command per study** (`reproduce.py`, stdlib argparse. Each step shells out to the module
CLIs, writes `results/<study>/provenance.json` with git sha, command, host, load average, timestamp and
input SHA-256s, and prints the runtime estimate first):
- `reproduce.py disk --quick`: tests, the `main.py --quick` path, and the multicorpus smoke run (3 traces x
  all modes x 50k). About 5 min.
- `reproduce.py disk`: C1 to C4, then analysis. Takes hours; E1.
- `reproduce.py disk --gpu`: E2.
- `reproduce.py realio --device /mnt/…`: E3 (calibrate, grid, report).
- `reproduce.py web --quick`: prepare, train, `benchmark --validation-views 1000 --test-views 2000`.
  About 2 min.
- `reproduce.py web`: E4, plus E5 with `--gpu`.
- `reproduce.py live --store-dir …`: E6.
- `reproduce.py paper`: figures, tables and numbers from `results/` (no experiment runs). Under 1 min.

### 4.2 What to drop (from the `master` tree; history keeps them)

- `presentation/Review2_Adaptive_Disk_IO_Prefetching_v5.pptx`: outdated per AGENTS.md.
- `presentation/Review2_Adaptive_Disk_IO_Prefetching_2026-09-22_v2.pptx`: stale (old LSTM 7% rows, "7.3x" claim).
- `docs/STAGE2_RESULTS.md`: stale. It is replaced by generated `results/` and the paper.
- `docs/REVIEW2.md`: folded into `docs/PRESENTATION.md`.
- `data/samples/demo.csv`: stale pre-D10 generator (M17). Regenerate it with `export-demo`, or drop it from
  the dataset list.
- `pivot/outputs/*`: moved verbatim to `results/pivot_original/`, not deleted.
- `pivot/docs/*`: moved to `docs/web/`.
- `pivot/README.md`: merged into the root README.
- Not dropped:
  - `datasets.py`, `pipeline.py`: still imported; low value to churn.
  - the drift report: it stays in `main.py`, flagged "saturated, demonstrates nothing" and excluded from
    the paper.
- `.gitignore`:
  - replace the blanket `data/` with `data/block-traces/`, `data/raw/`, `data/processed/` and
    `data/MSRC-trace-003/`
  - add `outputs/` (already present), `.realio/`, `paper/*.aux|log|pdf`
  - keep `models/*` and `!models/msr_sample_gnb.json`
  - **do not** ignore `results/`

### 4.3 Git commit sequence (`master`, author AshvikMishra; push to `origin master` after each wave once tests are green)

1. `Bring the web study into master as src/webprefetch` (WP0: pivot code, tests renamed, demo, audit, NOTICE, docs/web, results/pivot_original).
2. `Unify packaging, gitignore and test runner for both studies` (WP0: pyproject, .gitignore, reproduce.py skeleton, drop stale pptx and STAGE2).
3. `Fix disk CLI demo, report labels, hard-coded narrative and LSTM leak` (WP1).
4. `Add oracleGeneral block-trace loader with unit detection and FIU page merging` (WP2).
5. `Add device-I/O accounting, cache hook, linux_ra and sweep fixes to the disk simulator` (WP3).
6. `Add shared utility and statistics module for both studies` (WP4).
7. `Web core: cache core, fresh and online index, parallel replay, novel metrics, feedback gate` (WP5).
8. `Paper skeleton and verified bibliography` (WP7).
9. Wave 2: one commit per WP (WP6, WP8, WP9, WP10, WP11, WP12, WP13).
10. Wave 3: `Results: <experiment>` commits, each adding only `results/<study>/…` plus provenance (E1–E6).
11. Wave 4: `Paper: <section>` commits, `Docs: unify AGENTS/README/DECISIONS for master`, `Presentation outline`.
12. Final: `Verify clean-checkout reproduction` (WP16), tagged `paper-v1`.

---

## 5. Paper plan (IEEE conference, IEEEtran, `\documentclass[conference]{IEEEtran}`)

**Working title:** *Depth, Freshness and Feedback, Not Prediction: What Makes Prefetching Pay, from Disk
Blocks to Product Pages*. The alternative is *Is Learning Worth It for Prefetching? A Two-Domain Study
with Real-Device and Real-HTTP Validation*. The authors are the two students, listed in IEEE author blocks
with VIT affiliation.

**Unifying thesis.** (Critic) This is a **hypothesis**, not a pre-written conclusion. The pilots do not
all support it: the GPU `learned_depth` beat VCF by +2.62 pp on the mean (single seed), so on disk a
learned predictor of *depth* may not be the smallest effect. Before E1–E6 run, WP10, WP6 and WP12 each
commit a `results/<study>/PREREGISTERED.md`. It lists the headline comparisons, their baselines, n, the
CI method and the decision rule (for example "the predictor is called the smallest lever iff its |Δ|
CI lies below the depth and freshness |Δ| CIs"). WP14 writes the title and abstract from whichever way
those comparisons come out, and changes the working title if they contradict it.
- One cache/prefetch abstraction covers both domains.
- One objective covers both: U(c) = (hits − c·issued)/requests. The disk cost model is exactly U at
  c = p·m/(m−h), which is 1.0101 at main's defaults. The web "cost" is U at an assumed c, which the live
  experiment measures.
- In both domains the *learned predictor* (GNB class router, bucketed-delta LSTM, logistic ranker, GRU) is
  the smallest effect. The decisive levers are:
  - (i) **how much** to prefetch: depth, or window growth as in Linux read-ahead
  - (ii) **feedback** on observed precision against the break-even c
  - (iii) on the web side, **freshness** of the statistics
- (iv) Simulated hit ratio overstates real benefit. The real-device testbed shows late prefetches and
  device contention. The real HTTP replay shows link-queue effects. Both are measured with exact parity
  to the simulators.

**Sections and their evidence:**

| § | Content | Feeds from |
|---|---|---|
| Abstract | 4 claims with numbers from `numbers.tex` | all |
| I. Introduction | The OS problem: page-cache read-ahead and browser/proxy prefetch. Contributions as a bulleted list (multi-corpus negative result, depth/I/O decoupling, real testbed with parity, web freshness/controller decomposition, live HTTP). | — |
| II. Background & Related Work | Linux read-ahead (wu2007readahead, linux_mmapi_readahead); sequential prefetch in storage caches (gill2005sarc, gill2007amp); informed and integrated prefetching (patterson1995informed, cao1995integrated); learned prefetching (chakraborttii2020learning, hashemi2018learning, wong2024baleen, savva2026adaptive, ramadhan2026capsule with the **corrected** title and venue, akgun2021kml as HotStorage'21, garg2024crossprefetch); cache theory (belady1966study, mattson1970evaluation, megiddo2003arc); web prefetching (padmanabhan1996using, kroeger1997exploring, fan1999web); sequence recommenders (hidasi2016session, kang2018sasrec). | `$EVAL/literature/refs.bib` (47 verified entries) |
| III. A Common Model of Prefetching | Cache abstraction; definitions (block hit ratio per 512 B; device_ios; precision; waste; pollution); U(c); derivation c = p·m/(m−h); per-policy c\*; kappa\*. Fig. 1: schematic (TikZ-free; a matplotlib or hand-built PDF diagram from `make_figures.py`). | `docs/METRICS.md`, `prefetch_common/utility.py` |
| IV. Systems | A. Disk simulator and policies (12 modes + linux_ra + learned_depth; causal loop, first-window gate). B. Real Linux testbed (Backend A page cache + WILLNEED + memory cgroup; Backend B O_DIRECT + async prefetch threads; online driver; the kernel facts: 256 KiB WILLNEED truncation, cgroup re-charge rule, cachestat vs RWF_NOWAIT). C. Web replay (per-session LRU, ranker, controllers, online index, feedback gate). D. Live HTTP (asyncio origin, precise timer, priority link, validity gate). | WP3/8/5/12 code |
| V. Methodology | Datasets table: 56 oracleGeneral traces in 4 corpora with reconstruction rules, twins and caps; RetailRocket splits. Protocols: chronological val/test; bootstrap (paired, cluster); repeats; hardware table (CPU, HDD and SSD model, kernel 6.14, MGLRU, GPU and its contention). | `results/disk/corpus_profile.csv`, `results/web/dataset_profile.json`, `results/realio/device_*.json` |
| VI. Disk Results | **T2** main table (policy x {mean hit, Δ vs none [CI], Δ vs sequential [CI], precision, waste, device-I/O ratio, blocks-moved ratio}, 1 MiB and 64 MiB, per-corpus sub-columns). **F2** hit ratio vs depth per cache size (per corpus). **F3** scatter of device-I/O ratio vs hit gain (hit ratio ≠ device work). **F4** CDF of kappa\* with measured HDD/SSD kappa lines. Classifier negative result (adaptive vs sequential −0.89 pp, worse on 52/56, re-measured at 1M). Gate as waste limiter. learned_depth vs VCF (seeds, CI). Merge sensitivity row. | E1, E2 |
| VII. Real-Device Validation | **T3** Backend B/A x mode x think (hit, full-hit fraction, mean/p50/p99 latency, late hits, device MiB, k). **F5** simulated hit ratio vs measured mean latency (closed vs spread), 2 HDDs plus SSD. Prediction-error statement; rank correlation; kernel_ra baseline. | E3 |
| VIII. Web Results | **T4** dataset and pair kinds. **F6** coverage@K vs freshness. **T5** decomposition ("what moves the needle", CIs). **T6** per-capacity validated comparisons (vs best fixed, vs fixed0, vs transition). Feedback gate. GRU vs counts (seeds). Calibration footnote. | E4, E5 |
| IX. Live HTTP Results | **T7** ms/view saved [CI], model-ideal Δ, bytes x, parity = 0 mismatches, validity. **F7** latency saved vs bytes wasted. FIFO ablation (fixed8 loses about 2 of 3.3 ms); hint on HDD; in-flight-at-view. | E6 |
| X. Cross-Domain Discussion | **F8** U(c) envelopes, disk and web panels side by side, log c ∈ [1e-3, 1.1], with markers at disk c = 1.0101 and the measured web c. **T8** cross-domain summary (best policy, gain over none, waste, c\*, gate wins?). | E1, E4, E6 |
| XI. Threats to Validity | oracleGeneral loses read/write and service time, and its timestamps are 1 s; FIU merge ambiguity; 250k/1M caps (C3 checks); twin traces; the instant-prefetch assumption in the simulators (bounded by E3/E6); MGLRU vs LRU; shared host and GPU; assumed web object sizes, RTT and bandwidth; per-session cache scope; heuristic 30-min sessions; single retailer; ranking metrics vs cache utility. | — |
| XII. Conclusion and future work | — | — |
| References | `paper/refs.bib` | literature |

Page budget: 8 pages plus references. If the course allows less, move T3, T6 and F4 to an appendix.
(Critic) 12 sections, 8 tables and 8 figures will not fit in 8 IEEE two-column pages. Many IEEE
conferences cap at 6, and the course limit is unknown: **ask the students for the actual limit before
WP14 starts.** Priority order when cutting:
1. Must keep: T2, F3, T3/F5, T5, T7 and F8.
2. Next: F2, T6 and the learned_depth paragraph.
3. Appendix or companion tech report: T4, F4, F6, F7, T8, the FIU merge sensitivity and the full
   threats list.
Merge §III into §IV.A/§IV.C if space is short.

**Figure and table scripts.**
- `paper/figures/make_figures.py`: matplotlib with the Agg backend, a colour-blind-safe palette, IEEE
  column width 3.5 in, vector PDF. One function per figure, each reading only `results/**`. Missing
  inputs produce a placeholder PDF stamped "MISSING: <file>", so the paper skeleton always builds.
- `paper/tables/make_tables.py`: writes booktabs-free IEEE `tabular` `.tex` files (IEEEtran does not ship
  booktabs, though Overleaf has it; use plain `\hline`).
- `paper/numbers.py`: writes `numbers.tex` with `\newcommand{\diskAdaptiveVsSeq}{$-0.89$\,pp}` and so on.
  **Every number in the prose is a macro.** `check.py` fails if a digit sequence with a decimal point or
  `%` appears in `sections/*.tex` outside an allow-list (years, section numbers, hardware model names).

**Honesty constraints (from AGENTS.md, extended), enforced in WP14 review:**
1. Every simulated number is labelled simulated. Latency comes from a cost model unless it comes from
   E3/E6, and those are labelled measured, with the device, the think-time model and the load.
2. Classification accuracy (0.9688 held out) is on synthetic windows only. Real-trace classifier quality
   is unknown, because the traces carry no labels.
3. The LSTM predicts deltas; compare cache metrics and inference cost, never its "accuracy" as workload
   classification.
4. Oracle recall is reported only together with hit ratio. The EDA "cacheable ceiling" is not a bound.
5. Hit ratios on oracle traces include writes replayed as reads, and are **never** compared numerically
   with the MSRC-trace-003 D17 numbers. D17 is quoted only as the earlier dataset's result.
6. The disk break-even 1.0101 is a property of the per-block, no-merge cost model, not of hardware.
7. Headline comparisons are made against the strongest fair baseline: VCF for learned_depth, the best fixed
   depth at the same capacity for the web controller, and transition top-1 for the ranker. Do not
   generalise from web_3, cap 2 or any single trace.
8. Web object sizes, RTT and bandwidth are ASSUMED. Live HTTP savings are "on loopback with emulated
   network", 3–5% of a modelled ~105 ms view.
9. Invalid or contaminated runs (web-realexp ramp-2 s, `contaminated/`, pivot-science `variants.csv`) are
   never cited.
10. Every quantitative claim has n, a CI and a seed or repeat count. GPU timings carry a contention note.
11. The Review-1 deck's mis-citations are corrected in the paper (docs/REFERENCES.md) and the deck itself is
    not edited.
12. No number is typed by hand into the paper, docs or README. Everything is regenerated from `results/`.
13. (Critic) Headline disk numbers use the **canonical** request reconstruction (§2.1 `merge="auto"`).
    Unmerged-FIU ("raw") and uncapped same-second views appear only as labelled sensitivity rows, and
    the scratch `sweep2.jsonl` pooled figures are never quoted (§0.6).
14. (Critic) A trained artifact (lstm, learned_depth, GRU) is scored only on a segment or trace it was
    not trained or selected on. Each results row names the artifact's training range.
15. (Critic) The realio prediction-error and `k_eff` figures come from one trace prefix per device;
    they are stated per trace and per think model, never as a general property of HDDs or SSDs.

---

## 6. Work packages

Conventions:
- (Critic) **One git worktree and branch per WP.** Each WP runs in `git worktree add
  $SCR/wt/<wp> -b wp/<wp> master` and never in `$REPO` itself. Parallel agents sharing one working tree
  and index would race on `git add`/`commit`, and each would run the test suite against the others'
  half-edited files. The lead merges finished WP branches into `master` **one at a time**, rebasing as
  needed and re-running the full suite after each merge; only then does the next merge start. Files
  under `data/` (gitignored, 5.3 GB) are reached through the absolute `$REPO/data` path or a symlink in
  each worktree, never copied.
- "Owns" means only that WP may create or modify those paths while its wave runs.
- Every WP must leave the whole suite green:
  `PYTHONPATH=src /home/antpc/Ashvik/OS-Project/OS-Project/.venv/bin/python -m unittest discover -s tests`.
- Each code WP finishes with one commit, message as in §4.3.
- Long experiments produce results only. No code edits are allowed, except a WP-scoped `results/` folder.

### Wave 0 (serial, blocking)

**WP0 — Merge skeleton.** Owns everything at the root (`pyproject.toml`, `.gitignore`, `reproduce.py`
skeleton), `tests/test_web_*.py` (renames), `src/webprefetch/**` (copied verbatim), `demo/`, `audit/`,
`NOTICE.md`, `docs/web/`, `results/pivot_original/`, and the deletions listed in §4.2. Depends on nothing.

Tasks:
1. `git checkout master` and confirm it is clean.
2. Copy from `$SCR/pivot`: `src/webprefetch`, `tests` (renamed), `demo`, `audit`, `NOTICE.md`,
   `docs/*`→`docs/web/`, `outputs/*`→`results/pivot_original/`.
3. Write the unified `pyproject.toml` (§4.1).
4. Narrow `.gitignore` (§4.2).
5. Delete both pptx files, `docs/STAGE2_RESULTS.md` and `docs/REVIEW2.md`. Move the REVIEW2 content into
   a `docs/PRESENTATION.md` stub.
6. Add a `reproduce.py` skeleton with sub-commands that print "not yet implemented (WPn)".
7. Make `data/raw/events.csv` resolvable. It already exists at `$REPO/data/raw/events.csv`, and the pivot
   symlink points to it.
8. Add a tiny web fixture `data/samples/web_fixture.csv`: synthetic, about 200 rows, header matching
   RetailRocket, and labelled synthetic. Add it with `git add -f` if needed.
9. (Critic) **Separate the two studies' output and model directories.** Both `main.py:59,472` and
   `webprefetch/cli.py:16,23` write `outputs/eda.md`, so one study silently overwrites the other, and
   both use a flat `models/`. Change the webprefetch defaults (`cli.py:13-16`, `demo_server.py:27-42`) to
   `outputs/web/` and `models/web/`. Keep `data/processed/`, since main does not use it. Main keeps
   `outputs/` (AGENTS.md and the docs reference it). `models/web/` is already covered by the existing
   `models/*` ignore rule, so it stays untracked.
10. (Critic) **The demo parity test cannot pass on a fresh checkout.** `tests/test_demo_parity.py:15-17`
    builds `DemoData(root)`, which loads `models/*.joblib`, `data/processed/views.csv.gz` and six
    `outputs/*.csv|json` files. All of these are gitignored or are being moved to
    `results/pivot_original/`. Guard the test with `skipUnless` (artifacts present) and make
    `DemoData(root, models=…, processed=…, outputs=…)` take explicit paths. Until P15 lands, the
    reference stays `results/pivot_original/session_parity.json`; it is valid only against the
    committed full-run selection. A `--quick` run selects different penalties, so the reference must
    then be regenerated by `webprefetch parity` from that same run, never compared across runs.
11. (Critic) **Split `reproduce.py` so parallel WPs do not edit one file.** `reproduce.py` is a thin
    dispatcher that imports `repro/<study>.py`, and each module is owned by the WP that implements that
    study: `repro/disk.py` (WP10), `repro/gpu.py` (WP9 for disk, WP11 for web), `repro/realio.py`
    (WP8), `repro/web.py` (WP6), `repro/live.py` (WP12), `repro/paper.py` (WP13). WP0 writes stubs.

Acceptance:
- 147 tests run under the venv: all pass, except that `test_demo_parity` may report *skipped* when
  the web artifacts are absent. To check it, run `webprefetch prepare && webprefetch train` locally once
  (about 45 s) with `results/pivot_original/` restored to the outputs path the test reads.
- `python3 -m unittest discover -s tests` (system python, PYTHONPATH=src) runs the disk tests, and the web
  tests are skipped, not errors. This requires adding `skipUnless(pandas)` guards in the three web test
  modules.
- `PYTHONPATH=src .venv/bin/python -m webprefetch --help` and `python -m adaptive_prefetch --help` work.
- `git status` is clean after 2 commits.

Runtime: under 10 min.

### Wave 1 (parallel; disjoint files)

**WP1 — Disk CLI, report and EDA fixes.** Owns `main.py`, `src/adaptive_prefetch/{cli.py,report.py,eda.py,datasets.py}`,
`data/samples/demo.csv`, and the new `tests/test_disk_cli.py`. Depends on WP0.

Tasks:
- M1, M4, M5, M6, M7, M8, M17.
- Add `oracle` to the `--format` choices of `cli.py`. It errors gracefully until WP2 lands, and the
  routing test lives in WP2.
- New tests, run in a subprocess with `PYTHONPATH=src`:
  - `python -m adaptive_prefetch demo` exits 0
  - with `--cache-blocks 2048`, the oracle column of `main.py --quick` in `outputs/results.md` equals the
    console's
  - `--cost-model assumed` never prints "MEASURED"
  - the report contains none of the literals `18.99`, `4.10`, `1.0101` unless they were computed
  - the LSTM training set is disjoint from the benchmark datasets (unit-level, on `resolve_lstm`'s
    selection function, refactored to be testable)
  - `eda` no longer says "upper bound"

Acceptance: all tests green, and `main.py --quick` runs in under 30 s.

**WP2 — oracleGeneral loader.** Owns `src/adaptive_prefetch/trace.py` and `corpora.py` (new), and
`tests/test_disk_oracle.py`. Depends on WP0.

Tasks: §2.1 in full, plus M10.

Acceptance:
- Tests (a)–(j) pass; (h)–(j) run here, because the data is present.
- `python -c` streaming of msr_hm_0 (3,993,316 requests) finishes in under 30 s at under 300 MB RSS.
- fiu_ikki merged gives 2,009,285 ± 200 requests (scout: 2,009,285 numpy / 2,009,479 prototype).

**WP3 — Disk simulator core and sweep.** Owns
`src/adaptive_prefetch/{simulator.py,baselines.py,features.py,benchmark.py,artifacts.py,model.py,guard.py,sweep.py}`,
`sweep_modes.py`, `tests/test_project.py`, and the new `tests/test_disk_core.py`. Depends on WP0.
Integrates with WP2 only through `iter_csv_chunks(path, format, chunk_requests)`.

Tasks: §2.2 items 1–7, plus M2, M3, M9, M11–M14, M16.

Acceptance:
- All 131 original tests pass unchanged, except for **additions** to the parity test.
- New tests:
  - `cache_factory` identity across all MODES
  - device_ios on hand-built cases: one 8-block miss is 1 demand IO; a read-ahead run contiguous with a
    miss is mergeable
  - replay and replay_stream agree on the new fields and the inference counters
  - `linux_ra` scan and random behaviour, plus causality
  - `len(MODES)==13` with `rows[5]=="lstm"`
  - the bounded WindowContext is identical when the bound exceeds the footprint, and on a 100k-request
    synthetic low-reuse stream `len(last_seen)` never exceeds the bound (Critic: assert on the table
    size, not on process RSS, which is flaky inside a shared unittest process)
  - `sweep_modes.main([...,'--full','--modes','none','deep'])` and default modes with no LSTM artifact
    both exit 0 and write both the csv and the md (on a generated 3,000-line `.revised` file)
  - the QDA shrink round-trips, and a mismatch raises
  - `LatencyModel.measured(prefetch_multiple=0)` raises with no data
  - `make_model` priors are within 0.25 ± 0.05 and held-out accuracy is unchanged at 0.96875

**WP4 — Shared utility and statistics.** Owns `src/prefetch_common/**` and `tests/test_common.py`.
Depends on WP0.

Tasks:
- (Critic) `schemas.py`: the **results contract**. One tuple of column names, plus dtype notes, per
  results file the paper reads:
  - disk: `sweep.jsonl`, `table_main.csv`, `table_by_corpus.csv`, `deltas.csv`, `kappa.csv`,
    `corpus_profile.csv`, `disk_learned_depth.csv`
  - realio: `realio.csv`, `realio_summary.csv`, `device_<dev>.json`
  - web: `policy_metrics.csv`, `comparisons.csv`, `decomposition.csv`, `freshness.csv`, `seq_metrics.csv`
  - live: `live_metrics.csv`, `live_parity.csv`
  Producers (WP6, WP8, WP9, WP10, WP11, WP12) write through `schemas.write_rows(name, rows)`, which
  rejects unknown or missing columns. WP13 builds its fixtures from the same tuples, so the generators
  cannot drift from the producers in wave 2. A test asserts that every fixture under
  `tests/fixtures/results/` matches its schema.
- `utility.py`:
  - `U(hits, issued, requests, c)`
  - `c_star(policy_row, none_row)`
  - `cost_to_c(hit_us, miss_us, prefetch_us)`, returning 1/0.99 at main's defaults
  - `envelope(rows, cs)`
- `stats.py`:
  - `paired_bootstrap(a, b, reps=10000, seed=42)`
  - `cluster_bootstrap(values, clusters, weights)`
  - `session_bootstrap` (adapter for webprefetch's per-session dicts)
  - `wilcoxon` (scipy optional; returns None if absent)
- `provenance.py`: `write_provenance(dir, cmd, inputs)` (git sha, host, load average, nvidia-smi if
  present, SHA-256 of inputs).

Acceptance:
- `cost_to_c(1,100,100)` is 100/99, and `cost_to_c(5,100,50)` is 50/95.
- Fixtures show that the argmax of U at c=1.0101 equals the argmin of modelled cost on the D11 synthetic
  rows.
- Bootstrap CIs are deterministic for a given seed.
- Stdlib only, with scipy optional.

**WP5 — Web core.** Owns
`src/webprefetch/{simulator.py,model.py,data.py,cachecore.py,index.py,parallel.py}` and the new
`tests/test_web_core.py`. Depends on WP0 and WP4 (WP4's API is fixed in this plan, so they can run in
parallel and integrate at the end).

Tasks: §3.1 WP5 list, plus P2(a), P3, P4, P7, P10, P12, P14.

Acceptance:
- The 16 original tests pass.
- New tests:
  - cachecore re-export identity
  - `adaptive_ca` never counts resident slots
  - the online index never uses a future transition
  - `replay_parallel == replay` (fixture plus a real 20k prefix, `skipUnless` data)
  - global order equals session order for stateless policies
  - visitor scope persists across a visitor's sessions
  - novel metrics exclude revisits
  - the break-even direction is `dominates`
- Reproduction check (`skipUnless` data, `slow`): logistic fixed1 cap 8 on test gives 65,944 hits and
  445,796 issued.

**WP7 — Paper skeleton and bibliography.** Owns `paper/{main.tex,sections/*.tex,refs.bib,check.py,README.md}`
and `docs/REFERENCES.md`. Depends on WP0.

Tasks:
- `main.tex` with IEEEtran conference, author blocks, and `\input{numbers}`.
- Section stubs per §5, each with a TODO naming its results file.
- Copy `$EVAL/literature/refs.bib`, all 47 verified entries.
- `check.py`: every `\cite` key exists, every `\input`/`\includegraphics` target exists or has a declared
  generator, braces balance, and the hand-number lint.
- `docs/REFERENCES.md` with the 5 corrections: CAPSULE title and venue (PER 54(1):76-78, DOI
  10.1145/3820171.3806897), Savva & Herodotou BDCC 10(1):28, Akgun HotStorage'21, ECML PKDD LNCS 12460
  (2021), NVMW'22 as non-archival.
- `paper/README.md`: compile on Overleaf with IEEEtran; local compile is unavailable.

Acceptance: `python paper/check.py` passes on the skeleton, and no cite key is missing.

### Wave 2 (parallel; disjoint files; depends on wave 1)

**WP6 — Web benchmark, report and demo.** Owns
`src/webprefetch/{benchmark.py,cli.py,demo_server.py,diagnostics.py,scorers.py}`, `demo/`,
`tests/test_web_demo_parity.py`, and the new `tests/test_web_benchmark.py`. Depends on WP5 and WP4.

Tasks: §3.1 WP6 list, plus P1, P2(b, c), P5, P6, P8, P9, P11, P13, P15, and the `freshness`, `decompose`
and `parity` sub-commands. All outputs go through `prefetch_common.utility`.

Acceptance:
- On the fixture, per-capacity selection differs from cap-8 selection when constructed to, and
  comparisons carry vs-fixed0 and vs-best-fixed columns.
- Demo parity holds at costs 0, 0.01, 0.05 and 0.25.
- `webprefetch run --quick` runs in under 3 min.
- `slow` (real data): per-capacity selection reproduces `$EVAL/pivot-code/exp/percap.json` values to 1e-5,
  and the full benchmark with `--workers 96` takes under 10 min.

**WP8 — Real Linux disk testbed.** Owns `src/adaptive_prefetch/realio/**` and the new `tests/test_disk_realio.py`.
Depends on WP3 (cache_factory, device_ios, IOCostModel, linux_ra) and WP2 (loader).

Tasks: §2.6 in full, including `online.py` and `calibrate.py`.

Tests, which run on a temp file on whichever filesystem the test dir lives on, and skip if O_DIRECT gives
EINVAL or the platform is not Linux:
- `RecordingLRU` plan identity for all modes
- Backend B on a 64 MiB file with a synthetic sequential trace: late_hits > 0 when closed and 0 with 5 ms
  think, and `useful_prefetches == simulator`'s
- `CacheDivergence` raised on a tampered plan
- WILLNEED chunking makes a 1 MiB range resident (RWF_NOWAIT probe)
- the RegionMapper wrap counter and refusal
- the calibration arithmetic on a fixture
- CSV column stability
- the cgroup re-exec skips cleanly when `systemd-run --user` is unavailable

Acceptance smoke (manual, documented in the WP output):
`python -m adaptive_prefetch.realio run --trace data/block-traces/msr/msr_hm_0.oracleGeneral.zst --requests 5000 --device-dir /mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch --modes none,deep --backend odirect,pagecache --think closed`
finishes in under 2 min, and the B parity assertion holds.

`results/realio/realio.csv` columns:
- `device, backend, mode, think, repeat, trace, requests`
- `sim_hit_ratio, sim_precision, sim_device_ios`
- `real_hit_ratio, full_hit_request_frac, ready_hits, late_hits, late_wait_us`
- `lat_mean_us, lat_p50_us, lat_p95_us, lat_p99_us, lat_max_us`
- `device_MiB, prefetch_MiB`
- `think_total_s, think_error_us_p99, wall_s`
- `cg_file_MiB, cg_events_max, foreign_io_frac, wrapped`
- `pred_mean_us, pred_err, k_eff, load_avg`

**WP9 — GPU disk models.** Owns `src/adaptive_prefetch/{lstm.py,depth_model.py,cli.py}`, plus the
`learned_depth` registration lines in `benchmark.py`, `sweep.py` and `simulator.py`. Ownership of those
three files passes from WP3 to WP9 in wave 2, and nobody else touches them in that wave. Also owns
`tests/test_disk_gpu.py`. Depends on WP3, WP2 and WP1 (cli.py).
(Critic) WP9 also owns `tests/test_project.py` in wave 2, and only for two edits:
- add `learned_depth` (and `linux_ra`, if WP3 has not) to the hard-coded mode tuple in
  `test_every_supported_mode_agrees_across_both_paths` (`tests/test_project.py:1374-1380`). That tuple
  is **not** "every mode": it omits stride, markov, lstm and adaptive.
- keep `test_benchmark_includes_skipped_optional_lstm` passing.
With no artifact, `learned_depth` must produce a `skipped: no trained learned_depth artifact` row in
both `benchmark` and `sweep`, as `lstm` does. WP10, which runs in parallel and only imports `sweep`,
must not assume a fixed `RANKING_MODES` length.

Tasks: §2.5 in full, plus M15. CLI: `train-lstm --trace … --format oracle --device` and
`train-depth --traces … --device cuda --seed`.

Acceptance:
- The §2.5 tests, CPU-only, on a tiny model.
- `train-depth` on msr_hm_0 with `--n 200000 --epochs 1 --device cuda` finishes in under 3 min and writes
  the artifact plus the sidecar.
- A replay with `learned_depth` loads it.
- Stale-version rejection works.

**WP10 — Multi-corpus driver and analysis.** Owns `src/adaptive_prefetch/{multicorpus.py,analysis.py}`
and `tests/test_disk_multicorpus.py`. Depends on WP2, WP3 and WP4.

Tasks: §2.3. Configs C1–C4 are YAML-free Python dicts selected by `--config C1`. Resumable jsonl,
longest-first scheduling, an explicit latency model, both break-even settings for the gated modes, and the
analysis CSVs.

Acceptance:
- On the fixture, resume skips completed keys, and the analysis CIs are deterministic.
- Real (`slow`): the msr_hm_0 250k deep@8 cross-check equals 0.357231 and 901487.
- The 3x3x20k smoke run finishes in under 3 min.
- Cells already present in `$EVAL/main-science/sweep2.jsonl` match to 1e-9 for the same
  (trace, mode, cap, merge) at 250k.

**WP11 — Web GPU sequence model.** Owns `src/webprefetch/seqmodel.py` and `tests/test_web_seqmodel.py`.
Depends on WP5. It must expose `SequenceScorer.load(path)` and `precompute(views, splits) -> table` for
WP6's registry. (Critic) WP6 owns `webprefetch/cli.py` in the same wave, so WP11's training entry point
is `python -m webprefetch.seqmodel train …` (a `__main__` block in `seqmodel.py`), not a new `cli.py`
sub-command.

Tasks: §3.2.

Acceptance:
- The §3.2 tests on CPU with tiny V.
- (`slow`, GPU) a 1-epoch smoke run on split 0 plus 90% of split 1 finishes in under 2 min, and the
  reproduced logistic metrics match.

**WP12 — Web live HTTP.** Owns `src/webprefetch/live/**` and `tests/test_web_live.py`. Depends on WP5
(cachecore), and also (Critic: hidden dependency) on WP6's `scorers.py` registry, because
`plan.build_plan` needs `trans_online` candidates for the online-index policy and the
validation-selected penalties. Until WP6 merges, WP12 imports `index.OnlineIndex` from WP5 directly,
and switches to the registry at integration. The loopback-timing and validity-gate tests (2, 5, 6) are
load-sensitive, so they are marked `slow` and skip when `os.getloadavg()[0] > 32`.

Tasks: §3.3, with the 8 tests from the web-realexp recommendation:
1. cachecore parity over 300 trials
2. loopback timing: a miss is within 5 ms of the model
3. chunked WILLNEED
4. priority pool ordering
5. link schedule without drift
6. the validity gate trips on injected delay
7. plan determinism
8. 20-session end-to-end parity

Acceptance:
- All tests green in under 60 s.
- Manual smoke:
  `python -m webprefetch.live run --sessions 200 --min-views 2 --policies fixed0,fixed4 --ramp-s 60 --reps 1`
  gives 0 parity mismatches and is valid.

**WP13 — Figure, table and number generators.** Owns `paper/figures/make_figures.py`,
`paper/tables/make_tables.py`, `paper/numbers.py` and `tests/test_paper_scripts.py`. Depends on WP7 and
the results schemas in this plan (§2.3, §2.6 WP8, §3.1, §3.3).

Tasks:
- Build the generators against **fixture** results under `tests/fixtures/results/`. These are small
  synthetic files with the exact schemas, labelled fixture.
- Implement F1–F8 and T1–T8 as listed in §5.
- Missing input produces a placeholder figure.

Acceptance:
- `reproduce.py paper` on the fixtures produces every PDF, every `.tex` table and `numbers.tex`.
- `paper/check.py` passes.

### Wave 3 (long experiments; each is its own WP; results-only commits)

Scheduling constraints:
- (Critic) **E0 — device calibration first.** `python -m adaptive_prefetch.realio calibrate` on sdb, sde
  and the SSD (about 10 min each, quiet host) runs as soon as WP8 merges and **before** E1's analysis.
  `kappa.csv` and F4 need the measured device kappa (`t_pos/t_xfer`) from
  `results/realio/device_*.json`. `analysis.py` is cheap, so re-run it after E3 if the devices were
  recalibrated.
- E1, E2 and E4 may overlap. (Critic: on 128 shared cores the earlier "E1 at 80 + E4 at 96" would
  oversubscribe the CPU, with E2's label building on top.) When they overlap, cap E1 at 64 workers, E4 at
  40 and E2's CPU label pool at 16. Cache results are deterministic, so oversubscription only stretches
  wall time; but E4's inference timings and E2's GPU timings taken during an overlap carry a contention
  flag and are never quoted as costs.
- E2 and E5 share the GPU, so run them sequentially.
- **E3 and E6 run alone on a quiet host**: load average below 32, nothing else of ours running.
- Order: {E1 ‖ E2 → E5 ‖ E4} → E3 → E6.

| WP | Experiment | Owns | Depends | Command | Est. runtime | Acceptance artifacts |
|---|---|---|---|---|---|---|
| **E1** | Disk multi-corpus sweep C1–C4 + analysis | `results/disk/` | WP10, WP3, WP2 | `reproduce.py disk` | C1 about 1–1.5 h, C2 about 30–45 min, C3 about 20 min, C4 about 5 min, all on 100 workers. **About 2.5–3 h wall** including stragglers. | `sweep.jsonl`, `table_main.csv`, `table_by_corpus.csv`, `deltas.csv` (CI, cluster CI, Wilcoxon p, wins), `kappa.csv`, `corpus_profile.csv`, provenance. The hm_0 cross-check row passes. guard ≡ depth_adaptive at 1.0101 on 56/56. |
| **E2** | GPU: retrain `lstm` on 4 traces (honest split) + `learned_depth` on the 21-trace PAPER_SUBSET x 5 seeds + VCF / request-relative baselines | `results/gpu/disk_*.csv`, `models/` (ignored) | WP9 | `reproduce.py disk --gpu` | **About 2–3 h** (labels on CPU in parallel; one GPU job at a time; OOM retry) | `disk_learned_depth.csv` (trace, seed, val-chosen thresholds, test hit, issued, VCF depth, VCF hit, oracle-same-actions), `disk_lstm.csv`, `gpu_env.json` (nvidia-smi samples) |
| **E4** | Web full benchmark: per-capacity selection, all scorers except seq, freshness, decomposition, feedback gate, calibration, learning curve, visitor scope, parity | `results/web/` | WP6, WP5 | `reproduce.py web` | prepare 35 s + train 15 s + replay about 2–5 min on 96 workers + bootstrap about 5 min + learning curve about 10 min: **about 30 min** | `policy_metrics.csv`, `comparisons.csv`, `decomposition.csv`, `freshness.csv`, `calibration.csv`, `think_time.json`, `dataset_profile.json`, `session_parity.json`, provenance. Logistic fixed1 cap 8 = 65,944 / 445,796. |
| **E5** | Web GPU sequence model, 5 seeds, val protocol + final test | `results/web/seq_*.csv`, `models/` | WP11, E4 (for the scorer comparison rerun) | `reproduce.py web --gpu` | **About 1–1.5 h GPU** + a 5 min replay | `seq_metrics.csv` (seed, novel recall@8, hit@1, cache hit/U at caps 8 and 32, train s, us/req), then a re-run of `decomposition.csv` with the seq row |
| **E3** | Real disk testbed: calibrate 3 devices; full grid (5 traces x 8 B-modes + 11 A-modes x 2 think x 3 reps) on HDD sdb; reduced replica on sde; SSD closed-loop subset (Critic, §2.6) | `results/realio/` | WP8, (E2 optional for learned_depth) | `reproduce.py realio --device …` per device | Calibrate about 10 min per device; mkfile 32 GiB on sde at about 200 MB/s ≈ 3 min; grid **about 8 h** (sdb 6.6 h + sde 1 h + SSD 15 min; Critic recomputation) | `device_{sdb,sde,ssd}.json`, `realio.csv`, `realio_summary.csv` (mean ± CI over reps, k_eff, Spearman ρ), provenance with diskstats. B parity asserted on 100% of runs; foreign_io_frac < 1% on the HDDs. |
| **E6** | Live HTTP: headline (9 policies x 3 reps x SSD + HDD store), FIFO ablation, hint on HDD, optional sensitivity | `results/live/` | WP12, E4 (selected policies) | `reproduce.py live --store-dir /mnt/9792e947-f73e-45c8-a77f-0ba5cd89e363/os-project-web` | Headline about 5.4 h, ablation about 20 min, hint about 15 min, sensitivity about 3.5 h (optional). **About 6 h core.** | `live_metrics.csv` with valid=true for every cited row, `live_parity.csv` all 0, `live_config.json`, `disk_probe_{ssd,hdd}.json`, `live_report.md` |

### Wave 4 (integration)

**WP14 — Write the paper.** Owns `paper/sections/*.tex` and the final `paper/main.tex`. Depends on E1–E6
and WP13.

Tasks:
- Run `reproduce.py paper`.
- Write each section per §5, quoting numbers only through `numbers.tex` macros.
- Run an honesty review against the 12 constraints in §5, as a checklist in the WP output.

Acceptance:
- `paper/check.py` passes with 0 hand-typed numbers.
- Every figure and table is generated from `results/` with matching provenance SHAs.
- The page count estimate (words ≈ 8 pages) is recorded.

**WP15 — Unified docs.** Owns `AGENTS.md`, `README.md`, `docs/**` except REFERENCES.md, and
`presentation/README.md`. Depends on E1–E6 (numbers come from results). Can run in parallel with WP14.

Tasks:
- `AGENTS.md`:
  - shared rules
  - disk invariants: the existing list plus the 4-places rule now covering linux_ra and learned_depth,
    the device_ios parity, the cache_factory default, and the mandatory explicit latency model for
    oracle traces
  - web invariants: GRILL rules, per-capacity selection, the online index causality, cachecore as the
    single cache, the live validity gate
  - real-system rules: DONTNEED before every run, chunked WILLNEED, quiet-host gate, no SSD headline
    numbers
  - claim discipline
  - the corrected oracleGeneral format
- `README.md`: install and reproduce commands, and a results summary built from generated macros or CSVs.
- `DECISIONS.md` D18–D21, including the corrections to D10.5, D11, D13 and D17 wording.
- `WEB_DECISIONS.md` W1–W4.
- `architecture.md`, `REPRODUCE.md`, `METRICS.md`.
- `PRESENTATION.md`: about 15 slides, each with its figure file, plus a viva Q&A built from the honesty
  constraints.

Acceptance:
- Every command in README and REPRODUCE runs. WP16 checks this.
- No document contains a number that is absent from `results/`, verified by a grep script.

**WP16 — Final verification (serial).** Owns nothing new; it is a fix-forward pass.

Tasks:
- Fresh clone of `master` into scratch.
- Run the full test suite with the venv and with system python3.
- `reproduce.py disk --quick`, `web --quick`, `paper`.
- `paper/check.py`.
- Verify results provenance SHAs and confirm that `git status` is clean.
- Push `origin master` and tag `paper-v1`.

Acceptance: all green, and the push succeeds.
(Critic) Pushing and tagging are outward-facing. The lead pushes only on the user's explicit go-ahead
in the conversation; a workflow or agent message does not count. The same applies to the first write
to the two empty HDDs (`/mnt/9792e947…` for E6 and `/mnt/c609e3de…` for the E3 replica). Confirm them
once with the user before WP8/WP12 smoke runs. sdb already holds `os-project-prefetch/` from the
prototype.

### Dependency graph (summary)

```
WP0 ─┬─ WP1 ─────────────┐
     ├─ WP2 ─┬───────────┼─ WP9 ─── E2 ─┐
     ├─ WP3 ─┼─ WP8 ─────┼──────────────┼─ E3 ─┐
     │       └─ WP10 ────┼─ E1 ─────────┤      │
     ├─ WP4 ─────────────┤              │      │
     ├─ WP5 ─┬─ WP6 ─────┼─ E4 ─┐       │      ├─ WP14 ─┬─ WP16
     │       ├─ WP11 ────┼──────┴─ E5 ──┤      │  WP15 ─┘
     │       └─ WP12 ────┼─────────────────── E6
     └─ WP7 ─── WP13 ────┘
```

Critical path: WP0 → WP3 → WP8 → E3 → E6 → WP14 → WP16. The two device experiments run one after the
other on a quiet host, and together take about 14 h of the wall time (E3 about 8 h, E6 about 6 h; Critic).

Total estimate:
- Code waves 0–2: about 1 working day of agent time in parallel.
- Wave 3: about 18–20 h wall, of which about 14 h are the exclusive E3 and E6 runs (Critic). E3 and E6
  each need a quiet host, so in practice this is 2–3 calendar days on a shared machine.
- Wave 4: about half a day.

---

## 7. Risks and mitigations

- **Shared host or GPU contention** invalidates timings. E3 and E6 gate on load average and abort
  otherwise. GPU jobs retry on OOM, and their timings carry contention notes.
- **HDD space and permission.** The HDD paths (`/mnt/9792e947…`, `/mnt/c609e3de…`) were described as
  writable and empty in the task context. WP8 and WP12 write only under an `os-project-*` subdirectory
  there, each with a README. If the user objects, fall back to sdb only, where the prototype's directory
  already exists.
- **The learned_depth gain may shrink against VCF across seeds.** Report it whatever it is. The paper's
  thesis does not depend on it.
- **The web GRU may not beat online counting.** That outcome supports the thesis; report it.
- **The E1 grid could overrun.** It is resumable, and C2 can drop the 1048576-block capacity if needed;
  any reduction is recorded.
- **Overleaf compile is untested locally.** `check.py` plus a minimal IEEEtran template reduce the risk,
  and the students compile once early (after WP7) to confirm.

---

## Critic changes

The completeness and skeptical review was checked against the worktrees (`mainref` @ db05d74,
`pivot` @ 8569cdf), `$EVAL/*` and the machine (`df`, mounts, `python-pptx` availability). The changes
above are marked "(Critic)" in place. Summary:

**Correctness and honesty**
- C1. **The FIU numbers quoted from `sweep2` are inflated.** `main-science/run2.py:94-112` "raw" leaves
  FIU 4 KiB pages unmerged, and "merged" coalesces every corpus with no cap. The pooled §1.1 figures
  (+16.32, +42.47, +31.27 pp, 0.659x) are now marked orientation-only (§0.6, §1.1). A merge vocabulary
  `auto|none|same_second(uncapped)` maps sweep2 cells onto master for the cross-checks (§2.1, C2).
  Honesty constraint 13 is added.
- C2. **LSTM and `learned_depth` leakage.** E2 trains on the first 60% of each trace's 2M prefix, and
  C1 replays the first 1M requests of the same traces. Both modes are now always `skipped` in C1.
  E3 scores `learned_depth` on the E2 test segment via `skip_requests`, with matched baseline rows.
  msr_prn_0 is not in `PAPER_SUBSET`, so E2 must also train it. Honesty constraint 14 is added.
- C3. **Fair action space for `learned_depth`.** The pilot capped segments and VCF at 64 blocks, and
  VCF hit the cap on 7/8 traces. If the VCF grid goes to 256, the model gets [64,128) and [128,256)
  segments. Both the capped and uncapped comparisons are reported. "2M" means canonical requests, and
  short FIU traces use their whole length.
- C4. **The thesis is now a pre-registered hypothesis**, because the `learned_depth` pilot already cuts
  against "the predictor is the smallest lever". `PREREGISTERED.md` per study must be committed before
  E1–E6 start.
- C5. **M16 (GNB prior skew) is opt-in.** Changing `make_model` would break the sweep2 `adaptive`
  cross-check, the committed `msr_sample_gnb.json` reproduction and the CLASSIFIER_EVAL figures, all
  for a change with no measured accuracy effect. The M9 bound is documented as a behaviour change on
  traces whose footprint exceeds 4 M blocks, and each row records the bound.
- C6. **Belady/MIN demand reference added to C1** (the gap for M8). It is built from the block stream,
  not from `next_access_vtime`, which is keyed per obj_id and not per block.
- C7. Honesty constraint 15 is added: realio prediction error and k_eff are stated per trace and per
  think model, not as device properties. The "no SSD headline" rule is scoped to E3 only.

**Infeasible or underspecified as written**
- C8. **`linux_ra` cannot be a `next_candidates(request)` predictor.** It reads cache state: the miss
  list, the marker block still resident and unused, and a backwards residency scan
  (`run2.py:120-185`). WP3 adds a read-only `observe(request, missed_blocks, cache_view)` hook to both
  replay paths, and its acceptance test is exact reproduction of the sweep2 `linuxra` rows on
  non-FIU traces. In realio, `linux_ra` keeps its state against the shadow LRU, and the kernel's own
  policy is measured by `kernel_ra` and `kernel_seq`.
- C9. `sweep_collection` (`sweep.py:69-77`) has no `read_ahead_depth` parameter, so one is added
  together with `max_blocks`.
- C10. **E3 capacity is set to 65,536 blocks (32 MiB).** The cgroup budget is MemoryMax minus about
  7 MiB, so a 1 MiB cache is not realisable in Backend A. The capacity mismatch with E1 must be stated.
- C11. **E3 runtime recomputed** from the prototype timings: 19 runs per cell, plus a 13 s closed run,
  a 65 s spread run and a 5 s flush, gives about 6.6 h per full HDD, not "4–6 h" in total. The replica
  HDD and the SSD get reduced grids, for a total of about 8 h. The wave-3 total becomes about 18–20 h
  wall, and the exclusive E3+E6 time about 14 h.
- C12. **CPU caps for overlapping E1/E2/E4** (64/16/40), because 80+96 workers oversubscribe 128 shared
  cores. Timings taken during an overlap are flagged and not quoted.
- C13. The contiguity guard raises only on a **strictly** lower rate, so very random probes do not
  fail on a 0-vs-0 tie. Short FIU traces are `n/a` for the C3 second slice. WP3's memory test asserts
  the age-table size instead of RSS, which is flaky in a shared test process.
- C14. **No `.pptx` is possible**: `python-pptx` is absent from both interpreters and cannot be
  downloaded. The presentation deliverable is `docs/PRESENTATION.md`, plus an optional Beamer
  `presentation/slides.tex`, owned by WP15. It uses `paper/figures/*.pdf` and is compiled on Overleaf
  with the same toolchain as the paper.

**Conflicts between parallel WPs and hidden dependencies**
- C15. **One git worktree and branch per WP**, merged serially by the lead with a full-suite run after
  each merge. As written, parallel agents would share one working tree and index.
- C16. **`outputs/eda.md` collision.** `main.py:472` and `webprefetch/cli.py:23` both write it. Web
  defaults move to `outputs/web/` and `models/web/` (WP0 task 9).
- C17. **Demo parity test vs fresh checkout.** `tests/test_demo_parity.py:15-17` needs gitignored
  models, `data/processed` and six `outputs/` files that WP0 moves to `results/pivot_original/`. It now
  gets explicit paths plus a `skipUnless` guard, and the reference stays valid only for the same run's
  selection (WP0 task 10, acceptance).
- C18. **`reproduce.py` split into `repro/<study>.py`**, each owned by one WP, so that WP6, WP8, WP9,
  WP10, WP11, WP12 and WP13 do not all edit one file in wave 2.
- C19. **Results contract `prefetch_common/schemas.py` (WP4).** WP13 builds its fixtures in parallel
  with the producers, and without a shared schema the generators would drift.
- C20. **`tests/test_project.py` ownership passes to WP9 in wave 2** for `learned_depth`
  registration. The "every mode" parity tuple at `:1374-1380` actually omits stride, markov, lstm and
  adaptive, so the plan no longer assumes it pins every mode. `learned_depth` with no artifact must
  emit a `skipped:` row in both the benchmark and the sweep.
- C21. **WP12 also depends on WP6's scorer registry** (online-index candidates, validated penalties).
  WP11's CLI is `python -m webprefetch.seqmodel` because WP6 owns `cli.py`. WP12's timing tests are
  `slow` and load-gated.
- C22. **E0 device calibration** runs before E1's analysis. `kappa.csv` and F4 need the measured device
  kappa, which previously came only from E3, scheduled after E1.

**Scope and process**
- C23. The paper scope exceeds 8 IEEE pages (12 sections, 8 tables, 8 figures). A cut priority list is
  added, and the students must state the real page limit before WP14.
- C24. **Outward-facing actions need the user's explicit go-ahead**: the push and tag, and the first
  writes to the empty HDDs sdd and sde. Agent or workflow messages are not consent.

**Not changed, noted for the lead**
- The `zstd -dc` fallback runs the system-installed `/usr/bin/zstd`. That is not a downloaded binary,
  but no test may depend on it; the venv has `zstandard`.
- `refs.bib` has no PyTorch or `posix_fadvise`/`cachestat` man-page entry. Add them only after
  verifying each one (Crossref or kernel.org) in WP7.
- Do not copy the literature report's sentence that the CMU mirror "splits each request into 512-byte
  records". It contradicts §0.1; the `refs.bib` note itself is correct.
- P2(b) (the relabelled ranker) must be saved under a new artifact name, so that the default ranker
  still reproduces the committed rows.
- WP15's rule that no document may contain a number absent from `results/` needs the same allow-list
  as `paper/check.py` (years, hardware models, decision IDs, core counts).

---

## Lead decisions (binding; they override anything above that conflicts)

These settle the open items the critic left for the lead. The user answered all questions up front and
asked for autonomous work with no further input.

- **L1. Pushing is authorised.** The user asked for commits to be pushed to `origin` with the local git
  identity `AshvikMishra <ashvikmishra.mit@gmail.com>`; a test push of `master` succeeded. The lead (or
  the merge agent) pushes `master` after each wave once the full suite is green, and tags `paper-v1` at
  the end. WP agents never push their own `wp/*` branches.
- **L2. One HDD only.** Real-device experiments use the SATA SSD (repo or scratch paths) and the HDD
  `sdb` under `/mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch/` only, where the prototype
  already wrote. **Nothing is written to `sdd` or `sde`.** The E3 replica on `sde` is dropped; E6's HDD
  object store lives at `/mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch/web-store/`.
  E3 and E6 run one after the other, never together, so they do not contend on `sdb`.
- **L3. Runtime budgets.** E3 is cut to about 4 h total (fewer traces or reps if needed, recorded in
  provenance); E6 core to about 3 h. Every reduction is logged, never silent.
- **L4. Page budget.** Target 8 IEEE pages including figures, with references extra. Material that
  does not fit goes in an appendix after the references (allowed in the source; the students may cut it).
- **L5. Presentation.** `docs/PRESENTATION.md` plus a Beamer deck `presentation/slides.tex` that reuses
  `paper/figures/*.pdf`. No `.pptx` (python-pptx is unavailable).
- **L6. LaTeX.** No local compiler can be installed. `paper/check.py` is the local gate; the students
  compile on Overleaf.

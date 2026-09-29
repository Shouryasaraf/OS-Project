# AGENTS.md

ML-based adaptive disk I/O prefetching — a **user-space simulation** (replay of
recorded I/O against a modelled LRU cache). It does not touch the OS or a real
device. Read this before changing claims, numbers, or docs.

## Commands

The package lives in `src/` and is **not installed** in this environment. From
the repo root, `python -m adaptive_prefetch ...` fails with
`No module named adaptive_prefetch` unless you set the path first
(PowerShell):

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v     # 79 tests, ~20s
python main.py                              # full research run, no prompts, ~6 min
python main.py --quick                      # skip LSTM + regime sweep
python main.py --interactive                # guided dataset/LSTM picker
python main.py --cost-model assumed         # vs the 5/100/50 us model
python -m adaptive_prefetch demo
python -m adaptive_prefetch benchmark --cost-model measured
python -m adaptive_prefetch evaluate        # classifier x regime accuracy + cost
python -m adaptive_prefetch eda <trace>     # profile one trace
python -m adaptive_prefetch shift <trace>   # domain shift vs synthetic region
python sweep_modes.py                       # all 32 MSRC traces x all modes
python sweep_modes.py --full                # every request in every file (hours)
python sweep_modes.py --per-trace 50000     # cheaper sample
```

- `main.py` self-bootstraps `sys.path`; every other entry point needs
  `PYTHONPATH=src` or `pip install -e .`.
- **`main.py` is the primary entry point.** It runs EDA, classifier
  selection, domain shift, the 7-policy benchmark against a
  perfect-predictor bound, and drift. The console deliberately shows
  *conclusions*; every raw row is written to `outputs/` by `report.py`. Do not
  dump wide tables back to the console — put them in the file map instead.
- Single test / class: `python -m unittest tests.test_project.SimulatorTests` or
  `python -m unittest discover -s tests -k LSTMTests`.
- The unit suite is the **only** verification gate. There is no CI, no linter,
  no formatter, no typechecker, no `pytest` config — do not assume any of these
  run automatically. numpy/sklearn/matplotlib are **not** installed; the core
  is stdlib-only by design. `torch` (2.14 CPU) is installed, so the LSTM path
  is live, but it still needs a *trained* artifact.
- `models/lstm_delta.pt` and `outputs/` are gitignored. The LSTM artifact is
  versioned (`ARTIFACT_VERSION`); an older one is rejected with a "retrain"
  message rather than crashing.

## Invariants the tests pin (do not break)

- `simulator.replay()` is the single causal loop shared by all modes in
  `benchmark.MODES` (`none, sequential, strided, stride, markov, lstm,
  adaptive, adaptive_evidence`).
  Prefetches are gated on `index >= window_size` — the **first window never
  prefetches** and a window's policy only affects later requests. Three tests
  assert this; keep the gate.
- `adaptive_evidence` (`simulator.route_window`) routes on the window's own
  features — contiguity, repeated stride, jumpiness — and needs **no trained
  model**. Its thresholds were calibrated against the measured real-trace
  feature distribution, not invented. Since the generator was corrected (D10)
  it is much closer to the classifier than it was, and it wins outright on
  `msrc_hm_1`.
- Adding, removing, or **reordering** a mode breaks
  `test_benchmark_includes_skipped_optional_lstm`, which asserts
  `rows[0]["mode"] == "none"` and `rows[5]["status"].startswith("skipped")`.
  It now checks `len(rows) == len(MODES)`, so appending is safe but reordering
  is not.
- `Metrics.prefetch_recall` is backed by a future-read `Counter` built at replay
  start. It is **measurement only**. Never feed it into candidate generation.
- Classifier updates require trustworthy labels:
  `replay(..., online_updates=True)` raises unless `labels` covers every complete
  window. On real traces the model is frozen. `--online-real` /
  `adaptive_pseudo` self-trains at `pseudo_label_threshold=0.95` and is not
  accuracy evidence.
- A missing/failed optional baseline must produce a row with
  `status="skipped: <reason>"` and `None` metrics — never a substituted or
  synthetic value.
- Benchmark rows carry private `_read_blocks`, `_prefetchable_misses`,
  `_total_latency_us`, `_inference_calls`, `_total_inference_us`, `_requests`
  keys that `aggregate_results` depends on; `write_results_csv` strips them via
  `COLUMNS`.
- Two candidate sources feed the same cache: window-based `simulator.candidates()`
  and per-request `next_candidates(request)` predictors (`baselines.py`,
  `lstm.py`). A new baseline = implement `next_candidates` + register the mode
  in `replay`.

## Do not swap in a "better" classifier on intuition

`DEFAULT_CLASSIFIER` is `gnb`, and that was measured, not assumed. QDA wins
only on windows that straddle a phase change (86.6% vs 60.5%) and **loses
under the covariate shift this project faces** — 91.7% vs 100% when block size
changes 1→128. Reasoning is in `docs/DECISIONS.md` D1.

Note the pre-D10 figures quoted in `docs/CLASSIFIER_EVAL.md` (100.00% ± 0.00
held-out accuracy) were a property of the *degenerate* generator. The
generator now produces realistic within-class variance and held-out accuracy
is **0.9688**, which is the intended outcome, not a regression.

## Artifacts and feature changes

- Classifier persistence is **JSON, not pickle** (`artifacts.py`, format
  version 2). Payloads are tagged with `classifier` (`gnb` / `qda`) and carry
  the statistics that kind needs. `load_model` rejects a mismatch in format
  version, `FEATURE_NAMES`, `CLASSES`, classifier tag, `decay_factor`, or
  `shrink` — with a message telling you to regenerate.
- Changing `FEATURE_NAMES` or `CLASSES` invalidates the committed
  `models/msr_sample_gnb.json`; regenerate it with
  `python -m adaptive_prefetch train-msr-sample`.
- `FEATURE_NAMES` is now **12**: 8 stream features plus 4 contextual ones
  (`reuse_ratio`, `reuse_distance_log`, `contiguous_delta`, `stride_persist`).
  They are produced by `features.WindowContext`, which carries a block-age
  table across windows. `extract()` still returns only the 8 stream features
  (`STREAM_FEATURES`), so any caller using `extract` against a 12-feature
  model will raise. `benchmark.make_model` lays training windows out as one
  continuous stream precisely so train and replay build the same vector.
- `models/*` (except `msr_sample_gnb.json`) and `outputs/` are gitignored, so
  regenerated LSTM/result artifacts will not be committed by accident.
- `models/msr_sample_gnb.json` was adapted **in-sample** on
  `msr-cambridge1-sample.csv` with a `random` proxy label. Replaying that same
  sample with `--model-path` is a load check, not a held-out evaluation.

## Cost model: measured, not assumed

`LatencyModel.measured()` calibrates from the trace's own recorded device
service times (`.revised` column 6, retained on `Request.service_ms`).

- It uses the **mean**, never the median. 65-87% of real requests complete in
  under a microsecond, so a median-based model collapses to a near-zero cost
  and makes every policy look free. There is a regression test for this.
- Measured means are **260-5491 µs**, so the original assumed 100 µs
  demand-miss cost understated a miss by 3-50x.
- The **hit cost is still an approximation** (1% of mean service) and is
  labelled as such — the traces record device time only, never cache time.
- `Request.pattern` holds the trace's own `seq`/`rand` flag. It is for
  ground-truth scoring **only**; routing on it is not a result.

## The MSRC sweep: streaming, and why it is chunked

`sweep_modes.py` replays **every** `.revised` trace under every mode. The
collection is 11.86 GB over 32 files and a `Request` costs **~430 bytes**
resident, so it cannot be loaded.

- `trace.iter_csv_chunks` streams bounded chunks; `simulator.replay_stream`
  carries cache, metrics, partial-window state, block ages, stride and policy
  across chunk boundaries. Verified bit-identical to `replay()` across 80
  chunk boundaries on a 161k-request trace.
- **Chunk by request count, never by bytes.** A 500 MB slice of a `.revised`
  file is ~8.9M requests, which costs ~3.8 GB resident and will exhaust a
  15 GB machine. `DEFAULT_CHUNK = 250_000` requests is ~108 MB.
- Warm-up semantics match `replay()` exactly (`seen >= window_size`), *not*
  "first window closed". These differ when a stream's last partial window
  matters, and getting it wrong silently changed hit ratios.
- `prefetch_recall` is **not reported** by the streamed path: it needs
  whole-stream future knowledge. Hit ratio, precision, wasted I/O and
  modelled cost are exact, and those are what rank policies.
- The default caps at 250k requests per trace and says so in the output.
  `--full` removes the cap; the collection alone is ~94 min to parse at the
  measured 2.1 MB/s, before any replay.

## Data gotchas

- `data/MSRC-trace-003/final-trace/*.revised` is present locally but **gitignored**
  (32 files, 40 KB to 2.6 GB). `load_csv(..., "revised")` materializes the whole
  file as a `list[Request]` — measured ~5.6 MB/s and ~17M dataclass instances per
  GB. `main.py` samples five of them and caps each at 40k requests; do not
  raise that without a runtime reason.
- The default cache is `simulator.DEFAULT_CAPACITY` = **2048 blocks** (1 MiB
  at 512 B). It was 128, which was calibrated for the old generator's 1-3
  block requests; with realistic 8-128-block requests 128 blocks held ~16
  whole requests and starved every policy. Hit ratios saturate by 2048. See
  `docs/DECISIONS.md` D10.5.
- Five loader profiles: `normalized, msr, iotta8, alibaba, revised`. `auto`
  detects by column name only; headerless `iotta8` and `.revised` need an
  explicit `--format`. Loaders validate ascending timestamps and report the
  offending line number; the MSR loader re-sorts by raw `Timestamp` first
  because merged samples concatenate hosts.
- Committed samples under `data/samples/` are small (1k requests). Their
  provenance is explicitly unverified — see `data/samples/README.md`.

## Claim discipline (the repo is explicit about this)

This is a graded lab submission (`docs/reference/*.pdf`). Several docs
independently forbid overclaiming. If you touch numbers, README, docs, or the
slide deck, preserve:

- Latency/speedup comes from a **cost model**. `LatencyModel.measured()`
  calibrates hit/miss/prefetch from the trace's own recorded device service
  times (means 260-5491 µs, vs the 100 µs the original assumption used). The
  **hit** cost is still an approximation (1% of mean service) because the
  traces record device time only, never cache time. No queueing, bandwidth,
  or prefetch-completion modelling.
- Classification accuracy is on **independently generated synthetic windows**,
  never verified real-trace accuracy. The MSR sample has no four-class labels.
- LSTM predicts address **deltas**, not workload classes — compare cache metrics
  and inference cost, never its "accuracy".
- Oracle re-access recall can look high while hit ratio is low; report both.
- `docs/STAGE2_RESULTS.md`, `docs/REVIEW2.md`, and `presentation/README.md` all
  state that stored numbers are from an earlier run and must be regenerated from
  a fresh benchmark before presenting. Re-run the benchmark; do not hand-edit
  numbers into docs or `.pptx`.
- `presentation/Review2_Adaptive_Disk_IO_Prefetching_v5.pptx` is an outdated
  draft — do not update it.

## Known current state (fix or preserve deliberately)

- **`docs/STAGE2_RESULTS.md` and both `presentation/*.pptx` decks are stale.**
  They carry pre-fix numbers including the old LSTM rows (~7% hit ratio) and
  the old "7.3x faster inference" claim (the real gap is ~640x per request).
  They must be regenerated from a fresh `python main.py` before presenting.
  `docs/CLASSIFIER_EVAL.md` and `docs/DECISIONS.md` are current.
- **The drift report demonstrates nothing.** All eight rows are lag 0,
  accuracy 1.000 — frozen and online are identical because the task is
  saturated. The console prints that warning, but a reviewer skimming the
  table may miss it. The `transition_heavy` regime is the non-saturated task.
- **The real-workload result is still mostly negative** and is reported as
  such: fixed read-ahead beats the adaptive policy on 6 of 7 real traces at
  the corrected 2048-block cache. `msrc_hm_1` is the one dataset where
  adaptive wins. Do not soften this. See `docs/DECISIONS.md` D9 for the three
  hypotheses tested and rejected, and D10 for why the generator fix changed
  the numbers but not this conclusion.
- `docs/architecture.md` claims `main.py` applies a "per-trace request limit"
  when training the LSTM; it does not — it trains on the whole selected sample
  and prints a duration warning.

## Where the detail lives

`docs/DECISIONS.md` is the decision and iteration log — read it before
changing the classifier, the LSTM, or the report layout; it records what was
tried, what the measurement said, and why the default stayed Naive Bayes.
`docs/architecture.md` is the best module map, data-flow, and per-metric limits
reference. `docs/CLASSIFIER_EVAL.md` holds the detailed classifier comparison,
the LSTM rework, the full bug list, and the measured limits of the available
data. `docs/PIPELINE.md` is the short pipeline/7-mode summary, `docs/REVIEW2.md`
the demo/viva script, `docs/implementation-stage2.md` and
`implementation-stage3.md` the executed work plans, `docs/MSR_SAMPLE_TRAINING.md`
the weak-supervision record. `README.md` is the user-facing command reference.

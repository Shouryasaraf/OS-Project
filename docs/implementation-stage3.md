# Implementation — Stage 3 (Final Version)

Closes **every** gap in the checklist and delivers the "final version" =
full functionality per the OS Lab Submission 2 (Review 1 PDF) spec, on top
of the completed Review 2 prototype. Design rationale for each item lives in
`implementation-stage2.md` (root); this doc is the execution plan with
Definition of Done.

---

## Definition of Done (the final version, all checked)

- [ ] All 7 modes run in one benchmark matrix: `none, sequential, strided, stride, markov, lstm, adaptive`
- [ ] All 6 metrics per mode: `hit_ratio, precision, recall, unused, mean_latency_us, latency_speedup`
- [ ] Datasets: synthetic (multi-seed), MSR sample, + at least one full MSR/IOTTA trace normalized via `normalize`
- [ ] LSTM optional (torch extra); benchmark skips gracefully when absent; stdlib core intact
- [ ] Drift benchmark: frozen vs online-updated adaptive, switch-lag reported
- [ ] `tests/test_project.py` green; `benchmark` prints full matrix on MSR sample
- [ ] README + Pipeline.md updated; demo command chain documented

## Work items (execute in order)

### W1 — Baselines (`src/adaptive_prefetch/baselines.py`, new) · stdlib
- `StridePrefetcher`: delta recurrence ≥2× → prefetch `lba + k·stride`, k=1..degree(2).
- `MarkovPrefetcher`: bucketed-delta state chain, top-2 next-delta candidates ≥ confidence 0.3; optional 2nd-order `(d₋₂,d₋₁)` key.
- `CandidatePredictor` protocol shared with replay.
- **Accept:** strided trace → stride steadies; markov learns transition on synthetic mixed stream.

### W2 — Metrics + latency (`simulator.py`, modify)
- `Metrics.prefetch_recall` (oracle-lookahead: `useful / (useful + later-reaccessed misses)` — measurement only, documented).
- `LatencyModel`: configurable `t_hit=5µs, t_miss=100µs, t_prefetch=50µs` → `mean_latency_us` + `latency_speedup_vs_none`.
- **Accept:** recall ∈ [0,1] sanity; latency monotonic with hit ratio (same cache, same stream).

### W3 — Benchmark harness + IOTTA (`cli.py`, `trace.py`)
- `replay`/demo internals → mode registry incl. `stride, markov, lstm`.
- `benchmark` subcommand: loops datasets × modes, prints Markdown/CSV matrix.
- `normalize` subcommand: MSR (done) + **IOTTA** loader (auto-detect `sector/size/op/timestamp` variants) → normalized CSV.
- **Accept:** `benchmark --trace msr-cambridge1-sample.csv` → 7-row table, no crashes on absent LSTM.

### W4 — Offline LSTM (`src/adaptive_prefetch/lstm.py`, new; optional)
- 1-layer LSTM h=64, window of 16 deltas → top-64 bucketed next-delta distribution, top-2 prefetch candidates.
- `train-lstm` CLI → saves `models/lstm_delta.pt` (gitignored); `pyproject.toml` `[optional-dependencies] lstm=["torch"]`.
- **Accept:** trained on synthetic; replay `--mode lstm` loads artifact; missing torch/artifact → clear error, benchmark row skipped.

### W5 — Online-learning completeness (`simulator.py`, `cli.py`)
- Drift benchmark (labelled synthetic transition): switch-lag windows, frozen vs online-updated GNB vs LSTM-sliding-retrain (slow reference).
- `--online-real` flag: confidence-gated pseudo-label updates (≥0.95), never from prefetch success; measured effect vs frozen, no correctness claim.
- **Accept:** lag numbers printed per phase; frozen < updated on stable phase-length sweep.

### W6 — Finalization (docs + tests)
- Tests: predictor correctness, recall/latency arithmetic, IOTTA normalization, 7-mode bench run, causality re-check for new modes.
- Docs: `Pipeline.md` (7-mode matrix, metrics incl. recall/latency), `README.md` (benchmark/normalize/train-lstm commands, lstm extra note).
- **Accept:** unit tests green (`python -m unittest discover -s tests -v`); demo→benchmark→replay chain runnable per README.

## Runbook (final-version demo)

```powershell
python main.py                                  # guided run incl. demo + benchmark + drift
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v         # unit tests
python -m adaptive_prefetch benchmark           # full matrix, synthetic + msr sample
python -m adaptive_prefetch normalize --input trace.csv --format iotta8 --output out.csv
python -m adaptive_prefetch train-lstm --save models/lstm_delta.pt   # if torch extra installed
python -m adaptive_prefetch benchmark --lstm-model models/lstm_delta.pt   # adds LSTM row
```

## Non-goals (unchanged from stage 2)

No kernel/device I/O, no calibrated probabilities, no production-label
claims on real traces, no GPU/distributed training. LSTM stays offline CPU.

## Final-version mapping (PDF slide → deliverable)

| PDF requirement | Deliverable | Work item |
| --- | --- | --- |
| Classify in real time (slide 8) | GNB + held-out accuracy/confusion | done (Review 2) |
| Adapt policy per class (8) | class→policy routing in replay | done (Review 2) |
| Learn online (8) | drift benchmark + pseudo-label flag | W5 |
| Benchmark vs stride/Markov/LSTM (8, 11) | 7-mode matrix | W1, W3, W4 |
| Precision/recall, hit ratio, latency (11) | 6 metrics incl. recall + LatencyModel | W2 |
| MSR + IOTTA datasets (11) | normalize + loaders | W3 |
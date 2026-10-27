# Reproducing every result

This page lists every command that produces a file under `results/` or
`paper/`, how long it took here, what it needs, and what it writes. Each
study's `results/<study>/provenance.json` records the command, the git
commit, the host, its load average at start and end, the wall time, and the
SHA-256 of every input and output, so any committed file can be traced to the
run that wrote it. `prefetch_common.provenance.verify_provenance` re-hashes
the outputs, so a hand-edited results file is detectable.

Run everything from the repository root. The package is not installed in
this environment: either set `PYTHONPATH=src` or `pip install -e ".[all]"`.

## 1. Environment

| Item | This machine | Needed for |
|---|---|---|
| Python | `3.12.3` (`.venv/`, every extra installed) | everything; `>=3.10` |
| disk core | standard library only | `main.py`, the disk simulator, the paper lint |
| `[traces]` extra | `zstandard` | reading `*.oracleGeneral.zst` (falls back to the system `zstd -dc`) |
| `[web]` extra | numpy, pandas, scikit-learn, joblib | the web study, live HTTP plan |
| `[gpu]` extra | torch with CUDA | E2 (disk `learned_depth`, LSTM) and E5 (web GRU) |
| `[paper]` extra | matplotlib, scipy | figures, Wilcoxon p-values |
| OS | Linux kernel `6.14.0-37-generic`, MGLRU | the real-device testbed (Linux only; it needs `O_DIRECT`, `posix_fadvise`, `cachestat(2)` from `6.5`, and a user systemd manager for the memory cgroup) |
| LaTeX | none installed (lead decision L6) | the paper and the slides compile on Overleaf |

<!-- source: results/disk/provenance.json; results/gpu/provenance.json -->
Host recorded in the provenance files: 128 logical CPUs, an
`NVIDIA RTX PRO 6000 Blackwell Workstation Edition` GPU, both **shared with
other users**. Load averages at the start and end of every run are in each
`provenance.json`; the GPU was 92-99% busy with other tenants' work during
E2 and E5, so their training and inference times are contended and are not
quoted as costs.
<!-- source: none -->

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[all]"   # or use the existing .venv
export PYTHONPATH=src
PY=.venv/bin/python
```

## 2. Data (gitignored; obtained separately)

| Data | Where it goes | How it is checked |
|---|---|---|
| 56 libCacheSim `oracleGeneral` block traces (MSR 14, FIU 10, Alibaba 20, Tencent 12), from the CMU PDL cache-datasets mirror (sources in `NOTICE.md`) | `data/block-traces/<corpus>/*.oracleGeneral.zst` | SHA-256 of every trace used is in `results/disk/provenance.json` (`inputs`) |
| RetailRocket `events.csv`, Kaggle version 2 | `data/raw/events.csv` | the SHA-256 is pinned in `src/webprefetch/data.py`; a different file is refused |
| MSRC-trace-003 `.revised` files (legacy disk sweep only) | `data/MSRC-trace-003/final-trace/` | not present on this machine; nothing in `results/` uses them |

The committed samples in `data/samples/` (MSR-format traces of a thousand
requests each, a synthetic demo trace and a synthetic web fixture) need
nothing extra.

## 3. Tests: the only verification gate

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests   # every test; web/GPU/device tests run
PYTHONPATH=src python3 -m unittest discover -s tests            # stdlib only: web and GPU tests skip
python3 tools/check_doc_numbers.py                              # docs quote only numbers in results/
python3 paper/check.py --strict                                 # paper lint (after reproduce.py paper)
```

There is no CI, linter, formatter or type checker. Device and live-HTTP
timing tests are skipped unless Linux, `O_DIRECT` and a quiet host are
available. The suite takes about a minute and a half with the venv.

## 4. One command per study

`reproduce.py` dispatches to `repro/<study>.py` and prints its runtime
estimate before starting. `--quick` is always a bounded smoke run that writes
under `outputs/` and never touches `results/`: a smoke run is not a result.

| Command | Experiment | Writes | Needs |
|---|---|---|---|
| `python reproduce.py disk --quick` | unit suite, `main.py --quick`, 3 traces x every mode x 50k requests plus analysis | `outputs/disk-quick/` | block traces |
| `python reproduce.py disk --workers 64` | **E1**: multi-corpus sweep C1-C4, corpus profile, analysis | `results/disk/` | block traces; many cores |
| `python reproduce.py disk --gpu` | **E2**: `learned_depth` per trace and seed, the honest-split LSTM | `results/gpu/`, `models/` | block traces, CUDA |
| `python reproduce.py realio --device DIR` | **E0 + E3**: device calibration, then the real-device grid on the device holding `DIR` | `results/realio/` | Linux, a backing file (`python -m adaptive_prefetch.realio mkfile`), a quiet host |
| `python reproduce.py web` | **E4**: prepare, train, per-capacity benchmark, freshness, decomposition, parity, learning curve | `results/web/`, `models/web/`, `outputs/web/` | `events.csv`, `[web]` |
| `python reproduce.py web --gpu` | **E5**: GRU sequence model, 5 seeds, and the decomposition `sequence_model` lever | `results/web/seq_metrics.csv`, `results/web/seq/` | E4's prepared views, CUDA |
| `python reproduce.py live --store-dir DIR` | **E6**: real HTTP replay with simulator parity | `results/live/` | E4's `comparisons.csv`, prepared views and ranker; a quiet host |
| `python reproduce.py msrc` | **E7**: MSRC light sample, labelled-window transfer check (seq flag) and sanity replay; see `results/msrc_samples/README.md` for its wall time | `results/msrc_samples/` | `data/samples/msrc/` (tracked), `[web]` |
| `python reproduce.py paper` | figures, tables and `numbers.tex` from `results/` (no experiment) | `paper/` | `[paper]` for figures |
| `python reproduce.py all [--quick]` | every study above in order; a step that cannot run here is passed over and the exit status says so | as above | |

### Measured wall times of the committed results

<!-- source: results/disk/provenance.json; results/gpu/provenance.json; results/realio/provenance.json; results/web/provenance.json; results/web/seq/provenance.json; results/live/provenance.json -->
| Run | Command recorded | Wall time | Notes |
|---|---|---:|---|
| E1 | `reproduce.py disk --workers 64` | 4131 s | 9324 sweep rows; load average rose from 3.96 to 83.18 during the run (other users) |
| E2 | `reproduce.py disk --gpu` | 4694 s | GPU shared, 92-99% busy |
| E0 (SSD calibration) | `realio calibrate --device-dir outputs/realio/ssd` | 4.426 s | the HDD calibration is in `device_sdb.json` |
| E3, HDD part 1 | `reproduce.py realio --device .../os-project-prefetch` | 4614.721 s | 255 runs, closed loop, 5 traces |
| E3, HDD part 2 | the same, think model `spread:1:5000` | 4120.688 s | 51 runs on msr_hm_0 |
| E3, SSD subset | `reproduce.py realio --device outputs/realio/ssd --device-name ssd` | 509.45 s | 54 runs |
| E4 | `reproduce.py web --workers 40` | 1319.366 s | recorded under `extra.e4_provenance` |
| E5 | `python -m webprefetch.seqmodel experiment` | 6654.864 s | `results/web/seq/provenance.json` |
| E6, last invocation | `webprefetch.live run ... --experiment hint` | 757.888 s | 42 passes in total over several invocations; every pass's `wall_s` is in `live_config.json` |
<!-- source: none -->

Each E3 and E6 invocation appends to the same results directory; the earlier
invocations' commands, times and git states are kept under
`extra.earlier_invocations` (E3) or in `live_config.json` (E6).

### What each run reads and writes

- **E1** (`repro/disk.py`): `python -m adaptive_prefetch.multicorpus --config
  C1..C4` writes `results/disk/sweep.jsonl` (resumable; a torn last line is
  repaired), then `python -m adaptive_prefetch.analysis` writes
  `table_main.csv`, `table_by_corpus.csv`, `deltas.csv`, `winners.csv` and
  `kappa.csv`, and `--profile` writes `corpus_profile.csv`. The four
  validity gates of `results/disk/PREREGISTERED.md` are stored in
  `provenance.json` under `extra.checks`; all four passed.
- **E2** (`repro/gpu_disk.py`, also `python -m repro.gpu_disk`): each trace's
  first canonical requests are split 60/20/20 chronologically; models train
  on the first part, stop early and select thresholds on the second, and are
  scored once on the last. Writes `disk_learned_depth.csv`,
  `disk_depth_baselines.csv`, `disk_lstm.csv` and `gpu_env.json`. Artifacts
  go to `models/` with `.training.json` sidecars that bind their training
  range by SHA-256.
- **E3** (`repro/realio.py`): `python -m adaptive_prefetch.realio grid`
  runs every (backend, mode) of the pre-registered grid in a fresh child
  process, three repeats, shuffled order; `report` rebuilds
  `realio_summary.csv` from `realio.csv`. Per-run detail goes to
  `outputs/realio/runs.jsonl` (gitignored). `--grid auto` picks the `hdd`
  plan on a rotational device and the `ssd` subset otherwise; `--recalibrate`
  re-measures the device even when `device_<dev>.json` exists. The module's
  own `grid` command takes `--plan {full,hdd,ssd}` and `--max-load-wait`
  (how long to wait for a quiet host before stopping).
- **E4** (`repro/web.py`): `python -m webprefetch prepare`, `train
  --novel-ranker`, `benchmark` (logistic, relabelled logistic and three
  transition scorers, selection per capacity on validation), `freshness`,
  `decompose`, `parity`, `learning-curve`.
- **E5** (`repro/gpu.py`, web half): `python -m webprefetch.seqmodel
  experiment` trains the GRU for 5 seeds under two protocols and writes
  `seq_metrics.csv`; artifacts are `models/web/seq/gru_<protocol>_s<seed>/`
  (`weights.npz`, `vocab.npy`, `config.json`; no pickle), for example
  `gru_validation_s0` and `gru_final_s0`. The decomposition is then re-run
  with the `sequence_model` lever, using `--seq-model
  models/web/seq/gru_final_s0`.
- **E6** (`repro/live.py`): `python -m webprefetch.live run` builds the plan
  and its simulator reference (`plan.json`, `sim.json`), the object store,
  starts the origin, and runs the passes one at a time. `--selection
  results/web/comparisons.csv` resolves `validated_*` policy names to what
  E4 selected on validation; there is no built-in fallback. It exits 1 on a
  parity mismatch and 2 when the planned request rate is refused.
- **paper** (`repro/paper.py`): `paper/numbers.py`,
  `paper/tables/make_tables.py` and `paper/figures/make_figures.py` (sharing
  `paper/genlib.py`, loaded by path), then `paper/check.py --strict`. With
  `--fixtures` it reads the synthetic `tests/fixtures/results/` tree and
  writes `outputs/paper-fixture/`, never `paper/`.

## 5. Rules every run follows

- **Quiet host.** E3 and E6 wait while the 1-minute load average is above 32
  and record the load of every run; never run them concurrently with each
  other, with E1, or with anything heavy.
- **Devices (lead decision L2).** Real-device runs use only the SATA SSD (the
  repository disk) and the HDD `sdb` under
  `/mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch/`. Nothing
  is ever written to `sdd` or `sde`.
- **Drop before every run.** The backing file (E3) or the object store (E6)
  is dropped from the page cache with `POSIX_FADV_DONTNEED` before every run
  or pass, and the HDD's own cache is displaced by reading an unused region
  with `O_DIRECT`.
- **Bounded runs never write `results/`.** `--quick`, smoke and test runs go
  to `outputs/`. A results file is regenerated only by its full command.
- **CPU.** The host has 128 cores shared with other users: cap workers at 64
  unless the work package says otherwise.

## 6. The paper and the slides

```bash
PYTHONPATH=src .venv/bin/python reproduce.py paper   # under 1 min
python3 paper/check.py --final                       # submission gate
python3 tools/check_doc_numbers.py                   # also lints presentation/slides.tex
```

Then compile on Overleaf: `paper/README.md` for the paper,
`presentation/README.md` for the Beamer deck. Both read
`paper/numbers.tex`, `paper/figures/*.pdf` and `paper/tables/*.tex`, so the
paper step must have run first.

## 7. Legacy disk entry points

`main.py` (EDA, classifier selection, domain shift, the policy benchmark and
the drift report on the bundled samples and synthetic traces) and
`sweep_modes.py` (streamed MSRC-trace-003 sweep) still run and are tested,
but write only to `outputs/`. Their numbers are not results of this paper.

```bash
python main.py --quick            # skips LSTM training and the regime sweep; MSRC traces capped at 5k requests
python main.py                    # full run; trains the LSTM (20% held out for validation)
python sweep_modes.py --per-trace 50000   # needs data/MSRC-trace-003
```

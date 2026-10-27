# Prefetching from disk blocks to product pages: two studies

Ashvik Mishra and Shourya Saraf, School of Computer Science and Engineering,
Vellore Institute of Technology. Operating Systems course project and the
source of one IEEE conference paper (`paper/`).

Prefetching guesses what will be read next and fetches it before it is asked
for. This repository asks the same question in two places an operating
system meets it, and answers it with simulation **and** with real I/O:

- **Disk study** (`src/adaptive_prefetch`): block read-ahead for a page cache,
  replayed causally over 56 public block traces from four corpora (MSR
  Cambridge, FIU, Alibaba, Tencent), then re-run on a real HDD and SSD with
  the kernel page cache, `O_DIRECT` and `posix_fadvise`.
- **Web study** (`src/webprefetch`): product-page prefetching into a browser
  cache, replayed over the RetailRocket clickstream, then re-run as real HTTP
  fetches with exact parity to the simulator.

Both studies use one cache abstraction, one objective
`U(c) = (hits - c * issued) / requests`, one statistics module and one
results contract (`src/prefetch_common`). The question each tests, fixed in
advance in `results/*/PREREGISTERED.md`, is whether a *learned predictor*
matters as much as *how much* to prefetch, *feedback* on observed precision,
and (on the web) *freshness* of the statistics.

## What we found

Every number below is read from a file under `results/` named next to it,
and `tools/check_doc_numbers.py` verifies that. Simulated numbers are cache
statistics; latency is measured only on the real systems.

**Disk, simulated** (C1: first 1,000,000 canonical requests of each of 56
traces; block hit ratio; paired bootstrap over traces;
`results/disk/table_main.csv`).

<!-- source: results/disk/table_main.csv config=C1 merge=auto latency=assumed corpus=all capacity=2048 -->
| Policy, 1 MiB cache | Mean hit ratio | vs `none` (pp) | Precision | Device I/Os vs `none` | Blocks moved vs `none` |
|---|---:|---:|---:|---:|---:|
| `none` | 14.85% | | | | |
| `sequential` (2-block read-ahead) | 16.71% | +1.86 | 49.00% | 1.055x | 1.042x |
| `adaptive` (trained classifier router) | 15.77% | +0.92 | 59.16% | 1.078x | 1.008x |
| `guard` (feedback gate) | 16.26% | +1.40 | 51.47% | 1.026x | 1.010x |
| `deep@8` (fixed depth 8) | 21.03% | +6.17 | 47.76% | 1.066x | 1.169x |
| `deep@256` | 50.41% | +35.56 | 19.28% | 1.309x | 7.015x |
| `linux_ra@256` (Linux read-ahead policy) | 38.62% | +23.77 | 76.98% | 0.740x | 1.086x |
| `belady_min` (demand paging only; not a prefetch bound) | 22.79% | | | | |
<!-- source: none -->

- **The trained classifier is worse than plain read-ahead.**
  <!-- source: results/disk/deltas.csv config=C1 corpus=all latency=assumed metric=hit_ratio policy=adaptive baseline=sequential -->
  `adaptive` minus `sequential` is -0.94 pp [-1.31, -0.58] at 1 MiB and
  -1.21 pp [-1.52, -0.92] at 64 MiB; the cluster intervals over corpora also
  exclude 0.
  <!-- source: none -->
- **Depth buys hit ratio with device work.** Deeper fixed read-ahead raises
  the hit ratio steadily, but `deep@8` issues *more* device I/Os than `none`
  at 1 MiB and `deep@256` moves seven times the blocks. Only the Linux
  read-ahead window (`linux_ra`) raises hit ratio *and* cuts device I/Os
  (`docs/DECISIONS.md` D18, D20).
- **The predictor is not the smallest lever on disk.** It is smaller than
  depth and window growth, but at 64 MiB the feedback gate's effect is
  smaller still (D20). The learned depth model (`learned_depth`, GPU, 5
  seeds) did worse than a validation-chosen fixed depth on most of 21 traces
  (D19, `results/gpu/disk_learned_depth.csv`).

**Disk, real device** (`results/realio/`; HDD `sdb` is the headline device,
20,000-request prefixes of 5 traces, 3 repeats, Backend B parity held on
every run): in **none** of the 16 measured cells was the mode with the
highest simulated hit ratio the fastest, and doing nothing (`none`) was the
fastest in 6. Simulated hit ratio did not rank policies by measured latency
in any HDD cell (D21).

**Web, simulated** (test split, per-session cache of 8 objects, 95% paired
session-bootstrap intervals as pre-registered in `results/web/PREREGISTERED.md`;
`results/web/decomposition.csv`):

<!-- source: results/web/decomposition.csv metric=hit_ratio -->
| Lever (hit ratio) | Difference [interval] |
|---|---|
| freshness: causal online transition counts vs the split-0 logistic ranker | +1.52 pp [+1.49, +1.56] |
| depth: best fixed depth at no cost vs depth 1 | +0.80 pp [+0.77, +0.83] |
| ranker: transition counting vs the trained logistic ranker | +0.07 pp [+0.06, +0.09] |
| GRU sequence model vs online counts | -0.79 pp [-0.83, -0.76] |
| feedback gate vs the validated adaptive controller | -0.11 pp [-0.13, -0.10] |
<!-- source: none -->

On the web the learned ranker's effect lies below both the depth and the
freshness levers, but the GPU sequence model's overlaps depth, so by the
pre-registered rule 2 (which takes both predictors) the predictor is not called
the smallest lever; the sequence model is worse than counting, and the validated adaptive controller beats the best
fixed depth at capacities 8 and 32 at an assumed cost of 0.01 per prefetch
(`results/web/comparisons.csv`; `docs/web/WEB_DECISIONS.md` W1-W3).

**Web, real HTTP** (`results/live/`): 42 replay passes, all valid, with 0
sessions whose live counters differ from the simulator's. Over loopback with
an emulated network (RTT 50 ms, 10 Mbps, assumed object sizes) the online-index
controller saved about 8 ms per view at 2.20x the bytes, where fixed depth 4
saved about the same at 6.71x (W4).

## Install

Python 3.10 or newer. The disk core needs only the standard library.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[all]"        # or pick extras: traces, web, gpu, paper
export PYTHONPATH=src                     # if you do not install
```

On this machine `.venv/` already has every extra (numpy, pandas,
scikit-learn, joblib, zstandard, torch with CUDA, matplotlib, scipy) and the
package is not installed, so every command below runs with `PYTHONPATH=src`.
No LaTeX is installed: the paper and the slides compile on Overleaf.

The data is not in git. Put the 56 `*.oracleGeneral.zst` block traces under
`data/block-traces/<corpus>/` and RetailRocket's `events.csv` (Kaggle,
version 2; its SHA-256 is checked) at `data/raw/events.csv`. Sources and
licences are in `NOTICE.md`.

## Reproduce

One command per study; each prints its runtime estimate first and records a
`provenance.json` (command, commit, host load, input and output hashes).
`--quick` runs are smoke tests that write to `outputs/`, never to `results/`.
Measured wall times of the committed runs are in `docs/REPRODUCE.md`.

```bash
PY=.venv/bin/python
PYTHONPATH=src $PY -m unittest discover -s tests   # the test suite (about 90 s)
$PY reproduce.py disk --quick              # minutes: tests, main.py --quick, 3-trace sweep
$PY reproduce.py disk --workers 64         # E1: multi-corpus sweep and analysis (about 70 min here)
$PY reproduce.py disk --gpu                # E2: learned_depth and LSTM on the GPU (about 80 min)
$PY reproduce.py realio --device DIR       # E0+E3: calibrate and run the real-device grid
$PY reproduce.py web                       # E4: RetailRocket benchmark (about 22 min)
$PY reproduce.py web --gpu                 # E5: GRU sequence model (about 2 h on a shared GPU)
$PY reproduce.py live --store-dir DIR      # E6: real HTTP replay (about 3 h)
$PY reproduce.py paper                     # figures, tables, numbers.tex (under a minute)
python3 paper/check.py --final             # paper lint
python3 tools/check_doc_numbers.py         # docs quote only numbers in results/
```

E3 and E6 need Linux, a quiet host (they wait while the load average is
above 32) and a directory on the device under test; they never write to any
disk except the SSD and the HDD `sdb`.

## Repository layout

| Path | What |
|---|---|
| `src/adaptive_prefetch/` | disk study: trace loaders, causal simulator (14 modes), multi-corpus sweep and analysis, learned depth, LSTM, `realio/` real-device testbed |
| `src/webprefetch/` | web study: data preparation, scorers, browser-cache core, benchmark, GRU, demo, `live/` HTTP replay |
| `src/prefetch_common/` | `U(c)`, bootstraps, provenance, results schemas |
| `reproduce.py`, `repro/` | one reproduce module per study |
| `results/` | **tracked** aggregates the paper reads, one directory per study, each with `PREREGISTERED.md` and `provenance.json`; `results/pivot_original/` is the web branch's original run, verbatim |
| `paper/` | IEEEtran paper, generators for `numbers.tex`, tables and figures, `check.py` |
| `presentation/` | Beamer deck reusing the paper's numbers and figures |
| `docs/` | architecture, metrics, reproduction, decision logs, presentation outline |
| `tests/` | the unit suite and synthetic results fixtures |
| `tools/` | `check_doc_numbers.py` |
| `main.py`, `sweep_modes.py` | the disk study's original entry points (see below) |
| `demo/`, `audit/` | web demo page; the web branch's audit log |
| `data/samples/` | small committed samples; the large data directories are gitignored |
| `outputs/`, `models/` | gitignored scratch output and trained artifacts (`models/msr_sample_gnb.json` is tracked) |

## Documentation

| Document | Read it for |
|---|---|
| [docs/architecture.md](docs/architecture.md) | how every component works, module by module, and the data flow |
| [docs/METRICS.md](docs/METRICS.md) | the definition and limits of every metric |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | every command, runtime, requirement and output |
| [docs/DECISIONS.md](docs/DECISIONS.md) | disk decision log: D1-D17 (history on `main`), D18-D21 (corrections from the multi-corpus and real-device results) |
| [docs/web/WEB_DECISIONS.md](docs/web/WEB_DECISIONS.md) | web decision log W0-W4 |
| [docs/PRESENTATION.md](docs/PRESENTATION.md) | slide-by-slide outline, speaker notes and viva questions |
| [AGENTS.md](AGENTS.md) | the invariants and claim rules for anyone (or any agent) changing the code |
| [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) | the work plan this branch was built to |
| [paper/README.md](paper/README.md), [presentation/README.md](presentation/README.md) | building the paper and the deck on Overleaf |

## The disk study's original entry points

`main.py` is the end-to-end run of the original disk study: EDA,
classifier selection, domain shift, the policy benchmark against a
perfect-predictor reference, and the drift report, on the bundled samples
and synthetic traces. It writes raw rows to `outputs/` and prints
conclusions. `sweep_modes.py` sweeps the MSRC-trace-003 `.revised`
collection, which is not on this machine. Their figures are not results of
the paper.

```bash
python main.py --quick                   # skips LSTM training and the regime sweep
python main.py                           # full run
python main.py --cost-model assumed      # the 5/100/50 us model instead of measured service times
PYTHONPATH=src python -m adaptive_prefetch demo            # synthetic classifier demo
PYTHONPATH=src python -m adaptive_prefetch benchmark       # every mode on synthetic traces
PYTHONPATH=src python -m adaptive_prefetch replay data/samples/msr-cambridge1-sample.csv --format msr
PYTHONPATH=src python -m adaptive_prefetch evaluate        # classifier x regime accuracy (synthetic)
PYTHONPATH=src python -m webprefetch serve                 # local web demo (needs prepared data and a model)
```

`main.py`'s modelled-cost column is degenerate at the default
`--prefetch-multiple 1.0` (no prefetcher can reach the break-even precision);
read its hit ratio, precision and wasted I/O instead.

## Limits, stated once

- Simulated hit ratios are cache statistics. The oracle traces carry no
  read/write flag, so writes replay as reads; they carry no service times.
- Classifier accuracy exists only on synthetic windows; the real traces have
  no workload labels.
- Real-device results are one HDD and one shared SSD, five trace prefixes,
  one think model per cell, on a shared host. They are per trace and device,
  not properties of HDDs or SSDs.
- Web costs, object sizes, RTT and bandwidth are assumed; one retailer; a
  per-session cache.
- No number in this README, the docs or the paper is typed by hand; each is
  regenerated from `results/`.

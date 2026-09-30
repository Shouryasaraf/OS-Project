# ML-Based Adaptive Disk I/O Prefetching

A user-space, simulation-first prototype for **Machine Learning-Based Adaptive
Disk I/O Prefetching Using Workload Pattern Classification**. The central path
is: trace requests → window features → online-capable classifier → pattern-based
prefetch policy → LRU cache simulation. It does **not** modify the operating
system or perform physical disk reads.

## Repository layout

| Location | Contents |
| --- | --- |
| `src/adaptive_prefetch/` | Trace loading, features, models, policies, replay, benchmark, and CLI |
| `src/adaptive_prefetch/guard.py` | Feedback-gated policies: observed prefetch precision vs break-even |
| `tests/` | Automated checks (131) |
| `data/samples/` | Small demonstration traces and format notes |
| `data/MSRC-trace-003/final-trace/` | The 32-file, 11.86 GB collection (gitignored) |
| `docs/` | Pipeline, architecture, decision log, classifier analysis, Review 2 guide, results |
| `docs/reference/` | Original lab brief |
| `presentation/` | Review 2 slide decks (both currently stale) |
| `outputs/` | Generated results (gitignored) |

Run commands below from the repository root. Keep new trace datasets under
`data/` and generated benchmark outputs under `outputs/` (create that ignored
folder before writing results there).

## What is implemented

- Reproducible synthetic sequential, strided, random, and mixed I/O traces.
- Validated CSV trace import and export.
- Window-level, size-aware sequentiality and stride features. The jump and
  short-run features are measured *relative to the request size*, so the same
  access pattern classifies identically on devices with different block sizes.
- A supervised Gaussian Naive Bayes model with incremental updates, plus an
  optional shrinkage-regularized online QDA (`--classifier qda`) for
  workloads whose classes overlap in correlated feature space.
- Class-to-policy routing, LRU cache replay, and **twelve comparison modes**
  in four families:
  - *fixed* -- `none`, `sequential`, `deep` (depth-8 read-ahead), `strided`
  - *online per-request* -- `stride`, `markov`, optional offline `lstm`
  - *classifier-routed* -- `adaptive`
  - *evidence-routed* -- `adaptive_evidence` (no trained model)
  - *feedback-gated* -- `guard`, `depth_adaptive`, `correlate`, which set
    read-ahead depth from the **realised** precision of their own prefetches.
    Causal, needs no model and no future knowledge.
- Configurable decision cadence (`policy_interval`) so the gate can react
  inside a classifier window, and a configurable read-ahead depth for `deep`.
- Benchmark tables with oracle re-access recall, configurable modelled latency
  (calibrated from each trace's own recorded service times), inference timing,
  synthetic drift comparison, and optional pseudo-label tests.
- A memory-bounded streaming sweep (`sweep_modes.py`) over the whole MSRC
  collection, bit-identical to the in-memory replay across chunk boundaries.
- Exploratory data analysis: trace profiling, locality and reuse ceilings,
  cluster structure, and domain-shift measurement against the synthetic
  training region.
- Held-out synthetic classification test, confusion matrix, workload-transition
  demonstration, cache hit ratio, prefetch precision, and unused-prefetch count.
- Standard-library unit tests. No packages need to be downloaded to run from
  source. The core is stdlib-only by design; `torch` is installed here, so the
  LSTM path is live, but `models/lstm_delta.pt` is gitignored and must be
  trained before that row reports anything other than `skipped`.

## Run

From the repository root. The **complete research run** — no prompts. It
profiles every dataset, evaluates the classifier against alternative models,
measures domain shift, replays all twelve policies against a perfect-predictor
bound, and reports drift adaptation. The console shows conclusions; every raw
row goes to `outputs/`:

```powershell
python main.py                          # full run, ~6 min (11 datasets)
python main.py --quick                  # skip LSTM training + regime sweep
python main.py --interactive            # guided dataset/LSTM picker
python main.py --cost-model assumed     # compare vs the 5/100/50 us model
python main.py --prefetch-multiple 0.5   # model overlapped prefetch
```

The default run calibrates the cost model from the traces' own recorded device
service times (`.revised` columns 6-8) and prints a `COST MODEL` section with
the break-even precision **before** any cost table.

**At the default `--prefetch-multiple 1.0` the modelled-cost column is
degenerate.** A speculative read does the same device work as the demand read
it replaces and saves strictly less (the hit still costs a cache lookup), so
break-even precision is 1.0101 and no prefetcher can reach it. `none` wins that
column by arithmetic, not by result. Values below 1.0 model *overlapped*
prefetch -- identical device work, requester not blocked -- which is an
assumption these traces cannot support, since they record demand-read service
time only. Use `--prefetch-multiple` to explore it. See `docs/DECISIONS.md` D11.

**Read hit ratio, precision and wasted I/O; treat modelled cost as degenerate
unless the break-even is printed.**

### What wins across the whole collection

Full MSRC sweep, 32 traces x 250,000 requests, 12 modes, 384 rows. Unweighted
mean hit ratio per trace, so one huge trace cannot dominate:

| mode | mean hit | vs `none` |
| --- | ---: | ---: |
| `deep` (fixed depth-8 read-ahead) | 49.87% | **+4.94 pp** |
| `sequential` (depth-2 read-ahead) | 45.45% | +0.52 pp |
| ... eight other modes | 44.96-45.13% | +0.02 to +0.20 pp |
| `none` | 44.93% | - |
| `adaptive` (trained classifier) | 44.89% | **-0.04 pp** |

**All eleven non-deep modes fall within 0.55 points of each other, and the
trained classifier is the worst of them.** `deep` sits 4.42 points above the
best of them -- eight times the entire spread of policy choice. Per-trace
winners: `deep` 29, `none` 2, `adaptive` 1.

Choosing *which* prefetching heuristic to apply is worth at most half a
point. Choosing how *deep* to prefetch is worth five. Every earlier
conclusion in this project was measured correctly but against a degenerate
baseline that prefetched 2 blocks.

The caveat is that `deep` issues 11.2 million prefetches across the sweep, of
which 5.86 million are never read, and its modelled us/read is 4952 against
`none`'s 4105. The hit-ratio win is real; the claim that it is worth having
is not supported. Report both. See `docs/DECISIONS.md` D17.

### What actually wins on real traces

Measured over 8 real traces capped at 40k requests (`docs/DECISIONS.md` D14):

* **`deep`** (fixed depth-8 read-ahead) has the highest hit ratio on 6 of 8,
  and is by far the most wasteful. Read-ahead *depth* is worth **+4.10
  points** of mean hit ratio on its own -- an order of magnitude more than the
  entire policy-selection question, because every earlier mode prefetched at
  most 2 blocks. (Depth 16 scores +4.66 on the mean but issues 4x the
  prefetches and loses 7.2 points on `rsrch_2`, which is why 8 is the
  default and not the argmax.)
* **The feedback gate** (`guard`, `depth_adaptive`, `correlate`) wins on
  exactly one trace: `web_3`, where read-ahead wrecks the cache. It takes
  +18.99 points of hit ratio over `sequential` and cuts wasted I/O 86.9%. It
  does not beat `none` there (50.61%). On the other seven traces it *loses* to
  `sequential` by 0.00-3.00 points, in exchange for 12-19% of its wasted I/O
  on five of them.
* **`none`** still wins on `web_3` and `msrc_hm_1`.
* Note that `depth_adaptive` is byte-identical to `guard` at the default cost
  model, because break-even 1.0101 exceeds any achievable precision. It
  diverges only under `--prefetch-multiple 0.5`. See `DECISIONS.md` D14.1.

This is a targeted result, not a general win for adaptive prefetching. The
trained `adaptive` classifier remains behind fixed read-ahead; see
`docs/DECISIONS.md` D9-D10 for why, and D12-D14 for the inverted target and
the depth finding.

| Output file | Contents |
| --- | --- |
| `outputs/results.csv` | every benchmark row (all datasets × all modes) |
| `outputs/results.md` | narrative report with the full 12-mode table and interpretation |
| `outputs/eda.md` | per-trace request stream, locality, feature distributions, cluster structure |
| `outputs/classifier_eval.csv` | classifier × regime accuracy, spread, predict and train cost |
| `outputs/domain_shift.csv` | per-feature shift and coverage vs the synthetic training region |
| `outputs/drift.csv` | frozen vs online switch lag per phase |
| `outputs/label_support.md` | which classes the labelled population actually covers |

For specific tasks, call the modules directly (set `PYTHONPATH=src` first,
or install with `pip install -e .`):

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v          # unit tests
python -m adaptive_prefetch demo                 # classifier accuracy + confusion matrix
python -m adaptive_prefetch benchmark            # 12-mode comparison matrix
python -m adaptive_prefetch replay path\to\trace.csv
python -m adaptive_prefetch replay path\to\trace.csv --format alibaba
python -m adaptive_prefetch replay data/samples/msr-cambridge1-sample.csv --model-path models/msr_sample_gnb.json
python -m adaptive_prefetch train-msr-sample     # weakly adapt classifier on the MSR sample
python -m adaptive_prefetch normalize --input trace.csv --format alibaba --output normalized.csv
python -m adaptive_prefetch export-demo demo.csv # export a synthetic transition trace
```

Full-collection sweep (every MSRC trace, every mode, streamed so memory stays
flat regardless of file size):

```powershell
python sweep_modes.py                    # 250k requests x 32 traces, all modes
python sweep_modes.py --full             # every request in every file (hours)
python sweep_modes.py --per-trace 50000  # cheaper sample
python sweep_modes.py --modes none sequential adaptive adaptive_evidence
```

Writes `outputs/msrc_sweep.csv` (per-trace rows) and
`outputs/msrc_sweep.md` (winner table plus the full per-trace matrix).
Prefetch recall is omitted on streamed runs because it needs whole-stream
future knowledge; every other metric is exact.

Research and analysis commands:

```powershell
python -m adaptive_prefetch evaluate            # classifier x regime accuracy + cost table
python -m adaptive_prefetch evaluate --classifier qda --regime transition_heavy
python -m adaptive_prefetch eda data\samples\msr-cambridge1-sample.csv --format msr
python -m adaptive_prefetch shift data\samples\msr-cambridge1-sample.csv --format msr
python -m adaptive_prefetch benchmark --classifier qda
python -m adaptive_prefetch replay trace.csv --smoothing 3   # consecutive-run policy hysteresis
```

On macOS/Linux, prefix the same way
(`PYTHONPATH=src python -m adaptive_prefetch demo` or `python main.py`).

## Trace data format

The normalized CSV format has one request per row:

```csv
timestamp_ms,lba,size_blocks,operation,stream_id
0.25,1000,2,R,process_A
0.91,1002,1,R,process_A
1.20,2000,1,W,process_B
```

`timestamp_ms` is a non-negative millisecond timestamp in ascending order.
`lba` is a **block index**, not a byte offset. `size_blocks` is the positive
number of contiguous blocks in that request. `operation` is `R` or `W`.
`stream_id` is optional and retained for future per-process/per-volume analysis;
the current model classifies the **aggregate** request stream. A real dataset
must be converted to these units before replay. The repository includes a
1,000-request MSR-format sample; it does not bundle complete MSR Cambridge or
SNIA IOTTA trace collections. The sample's provenance has not been
independently verified here.

The included `data/samples/msr-cambridge1-sample.csv` uses the original MSR headers
`Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime`. The loader
converts its byte offsets and sizes to 512-byte blocks, timestamps to elapsed
milliseconds, and `Read`/`Write` to `R`/`W` automatically.

`train-msr-sample` creates a small, reproducible JSON classifier artifact at
`models/msr_sample_gnb.json`. The four-class model is first trained on the
labelled synthetic generator. From the unlabelled MSR sample, it then learns
only from read-heavy windows with a conservative **random-like proxy label**.
This is weakly supervised adaptation, not four-class training from real ground
truth. The model can be loaded with `--model-path` for replay; replaying the
same sample is an in-sample demonstration, not an unbiased performance
evaluation.
See [the training record](docs/MSR_SAMPLE_TRAINING.md) for exact counts and
limitations.

`normalize` also supports two explicit IOTTA-related profiles: `alibaba`
(`device_id,opcode,offset,length,timestamp`; byte offsets/lengths and
microsecond timestamps) and `iotta8`
(`device,sector,size,op,offset,timestamp,lifetime,count`; sector indices/counts
and microsecond timestamps). The latter is the format proposed in the Stage 2
plan, **not** a universal SNIA IOTTA format. Check the particular trace's
metadata and units before selecting either profile. Headerless `iotta8` needs
`--format iotta8`. No full public IOTTA trace is bundled or benchmarked.

For the optional offline LSTM baseline, install the extra with
`python -m pip install -e ".[lstm]"` in a suitable Python environment, then
run `python -m adaptive_prefetch train-lstm --save models/lstm_delta.pt` and
`python -m adaptive_prefetch benchmark --lstm-model models/lstm_delta.pt`.
The core project and all other modes work without PyTorch; an unavailable
LSTM is shown as *skipped*, not silently replaced by another predictor. The
delta model must be trained on the distribution it will be scored against:
`main.py` trains it on the traces in the current run rather than mixing a
real-trace artifact into a synthetic comparison table.

## How the demonstration avoids a timing mistake

A window is classified only after all its requests have occurred. Its policy
can affect the **next** requests, never earlier ones. The first window uses no
prefetch. The model is trained on generated examples from one random seed and
tested on independently generated examples from another seed. Those results
describe the synthetic generator, not accuracy on production workloads.

Incremental updates require a trustworthy label. The demonstration provides
labels because it generated the workload. On real unlabelled CSV traces,
classifier updates are disabled; successful prefetches are not treated as
proof that a workload label was correct.

## Interpretation and limits

Hit ratio is read-block cache hits divided by requested read blocks. Prefetch
precision is prefetched blocks subsequently read divided by prefetches issued.
Oracle re-access recall divides useful prefetches by useful prefetches plus
read misses that are accessed again later; future-read information is used
only after replay for measurement, never for candidate generation. It is a
specialized diagnostic, not workload-classification recall.
`unused` counts all prefetches not used during the replay, including those still
resident at the end. Writes use a simplified write-allocate cache model but
never trigger prefetch. Every policy receives the same request stream, cache
capacity, first-window observation period, and LRU rules. The latency column
is a *configured cost model* (defaults: 5 µs hit, 100 µs demand miss, 50 µs
prefetch), including prefetch cost; its speedup is relative to no prefetch
under those assumptions. The simulation still omits queueing, prefetch
completion time, and bandwidth, and reports no measured device speedup.

`--online-real` adds an opt-in confidence-gated self-training row. Its model
updates from its own predictions, not verified labels, so it cannot establish
real-trace classification accuracy. The default real-trace adaptive model is
frozen. Synthetic drift results separately compare frozen with genuinely
labelled incremental updates; they do not automatically prove faster
adaptation. The LSTM predicts address deltas, not workload classes, so its
cache outcomes and compute cost can be compared, but classification accuracy
cannot be compared directly.

See [Review 2 guide](docs/REVIEW2.md) for architecture, module completion,
demonstration steps, and discussion questions. See the
[architecture document](docs/architecture.md) for the module map, data flow,
and design principles. See
[docs/DECISIONS.md](docs/DECISIONS.md) for the decision log: what was tried,
what the measurement said, and why the default classifier is still Naive
Bayes despite QDA being available and better on some regimes. See
[docs/CLASSIFIER_EVAL.md](docs/CLASSIFIER_EVAL.md) for the detailed
classifier comparison, the LSTM rework, the bug list, and the measured limits
of what the available data can support. See the
[Stage 2 verification snapshot](docs/STAGE2_RESULTS.md) for actual results
and explicit gaps; rerun the benchmark before presenting any numbers. The
[pipeline guide](docs/PIPELINE.md) and [Stage 2 plan](docs/implementation-stage2.md)
are kept under `docs/`; sample traces and their format notes are under
`data/samples/`.

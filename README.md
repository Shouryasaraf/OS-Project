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
| `tests/` | Automated checks |
| `data/samples/` | Small demonstration traces and format notes |
| `docs/` | Pipeline, implementation plan, Review 2 guide, and results |
| `docs/reference/` | Original lab brief |
| `presentation/` | Review 2 slide deck |

Run commands below from the repository root. Keep new trace datasets under
`data/` and generated benchmark outputs under `outputs/` (create that ignored
folder before writing results there).

## What is implemented

- Reproducible synthetic sequential, strided, random, and mixed I/O traces.
- Validated CSV trace import and export.
- Window-level, size-aware sequentiality and stride features.
- A supervised Gaussian Naive Bayes model with incremental updates.
- Class-to-policy routing, LRU cache replay, and seven comparison modes:
  no prefetch, fixed sequential, window-based strided, classic stride,
  delta Markov, optional offline LSTM, and adaptive classification.
- Benchmark tables with oracle re-access recall, configurable modelled latency,
  inference timing, synthetic drift comparison, and optional pseudo-label tests.
- Held-out synthetic classification test, confusion matrix, workload-transition
  demonstration, cache hit ratio, prefetch precision, and unused-prefetch count.
- Standard-library unit tests. No packages need to be downloaded to run from source.

## Run on Windows

From the repository root in PowerShell:

```powershell
.\run_review2.ps1
.\run_review2.ps1 -Test
.\run_review2.ps1 -Benchmark
```

The launcher uses a Python command on PATH or the bundled Codex Python on
this machine. To replay a normalized CSV trace:

```powershell
.\run_review2.ps1 -Trace path\to\trace.csv
.\run_review2.ps1 -Trace path\to\trace.csv -Format alibaba
```

To use Python directly or export a demonstration trace:

```powershell
$env:PYTHONPATH='src'
python -m adaptive_prefetch export-demo demo.csv
python -m adaptive_prefetch replay demo.csv
python -m adaptive_prefetch replay data/samples/msr-cambridge1-sample.csv
python -m adaptive_prefetch benchmark --datasets synthetic msr --output-csv results.csv
python -m adaptive_prefetch normalize --input trace.csv --format alibaba --output normalized.csv
```

On macOS/Linux, use `PYTHONPATH=src python -m adaptive_prefetch demo`.

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
LSTM is shown as *skipped*, not silently replaced by another predictor.

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
[Stage 2 verification snapshot](docs/STAGE2_RESULTS.md) for actual results
and explicit gaps; rerun the benchmark before presenting any numbers. The
[pipeline guide](docs/PIPELINE.md) and [Stage 2 plan](docs/implementation-stage2.md)
are kept under `docs/`; sample traces and their format notes are under
`data/samples/`.

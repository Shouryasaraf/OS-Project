# ML-Based Adaptive Disk I/O Prefetching

A user-space, simulation-first prototype for **Machine Learning-Based Adaptive
Disk I/O Prefetching Using Workload Pattern Classification**. The central path
is: trace requests → window features → online-capable classifier → pattern-based
prefetch policy → LRU cache simulation. It does **not** modify the operating
system or perform physical disk reads.

## What is implemented

- Reproducible synthetic sequential, strided, random, and mixed I/O traces.
- Validated CSV trace import and export.
- Window-level, size-aware sequentiality and stride features.
- A supervised Gaussian Naive Bayes model with incremental updates.
- Class-to-policy routing, LRU cache replay, and no-prefetch/fixed-policy baselines.
- Held-out synthetic classification test, confusion matrix, workload-transition
  demonstration, cache hit ratio, prefetch precision, and unused-prefetch count.
- Standard-library unit tests. No packages need to be downloaded to run from source.

## Run on Windows

From the repository root in PowerShell:

```powershell
.\run_review2.ps1
.\run_review2.ps1 -Test
```

The launcher uses a Python command on PATH or the bundled Codex Python on
this machine. To replay a normalized CSV trace:

```powershell
.\run_review2.ps1 -Trace path\to\trace.csv
```

To use Python directly or export a demonstration trace:

```powershell
$env:PYTHONPATH='src'
python -m adaptive_prefetch export-demo demo.csv
python -m adaptive_prefetch replay demo.csv
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
must be converted to these units before replay. The repository does not bundle
MSR Cambridge or SNIA IOTTA production traces.

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
`unused` counts all prefetches not used during the replay, including those still
resident at the end. Writes use a simplified write-allocate cache model but
never trigger prefetch. Every policy receives the same request stream, cache
capacity, and LRU rules. The simulation omits queueing, device latency,
prefetch completion time, and bandwidth, so it does not report real access
latency or speedup.

See [Review 2 guide](docs/REVIEW2.md) for architecture, module completion,
demonstration steps, and discussion questions.

# ML pipeline and Stage 2 comparison

This is a user-space **simulation** of adaptive disk-I/O prefetching, not an
OS modification or physical disk benchmark.

```text
CSV / synthetic requests -> normalized Request records -> completed windows
                         -> 12 features -> incremental Gaussian NB classifier
                         -> class-selected candidates -> shared LRU cache
Classic stride / Markov / optional offline LSTM --^ (alternative candidates)
Feedback gate: observed prefetch precision -> read-ahead depth --^ (self-tuning)
```

`trace.py` loads normalized CSV, MSR-format CSV, an Alibaba EBS schema, and
the explicitly selected eight-column profile from the Stage 2 plan. The two
IOTTA-related profiles require checking source metadata for field units; SNIA
IOTTA itself has no single universal CSV layout. The current classifier uses
the aggregate request stream, not separate per-device models.

`features.py` extracts 12 features per complete window: 8 stream features
(size-aware sequentiality, dominant stride, jump and short-run ratios,
uniqueness, timing variation, burstiness, read ratio) plus 4 contextual ones
(`reuse_ratio`, `reuse_distance_log`, `contiguous_delta`, `stride_persist`)
carried across windows by `WindowContext`. `model.py` trains an incremental
Gaussian Naive Bayes classifier on labelled synthetic windows. Classes are
sequential, strided, random, and mixed. `simulator.py` maps these classes to
two-block read-ahead, a learned window stride, no prefetch, and one-block
read-ahead, respectively. The first window observes requests without
prefetching; its prediction only affects later requests.

`guard.py` adds a family of modes that use **no classifier at all**. They
compare the *realised* precision of their own prefetches — directly
observable from `LRUCache.useful_prefetches`, so causal and needing no future
knowledge — against `LatencyModel.break_even_precision()`, and set read-ahead
depth from the margin. The decision is therefore to throttle, not to choose
a policy; see [DECISIONS.md](DECISIONS.md) D12 for why the target was
inverted from "which policy is best" to "will prefetching hurt".

## Twelve policy rows

| Mode | Source of candidates | Learning/availability |
| --- | --- | --- |
| `none` | No candidates | Control |
| `sequential` | Fixed two-block read-ahead | Fixed heuristic |
| `deep` | Fixed `DEFAULT_READ_AHEAD_DEPTH` (8) read-ahead | Fixed heuristic |
| `strided` | Window-derived stride, one block | Fixed heuristic |
| `stride` | Consecutive-delta detection, two blocks | Online per-request baseline |
| `markov` | Observed delta-transition probabilities, top two | Online per-request baseline |
| `lstm` | Offline LSTM next-delta prediction, top two | Optional PyTorch + saved artifact; otherwise skipped |
| `adaptive` | Classifier chooses a policy after each window | Frozen on unlabelled traces by default |
| `adaptive_evidence` | Window's own features route a policy | No model, thresholds calibrated on real feature distribution |
| `guard` | Depth 2 while observed precision clears break-even | Self-tuning, no model |
| `depth_adaptive` | Depth 0-8 scaled by the margin above break-even | Self-tuning, no model |
| `correlate` | `guard` plus a stride confirmed over a longer history | Self-tuning, no model |

All modes use the same input requests, LRU capacity, and first-window warmup.
The predictor objects in `baselines.py` and `lstm.py` expose
`next_candidates(request)`. Unlike the classifier, the LSTM predicts *deltas*,
not workload classes. Compare them using cache behavior and compute time, not
the four-class accuracy metric.

**Read-ahead depth matters more than policy choice.** Every mode that existed
before `deep` prefetched at most 2 blocks, which is why a perfect window
router could find only +0.4 points of headroom. Fixed read-ahead at depth 8
gives **+4.10 points of *mean* hit ratio** over that depth-2 baseline across
8 real traces, up to +11.05 on `src2_2` ([DECISIONS.md](DECISIONS.md) D13).
`deep` exists to make that lever visible in the shipped benchmark; the gate
has to beat it.

## What the benchmark measures

`benchmark.py` reports hit ratio, prefetch precision, oracle re-access recall,
unused prefetches, modelled mean access cost, speedup against `none` under
the same cost parameters, and measured predictor time per call. Its recall
denominator includes misses on blocks that will be read again later. This
future knowledge is for scoring only. It can look high even when the overall
hit ratio is low, so interpret it together with hit ratio and wasted work.

`LatencyModel.measured()` calibrates hit, demand-miss and prefetch cost
from the trace's own recorded device service times (means of 260-5491 µs),
falling back to the original assumed 5/100/50 µs model when unavailable. The
**hit** cost is still an approximation — 1% of mean service — because these
traces record device time only, never cache service time. Prefetches are
inserted immediately; queueing, bandwidth, asynchronous completion and device
scheduling are not modelled. The reported speedup is *modelled*.

**The modelled-cost column is degenerate at the default
`prefetch_multiple=1.0`.** Break-even precision is `1.0101`, so no prefetcher
can pay for itself and `none` wins that column by arithmetic rather than by
result. Only `--prefetch-multiple < 1.0` makes prefetching feasible, and that
models overlapped prefetch — an assumption these traces cannot support. Read
hit ratio, precision and wasted I/O; see [DECISIONS.md](DECISIONS.md) D11.

Held-out four-class accuracy and the confusion matrix use independent
synthetic seeds; they do not establish accuracy on real traces. The synthetic
drift report compares a frozen classifier with one updated using generator
labels after each prediction. `--online-real` is an opt-in self-training
experiment using high-confidence model predictions as pseudo-labels, with no
claim that those labels are correct. The default real-trace run makes no
classifier updates.

## Run

```powershell
python main.py                       # guided run: demo + benchmark + drift + outputs/
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m adaptive_prefetch benchmark
python -m adaptive_prefetch replay .\data\samples\msr-cambridge1-sample.csv
```

For custom datasets, normalization, latency parameters, CSV export, and the
optional LSTM commands, see [README.md](../README.md). The bundled MSR-format
sample has unverified provenance. No full IOTTA trace has been supplied, so
full-trace outcome claims remain open. The Review 2 slide deck predates this
Stage 2 benchmark; refresh its numbers from a current run before presenting.

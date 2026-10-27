# Disk pipeline and the policy modes

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
the explicitly selected eight-column profile from the Stage 2 plan, the
MSRC `.revised` format and libCacheSim `oracleGeneral` block traces. The two
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

## Fourteen policy modes

`benchmark.MODES` and `sweep.RANKING_MODES`, in this order (appending a mode
is safe; reordering breaks `test_benchmark_includes_skipped_optional_lstm`):

| Mode | Source of candidates | Learning/availability |
| --- | --- | --- |
| `none` | No candidates | Control |
| `sequential` | Fixed two-block read-ahead | Fixed heuristic |
| `strided` | Window-derived stride, one block | Fixed heuristic |
| `stride` | Consecutive-delta detection, two blocks | Online per-request baseline |
| `markov` | Observed delta-transition probabilities, top two | Online per-request baseline |
| `lstm` | Offline LSTM next-delta prediction, top two | Optional artifact; otherwise `skipped:` |
| `adaptive` | Classifier chooses a policy after each window | Frozen on unlabelled traces by default |
| `adaptive_evidence` | Window's own features route a policy | No model |
| `guard` | Read-ahead while observed precision clears break-even | Self-tuning, no model |
| `depth_adaptive` | Depth 0-8 scaled by the margin above break-even | Self-tuning, no model |
| `correlate` | `guard` plus a stride confirmed over a longer history | Self-tuning, no model |
| `deep` | Fixed read-ahead of `read_ahead_depth` blocks (`deep@N` in the sweeps) | Fixed heuristic |
| `linux_ra` | Emulation of the Linux on-demand read-ahead policy, window ceiling `linux_ra_max_blocks` (`linux_ra@N`) | Fixed policy, reads cache state through `CacheView` |
| `learned_depth` | Trained per-segment read-ahead depth (`depth_model.py`) | Optional artifact (`depth_model=`); otherwise `skipped:` |

All modes use the same input requests, LRU capacity, and first-window warmup.
The predictor objects in `baselines.py`, `lstm.py` and `depth_model.py`
expose `next_candidates(request)`; `linux_ra` is cache-aware and is driven
through `observe(request, missed_blocks, cache_view)`. Unlike the
classifier, the LSTM predicts *deltas*, not workload classes. Compare them
using cache behaviour, device work and compute time, not the four-class
accuracy metric.

**Read-ahead depth matters more than policy choice, and depth is not free.**
Every mode that existed before `deep` prefetched at most 2 blocks. On the
multi-corpus block traces the hit ratio keeps rising with depth far past 8,
but blocks moved and (for fixed depth) device I/Os rise with it; only the
growing `linux_ra` window cuts device I/Os. See
[DECISIONS.md](DECISIONS.md) D18 and D20 for the measured figures.

## What the benchmark measures

`benchmark.py` reports hit ratio, prefetch precision, oracle re-access recall,
unused prefetches, modelled mean access cost, speedup against `none` under
the same cost parameters, and measured predictor time per call. Its recall
denominator includes misses on blocks that will be read again later. This
future knowledge is for scoring only. It can look high even when the overall
hit ratio is low, so interpret it together with hit ratio and wasted work.

`LatencyModel.measured()` calibrates hit, demand-miss and prefetch cost
from the trace's own recorded device service times (MSRC `.revised` files),
falling back to the original assumed 5/100/50 µs model when unavailable.
Oracle block traces have no service times, so every multi-corpus row names
its latency model explicitly. Device work is counted separately
(`device_ios`, `blocks_moved`) and costed by `IOCostModel` from a measured
device calibration (D18). The
**hit** cost is still an approximation (`mean_service * 0.01` in
`LatencyModel.measured`) because these
traces record device time only, never cache service time. Prefetches are
inserted immediately; queueing, bandwidth, asynchronous completion and device
scheduling are not modelled. The reported speedup is *modelled*.

**The modelled-cost column is degenerate at the default
`prefetch_multiple=1.0`.** Break-even precision is above 1, so no prefetcher
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
optional LSTM commands, see [README.md](../README.md) and
[REPRODUCE.md](REPRODUCE.md). The bundled MSR-format sample has unverified
provenance.

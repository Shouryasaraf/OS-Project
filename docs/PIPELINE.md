# ML pipeline and Stage 2 comparison

This is a user-space **simulation** of adaptive disk-I/O prefetching, not an
OS modification or physical disk benchmark.

```text
CSV / synthetic requests -> normalized Request records -> completed windows
                         -> 8 features -> incremental Gaussian NB classifier
                         -> class-selected candidates -> shared LRU cache
Classic stride / Markov / optional offline LSTM --^ (alternative candidates)
```

`trace.py` loads normalized CSV, MSR-format CSV, an Alibaba EBS schema, and
the explicitly selected eight-column profile from the Stage 2 plan. The two
IOTTA-related profiles require checking source metadata for field units; SNIA
IOTTA itself has no single universal CSV layout. The current classifier uses
the aggregate request stream, not separate per-device models.

`features.py` extracts size-aware sequentiality, dominant stride, jump and
short-run ratios, uniqueness, timing variation, burstiness, and read ratio
from each complete window. `model.py` trains an incremental Gaussian Naive
Bayes classifier on labelled synthetic windows. Classes are sequential,
strided, random, and mixed. `simulator.py` maps these classes to two-block
read-ahead, a learned window stride, no prefetch, and one-block read-ahead,
respectively. The first window observes requests without prefetching; its
prediction only affects later requests.

## Seven policy rows

| Mode | Source of candidates | Learning/availability |
| --- | --- | --- |
| `none` | No candidates | Control |
| `sequential` | Fixed two-block read-ahead | Fixed heuristic |
| `strided` | Window-derived stride, one block | Fixed heuristic |
| `stride` | Consecutive-delta detection, two blocks | Online per-request baseline |
| `markov` | Observed delta-transition probabilities, top two | Online per-request baseline |
| `lstm` | Offline LSTM next-delta prediction, top two | Optional PyTorch + saved artifact; otherwise skipped |
| `adaptive` | Classifier chooses a policy after each window | Frozen on unlabelled traces by default |

All modes use the same input requests, LRU capacity, and first-window warmup.
The predictor objects in `baselines.py` and `lstm.py` expose
`next_candidates(request)`. Unlike the classifier, the LSTM predicts *deltas*,
not workload classes. Compare them using cache behavior and compute time, not
the four-class accuracy metric.

## What the benchmark measures

`benchmark.py` reports hit ratio, prefetch precision, oracle re-access recall,
unused prefetches, modelled mean access cost, speedup against `none` under
the same cost parameters, and measured predictor time per call. Its recall
denominator includes misses on blocks that will be read again later. This
future knowledge is for scoring only. It can look high even when the overall
hit ratio is low, so interpret it together with hit ratio and wasted work.

The default modelled costs are 5 µs/cache hit, 100 µs/demand miss, and
50 µs/issued prefetch. They are assumptions, not measurements. Prefetches are
otherwise inserted immediately; queueing, bandwidth, asynchronous completion,
and device scheduling are not modelled. The reported speedup is *modelled*.

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

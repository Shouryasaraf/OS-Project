# Web study: decisions and corrections

The web study's own decision log, the counterpart of `docs/DECISIONS.md` for
the disk study. W0 summarises the audit the pivot branch already did
(`audit/GRILL.md`, kept verbatim); W1-W4 are what changed on `master`, each
with the measurement that decided it. Every figure below comes from a file
under `results/` named next to it; `tools/check_doc_numbers.py` checks them.

What every web number is: a **simulated** per-session browser cache replayed
over RetailRocket product views, with equal object sizes, instant prefetch
completion and an **assumed** cost `c` per prefetch, unless it is from
`results/live/`, where waits are **measured** over real HTTP on loopback with
an **emulated** network and **assumed** object sizes.

## W0. The audit the pivot branch passed (GRILL), and the rules it left

`audit/GRILL.md` records 14 findings (G01-G14) against the pivot's first
snapshot and how each was fixed. They are the web study's standing rules on
`master`, and the tests that pin them moved with the code
(`tests/test_web_audit_regressions.py`, `test_web_workflow.py`):

- **Sessions never straddle a split** (G01). Sessions break after 30 minutes
  of inactivity *and* at each chronological split boundary; the regenerated
  views have zero cross-split sessions.
- **A replay prefix never truncates a session** (G02, P8): bounded replays
  take complete sessions only (`simulator.complete_views`).
- **Selection and evaluation use whole splits** (G03); a bounded
  `--validation-views/--test-views` run is a smoke run only.
- **One split, one role.** Split 0 builds the candidate index, split 1
  trains the ranker, split 2 (validation) is the only split any choice is
  made on, split 3 (test) is replayed once with every choice frozen.
- **The ranker is isolated** (G04): transition top-1 is a first-class
  baseline; context ablation and gradient boosting are diagnostics, never
  replacements.
- **Calibration is measured** (G05, P6): a reliability table for p >= 0.1
  and a Brier skill score (`results/web/calibration.csv`); ECE is secondary
  because almost every candidate falls in the lowest bin.
- **Deterministic ties** (G11): popularity ties are broken by item id;
  bootstraps resample whole sessions with seed 42.
- **Limits are stated** (G14): one retailer, equal-size objects, instant
  prefetch; no page-load claim from the replay.

The pivot run's own outputs are kept in `results/pivot_original/` and are
never edited. Its headline (cache 8, cost 0.01, adaptive minus fixed1) is
reproduced exactly on `master`:

<!-- source: results/pivot_original/report.md; results/web/comparisons.csv split=test selection=per_capacity scorer=logistic capacity=8 cost=0.01 adaptive_policy=adaptive_0.01 -->
+0.358 pp of utility per view [+0.331, +0.385].
<!-- source: none -->

## W1. Policies are selected per capacity (corrects P1)

**What the pivot did.** It chose every policy on validation at capacity 8
and reused that choice at capacities 2 to 32. `master` keeps that rule as a
separately labelled `capacity_transfer` comparison and never as a headline.
`webprefetch.benchmark.select_policies` now selects separately at each
capacity, on validation, and every comparison is reported three ways: against
the best fixed depth chosen at the same capacity, against no prefetching
(`fixed0`), and against transition top-1.

**Why it matters** (test split, logistic ranker, `c` = 0.01, utility per view
in pp; `results/web/comparisons.csv`):

<!-- source: results/web/comparisons.csv split=test scorer=logistic cost=0.01 capacity=2 -->
- At capacity 2 the transferred controller **loses to no prefetching**
  (-2.056 pp against `fixed0`, interval [-2.112, -2.005]). Selected at
  capacity 2, the validated choice is a far more conservative controller
  (`adaptive_0.1`), and it beats `fixed0` by +0.035 pp [+0.025, +0.046].
<!-- source: results/web/comparisons.csv split=test scorer=logistic cost=0.01 capacity=4 -->
- At capacity 4 the transferred controller also loses to no prefetching
  (-0.071 pp [-0.100, -0.043]); the per-capacity choice beats it
  (+0.197 pp [+0.174, +0.218]).
<!-- source: results/web/comparisons.csv split=test scorer=logistic cost=0.0 capacity=4 selection=per_capacity -->
- At cost 0 and capacity 4 the validated controller *loses* to the best fixed
  depth (`fixed2`): -0.196 pp [-0.220, -0.168]. The pivot reported a win
  there only because it compared against the wrong depth.
<!-- source: none -->

**Headline comparisons** (pre-registered W1/W2, `results/web/PREREGISTERED.md`):

<!-- source: results/web/comparisons.csv split=test selection=per_capacity scorer=logistic cost=0.01 -->
| Capacity | validated adaptive | vs best fixed (same cap) | vs `fixed0` |
|---:|---|---|---|
| 8 | `adaptive_0.01` | +0.358 pp [+0.331, +0.385] | +0.444 pp [+0.413, +0.477] |
| 32 | `adaptive_0.01` | +0.581 pp [+0.555, +0.606] | +0.768 pp [+0.727, +0.806] |
<!-- source: none -->

By decision rule 1 the adaptive controller beats fixed prefetching at
capacities 8 and 32 at this cost, and the statement does not generalise to
other capacities or costs. The best fixed depth at `c` = 0.01 is depth 1 at
both (`best_fixed_policy` in `results/web/comparisons.csv`): deeper fixed
prefetching costs more than it gains.

**Demo (P5).** The demo's "adaptive" is now the validation-selected
controller for the chosen capacity and cost, read from `benchmark.json`. The
pivot's rule (penalty = cost, which behaves like `fixed8` at cost 0) is still
offered and labelled `adaptive_cost`. `webprefetch parity` regenerates
`session_parity.json` from the same run; it is never compared across runs.

## W2. Freshness: the candidate index was stale (corrects the pivot's "coverage ceiling")

**What the pivot did.** Its candidate index was built once from split 0, the
oldest 40% of the views by time (split cut points in
`results/web/PREREGISTERED.md`), and its report called coverage@8 a ceiling
on next-page prefetching.

**What was built (WP5).** `webprefetch.index.OnlineIndex` adds a transition
`a -> b` only once `b`'s view time is strictly earlier than the decision
time, so no replayed decision can see its own or any later transition. The
scorer registry (`webprefetch.scorers`) offers counts from split 0
(`trans_s0`), from all earlier splits (`trans_prior`) and online
(`trans_online`). An online index needs global time order, which the
session-sharded replay provides by giving every shard the whole split's
pending transitions.

**What it measured** (test split, eligible next views,
`results/web/freshness.csv`):

<!-- source: results/web/freshness.csv split=test -->
| Index | coverage@1 | coverage@8 |
|---|---:|---:|
| split 0 (the pivot's) | 3.33% | 9.08% |
| splits 0-2 | 4.95% | 13.99% |
| online, causal | 6.33% | 17.53% |
<!-- source: none -->

Most of the pivot's "ceiling" was staleness. In cache terms (decomposition
lever `freshness`, capacity 8, depth 4, hit ratio):

<!-- source: results/web/decomposition.csv lever=freshness metric=hit_ratio -->
online transition counts minus the split-0 logistic ranker:
+1.52 pp [+1.49, +1.56].
<!-- source: none -->

## W3. What moves the needle: the decomposition, the feedback gate and the GRU

`webprefetch decompose` measures each lever as one paired difference on the
test split at capacity 8 (`results/web/decomposition.csv`; pre-registered
W3-W8), with a 95% paired session-bootstrap interval
(`results/web/PREREGISTERED.md`). Hit ratio:

<!-- source: results/web/decomposition.csv metric=hit_ratio -->
| Lever | Comparison | Difference [interval] |
|---|---|---|
| ranker (W3) | transition top-1 (split-0 counts) minus logistic, depth 1 | +0.07 pp [+0.06, +0.09] |
| freshness (W4) | online counts minus logistic, depth 4 | +1.52 pp [+1.49, +1.56] |
| depth (W5) | validated best fixed depth at `c` = 0 (`fixed4`) minus depth 1 | +0.80 pp [+0.77, +0.83] |
| feedback (W6) | validated feedback gate minus validated adaptive | -0.11 pp [-0.13, -0.10] |
| sequence model (W8) | GRU minus online counts, depth 4 | -0.79 pp [-0.83, -0.76] |
| cache scope (W7) | one cache per visitor minus per session, no prefetch, unbounded cache | +6.73 pp [+6.64, +6.82] |
<!-- source: none -->

Reading them by the pre-registered rules (`results/web/PREREGISTERED.md`):

- **The learned ranker is the smallest lever.** Its interval lies entirely
  below the lower ends of both the depth and the freshness intervals, which
  is the rule's condition. Counting transitions does at least as well as the
  trained logistic ranker, which had learned mostly back-navigation (P2).
- **The GRU sequence model is worse than counting** (W8, E5): its interval
  lies below 0. Its interval does not lie below the depth lever's, so no
  ordering is claimed between the sequence model and depth; it does lie
  below the freshness lever's. E5 trained 5 seeds on a shared GPU
  (`results/web/seq_metrics.csv`); compare it with other rankers only on the
  `novel_*` columns, because its history mask never ranks an item already in
  the window.
- **The feedback gate** (W6), ported from the disk study's precision gate
  (`cachecore.FeedbackGate`: a candidate is issued only while the observed
  precision of its (rank, score-bin) bucket beats `c`), is *not* described as
  improving on the adaptive controller.

<!-- source: results/web/decomposition.csv lever=feedback metric=utility_per_view -->
  On utility at `c` = 0.01 the difference is -0.007 pp [-0.023, +0.007]: no
  detectable difference.
<!-- source: results/web/policy_metrics.csv split=test scorer=logistic cache_scope=session capacity=8 -->
  It does issue fewer prefetches: 0.629 per view against the adaptive
  controller's 0.734 (`adaptive_0.01`), for a hit ratio of 12.52% against
  12.63%. Both are stated, neither is inferred from the other (rule 3).
<!-- source: none -->

- **Cache scope** (W7) is an upper bound on reuse across a visitor's
  sessions with an unbounded cache, not a prefetching result (rule 4).

**Calibration footnote.** `results/web/calibration.csv` carries the
reliability table and the Brier skill score. The logistic ranker is
overconfident above p = 0.2:

<!-- source: results/web/calibration.csv split=test scorer=logistic bin=3 -->
on the test split, candidates scored in [0.3, 0.4) have a mean probability of
0.343 and an observed frequency of 0.192.
<!-- source: none -->

That is one more reason controller decisions are judged by cache outcomes,
not by probabilities.

## W4. Live HTTP replay: measured waits with exact simulator parity (E6)

**What was built (WP12, `webprefetch.live`).** Each sampled test session is
one emulated browser whose logical cache **is** `cachecore.Cache`, the class
the simulator replays; per-session hits, issued and useful prefetches must
equal the simulator's (`sim.json`) exactly. An asyncio origin reads every
object with a real `pread` from an object store on the SSD or the HDD `sdb`;
round-trip time, bandwidth and object sizes are emulated or assumed and are
tagged so in `results/live/live_config.json`. A pass is **valid** only if
server lag, origin and client event-loop lag p99 are all at most 2 ms, there
is no parity mismatch and no connection error; an invalid pass is written but
never compared. The store's pages are dropped before every pass, and a pass
waits for a quiet host.

**What it measured** (2000 sessions, 6685 views per pass, capacity 8, RTT 50
ms, 10 Mbps, object sizes lognormal with median 64 KiB, all ASSUMED; waits
MEASURED; `results/live/`):

<!-- source: results/live/live_parity.csv; results/live/live_metrics.csv -->
- 42 passes; every one valid, and 0 sessions whose live counters differ
  from the simulator's.
<!-- source: results/live/live_metrics.csv experiment=headline store=hdd -->
- On the HDD store, saved wait per view against `fixed0` in the same
  repetition (3 repetitions): the online-index controller
  `adaptive_0.1@trans_online` saved 8.13, 8.14 and 8.55 ms at 2.20x the
  bytes; `fixed4` saved 8.05, 7.95 and 8.53 ms at 6.71x the bytes;
  `fixed8` saved 7.25, 7.13 and 7.32 ms at 14.91x. The no-prefetch mean
  wait was about 114 ms (114.05, 114.12, 114.16).
<!-- source: results/live/live_metrics.csv experiment=fifo -->
- **FIFO ablation** (SSD store, one repetition): with a FIFO link and a FIFO
  client pool, `fixed8`'s change in wait against `fixed0` was -1.22 ms
  [-2.44, +0.14], an interval that includes 0, where the model-ideal change
  is -3.32 ms; `fixed4` saved 4.79 ms. Speculative bytes queue in front of
  demand fetches when nothing gives demand priority.
<!-- source: results/live/live_metrics.csv experiment=hint -->
- **Server-side hints** (`hint:`, HDD store, one repetition): issuing the
  prefetches as `POSIX_FADV_WILLNEED` at the origin and sending nothing saved
  1.48 ms (`hint:fixed4`) and 1.75 ms (`hint:fixed8`), all of it disk time,
  at 1.00x the bytes.
<!-- source: none -->

**Caveats that travel with every live number.** Loopback with an emulated
network; object sizes assumed; one retailer; one per-session cache size;
SSD-store rows whose foreign I/O exceeds 1% are flagged in
`results/live/live_report.md` and are cited only with that flag, so the HDD
rows (no foreign I/O) are the headline. The model-ideal column is what an
analytic simulator without contention would predict; the difference is the
link queue, disk time and timer lag that `live_report.md` decomposes.

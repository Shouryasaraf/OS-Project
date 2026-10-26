# Disk study (E1): pre-registered comparisons and decision rules

Written before experiment E1 is run (plan C4). It fixes, in advance, the
comparisons the paper reports for the multi-corpus disk sweep, their
baselines and sample sizes, the interval method, and the rule that turns an
interval into a sentence. It states no expected outcome. Any change made after
E1 starts is recorded in `results/disk/provenance.json` (`notes`) with the
reason, and the original rule stays in this file.

Code: `python -m adaptive_prefetch.multicorpus` (sweep) and
`python -m adaptive_prefetch.analysis` (tables, deltas, winners, kappa*),
run together by `python reproduce.py disk`.

## 1. Data, slices and what every number is

- **Traces.** The 56 libCacheSim oracleGeneral block traces under
  `data/block-traces/`: msr 14, fiu 10, alibaba 20, tencent 12. Twin families
  (`corpora.TWIN_GROUPS`, 15 traces in 5 families) are near-duplicates.
- **Requests.** Canonical reconstruction (`merge="auto"`): FIU 4 KiB page
  records merged back into requests (same second, contiguous, cap 1024
  sectors); every other corpus as recorded. The format has no read/write flag,
  so writes replay as reads, and timestamps are whole seconds.
- **Everything is simulated.** Hit ratios are block hit ratios of an LRU cache
  replay. Latency figures come from a cost model, never from a device.
  `device_ios` and `blocks_moved` count modelled device work.
- **Cost models.** Every row runs under `LatencyModel(5, 100, 50)`
  ("assumed", break-even precision 0.526). The feedback-gated modes (`guard`,
  `depth_adaptive`, `correlate`) also run under `LatencyModel(1, 100, 100)`
  ("break_even", break-even 1.0101). No other mode's counters depend on the
  cost model.
- **Trained modes.** `lstm` and `learned_depth` are `skipped: evaluated in E2
  only` in every configuration here (E2 trains on a prefix that overlaps C1).

| Config | Traces | Slice | Merge rule | Capacities (512-B blocks) | Policies |
|---|---|---|---|---|---|
| C1 | all 56 | first 1,000,000 canonical requests | auto | 2048 (1 MiB), 131072 (64 MiB) | every `sweep.RANKING_MODES` mode; `deep` at 8, 32, 256; `linux_ra` at 64, 256, 1024; Belady/MIN reference on `none` rows |
| C2 | all 56 | first 250,000 | auto; same_second (uncapped); none (FIU only) | 2048, 16384, 131072, 1048576 | `none`; `deep` at 0, 2, 8, 32, 64, 128, 256, 512; `linux_ra` at 64, 256, 1024 |
| C3 | the 21 `PAPER_SUBSET` traces | slice 2 (skip 1,000,000, next 1,000,000), and the whole file where it has at most 4,000,000 canonical requests | auto | 2048, 131072 | none, sequential, adaptive, guard, deep at 8/32/256, linux_ra at 256 |
| C4 | the 10 FIU traces | first 1,000,000 | auto with merge cap 256 and 1024 | 2048, 131072 | as C3 |

A slice that cannot be filled (a second slice shorter than 1,000,000, or a
whole file over 4,000,000 requests) is an `n/a` row, never a shorter slice.

## 2. Validity gates (the run is not reported if any fails)

1. `msr_hm_0`, first 250,000 requests, cache 2048, `deep@8`: hit ratio rounds
   to 0.357231 and 901,487 prefetches are issued (the earlier prototype's
   cell; C2 contains it).
2. `guard` and `depth_adaptive` are identical on every counter at the
   break-even cost model, on every trace and capacity where both ran (D14.1).
3. `deep@0` is identical to `none` on every C2 cell.
4. No sweep key appears twice in `sweep.jsonl`.

The analysis computes all four (`analysis.checks`) and the reproduce step
stores them in `provenance.json` (`extra.checks`).

## 3. Primary comparisons (C1, both capacities, pooled and per corpus)

Each is a paired comparison over the traces on which both policies ran, n =
56 pooled (per corpus: 14 / 10 / 20 / 12).

| # | Metric | Policy | Baseline | Statistic |
|---|---|---|---|---|
| P1 | block hit ratio | adaptive | sequential | mean difference |
| P2 | block hit ratio | guard (assumed and break_even) | sequential | mean difference |
| P3 | block hit ratio | deep@8, deep@32, deep@256 | none | mean difference |
| P4 | block hit ratio | linux_ra@64, @256, @1024 | none | mean difference |
| P5 | device I/Os | every policy | none | mean of per-trace ratios |
| P6 | blocks moved | every policy | none | mean of per-trace ratios |
| P7 | precision, wasted share | every prefetching policy | sequential | mean difference, traces where both issue prefetches |
| P8 | kappa* | every policy | none | per trace, reported as a distribution (no test) |

`deltas.csv` carries every policy against `none` (hit ratio, device I/Os,
blocks moved) and against `sequential` (hit ratio, precision, wasted share);
the rows above are the ones the paper's claims are allowed to rest on.

## 4. Intervals and tests

- **Paired bootstrap** over traces: 10,000 resamples, seed 42, 95% percentile
  interval (`prefetch_common.stats.paired_bootstrap`). Every trace counts once.
- **Cluster bootstrap**: resample corpora, then traces within each drawn
  corpus; each twin family is down-weighted to a total weight of 1
  (`stats.cluster_bootstrap` with `stats.twin_weights`). 10,000 resamples,
  seed 42, 95% percentile interval.
- **Wilcoxon signed-rank**, two-sided, on paired differences (ratios minus 1),
  reported as a raw p-value with the number of tests in the table. It is not
  used by any decision rule. No multiplicity correction is applied to the
  intervals; the count of comparisons is stated wherever p-values appear.
- **Wins/losses/ties** per comparison, where a win is a trace on which the
  policy is better (higher hit ratio or precision, fewer device I/Os, fewer
  blocks moved, lower waste).

## 5. Decision rules

For a difference with interval [lo, hi] (and for a ratio, [lo, hi] compared
with 1):

- "**better** than the baseline" iff both the paired and the cluster interval
  lie entirely on the better side (lo > 0, or for a lower-is-better ratio
  hi < 1);
- "**worse**" iff both lie entirely on the worse side;
- otherwise "**no difference detected**", stated with both intervals. A
  paired-only result is reported as "paired interval only; not robust to
  corpus clustering", never as better or worse.

Lever comparison (the plan's thesis is a hypothesis, tested here, not a
conclusion). On hit ratio against `sequential`, at each capacity, the levers
are: predictor = `adaptive`, feedback = `guard`, depth = `deep@8`, window
growth = `linux_ra@256`. For an interval [lo, hi] of a difference D, the
interval of |D| is [0, max(|lo|, |hi|)] when lo < 0 < hi, else
[min(|lo|, |hi|), max(|lo|, |hi|)]. One lever is called **smaller** than
another iff the upper end of its |D| paired interval is below the lower end of
the other's |D| paired interval, and the same holds for the cluster intervals.
"The predictor is the smallest lever" is written iff it is smaller than every
other lever at both capacities; otherwise the paper reports the ordering the
intervals support and says which pairs cannot be ordered.

Device work. A policy is said to "reduce device I/Os" iff its P5 ratio
interval (paired and cluster) lies below 1, and to "move more blocks" iff its
P6 interval lies above 1. A hit-ratio gain is never described as a reduction
in device work without P5.

Winners. `winners.csv` counts per-trace winners. A winner count is never
reported without the same policy's precision, wasted share and device-I/O
ratio. No policy is called "the winner" from hit ratio alone, and `linux_ra`
appears in every comparison table.

## 6. Robustness and sensitivity (reported against C1, no new claims)

- **C3 slice 2 / whole files.** A C1 decision (Section 5) on the
  `PAPER_SUBSET` traces is called **robust** iff the same decision holds on
  slice 2 at both capacities; it is called **prefix-dependent** otherwise.
  Whole-file rows are reported for the traces that qualify, with covered hours.
  `alibabaBlock_3` is the designated drift case and is discussed individually.
- **C4 FIU merge cap.** FIU decisions are called **cap-insensitive** iff they
  agree at merge caps 256 and 1024.
- **C2 grid.** Descriptive only: hit ratio, device-I/O ratio and blocks-moved
  ratio against depth and `linux_ra` ceiling per capacity and corpus, and the
  `same_second` / `none` (FIU) rows as labelled sensitivity views. The best
  depth per (capacity, corpus) is reported as the argmax of the pooled mean,
  with its interval against the next depth.

## 7. References, not policies

- `belady_min` (C1, `none` rows): the demand-only Belady/MIN hit ratio of the
  same block stream at the same capacity. It is reported beside `none` as the
  limit of demand replacement; it is not a bound on prefetching policies and
  is never described as one.
- kappa* is the positioning-to-transfer ratio above which a policy's modelled
  device time drops below `none`'s. It is compared with measured device
  kappa only when `results/realio/device_*.json` exists (E0), and then per
  device.

## 8. What will not be done

- No number from the scratch prototype (`sweep2.jsonl`) is quoted; it is used
  only as a code cross-check.
- No hit ratio here is compared numerically with the MSRC-trace-003 (D17)
  figures.
- No trace, capacity or subset is chosen after seeing results to strengthen a
  claim; every configuration above is reported in full.

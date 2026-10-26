# Real-device testbed (E3): pre-registered comparisons and decision rules

Written before experiment E3 is run (plan C4). It fixes the grid, the
comparisons the paper reports from the real Linux testbed, their baselines,
the interval method and the rule that turns an interval into a sentence. It
states no expected outcome. Any change after the first E3 run is appended to
the amendment log at the end, with its date and reason, and the original
text is left in place. Amendment A1 below was made before any E3 run.

Code: `python -m adaptive_prefetch.realio grid` (runs, `realio.csv`,
`realio_summary.csv`), driven by `python reproduce.py realio --device DIR`,
once per device. `python -m adaptive_prefetch.realio report` rewrites the
summary from the rows.

## 1. What is measured

- **Devices (lead decision L2).** Primary: the HDD `sdb`
  (ST10000NM017B-2T), backing file under
  `/mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch/`, idle.
  Secondary: the root SATA SSD (Micron 5300), backing file under
  `outputs/realio/ssd/`, which carries other tenants' I/O; every SSD row
  carries `foreign_io_frac`. Nothing is written to `sdd` or `sde`, so the
  plan's replica HDD is not run.
- **Calibration (E0).** `results/realio/device_{sdb,ssd}.json`, measured
  before this file was written. `reproduce.py realio` reuses them and does
  not recalibrate unless `--recalibrate` is given. They feed `IOCostModel`
  (simulated device time) only, never a measured column.
- **Traces.** The first 20,000 canonical requests (plan section 2.1
  reconstruction; FIU pages merged) of msr_hm_0, msr_web_2, msr_prn_0,
  alibabaBlock_17 and fiu_homes-110108-112108. The format has no read/write
  flag, so writes replay as reads.
- **Cache.** 65,536 blocks (32 MiB), not E1's 2,048: Backend A's memory
  cgroup cannot hold a 1 MiB page cache (plan C10). The simulator plan of
  every row is computed at the same 65,536 blocks, window 32, with the
  assumed cost model 5/100/50 us (so the gated modes' break-even is 0.526).
- **Backends.** Backend B (`odirect`): a user-space LRU with exactly the
  simulator's rules, O_DIRECT demand reads, asynchronous prefetch on 4
  threads; its hit/miss sequence is asserted equal to the plan on every
  request. Backend A (`pagecache`): the kernel page cache in a memory cgroup,
  prefetch as `POSIX_FADV_WILLNEED` in chunks of at most 256 KiB; its hit
  ratio is a measured divergence (MGLRU, 4 KiB pages), never matched.
- **Modes** (`realio.runner.GRID_MODES`): none, sequential, deep (depth 8),
  deep@64, linux_ra, guard, adaptive_evidence on both backends;
  kernel_ra (`FADV_NORMAL`), kernel_seq (`FADV_SEQUENTIAL`) and online-guard
  (policy run inline, no precomputed plan) on Backend A only. `linux_ra`
  keeps its window state against the user-space shadow LRU; the kernel's own
  read-ahead is what kernel_ra and kernel_seq measure.
- **Think time.** `closed` (next request as soon as the previous completes)
  and `spread:1:5000` (the requests of one recorded 1-s tick spread evenly
  over that second, each gap capped at 5 ms).
- **Run protocol.** Every run is a fresh child process. Before it: the
  trace file's pages are dropped, the drive cache is displaced by reading
  1 GiB of an unused region with O_DIRECT, and (Backend A) the whole
  backing file is dropped from the page cache and the cgroup limit is set.
  Within one (trace, think model, repeat) the order of the (backend, mode)
  runs is shuffled with seed 42. 3 repeats.
- **Measured per run** (`realio.csv`): mean, p50, p95, p99 and max request
  latency; measured hit ratio and full-hit request fraction; ready and late
  prefetch hits and the time waited on late ones; device and prefetch MiB;
  think-time error; foreign I/O fraction; load average at the start.

## 2. Validity gates (a run that fails one is not reported)

1. **Backend B parity.** On every `odirect` run the block hit/miss sequence
   and `useful_prefetches` equal the simulator's (asserted per request; the
   run raises `CacheDivergence` otherwise). A divergence is a defect: the
   affected rows are not reported until it is fixed and re-run.
2. **No aliasing.** `wrapped = 0` on every row (no `--allow-alias`).
3. **Quiet host.** The 1-minute load average is at most 32 when each run
   starts. The grid waits (up to 1 h at a time, with a log line) until it is;
   a row whose recorded `load_avg` exceeds 32 is not used.
4. **Foreign I/O.** On sdb, a row with `foreign_io_frac` above 1% is reported
   as contended and not used in a decision. On the SSD it is reported with
   every row; a cell is called contended where any of its rows exceeds 1%.

## 3. Comparisons

A *cell* is one (device, backend, think model, trace).

| # | What | Statistic |
|---|---|---|
| R1 | measured mean latency of each mode against `none`, per cell | `delta_vs_none_us`: mean over repeats of (mode - none of the same repeat), 95% Student-t interval over the 3 paired repeats (df 2) |
| R2 | does the simulator rank policies by measured latency? per cell | Spearman rho across modes of simulated hit ratio vs measured mean latency (`spearman_hit_ratio`), and of simulated `IOCostModel` device time vs measured mean latency (`spearman_device_time`), on repeat-mean latencies |
| R3 | Backend B parity | count of `odirect` runs and of runs whose parity held (must be all) |
| R4 | hit-ratio-only prediction | `pred_err`: (prediction from the run's full-hit fraction and its repeat's `none` hit/miss latencies - measured) / measured; mean over repeats |
| R5 | effective prefetch cost | `k_eff`: the `prefetch_multiple` at which the per-block cost model reproduces the measured mean, calibrated from the same repeat's `none`; mean over repeats |
| R6 | the kernel's own read-ahead | kernel_ra and kernel_seq against `none` on Backend A, as R1 |
| R7 | think time | on msr_hm_0 (sdb), each mode's R1 delta under `closed` and under `spread:1:5000`, side by side |

Late hits (`late_hits`, `late_wait_us`) and device MiB are reported with every
R1 row, because a prefetch that has not completed when demanded can cost
more than the miss it was meant to hide.

## 4. Decision rules

- **R1/R6.** In a cell, mode M "**lowers** mean latency relative to none"
  iff the upper end of its 95% interval is below 0; "**raises**" it iff the
  lower end is above 0; otherwise "no difference detected (3 repeats)". With
  df 2 the interval is wide; that is stated, never compensated for after
  the fact. Fewer than 2 paired repeats: "not established".
- **R2.** rho is always reported with its n (the modes with a simulator
  plan: 7 on Backend B, 8 on Backend A, where kernel_ra and kernel_seq have
  none). The simulator is said to "**rank policies consistently with
  measured latency**" in a cell iff rho_hit is negative (higher simulated hit
  ratio, lower latency) and |rho_hit| reaches the exact two-sided 5%
  permutation critical value for that n without ties: 0.886 (n = 6),
  0.786 (n = 7), 0.738 (n = 8), 0.700 (n = 9), 0.648 (n = 10). It "**ranks
  them against** measured latency" iff rho_hit is positive and reaches the
  same value. Otherwise "ranking not supported". With n = 4 (both backends of
  the SSD subset) no 5% permutation test exists and rho is descriptive only. The same rule applies to rho_device_time
  with "lower simulated device time, lower latency" as the consistent sign.
  No claim about hit ratio and latency is made across cells, devices or
  think models.
- **R4/R5.** Stated per trace, think model and device (plan honesty rule
  15), never as a property of HDDs or SSDs. `k_eff < 1` is described as "a
  prefetched block cost the requester less than a demand-read block in this
  run" and nothing more.
- **R7.** Descriptive: the two deltas with their intervals. "Think time
  changes the effect of M" is written only if one interval lies entirely
  below 0 and the other does not.
- **Simulated vs measured.** A simulated hit-ratio gain is never written as
  a latency gain without R1. Every latency number in the paper from E3 is
  labelled measured, with device, backend, think model, trace prefix and the
  load average.

## 5. Limitations fixed in advance

- `learned_depth` is not in the E3 grid (lead decision L3, runtime budget),
  so E3 says nothing about the learned depth model on a real device.
- One HDD (L2): there is no replica drive, so device-to-device variation is
  not measured.
- 20,000-request prefixes of five traces; one shuffle seed; 3 repeats.
- The 32 MiB cache differs from E1's 1 MiB; E3 rows are compared with E3
  simulator plans at the same capacity only, never with E1 numbers.
- Backend A's `prefetch_MiB` counts the bytes hinted, not the bytes the
  kernel read; its `device_MiB` comes from `/proc/diskstats`. Backend A's
  closed loop has a little extra slack per request compared with Backend B's
  (WP8 review).
- Each invocation's `provenance.json` keeps the earlier invocations' command,
  time, git state and notes under `extra.earlier_invocations`.

## 6. What will not be done

- No number from the WP8 smoke runs, the WPF1 timing run (A1) or the
  prototype (`$EVAL/realio/`) is quoted; they used other request counts,
  flush sizes or code.
- No trace, mode, cell or repeat is dropped or added after seeing E3
  results; every cell of the grid below is reported.

## Amendment log

### A1 (2026-10-26, before any E3 run): runtime budget

Reason: runtime budget (lead decision L3: E3 about 4 h wall in total, sdb
plus an SSD subset). The grid of plan section 2.6 (5 traces x 17 runs x 2
think models x 3 repeats on each HDD) is planned at about 9.9 h on sdb alone
by `realio.runner.estimate_seconds`, and the replica HDD is dropped by L2.
The grid that runs instead (`realio.runner.E3_PLANS`, chosen by
`reproduce.py realio --grid auto` from the device's rotational flag):

| Plan | Device | Traces | Modes | Think | Repeats | Runs | Planned |
|---|---|---|---|---|---|---|---|
| `hdd`, part 1 | sdb | all 5 | all 10 (17 backend x mode runs) | closed | 3 | 255 | about 1.8 h |
| `hdd`, part 2 | sdb | msr_hm_0 | all 10 (17 runs) | spread:1:5000 | 3 | 51 | about 1.2 h |
| `ssd` | SSD | msr_hm_0, msr_web_2 | none, deep, linux_ra, guard (both backends), kernel_ra (Backend A) | closed | 3 | 54 | about 0.3 h |

Planned total: about 3.3 h, leaving room for load-average waits.

How the cut was chosen, using run time only:

- The closed loop keeps every trace and mode: it is the main real-device
  table and costs about 20-35 s per run.
- A spread run costs its think time on top (53 s on msr_hm_0, 93-100 s on
  the other four traces, computed from the traces with `clock.gaps_us`). The
  spread model is kept on msr_hm_0 only: it is the cheapest of the five, and
  it is the trace whose prototype think-time findings plan section 2.6 lists
  for re-testing.
- The SSD subset is the plan's reduced SSD grid (Critic, section 2.6:
  msr_hm_0 and msr_web_2, closed loop, none, deep, linux_ra and guard on
  Backend B, kernel_ra on Backend A) with one addition: the four Backend B
  modes also run on Backend A, so that kernel_ra has a Backend A `none` of
  the same repeat to be paired with (R1, R6). That adds 24 of the 54 runs.

The planning constants (`realio.runner.E3_TIMING`) are the WP8 smoke's 1 GiB
flush times (9.6-10.5 s on sdb, 2.6-2.7 s on the SSD) and a WPF1 timing run on
2026-10-26: Backend B, closed loop, 5,000 requests of each trace under `none`
and `deep@64`, 64 MiB flush, on sdb at a load average of about 120, scaled to
20,000 requests and multiplied by 1.5. That run's latencies were printed to
the console and seen by the author of this amendment. They played no part in
the cut, which depends only on run time, and they are not results (other
request count, flush size and load).

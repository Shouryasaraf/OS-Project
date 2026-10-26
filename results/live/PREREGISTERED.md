# Pre-registration: live HTTP prefetch replay (E6)

Committed by WP12 before any E6 pass is run (plan C4). This file fixes the
comparisons, the baselines, the sample, the interval method and the decision
rules. It states no expected outcome. Any change after the first E6 pass is
appended to the amendment log at the end, with its date and reason, and the
original text is left in place.

## What is measured

`python -m webprefetch.live run` (driven by `reproduce.py live`) replays held-out
RetailRocket sessions as real HTTP/1.1 fetches over loopback. An origin process
serves objects with a real `pread` from an object store, and emulates RTT and a
per-session access link. Emulated browsers each keep a `cachecore.Cache` (the
simulator's cache class) and record the wait that every view perceives.

- **Sample.** Split 3 (test). Sessions with at least 2 views. 2000 sessions drawn
  with seed 42. Cache capacity 8 objects per session.
- **Policies (headline).** fixed0, fixed1, fixed4, fixed8, adaptive_0.002,
  adaptive_0.01, gate_0.01, feedback_0.01, and adaptive_0.01@trans_online (the
  online-index variant). Decisions come from the plan, which is checked
  session by session against `simulator.replay` before any pass runs.
- **Baseline.** fixed0 (no prefetch), in the same run and repetition.
- **Assumed inputs, never measured.** RTT 50 ms. Bandwidth 10 Mbps. Lognormal
  object sizes with median 64 KiB, sigma 1.0, clipped to 1 KiB-4 MiB. Recorded
  inter-view gaps capped at 2 s. Session starts spread uniformly over 180 s.
  Client pool of 6 connections, at most 2 of them speculative. Priority link.
  Cold store page cache at the start of every pass. `live_config.json` tags
  every input ASSUMED or MEASURED.
- **Stores.** SATA SSD (repository disk) and HDD `sdb` under
  `/mnt/b18edc65-4395-4a85-976a-bed3428c073b/os-project-prefetch/web-store/`
  (lead decision L2). Nothing is written to sdd or sde.
- **Repetitions.** 3 per store for the headline. The policy order is rotated in
  every repetition, and passes run one at a time.

## Primary outcome

The mean perceived wait per view (ms), policy minus fixed0, from passes of the
same repetition. Its 95% interval is a paired session bootstrap: the ratio
estimator sum(delta) / sum(views) over resampled sessions, 1000 resamples,
seed 42 (`prefetch_common.stats.session_bootstrap`).

Secondary outcomes, reported for every pass:

- the hit ratio
- ready, in-flight and miss view counts
- KiB per view, wasted KiB per view, and bytes relative to fixed0
- the model-ideal wait (0 on a hit; rtt + size/bandwidth on a miss) and its
  delta against fixed0
- the excess of the measured wait over the model-ideal wait
- the demand-fetch decomposition: disk, link queue, timer lag and time
  outside the server

## Validity (fixed before any pass)

A pass is valid only if all of the following hold:

- the per-request server lag p99 is at most 2 ms
- the origin and client event-loop lag p99 are each at most 2 ms
- parity has 0 mismatched sessions
- there are 0 connection errors
- the 1-minute load average was below 32 when the pass started; otherwise the
  pass is aborted and logged

An invalid pass is written with `valid=False`. It is never compared and never
cited. The paper cites only rows with `valid=True`, each together with its
`foreign_io_frac` and load average.

## Hypotheses and decision rules

- **H1 (parity).** In every pass, the live per-session hits, issued, views and
  useful counts equal `sim.json` exactly. Decision: any mismatch is a defect.
  The affected run is not reported until it is fixed and re-run.
- **H2 (perceived latency, per policy and store).**
  - "P lowers the mean wait relative to fixed0" iff the 95% CI upper bound is
    below 0 in every valid repetition.
  - "P raises it" iff the CI lower bound is above 0 in every valid repetition.
  - Otherwise: "no consistent difference".
  - Fewer than 2 valid repetitions: "not established".
- **H3 (simulation vs measurement).** For each prefetching policy, the measured
  delta is compared with the model-ideal delta of the same pass. The
  model-ideal delta assumes instant prefetch and no contention.
  - "The instant-prefetch model overstates P's benefit" iff the measured
    delta's CI lower bound is above the model-ideal delta in every valid
    repetition.
  - "It understates it" iff the CI upper bound is below the model-ideal delta.
  - Otherwise: "consistent with the model".
- **H4 (latency vs bytes).** Descriptive only. The figure plots the mean wait
  saved against bytes relative to fixed0, per policy and store. On the
  per-repetition means, the paper lists which policies no other policy beats
  on both axes. No significance claim is made from this figure.
- **H5 (scheduling, FIFO ablation).** fixed4 and fixed8 run under a FIFO link
  and a FIFO pool (1 repetition, SSD), and are compared with the headline
  priority run of the same store. "Demand priority changes P's delta" iff the
  two 95% CIs (each P vs its own run's fixed0) do not overlap.
- **H6 (server-side hints, HDD).** hint:fixed4 and hint:fixed8 are each
  compared with fixed0 of the same run, using the H2 rule (1 repetition). With
  only one repetition, the result is stated as "in one repetition".
- **H7 (store).** For each policy, the SSD and HDD deltas are compared with the
  H5 non-overlap rule, using the first valid repetition of each store.

## What is not claimed

- Savings are stated as measured "on loopback with an emulated network".
  Object sizes, RTT and bandwidth are assumptions, so no figure is a measured
  page-load speedup of a real site.
- The feedback gate and the online index take their decisions in trace time
  order. Those decisions are replayed on the compressed live schedule. The live
  run measures the latency of those decisions; it does not re-learn the gate in
  live time.
- The gate learns only from the 2000 sampled sessions.
- Nothing is generalised beyond one retailer, one split, or these assumptions.

## Budget and reductions

Lead decision L3 sets the E6 core budget at about 3 h. Any reduction is chosen
before the first pass: fewer repetitions or policies for a store, or a dropped
ablation. It is recorded in this file's amendment log and in
`live_config.json` (reps, policies and aborted passes for each run). No
reduction is made after looking at results.

## Amendment log

(none)

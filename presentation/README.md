# Project Review 2 presentation

Use [Review2_Adaptive_Disk_IO_Prefetching_2026-09-22_v2.pptx](Review2_Adaptive_Disk_IO_Prefetching_2026-09-22_v2.pptx) for the current review. It follows the Review 1 visual style and covers the four Review 2 marking criteria with results from the current repository.

`Review2_Adaptive_Disk_IO_Prefetching_v5.pptx` is an earlier draft with outdated results; retain it only for historical reference.

The reported 160/160 classification result is on independently generated labelled windows, not verified real-trace accuracy. Cache costs are modelled, and the optional LSTM baseline was not run.

**Both decks are out of date** and must be rebuilt from a fresh `python main.py` before presenting. They predate the following, all of which change the headline result:

- The benchmark now runs **12 modes**, not 7. Four were added: `guard`,
  `depth_adaptive`, `correlate` (feedback-gated, see
  `../docs/DECISIONS.md` D12) and `deep` (fixed depth-8 read-ahead, D13).
- **Fixed depth-8 read-ahead has the highest hit ratio on 6 of 8 real
  datasets** and on 9 of 11 overall. It is not adaptive -- it never looks at
  the workload. Any slide claiming the learned classifier leads on real
  traces is wrong.
- The modelled-cost column is **degenerate** at the default cost model:
  break-even precision is 1.0101, so no prefetcher can pay for itself and
  "no prefetch" wins it by arithmetic. Do not present a cost ranking
  without printing the break-even alongside it.
- An earlier "no routing decision can help" conclusion has been **retracted**
  -- it was an artefact of every mode prefetching at most 2 blocks
  (`../docs/DECISIONS.md` D13, D16).

Do not hand-edit numbers into the `.pptx` files. Re-run the benchmark and
rebuild the slides from `outputs/`.


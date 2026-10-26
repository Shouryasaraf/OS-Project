# `master` branch

This branch combines the two lines of work in this repository so the
project can be presented in one place:

- `main`: adaptive **block-level disk I/O** prefetching (MSR Cambridge style
  block traces, LRU page-cache simulation, twelve prefetch policies).
- `pivot/retailrocket-adaptive-prefetch`: adaptive **web object** prefetching
  on the RetailRocket clickstream (per-session object cache, learned ranker,
  adaptive depth controller).

Work on this branch replicates the useful code from both, fills the gaps that
each branch documents (real-system validation, LSTM training, cross-domain
comparison), and carries the IEEE-format paper. Progress is recorded in the
commit history and in `docs/`.

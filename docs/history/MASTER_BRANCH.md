# `master` branch

`master` combines the two lines of work in this repository so the project is
presented, reproduced and written up in one place:

- `main`: adaptive **block-level disk I/O** prefetching (`src/adaptive_prefetch`;
  an LRU page-cache simulation that started with twelve policies on
  MSRC-trace-003). Its history and decision log (`docs/DECISIONS.md`
  D1-D17) are kept.
- `pivot/retailrocket-adaptive-prefetch`: adaptive **web object** prefetching
  on the RetailRocket clickstream (`src/webprefetch`; per-session object cache,
  learned ranker, adaptive depth controller). Its docs moved to `docs/web/`,
  its audit to `audit/GRILL.md`, and its committed outputs, verbatim, to
  `results/pivot_original/`.

On top of both, `master` added what each branch documented as missing:

- a multi-corpus disk study over 56 public block traces with paired and
  cluster bootstrap intervals (E1), device-I/O accounting and the Linux
  read-ahead policy as a baseline;
- GPU-trained models, scored only on data they were not trained on: the
  learned read-ahead depth and the LSTM (E2), and a GRU for the web (E5);
- a real Linux testbed that replays the simulator's decisions on an HDD and
  an SSD (E0, E3), and a real HTTP replay with exact simulator parity (E6);
- per-capacity policy selection, a causal online index and a feedback gate
  for the web study (E4);
- one results contract (`prefetch_common.schemas`), provenance for every
  results directory, pre-registered decision rules, the IEEE paper
  (`paper/`) and the Beamer deck (`presentation/`).

The work plan is `docs/MASTER_PLAN.md`; what changed in each study's
conclusions is `docs/DECISIONS.md` D18-D21 and `docs/web/WEB_DECISIONS.md`.
The rules for changing anything are in `AGENTS.md`.

# Data sources, attribution and derived outputs

This repository holds two studies. Neither bundles its full source data in
Git: the block traces (`data/block-traces/`, `data/MSRC-trace-003/`) and the
RetailRocket events (`data/raw/`) are gitignored and must be obtained from the
sources below.

## Web study: RetailRocket events

Source: RetailRocket, [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), `events.csv`, version 2. The Kaggle listing identifies its license as [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

This project sorts and deduplicates source events, segments them into sessions and time partitions, learns transition counts and trained models, and publishes aggregate metrics and a bounded demonstration drawn from the data. These are changes from the source dataset; RetailRocket does not endorse this project.

If you share dataset-derived data or adapted material, retain appropriate credit, link the license, indicate changes, observe the NonCommercial restriction, and apply the ShareAlike condition where the license requires it. Review whether a specific model, export, or dataset-derived artifact is adapted material before redistribution; this notice does not declare a blanket license for independently written source code. The repository does not bundle the full dataset in Git. The static demo export contains real derived sessions and should carry this notice when hosted.

## Disk study: block I/O traces

The trace files are used unmodified on disk; the loaders reconstruct requests
from them at replay time. Disk results committed under `results/` are meant
to be aggregate metrics only. Check each source's own terms before
redistributing a trace or a per-request derivative.

- **oracleGeneral collection** (`data/block-traces/`, files
  `*.oracleGeneral.zst`): CMU Parallel Data Lab, *Cache Datasets in
  oracleGeneral Format*,
  <https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/>. The
  format is the one read by libCacheSim
  (<https://github.com/1a1a11a/libCacheSim>); no libCacheSim code is included
  here. The collection repackages these original trace sets:
  - `msr_*`: Microsoft Research Cambridge enterprise server traces, released
    with D. Narayanan, A. Donnelly and A. Rowstron, "Write Off-Loading:
    Practical Power Management for Enterprise Storage", USENIX FAST 2008.
  - `fiu_*`: Florida International University traces, released with
    R. Koller and R. Rangaswami, "I/O Deduplication: Utilizing Content
    Similarity to Improve I/O Performance", USENIX FAST 2010.
  - `alibabaBlock_*`: Alibaba Cloud block storage traces, described in J. Li,
    Q. Wang, P. P. C. Lee and C. Shi, "An In-Depth Analysis of Cloud Block
    Storage Workloads in Large-Scale Production", IEEE IISWC 2020.
  - `tencentBlock_*`: Tencent Cloud Block Storage traces, SNIA IOTTA
    Repository trace set 27917 (<https://iotta.snia.org/traces/27917>),
    released with Y. Zhang et al., "OSCA: An Online-Model Based Cache
    Allocation Scheme in Cloud Block Storage Systems", USENIX ATC 2020.
- **MSRC-trace-003** (`data/MSRC-trace-003/`, `.revised` files): the MSR
  Cambridge traces in the form used by the earlier disk results (D17 in
  `docs/DECISIONS.md`). Absent on some machines.
- **Committed samples** (`data/samples/`): 1,000-request MSR-format samples
  whose original source and selection method have not been independently
  verified (see `data/samples/README.md`), a generated synthetic trace, and a
  small synthetic web fixture (`web_fixture.csv`) that contains no
  RetailRocket data.

# References: verification record and Review 1 corrections

`paper/refs.bib` is the paper's only bibliography. This file records how each
entry was checked, and which citations in the Review 1 deck
(`docs/reference/Operating Systems Lab Submission 2.pdf`) were wrong. The
numbers in this file are bibliographic metadata (volumes, pages, DOIs), not
results.

## How the entries were verified

`refs.bib` holds 55 entries in seven groups.

- **Groups A to F (47 entries)** are the literature evaluation's verified list,
  copied unchanged. Each one was checked on 2026-10-26 against Crossref
  (`api.crossref.org`), the publisher or USENIX page, PMLR, arXiv, or the
  official dataset page. A field that could not be confirmed was left out
  rather than guessed. Several USENIX page ranges are missing for this
  reason.
- **Group G (8 entries)** was added by WP7 and checked on 2026-10-26 against
  the pages below. A man page's `year` is the date in that page's footer.

| key | checked against | what was confirmed |
|---|---|---|
| `paszke2019pytorch` | NeurIPS proceedings abstract page and its BibTeX export (`proceedings.neurips.cc`, hash `bdbca288…`) | title, all 21 authors, editors, volume 32, publisher. The export gives no page range, so the entry has none. |
| `man_posix_fadvise` | `man7.org/linux/man-pages/man2/posix_fadvise.2.html` (man-pages 6.19, 2026-02-08) | name line; WILLNEED starts a non-blocking read into the page cache; DONTNEED frees cached pages |
| `man_readahead` | `man7.org/.../readahead.2.html` (man-pages 6.19, 2026-02-08) | name line; Linux-specific; added in Linux 2.4.13 |
| `man_cachestat` | `man7.org/.../cachestat.2.html` (man-pages 6.19, 2025-06-28) | name line; added in Linux 6.5 |
| `man_readv` | `man7.org/.../preadv2.2.html`, which renders `readv(2)` (man-pages 6.19, 2026-02-22) | `RWF_NOWAIT` for `preadv2` since Linux 4.14 |
| `man_open` | `man7.org/.../open.2.html` (man-pages 6.19, 2026-02-08) | `O_DIRECT` and its semantics; supported since Linux 2.4.10 |
| `linux_cgroup_v2` | `docs.kernel.org/admin-guide/cgroup-v2.html` | `memory.max` is the hard limit; a page is charged to the cgroup that instantiated it, and stays charged until it is released |
| `linux_mglru` | `docs.kernel.org/admin-guide/mm/multigen_lru.html` | multi-gen LRU is an alternative LRU for page reclaim; its state is in `/sys/kernel/mm/lru_gen/enabled` |

The `cmupdl_cachedatasets` note describes the oracleGeneral record layout
correctly: 24-byte records of (timestamp, obj_id, obj_size,
next_access_vtime). Do not describe the CMU mirror as splitting each request
into 512-byte records. For MSR, Alibaba and Tencent one record is one
original request. For FIU one record is one 4 KiB page (MASTER_PLAN §0.1).

## Corrections to the Review 1 deck

The deck is a historical record and is **not edited** (MASTER_PLAN §5,
constraint 11). The paper cites the corrected entries below. Only the
bibliographic data was corrected here. The deck's summaries of what each
paper found were not re-verified, and the paper does not repeat them.

| deck cites it as | correct citation | key |
|---|---|---|
| "CAPSULE: Clustering-Assisted Prefetching for Cloud Storage", Ramadhan et al., ACM SIGMETRICS 2026 | A. R. Ramadhan, S. Yoo, J. Choi, "CAPSULE: A Storage Prefetcher Harnessing Spatio-Temporal Locality for Cloud-Scale Workloads", *ACM SIGMETRICS Performance Evaluation Review* 54(1):76–78, 2026, DOI 10.1145/3820171.3806897. The title is different, and the venue is the PER journal issue, not the conference proceedings. | `ramadhan2026capsule` |
| "Adaptive Data Prefetching for File Storage Systems Using Online ML", "Cyprus Univ. of Technology, 2026", with no authors or venue | G. Savva, H. Herodotou, *Big Data and Cognitive Computing* (MDPI) 10(1):28, 2026, DOI 10.3390/bdcc10010028 | `savva2026adaptive` |
| Akgun et al., "arXiv 2021" | I. U. Akgun, A. S. Aydin, A. Shaikh, L. Velikov, E. Zadok, "A Machine Learning Framework to Improve Storage System Performance", *HotStorage '21*, ACM, pp. 94–102, DOI 10.1145/3465332.3470875. It is the archival workshop paper, not an arXiv preprint. | `akgun2021kml` |
| Chakraborttii & Litz, "ECML 2020" | "Learning I/O Access Patterns to Improve Prefetching in SSDs", ECML PKDD 2020, Applied Data Science Track Part IV, *LNCS* vol. 12460, Springer, **2021**, pp. 427–443, DOI 10.1007/978-3-030-67667-4_26. The conference took place in 2020, and the proceedings volume is dated 2021. | `chakraborttii2020learning` |
| Chakraborttii & Litz, "NVMW 2022" | "Deep Learning-based Prefetching for Flash", Non-Volatile Memories Workshop 2022. This is a **non-archival** workshop abstract (NSF PAR record 10332957), so it is cited as such and never as a peer-reviewed paper. The NVMW 2022 call (sigarch.org, checked 2026-10-26) asked for 2-page extended abstracts, posted them online with no printed proceedings, and allowed the same work to be published later. | `chakraborttii2022deep` |

These deck citations were confirmed correct as given:

- Baleen (Wong et al., FAST '24): `wong2024baleen`
- Soundararajan et al. (USENIX ATC '08): `soundararajan2008context`
- Lucas Filho et al. (AI4Sys '23): `lucasfilho2023streaming`
- CrossPrefetch (ASPLOS '24): `garg2024crossprefetch`

## Adding a reference

1. Verify it against a primary source: Crossref, the publisher, USENIX,
   kernel.org or man7.org. Leave out any field you cannot confirm.
2. Add it to `paper/refs.bib`, with the source and access date in its `note`
   or in the group comment.
3. Add a row to the table above.
4. Run `python paper/check.py`. It fails on a duplicate key, on an entry with
   no title, and on any `\cite` key that is not in `refs.bib`.

## Classifier study and E7 references (paper-v2)

The tracks CLF, HSR and WRK and the MSRC light sample (E7) proposed their
references in `paper/refs_CLF.bib`, `refs_HSR.bib`, `refs_WRK.bib` and
`refs_MSRC.bib`. They were merged into `paper/refs.bib` and the four files
removed. Where two files held the same key, the entries agreed on authors,
title, venue, volume, pages and year and differed only in author-name order
or venue spelling; the entry with the most fields was kept. The CART book had
two keys (`breiman1984classification` in CLF, `breiman1984cart` elsewhere);
`breiman1984cart` is kept, `clfzoo/zoo.py` cites it, and the
`results/classifier/README.md` table still shows the old key because that
README is hashed in its provenance. `tests/test_paper_check.py` checks that
every `clfzoo.zoo.CITATIONS` key is in `refs.bib` and that no
`refs_<TRACK>.bib` file is left.

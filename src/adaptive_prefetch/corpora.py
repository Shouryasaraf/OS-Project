"""Block-trace corpora: names, near-duplicate groups and the paper subset.

Data and pure functions only. Nothing here opens a trace; the loader is
``trace.iter_oracle_chunks``. The files are libCacheSim ``oracleGeneral``
conversions kept under ``data/block-traces/<corpus>/`` (gitignored).

Four corpora are present. Their records are not all alike, which is why the
loader detects units and merges pages (``trace.oracle_profile``):

* ``msr``     MSR Cambridge. Byte sizes, one record per original request.
* ``fiu``     FIU SRCMap/IODedup. One record per 4 KiB page, so requests are
              re-assembled. Sizes are 512-B sectors, except ``fiu_cheetah``,
              which stores bytes.
* ``alibaba`` Alibaba EBS. Byte sizes, a volume tag in bits 40 and up.
* ``tencent`` Tencent CBS. Byte sizes, a volume tag in bits 40 and up.
"""

from __future__ import annotations

from pathlib import Path

#: Default location of the (gitignored) trace collection, relative to the
#: repository root.
BLOCK_TRACE_ROOT = "data/block-traces"
#: Every file in the collection carries this marker before any compression
#: suffix: ``msr_hm_0.oracleGeneral.zst``.
ORACLE_MARKER = ".oracleGeneral"

#: File-name prefix -> corpus. Checked in order; prefixes do not overlap.
_PREFIXES = (("msr_", "msr"), ("fiu_", "fiu"),
             ("alibabaBlock_", "alibaba"), ("tencentBlock_", "tencent"))
CORPORA = tuple(corpus for _, corpus in _PREFIXES)


def trace_stem(path: str | Path) -> str:
    """The trace name without directory, marker or compression suffix.

    ``data/block-traces/msr/msr_hm_0.oracleGeneral.zst`` -> ``msr_hm_0``.
    FIU names keep their date range and host suffix, because two FIU files
    can share a host (``fiu_webmail`` and ``fiu_webmail+online``).
    """
    name = Path(path).name
    if ORACLE_MARKER in name:
        return name.split(ORACLE_MARKER)[0]
    return name[:-4] if name.endswith(".zst") else name


def CORPUS_OF(stem: str) -> str:
    """Corpus of a trace stem; raises for a name outside the four corpora."""
    stem = trace_stem(stem)
    for prefix, corpus in _PREFIXES:
        if stem.startswith(prefix):
            return corpus
    raise ValueError(f"unknown block-trace corpus for {stem!r}; expected a "
                     f"prefix in {[p for p, _ in _PREFIXES]}")


#: Near-duplicate traces. Members of a group have record counts within about
#: 5% of each other and near-equal mean request sizes (blocktrace-scout
#: ``totals.jsonl``): tencent 1162/1591/1911 are 3.047-3.049M records
#: averaging 127.99 sectors, and alibaba 13/16/18/20/22 are 14.1-14.9M records
#: averaging 72.2-72.9 sectors. A collection-wide statistic that counts each
#: member as an independent workload overweights them, so analyses should
#: cluster on these.
TWIN_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(f"alibabaBlock_{n}" for n in (13, 16, 18, 20, 22)),
    frozenset(f"alibabaBlock_{n}" for n in (11, 15)),
    frozenset(f"alibabaBlock_{n}" for n in (1, 5)),
    frozenset(f"alibabaBlock_{n}" for n in (8, 19, 24)),
    frozenset(f"tencentBlock_{n}" for n in (1162, 1591, 1911)),
)


def twin_group(stem: str) -> frozenset[str] | None:
    """The near-duplicate group containing ``stem``, or ``None``."""
    stem = trace_stem(stem)
    for group in TWIN_GROUPS:
        if stem in group:
            return group
    return None


#: Traces chosen by the blocktrace-scout for diversity in sequentiality,
#: in the order of plan §2.1. The plan names 20 traces but calls the subset
#: 21; the lead's ruling made ``msr_prn_0`` the 21st (it is also the trace E2
#: must train on, plan C2). ``alibabaBlock_3`` is the drift case: its
#: next-record contiguity is 0.027 over the first 18 h but 0.51 over the
#: whole 744 h file (blocktrace-scout ``stats_all.jsonl``), so a prefix does
#: not represent it. FIU entries are replayed as canonical (page-merged)
#: requests.
PAPER_SUBSET: tuple[str, ...] = (
    "msr_hm_0", "msr_prn_1", "msr_proj_0", "msr_web_2", "msr_prxy_0",
    "msr_prn_0",
    "fiu_casa-110108-112108", "fiu_homes-110108-112108",
    "fiu_online.cs.fiu.edu-110108-113008", "fiu_madmax-110108-112108",
    "alibabaBlock_23", "alibabaBlock_19", "alibabaBlock_14",
    "alibabaBlock_17", "alibabaBlock_13", "alibabaBlock_3",
    "tencentBlock_1791", "tencentBlock_1757", "tencentBlock_2145",
    "tencentBlock_1162", "tencentBlock_1894",
)


def discover_block_traces(root: str | Path = BLOCK_TRACE_ROOT) -> dict[str, Path]:
    """Map stem -> path for every oracleGeneral trace under ``root``.

    Looks one directory deep (``<root>/<corpus>/<file>``) and at ``root``
    itself. Files whose stem matches no corpus are skipped rather than
    guessed. A missing root gives an empty mapping, because the collection is
    gitignored and absent on a fresh checkout. Two files with the same stem
    (say, a ``.zst`` and its decompressed copy) raise instead of one silently
    shadowing the other.
    """
    root = Path(root)
    if not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted([*root.glob(f"*{ORACLE_MARKER}*"),
                        *root.glob(f"*/*{ORACLE_MARKER}*")]):
        if not path.is_file():
            continue
        stem = trace_stem(path)
        try:
            CORPUS_OF(stem)
        except ValueError:
            continue
        if stem in found:
            raise ValueError(f"two files for trace {stem!r}: {found[stem]} and {path}")
        found[stem] = path
    return dict(sorted(found.items()))

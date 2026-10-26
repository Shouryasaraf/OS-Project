"""oracleGeneral block-trace loader and the corpora table (plan §2.1, WP2).

Unit tests build tiny traces with ``struct.pack`` in a tempdir. The
real-data tests (h) and (j) run only when the gitignored collection is
present under ``data/block-traces``; (i) runs only when ``zstandard`` is
importable.
"""

import importlib.util
import shutil
import struct
import tempfile
import unittest
from itertools import chain
from pathlib import Path
from statistics import median
from unittest import mock

from adaptive_prefetch import trace
from adaptive_prefetch.corpora import (CORPORA, CORPUS_OF, PAPER_SUBSET, TWIN_GROUPS,
                                       discover_block_traces, trace_stem, twin_group)
from adaptive_prefetch.trace import (ORACLE_TAG_MASK, Request, iter_csv_chunks,
                                     iter_oracle_chunks, load_csv, oracle_profile,
                                     write_csv)

REPO = Path(__file__).resolve().parents[1]
BLOCK_TRACES = REPO / "data" / "block-traces"
HM_0 = BLOCK_TRACES / "msr" / "msr_hm_0.oracleGeneral.zst"
CHEETAH = BLOCK_TRACES / "fiu" / "fiu_cheetah.cs.fiu.edu-110108-113008.oracleGeneral.zst"
MSR_SAMPLE = REPO / "data" / "samples" / "msr-cambridge1-sample.csv"
HAVE_ZSTANDARD = importlib.util.find_spec("zstandard") is not None

RECORD = struct.Struct("<IQIq")


def write_oracle(path, records):
    """Write (ts, obj_id, obj_size) triples as oracleGeneral records."""
    with open(path, "wb") as handle:
        for ts, obj_id, obj_size in records:
            handle.write(RECORD.pack(ts, obj_id, obj_size, -1))
    return Path(path)


def byte_scan(count, ts=0, start=0, size=4096):
    """``count`` contiguous byte-unit records in one second."""
    return [(ts, start + i * size, size) for i in range(count)]


def mixed_byte_records(count):
    """Byte records with runs, jumps and rising time; contiguity is ~2/3."""
    out, offset = [], 1 << 20
    for i in range(count):
        if i % 3 == 0:
            offset = (i * 7919 % 5000 + 1) * 4096
        out.append((i // 10, offset, 4096))
        offset += 4096
    return out


def flatten(chunks):
    return list(chain.from_iterable(chunks))


class OracleLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # (a)
    def test_byte_records_are_one_request_each(self):
        # Contiguous same-second records are distinct requests outside FIU:
        # auto merging must leave them alone.
        path = write_oracle(self.dir / "msr_t.oracleGeneral",
                            byte_scan(3, ts=5, start=8192) + [(7, 1 << 30, 512)])
        requests = load_csv(path, "oracle")
        self.assertEqual(requests, [
            Request(0.0, 16, 8, "R", "msr_t"),
            Request(0.0, 24, 8, "R", "msr_t"),
            Request(0.0, 32, 8, "R", "msr_t"),
            Request(2000.0, (1 << 30) // 512, 1, "R", "msr_t"),
        ])
        self.assertTrue(all(r.service_ms is None and r.pattern is None for r in requests))
        profile = oracle_profile(path)
        self.assertEqual((profile["size_unit"], profile["merge"], profile["corpus"]),
                         ("bytes", "none", "msr"))

    # (b)
    def test_fiu_sector_pages_merge_into_requests(self):
        records = [(0, 0, 8), (0, 4096, 8), (0, 8192, 8),      # one 24-sector run
                   (1, 12288, 8),                              # contiguous, new second
                   *[(2, 16384 + i * 4096, 8) for i in range(129)]]  # 1032 sectors
        path = write_oracle(self.dir / "fiu_t-1.oracleGeneral", records)
        profile = oracle_profile(path)
        self.assertEqual((profile["size_unit"], profile["merge"], profile["corpus"]),
                         ("sectors", "pages", "fiu"))
        merged = load_csv(path, "oracle")
        self.assertEqual([(r.timestamp_ms, r.lba, r.size_blocks) for r in merged], [
            (0.0, 0, 24), (1000.0, 24, 8), (2000.0, 32, 1024), (2000.0, 1056, 8)])
        raw = flatten(iter_oracle_chunks(path, merge="none"))
        self.assertEqual(len(raw), len(records))
        uncapped = flatten(iter_oracle_chunks(path, merge="same_second",
                                              max_merge_sectors=None))
        self.assertEqual([r.size_blocks for r in uncapped], [24, 8, 1032])
        smaller = flatten(iter_oracle_chunks(path, max_merge_sectors=256))
        self.assertEqual([r.size_blocks for r in smaller], [24, 8, 256, 256, 256, 256, 8])
        self.assertEqual(sum(r.size_blocks for r in merged), 8 * len(records))

    def test_a_non_contiguous_page_or_a_record_over_the_cap_breaks_a_run(self):
        records = [(0, 0, 8), (0, 8192, 8), (0, 12288, 2048), (0, 12288 + 2048 * 512, 8)]
        path = write_oracle(self.dir / "fiu_gap.oracleGeneral", records)
        got = load_csv(path, "oracle")
        # A gap starts a new request; a single record above the cap is kept
        # whole, and the contiguous page after it cannot join it.
        self.assertEqual([(r.lba, r.size_blocks) for r in got],
                         [(0, 8), (16, 8), (24, 2048), (2072, 8)])

    def test_explicit_merge_applies_to_any_corpus(self):
        path = write_oracle(self.dir / "msr_m.oracleGeneral", byte_scan(4))
        self.assertEqual(len(load_csv(path, "oracle")), 4)
        for merge in ("pages", "same_second"):
            got = flatten(iter_oracle_chunks(path, merge=merge))
            self.assertEqual([(r.lba, r.size_blocks) for r in got], [(0, 32)])

    # (c)
    def test_volume_tag_is_masked_off(self):
        tag = 130 << 40
        path = write_oracle(self.dir / "alibabaBlock_13.oracleGeneral",
                            [(1577808000, tag + 4096, 4096), (1577808000, tag + 8192, 4096)])
        self.assertEqual(ORACLE_TAG_MASK, (1 << 40) - 1)
        requests = load_csv(path, "oracle")
        self.assertEqual([r.lba for r in requests], [8, 16])
        self.assertEqual(requests[0].timestamp_ms, 0.0)
        self.assertTrue(oracle_profile(path)["tagged"])
        self.assertEqual(oracle_profile(path)["corpus"], "alibaba")

    # (d)
    def test_truncated_file_raises_with_the_record_index(self):
        path = write_oracle(self.dir / "msr_trunc.oracleGeneral", byte_scan(3))
        with open(path, "ab") as handle:
            handle.write(b"\x00" * 10)
        with self.assertRaisesRegex(ValueError, r"truncated oracleGeneral record at index 3"):
            load_csv(path, "oracle")
        # Past the probe too, so the error comes from the streaming loop.
        big = write_oracle(self.dir / "msr_trunc2.oracleGeneral", mixed_byte_records(5000))
        with open(big, "ab") as handle:
            handle.write(b"\x01" * 23)
        with self.assertRaisesRegex(ValueError, r"index 5000"):
            list(iter_oracle_chunks(big, 1000))

    def test_empty_and_malformed_records_raise(self):
        empty = self.dir / "msr_empty.oracleGeneral"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            load_csv(empty, "oracle")
        unaligned = write_oracle(self.dir / "msr_u.oracleGeneral",
                                 byte_scan(3) + [(1, 4097, 4096)])
        with self.assertRaisesRegex(ValueError, r"record 3: offset 4097"):
            load_csv(unaligned, "oracle")
        zero = write_oracle(self.dir / "msr_z.oracleGeneral", byte_scan(3) + [(1, 1 << 20, 0)])
        with self.assertRaisesRegex(ValueError, "record 3"):
            load_csv(zero, "oracle")

    # (e)
    def test_decreasing_timestamp_raises(self):
        path = write_oracle(self.dir / "msr_back.oracleGeneral",
                            byte_scan(3, ts=10) + [(9, 1 << 20, 4096)])
        with self.assertRaisesRegex(ValueError, r"record 3: timestamp 9 s precedes 10 s"):
            load_csv(path, "oracle")

    def test_decreasing_timestamp_raises_across_a_chunk_boundary(self):
        # Records 0-999 fill the first chunk exactly; record 1000 goes back.
        records = [(10, i * 1_000_000 * 512, 4096) for i in range(1000)]
        records += [(5, (1000 + i) * 1_000_000 * 512, 4096) for i in range(1000)]
        records[1] = (10, records[0][1] + 4096, 4096)     # a little contiguity
        path = write_oracle(self.dir / "msr_edge.oracleGeneral", records)
        with self.assertRaisesRegex(ValueError, r"record 1000: timestamp 5 s"):
            list(iter_oracle_chunks(path, 1000))
        with self.assertRaisesRegex(ValueError, r"record 1000"):
            list(iter_csv_chunks(path, "oracle", 1000))

    def test_revised_stream_checks_order_across_a_chunk_boundary(self):
        # M10's cross-chunk gap: each chunk is sorted on its own, but the
        # second starts before the first ended.
        path = self.dir / "edge.revised"
        rows = [f"{10 + i * 0.001:.3f} RS {i * 8} 8 rand 0.1 0.01 0.0" for i in range(1000)]
        rows += [f"{10.5 + i * 0.001:.3f} RS {i * 8} 8 rand 0.1 0.01 0.0" for i in range(1000)]
        path.write_text("\n".join(rows) + "\n")
        with self.assertRaisesRegex(ValueError, "chunk boundary"):
            list(iter_csv_chunks(path, "revised", 1000))

    # (f)
    def test_streaming_and_load_csv_agree(self):
        path = write_oracle(self.dir / "msr_s.oracleGeneral", mixed_byte_records(2500))
        whole = load_csv(path, "oracle")
        self.assertEqual(len(whole), 2500)
        chunks = list(iter_oracle_chunks(path, 1000))
        self.assertEqual([len(c) for c in chunks], [1000, 1000, 500])
        self.assertEqual(flatten(chunks), whole)
        self.assertEqual(flatten(iter_csv_chunks(path, "oracle", 1000)), whole)
        self.assertEqual(flatten(iter_oracle_chunks(path, 1000, skip_requests=500,
                                                    max_requests=700)),
                         whole[500:1200])
        self.assertEqual(flatten(iter_oracle_chunks(path, max_requests=10)), whole[:10])
        self.assertEqual(flatten(iter_oracle_chunks(path, skip_requests=5000)), [])

    def test_a_merge_run_crossing_a_chunk_boundary_stays_one_request(self):
        # 999 isolated pages, then a 3-page run that straddles request 1000.
        records = [(i, (i * 64 + 1) * 4096, 8) for i in range(999)]
        records[1] = (1, records[0][1] + 4096, 8)
        records += [(999, 10 ** 9, 8), (999, 10 ** 9 + 4096, 8), (999, 10 ** 9 + 8192, 8)]
        records += [(1000 + i, (10 ** 6 + i * 64) * 4096, 8) for i in range(200)]
        path = write_oracle(self.dir / "fiu_cross.oracleGeneral", records)
        small = list(iter_oracle_chunks(path, 1000))
        whole = flatten(iter_oracle_chunks(path, 100_000))
        self.assertEqual(flatten(small), whole)
        self.assertEqual(small[0][-1].size_blocks, 24)
        self.assertEqual(sum(r.size_blocks for r in whole), 8 * len(records))

    # (g)
    def test_wrong_unit_override_is_refused(self):
        fiu = write_oracle(self.dir / "fiu_g.oracleGeneral",
                           [(0, i * 4096, 8) for i in range(50)])
        with self.assertRaisesRegex(ValueError, "size unit is wrong"):
            list(iter_oracle_chunks(fiu, size_unit="bytes"))
        msr = write_oracle(self.dir / "msr_g.oracleGeneral", mixed_byte_records(300))
        with self.assertRaisesRegex(ValueError, "size unit is wrong"):
            list(iter_oracle_chunks(msr, size_unit="sectors"))
        self.assertEqual(len(load_csv(msr, "oracle")), 300)

    def test_a_contiguity_tie_warns_instead_of_raising(self):
        scattered = write_oracle(self.dir / "msr_tie.oracleGeneral",
                                 [(0, (i * 7919 % 1000 + 1) << 22, 4096)
                                  for i in range(20)])
        with self.assertWarnsRegex(UserWarning, "cannot be confirmed"):
            self.assertEqual(len(load_csv(scattered, "oracle")), 20)

    def test_option_validation(self):
        path = write_oracle(self.dir / "msr_v.oracleGeneral", byte_scan(3))
        for kwargs in ({"size_unit": "kb"}, {"merge": "all"}, {"max_merge_sectors": 0},
                       {"max_requests": -1}, {"skip_requests": -1}):
            with self.assertRaises(ValueError):
                list(iter_oracle_chunks(path, **kwargs))
        with self.assertRaises(ValueError):
            list(iter_oracle_chunks(path, 10))
        with self.assertRaises(ValueError):
            oracle_profile(path, probe=1)
        self.assertEqual(flatten(iter_oracle_chunks(path, max_requests=0)), [])
        self.assertEqual({r.stream_id for r in flatten(iter_oracle_chunks(path, stream_id="x"))},
                         {"x"})

    def test_format_routing(self):
        path = write_oracle(self.dir / "msr_r.oracleGeneral", byte_scan(3))
        self.assertIn("oracle", trace.FORMATS)
        # oracle is never auto-detected, and the error says what to choose.
        with self.assertRaisesRegex(ValueError, "--format oracle"):
            load_csv(path)
        with self.assertRaisesRegex(ValueError, "format must be one of"):
            load_csv(path, "binary")
        with self.assertRaises(TypeError):
            list(iter_csv_chunks(path, "revised", 1000, merge="none"))
        self.assertEqual(flatten(iter_csv_chunks(path, "oracle", 1000, merge="pages")),
                         [Request(0.0, 0, 24, "R", "msr_r")])

    def test_csv_profiles_are_parsed_once_when_chunked(self):
        # M10: the old loop re-parsed the whole file twice per chunk.
        path = self.dir / "n.csv"
        write_csv(path, [Request(i, i * 8, 8) for i in range(2500)])
        with mock.patch.object(trace, "load_csv", wraps=trace.load_csv) as spy:
            chunks = list(iter_csv_chunks(path, "normalized", 1000))
        self.assertEqual(spy.call_count, 1)
        self.assertEqual([len(c) for c in chunks], [1000, 1000, 500])
        self.assertEqual(flatten(chunks), load_csv(path, "normalized"))

    # (i)
    @unittest.skipUnless(HAVE_ZSTANDARD, "zstandard is not installed")
    def test_zst_round_trip(self):
        import zstandard
        raw = write_oracle(self.dir / "fiu_z.oracleGeneral",
                           [(i // 3, i * 4096, 8) for i in range(3000)])
        packed = self.dir / "fiu_z.oracleGeneral.zst"
        packed.write_bytes(zstandard.ZstdCompressor().compress(raw.read_bytes()))
        self.assertEqual(trace_stem(packed), "fiu_z")
        self.assertEqual(load_csv(packed, "oracle"), load_csv(raw, "oracle"))
        self.assertEqual(len(load_csv(packed, "oracle")), 1000)
        self.assertEqual(flatten(iter_oracle_chunks(packed, max_requests=5)),
                         load_csv(raw, "oracle")[:5])

    @unittest.skipUnless(HAVE_ZSTANDARD and shutil.which("zstd"),
                         "needs zstandard to write and the zstd binary to read")
    def test_zstd_binary_fallback(self):
        import zstandard
        raw = write_oracle(self.dir / "msr_b.oracleGeneral", mixed_byte_records(1500))
        packed = self.dir / "msr_b.oracleGeneral.zst"
        packed.write_bytes(zstandard.ZstdCompressor().compress(raw.read_bytes()))
        with mock.patch.object(trace, "_zstd_module", return_value=None):
            self.assertEqual(load_csv(packed, "oracle"), load_csv(raw, "oracle"))
            self.assertEqual(len(flatten(iter_oracle_chunks(packed, max_requests=3))), 3)

    def test_zst_without_any_decompressor_raises(self):
        packed = self.dir / "msr_n.oracleGeneral.zst"
        packed.write_bytes(b"\x28\xb5\x2f\xfd")
        with mock.patch.object(trace, "_zstd_module", return_value=None), \
                mock.patch.object(trace.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "zstandard"):
                load_csv(packed, "oracle")

    # (h)
    @unittest.skipUnless(HM_0.is_file() and HAVE_ZSTANDARD, "msr_hm_0 trace not present")
    def test_hm_0_matches_the_committed_msr_sample(self):
        sample = load_csv(MSR_SAMPLE, "msr")
        oracle = flatten(iter_oracle_chunks(HM_0, max_requests=len(sample)))
        self.assertEqual(len(sample), 1000)
        self.assertEqual(len(oracle), 1000)
        self.assertEqual(sum(a.lba == b.lba for a, b in zip(sample, oracle)), 1000)
        profile = oracle_profile(HM_0)
        self.assertEqual((profile["size_unit"], profile["merge"], profile["tagged"]),
                         ("bytes", "none", False))

    # (j)
    @unittest.skipUnless(CHEETAH.is_file() and HAVE_ZSTANDARD, "fiu_cheetah trace not present")
    def test_fiu_cheetah_is_bytes_and_merges_to_multi_page_requests(self):
        profile = oracle_profile(CHEETAH)
        self.assertEqual((profile["size_unit"], profile["merge"]), ("bytes", "pages"))
        requests = flatten(iter_oracle_chunks(CHEETAH, max_requests=20_000))
        self.assertEqual(len(requests), 20_000)
        self.assertGreaterEqual(median(r.size_blocks for r in requests), 8)
        self.assertLessEqual(max(r.size_blocks for r in requests), 1024)


class CorporaTests(unittest.TestCase):
    def test_stems_and_corpora(self):
        self.assertEqual(trace_stem("data/block-traces/msr/msr_hm_0.oracleGeneral.zst"),
                         "msr_hm_0")
        self.assertEqual(trace_stem("fiu_webmail+online.cs.fiu.edu-110108-113008"
                                    ".oracleGeneral.zst"),
                         "fiu_webmail+online.cs.fiu.edu-110108-113008")
        self.assertEqual([CORPUS_OF(s) for s in ("msr_hm_0", "fiu_ikki-110108-112108",
                                                 "alibabaBlock_3", "tencentBlock_1162")],
                         ["msr", "fiu", "alibaba", "tencent"])
        with self.assertRaises(ValueError):
            CORPUS_OF("systor_17")

    def test_twin_groups_and_paper_subset(self):
        members = [stem for group in TWIN_GROUPS for stem in group]
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(twin_group("tencentBlock_1591"),
                         frozenset({"tencentBlock_1162", "tencentBlock_1591",
                                    "tencentBlock_1911"}))
        self.assertIsNone(twin_group("msr_hm_0"))
        self.assertEqual(len(PAPER_SUBSET), len(set(PAPER_SUBSET)))
        self.assertEqual({CORPUS_OF(s) for s in PAPER_SUBSET}, set(CORPORA))
        self.assertIn("alibabaBlock_3", PAPER_SUBSET)

    def test_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(discover_block_traces(root / "absent"), {})
            (root / "msr").mkdir()
            (root / "fiu").mkdir()
            write_oracle(root / "msr" / "msr_a.oracleGeneral.zst", [])
            write_oracle(root / "fiu" / "fiu_b-1.oracleGeneral", [])
            write_oracle(root / "msr" / "other.oracleGeneral", [])
            (root / "msr" / "notes.txt").write_text("x")
            found = discover_block_traces(root)
            self.assertEqual(list(found), ["fiu_b-1", "msr_a"])
            write_oracle(root / "msr" / "msr_a.oracleGeneral", [])
            with self.assertRaisesRegex(ValueError, "two files"):
                discover_block_traces(root)

    @unittest.skipUnless(BLOCK_TRACES.is_dir(), "block-trace collection not present")
    def test_paper_subset_is_present_locally(self):
        found = discover_block_traces(BLOCK_TRACES)
        self.assertEqual([s for s in PAPER_SUBSET if s not in found], [])
        for group in TWIN_GROUPS:
            self.assertTrue(group <= set(found), group)


if __name__ == "__main__":
    unittest.main()

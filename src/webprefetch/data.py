"""Validate RetailRocket events and build causal view records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

EXPECTED = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
EVENTS = {"view", "addtocart", "transaction"}
SESSION_GAP_MS = 30 * 60 * 1000
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset/events.csv?datasetVersionNumber=2"
EXPECTED_CSV_SHA256 = "3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7"
EXPECTED_ROWS = 2_756_101


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(raw: str | Path) -> Path:
    """Fetch the pinned public Kaggle file, preserving the archive for provenance."""
    raw = Path(raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    archive = raw.with_suffix(raw.suffix + ".zip")
    if raw.exists():
        if sha256(raw) != EXPECTED_CSV_SHA256:
            raise ValueError("existing raw CSV differs from the validated Kaggle version 2 file")
        return raw
    with urlopen(DATASET_URL, timeout=120) as response, archive.open("wb") as target:
        while block := response.read(1024 * 1024):
            target.write(block)
    with ZipFile(archive) as zipped:
        if zipped.namelist() != ["events.csv"]:
            raise ValueError(f"unexpected dataset archive contents: {zipped.namelist()}")
        with zipped.open("events.csv") as source, raw.open("wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
    if sha256(raw) != EXPECTED_CSV_SHA256:
        raw.unlink()
        raise ValueError("downloaded CSV checksum differs from validated Kaggle version 2 file")
    return raw


def partition_sessions(views: pd.DataFrame, cuts) -> pd.DataFrame:
    """Cut complete replay units at period boundaries, then rebuild causal context."""
    views = views.copy()
    views["split"] = np.searchsorted(cuts, views.timestamp.to_numpy(), side="right").astype("int8")
    # Original session IDs are globally unique; multiplying by four keeps each
    # partition segment distinct without depending on row traversal order.
    views["session_id"] = (views.session_id.astype("int64") * 4 + views.split).astype("int32")
    ordered = views.sort_values(["session_id", "timestamp", "source_row"], kind="mergesort")
    group = ordered.groupby("session_id", sort=False)
    ordered["position"] = group.cumcount().astype("int16")
    ordered["prev_item"] = group.itemid.shift(1).fillna(-1).astype("int32")
    ordered["prev2_item"] = group.itemid.shift(2).fillna(-1).astype("int32")
    ordered["next_item"] = group.itemid.shift(-1).fillna(-1).astype("int32")
    ordered["next_timestamp"] = group.timestamp.shift(-1).fillna(-1).astype("int64")
    ordered["gap_ms"] = group.timestamp.diff().fillna(0).clip(lower=0).astype("int64")
    ordered["eligible"] = ordered.next_item.ge(0)
    return ordered.sort_values(["timestamp", "source_row"], kind="mergesort")


def prepare(raw: str | Path, output_dir: str | Path, session_gap_minutes: int = 30) -> dict:
    """Prepare all real view events; never synthesize missing rows or labels."""
    raw = Path(raw)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not raw.is_file():
        raise FileNotFoundError(f"RetailRocket events file missing: {raw}")
    raw_digest = sha256(raw)
    if raw_digest != EXPECTED_CSV_SHA256:
        raise ValueError("raw CSV SHA-256 differs from validated Kaggle version 2")

    frame = pd.read_csv(raw, dtype={"timestamp": "int64", "visitorid": "int32",
                                     "event": "category", "itemid": "int32"})
    if list(frame.columns) != EXPECTED:
        raise ValueError(f"expected columns {EXPECTED}; found {list(frame.columns)}")
    if frame[EXPECTED[:4]].isna().any().any():
        raise ValueError("required event fields contain nulls")
    if not set(frame.event.astype(str).unique()).issubset(EVENTS):
        raise ValueError("unexpected event type")
    if (frame.timestamp < 0).any() or (frame.visitorid < 0).any() or (frame.itemid < 0).any():
        raise ValueError("negative timestamp, visitor ID, or item ID")
    invalid_transactions = frame.event.eq("transaction") & frame.transactionid.isna()
    if invalid_transactions.any():
        raise ValueError(f"{int(invalid_transactions.sum())} transactions lack IDs")
    unexpected_ids = frame.event.ne("transaction") & frame.transactionid.notna()
    if unexpected_ids.any():
        raise ValueError(f"{int(unexpected_ids.sum())} non-transaction events contain transaction IDs")

    raw_rows = len(frame)
    if raw_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} source rows; found {raw_rows:,}")
    duplicates = int(frame.duplicated(subset=EXPECTED).sum())
    frame = frame.drop_duplicates(subset=EXPECTED, keep="first").copy()
    frame["source_row"] = frame.index.astype("int32")
    event_counts = {str(k): int(v) for k, v in frame.event.value_counts().items()}
    frame.sort_values(["visitorid", "timestamp", "source_row"], kind="mergesort", inplace=True)
    gap = frame.groupby("visitorid", sort=False).timestamp.diff()
    if session_gap_minutes <= 0:
        raise ValueError("session gap must be positive")
    new_session = gap.isna() | gap.gt(session_gap_minutes * 60 * 1000)
    frame["session_id"] = new_session.cumsum().astype("int32")
    frame["cart_count"] = frame.event.eq("addtocart").astype("int16")
    frame["cart_count"] = frame.groupby("session_id", sort=False).cart_count.cumsum().astype("int16")

    views = frame.loc[frame.event.eq("view"),
                      ["timestamp", "visitorid", "itemid", "session_id", "source_row", "cart_count"]].copy()
    views.sort_values(["timestamp", "source_row"], kind="mergesort", inplace=True)
    cuts = np.quantile(views.timestamp.to_numpy(), [0.4, 0.6, 0.8]).astype("int64")
    if len(set(cuts)) != 3:
        raise ValueError("time split boundaries are not distinct")
    views = partition_sessions(views, cuts)

    path = output_dir / "views.csv.gz"
    views.to_csv(path, index=False, compression="gzip")
    lengths = views.groupby("session_id", sort=False).size()
    profile = {
        "source": "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
        "dataset_version": 2,
        "session_gap_minutes": session_gap_minutes,
        "raw_sha256": raw_digest,
        "raw_rows": raw_rows,
        "duplicate_rows_removed": duplicates,
        "event_counts_after_dedup": event_counts,
        "views": int(len(views)),
        "visitors": int(frame.visitorid.nunique()),
        "sessions": int(len(lengths)),
        "sessions_with_2plus_views": int(lengths.ge(2).sum()),
        "sessions_with_3plus_views": int(lengths.ge(3).sum()),
        "sessions_with_5plus_views": int(lengths.ge(5).sum()),
        "eligible_next_view_pairs": int(views.eligible.sum()),
        "session_length_quantiles": {str(q): float(lengths.quantile(q)) for q in [0.5, 0.75, 0.9, 0.95, 0.99]},
        "split_cutoffs_ms": [int(x) for x in cuts],
        "views_by_split": {str(k): int(v) for k, v in views.split.value_counts().sort_index().items()},
        "eligible_pairs_by_split": {str(k): int(v) for k, v in views.loc[views.eligible, "split"].value_counts().sort_index().items()},
        "prepared_path": str(path),
    }
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def load_views(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"timestamp", "itemid", "session_id", "split", "eligible", "next_item"}.issubset(frame):
        raise ValueError("prepared views schema is incomplete")
    return frame


def eda_markdown(profile: dict, views: pd.DataFrame) -> str:
    repeated = views.prev_item.eq(views.itemid) & views.prev_item.ge(0)
    eligible = views.loc[views.eligible]
    transitions = eligible.groupby("itemid").next_item.nunique()
    pair_counts = (views.loc[views.split.eq(0) & views.eligible]
                   .groupby(["itemid", "next_item"]).size())
    top_pair_share = (pair_counts.sort_values(ascending=False).head(100).sum() / pair_counts.sum()
                      if len(pair_counts) else 0.0)
    first = pd.to_datetime(views.timestamp.min(), unit="ms", utc=True)
    last = pd.to_datetime(views.timestamp.max(), unit="ms", utc=True)
    return "\n".join([
        "# RetailRocket exploratory analysis", "",
        f"Raw SHA-256: `{profile['raw_sha256']}`", "",
        f"Events: {profile['raw_rows']:,}; exact duplicates removed: {profile['duplicate_rows_removed']:,}.",
        "Event types after deduplication: " + ", ".join(
            f"{kind} {count:,}" for kind, count in profile["event_counts_after_dedup"].items()) + ".",
        f"Time range (UTC): {first} to {last}.",
        f"View requests: {profile['views']:,}; visitors: {profile['visitors']:,}; sessions: {profile['sessions']:,}.",
        f"Sessions with at least 2 / 3 / 5 views: {profile['sessions_with_2plus_views']:,} / "
        f"{profile['sessions_with_3plus_views']:,} / {profile['sessions_with_5plus_views']:,}.",
        f"Usable within-split next-view pairs: {profile['eligible_next_view_pairs']:,}.",
        f"Immediate repeated-item view share: {repeated.mean():.3%}.",
        f"Median distinct next items per source item: {transitions.median():.1f}.",
        f"The 100 most common history-split transitions account for {top_pair_share:.3%} of its eligible pairs.",
        "Session view-count quantiles: " + ", ".join(
            f"p{int(float(q)*100)}={value:.0f}" for q, value in profile["session_length_quantiles"].items()) + ".",
        "Chronological split view counts: " + ", ".join(
            f"split {part}={count:,}" for part, count in profile["views_by_split"].items()) + ".",
        "", "The data records page-view requests, not object bytes or measured network latency.",
        "Sessions use a 30-minute inactivity rule. The split is chronological by request timestamp.",
    ]) + "\n"

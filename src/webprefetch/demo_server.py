"""Read-only local API and bounded static export for the real-model replay."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from .data import load_views
from .model import FastScorer, load_artifacts
from .simulator import Cache, Counters, Policy


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class DemoData:
    def __init__(self, root: Path):
        self.root = root
        self.index, model = load_artifacts(root / "models")
        self.scorer = FastScorer(model)
        views = load_views(root / "data/processed/views.csv.gz")
        test = views.loc[views.split.eq(3)]
        columns = ("timestamp", "visitorid", "itemid", "position", "prev_item",
                   "gap_ms", "cart_count", "next_item")
        buckets: dict[int, list[dict]] = {}
        for values in test[["session_id", *columns]].itertuples(index=False, name=None):
            buckets.setdefault(int(values[0]), []).append(dict(zip(columns, values[1:])))
        self.sessions = {key: rows for key, rows in buckets.items() if len(rows) >= 2}
        self.ids = sorted(self.sessions)
        output = root / "outputs"
        self.summary = {name: _csv(output / f"{name}.csv") for name in
                        ("comparisons", "break_even", "break_even_curve", "cohorts",
                         "session_gap_sensitivity", "model_diagnostics")}
        self.summary["benchmark"] = json.loads((output / "benchmark.json").read_text())

    def listing(self, offset: int = 0, limit: int = 30, seed: int | None = None) -> dict:
        ids = self.ids if seed is None else random.Random(seed).sample(self.ids, min(limit, len(self.ids)))
        chosen = ids[max(0, offset): max(0, offset) + min(100, max(1, limit))]
        return {"total": len(self.ids), "sessions": [
            {"id": key, "views": len(self.sessions[key]),
             "visitor_token": hashlib.sha256(str(self.sessions[key][0]["visitorid"]).encode()).hexdigest()[:8]}
            for key in chosen]}

    def session(self, key: int) -> dict:
        if key not in self.sessions:
            raise ValueError("unknown held-out session")
        rows = self.sessions[key]
        return {"id": key, "source": "RetailRocket held-out test", "views": [
            {"item": int(row["itemid"]), "timestamp": int(row["timestamp"])} for row in rows]}

    def step(self, payload: dict) -> dict:
        capacity = int(payload.get("capacity", 8))
        if capacity not in (2, 4, 8, 16, 32):
            raise ValueError("capacity must be 2, 4, 8, 16, or 32")
        cost = float(payload.get("cost", .01))
        if not 0 <= cost <= 1:
            raise ValueError("cost must be between zero and one")
        name = str(payload.get("policy", "adaptive"))
        if name == "adaptive":
            policy = Policy(name, "adaptive", cost)
        elif name.startswith("fixed") and name[5:] in ("0", "1", "2", "4", "8"):
            policy = Policy(name, "fixed", int(name[5:]))
        elif name == "gate":
            policy = Policy(name, "gate", float(payload.get("threshold", .01)))
        else:
            raise ValueError("unknown policy")
        custom = payload.get("items")
        if custom is not None:
            if not isinstance(custom, list) or not 1 <= len(custom) <= 30:
                raise ValueError("items must be a list of 1 to 30 product IDs")
            items = [int(item) for item in custom]
            if any(item not in self.index.global_count for item in items):
                raise ValueError("custom items must be real product IDs from the history partition")
            rows = [dict(itemid=item, timestamp=0, position=i,
                         prev_item=items[i-1] if i else -1, gap_ms=0, cart_count=0,
                         next_item=items[i+1] if i + 1 < len(items) else -1)
                    for i, item in enumerate(items)]
            source = "hypothetical sequence of real product IDs; zero gaps and cart counts"
        else:
            key = int(payload["session_id"])
            if key not in self.sessions:
                raise ValueError("unknown held-out session")
            rows = self.sessions[key]
            source = "RetailRocket held-out test"
        target = int(payload.get("step", len(rows) - 1))
        if not 0 <= target < len(rows):
            raise ValueError("step outside session")
        counters = Counters()
        cache = Cache(capacity, counters)
        steps = []
        for i, row in enumerate(rows[:target + 1]):
            ranked = self.scorer.score(SimpleNamespace(**row), self.index)
            before = list(cache.blocks)
            hit = cache.demand(int(row["itemid"]))
            depth = policy.depth(ranked)
            old_issued = counters.issued
            old_evictions = counters.evicted_unused
            issued_items = []
            for item, _ in ranked[:depth]:
                prior = counters.issued
                cache.prefetch(item)
                if counters.issued > prior:
                    issued_items.append(item)
            next_item = int(row["next_item"])
            steps.append({"step": i, "timestamp": int(row["timestamp"]),
                          "item": int(row["itemid"]), "actual_next_item": next_item if next_item >= 0 else None,
                          "cache_before": before, "cache_after": list(cache.blocks),
                          "demand_hit": hit, "selected_depth": depth,
                          "candidates": [[item, round(score, 6)] for item, score in ranked[:8]],
                          "issued_items": issued_items, "issued_now": counters.issued - old_issued,
                          "unused_evicted_now": counters.evicted_unused - old_evictions,
                          "next_item_in_cache": next_item in cache.blocks if next_item >= 0 else None,
                          "running_hits": counters.hits, "running_issued": counters.issued,
                          "running_unused": counters.issued - counters.useful,
                          "running_utility": round((counters.hits - cost*counters.issued)/(i+1), 8)})
        return {"source": source, "policy": name, "capacity": capacity, "cost": cost,
                "total_views": len(rows), "steps": steps,
                "totals": {"views": counters.views, "hits": counters.hits,
                           "prefetch_hits": counters.prefetch_hits, "issued": counters.issued,
                           "useful": counters.useful, "unused": counters.issued - counters.useful,
                           "evicted_unused": counters.evicted_unused}}


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    data = DemoData(root)
    html = (root / "demo/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, value: object, content_type: str = "application/json") -> None:
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, html, "text/html")
                if url.path == "/api/sessions":
                    params = parse_qs(url.query)
                    return self._send(200, data.listing(int(params.get("offset", [0])[0]),
                                                        int(params.get("limit", [30])[0]),
                                                        int(params["seed"][0]) if "seed" in params else None))
                if url.path.startswith("/api/session/"):
                    return self._send(200, data.session(int(url.path.rsplit("/", 1)[-1])))
                if url.path == "/api/summary":
                    return self._send(200, data.summary)
                self._send(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError) as exc:
                self._send(400, {"error": str(exc)})

        def do_POST(self):
            if self.path != "/api/step":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 8192:
                    return self._send(413, {"error": "request too large"})
                payload = json.loads(self.rfile.read(length))
                self._send(200, data.step(payload))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Demo: http://{host}:{port}/ (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def export_demo(root: Path, limit: int = 20) -> Path:
    data = DemoData(root)
    chosen = data.listing(limit=limit, seed=42)["sessions"]
    artifact = {"sessions": chosen, "details": {str(row["id"]): data.session(row["id"])
                                                   for row in chosen},
                "summary": data.summary,
                "steps": {str(row["id"]): {name: data.step({"session_id": row["id"],
                    "step": len(data.sessions[row["id"]]) - 1, "policy": name,
                    "capacity": 8, "cost": .01}) for name in ("adaptive", "fixed0", "fixed1")}
                          for row in chosen}}
    target = root / "demo/export.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact), encoding="utf-8")
    return target

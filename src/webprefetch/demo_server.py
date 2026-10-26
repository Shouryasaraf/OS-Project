"""Read-only local API and bounded static export for the real-model replay.

The demo's "adaptive" policy is the **validation-selected** controller for
the chosen capacity and cost (P5), read from ``benchmark.json``. The pivot's
demo used ``penalty = cost`` instead, which at cost 0 behaves like fixed8;
that rule is still offered, labelled ``adaptive_cost`` ("adaptive (penalty =
cost)"). :func:`session_parity` checks the demo's own step loop against the
batch replay engine, session by session.
"""

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
from .paths import MODELS, OUTPUTS, PROCESSED
from .simulator import Cache, Counters, Policy, replay

CAPACITIES = (2, 4, 8, 16, 32)
#: Costs with a validation-selected penalty (benchmark.SCENARIO_COSTS).
SELECTED_COSTS = (0.0, 0.01, 0.05, 0.25)
SUMMARY_TABLES = ("comparisons", "capacity_transfer", "break_even", "break_even_curve",
                  "cohorts", "session_gap_sensitivity", "model_diagnostics")
PARITY_FIELDS = ("views", "hits", "prefetch_hits", "issued", "useful", "unused", "evicted_unused")
PARITY_FORMAT = "webprefetch.session_parity/2"


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def depth_histogram(steps: list[dict]) -> dict[str, int]:
    """``{depth: decisions}`` over a step list, keyed as the replay's depth_counts."""
    counts: dict[int, int] = {}
    for step in steps:
        counts[step["selected_depth"]] = counts.get(step["selected_depth"], 0) + 1
    return {str(depth): counts[depth] for depth in sorted(counts)}


def _cost_key(cost: float) -> str:
    return str(float(cost))


class DemoData:
    """Held-out test sessions, the frozen ranker and the run's summary tables.

    Every summary table is optional (a partial run may lack some); without a
    ``benchmark.json`` only the policies that need no selection work.
    """

    def __init__(self, root: Path, models: Path | None = None,
                 processed: Path | None = None, outputs: Path | None = None):
        self.root = root
        self.index, model = load_artifacts(models or root / MODELS)
        self.scorer = FastScorer(model)
        views = load_views((processed or root / PROCESSED) / "views.csv.gz")
        self.test = views.loc[views.split.eq(3)]
        columns = ("timestamp", "visitorid", "itemid", "position", "prev_item",
                   "gap_ms", "cart_count", "next_item")
        buckets: dict[int, list[dict]] = {}
        for values in self.test[["session_id", *columns]].itertuples(index=False, name=None):
            buckets.setdefault(int(values[0]), []).append(dict(zip(columns, values[1:])))
        self.sessions = {key: rows for key, rows in buckets.items() if len(rows) >= 2}
        self.ids = sorted(self.sessions)
        output = Path(outputs or root / OUTPUTS)
        self.summary = {name: _csv(output / f"{name}.csv") for name in SUMMARY_TABLES
                        if (output / f"{name}.csv").exists()}
        path = output / "benchmark.json"
        self.benchmark_path = path if path.exists() else None
        self.summary["benchmark"] = json.loads(path.read_text()) if path.exists() else None

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

    def selected_adaptive(self, capacity: int, cost: float) -> tuple[str, str]:
        """(policy name, where it came from) for the validated adaptive controller."""
        benchmark = self.summary.get("benchmark")
        if not benchmark:
            raise ValueError("no benchmark.json: the validated adaptive policy is unavailable; "
                             "use adaptive_cost")
        key = _cost_key(cost)
        if key not in {_cost_key(c) for c in SELECTED_COSTS}:
            raise ValueError("validated penalties exist only for costs 0, 0.01, 0.05 and 0.25; "
                             "use adaptive_cost (penalty = cost) for other costs")
        per_capacity = benchmark.get("validation_selection_per_capacity")
        if per_capacity is not None:
            # The demo scores with the logistic ranker, so only the logistic
            # selection applies; another scorer's penalty would be mislabelled.
            if "logistic" not in per_capacity:
                raise ValueError("benchmark.json has no logistic selection; the demo scores with "
                                 "the logistic ranker, so use adaptive_cost")
            chosen, source = per_capacity["logistic"][str(capacity)], f"validation, capacity {capacity}"
        else:
            # A pivot-layout benchmark selected at capacity 8 only.
            chosen, source = benchmark["validation_selection"], "validation, capacity 8 (transferred)"
        return chosen[key]["adaptive"], source

    def policy(self, name: str, capacity: int, cost: float, threshold: float = .01) -> tuple[Policy, str]:
        """The policy a UI name means, and a note on how it was chosen."""
        if name == "adaptive":
            chosen, source = self.selected_adaptive(capacity, cost)
            return Policy(chosen, "adaptive", float(chosen.rsplit("_", 1)[1])), source
        if name == "adaptive_cost":
            return Policy(name, "adaptive", cost), "penalty = cost (not validated)"
        if name.startswith("fixed") and name[5:] in ("0", "1", "2", "4", "8"):
            return Policy(name, "fixed", int(name[5:])), "fixed depth"
        if name == "gate":
            return Policy(name, "gate", threshold), "confidence gate (threshold from the UI)"
        raise ValueError("unknown policy")

    def step(self, payload: dict) -> dict:
        capacity = int(payload.get("capacity", 8))
        if capacity not in CAPACITIES:
            raise ValueError("capacity must be 2, 4, 8, 16, or 32")
        cost = float(payload.get("cost", .01))
        if not 0 <= cost <= 1:
            raise ValueError("cost must be between zero and one")
        name = str(payload.get("policy", "adaptive"))
        policy, selection_note = self.policy(name, capacity, cost, float(payload.get("threshold", .01)))
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
        return {"source": source, "policy": name, "resolved_policy": policy.name,
                "penalty": policy.parameter if policy.kind == "adaptive" else None,
                "selection": selection_note, "capacity": capacity, "cost": cost,
                "total_views": len(rows), "steps": steps,
                "totals": {"views": counters.views, "hits": counters.hits,
                           "prefetch_hits": counters.prefetch_hits, "issued": counters.issued,
                           "useful": counters.useful, "unused": counters.issued - counters.useful,
                           "evicted_unused": counters.evicted_unused,
                           "depth_counts": depth_histogram(steps)}}


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765, models: Path | None = None,
          processed: Path | None = None, outputs: Path | None = None) -> None:
    data = DemoData(root, models=models, processed=processed, outputs=outputs)
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


def export_demo(root: Path, limit: int = 20, models: Path | None = None,
                processed: Path | None = None, outputs: Path | None = None) -> Path:
    data = DemoData(root, models=models, processed=processed, outputs=outputs)
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


def session_parity(data: DemoData, session_ids, capacity: int = 8,
                   costs=SELECTED_COSTS, policy: str = "adaptive") -> list[dict]:
    """The batch replay's counters for each (session, cost), next to the demo's.

    The reference comes from :func:`webprefetch.simulator.replay` on the
    session's own rows: caches are session-local and the scorer is frozen,
    so that is exactly the session's share of a full-split replay. The demo
    side is :meth:`DemoData.step`, a separate loop. ``match`` is True when
    every counter and the depth histogram agree.
    """
    cases = []
    for session_id in session_ids:
        frame = data.test.loc[data.test.session_id.eq(session_id)]
        if frame.empty:
            raise ValueError(f"session {session_id} is not in the test split")
        for cost in costs:
            resolved, _ = data.policy(policy, capacity, cost)
            rows, _, _ = replay(frame, data.index, None, [resolved], capacity, max_views=0,
                                scorer=data.scorer)
            reference = {key: rows[0][key] for key in PARITY_FIELDS}
            reference["depth_counts"] = {str(k): v for k, v in sorted(rows[0]["depth_counts"].items())}
            demo = data.step({"session_id": int(session_id), "step": len(frame) - 1,
                              "capacity": capacity, "cost": cost, "policy": policy})["totals"]
            cases.append({"session_id": int(session_id), "policy": policy,
                          "resolved_policy": resolved.name, "capacity": capacity, "cost": cost,
                          **reference,
                          "match": all(demo[key] == reference[key] for key in PARITY_FIELDS)
                          and demo["depth_counts"] == reference["depth_counts"]})
    return cases


def run_parity(root: Path, target: Path, models: Path | None = None,
               processed: Path | None = None, outputs: Path | None = None,
               sessions: int = 20, capacity: int = 8, costs=SELECTED_COSTS) -> dict:
    """Write ``session_parity.json`` (P15) for the run in ``outputs``.

    The sessions are the static export's (a seed-42 sample of held-out
    sessions), not picked for agreement. The file names the benchmark.json
    it used, by SHA-256: a reference is only valid for the run whose
    validation selection produced it.
    """
    data = DemoData(root, models=models, processed=processed, outputs=outputs)
    if data.benchmark_path is None:
        raise ValueError("no benchmark.json in the outputs directory; run webprefetch benchmark first")
    chosen = [row["id"] for row in data.listing(limit=sessions, seed=42)["sessions"]]
    cases = session_parity(data, chosen, capacity, costs)
    digest = hashlib.sha256(data.benchmark_path.read_bytes()).hexdigest()
    record = {"format": PARITY_FORMAT,
              "selection_rule": f"{len(chosen)} held-out sessions sampled with seed 42 (the static export's)",
              "benchmark_json_sha256": digest, "capacity": capacity,
              "costs": [float(c) for c in costs], "policy": "adaptive (validation-selected)",
              "cases": cases, "mismatches": sum(not case["match"] for case in cases)}
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record

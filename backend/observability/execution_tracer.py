"""Execution Tracer — observability covers every stage, not just the
final response: routing decision, plan, per-node timings, confidence
and escalation history are all recorded per request.

Phase 1 keeps traces in memory. Phase 2: emit the same dict shape to
Prometheus / a tracing backend.
"""
from __future__ import annotations

import threading


class ExecutionTracer:
    def __init__(self):
        self._traces: dict[str, dict] = {}
        self._lock = threading.Lock()

    def record(self, trace: dict) -> None:
        with self._lock:
            self._traces[trace["request_id"]] = trace

    def get(self, request_id: str) -> dict | None:
        return self._traces.get(request_id)

    def list(self) -> list[dict]:
        with self._lock:
            return list(self._traces.values())

    def summary(self) -> dict:
        traces = self.list()
        if not traces:
            return {"requests": 0}
        n = len(traces)
        return {
            "requests": n,
            "avg_latency_ms": round(sum(t["metrics"]["latency_ms"] for t in traces) / n, 1),
            "avg_cost_usd": round(sum(t["metrics"]["cost_usd"] for t in traces) / n, 6),
            "avg_tokens": round(sum(t["metrics"]["tokens"] for t in traces) / n, 1),
            "escalation_rate": round(
                sum(1 for t in traces if t["metrics"]["escalations"] > 0) / n, 3),
            "strategies": {s: sum(1 for t in traces if t["strategy"] == s)
                           for s in {t["strategy"] for t in traces}},
        }

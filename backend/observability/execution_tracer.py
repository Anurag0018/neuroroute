"""Execution Tracer — observability covers every stage, not just the
final response: routing decision, plan, per-node timings, confidence
and escalation history are all recorded per request.

Traces are kept in memory (this class is the seam where a database
backend would slot in); aggregates are exposed as JSON for the
dashboards and in Prometheus exposition format for scraping.
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

    def prometheus_text(self) -> str:
        """Prometheus exposition format, derived from the recorded
        traces: per-strategy request counters plus token/cost totals
        and latency/escalation gauges."""
        traces = self.list()
        lines = [
            "# HELP neuroroute_requests_total Requests processed, by strategy.",
            "# TYPE neuroroute_requests_total counter",
        ]
        by_strategy: dict[str, int] = {}
        for t in traces:
            by_strategy[t["strategy"]] = by_strategy.get(t["strategy"], 0) + 1
        for strategy, count in sorted(by_strategy.items()):
            lines.append(
                f'neuroroute_requests_total{{strategy="{strategy}"}} {count}')
        lines += [
            "# HELP neuroroute_tokens_total Total tokens consumed.",
            "# TYPE neuroroute_tokens_total counter",
            f"neuroroute_tokens_total {sum(t['metrics']['tokens'] for t in traces)}",
            "# HELP neuroroute_cost_usd_total Total modelled cost in USD.",
            "# TYPE neuroroute_cost_usd_total counter",
            f"neuroroute_cost_usd_total {sum(t['metrics']['cost_usd'] for t in traces):.6f}",
            "# HELP neuroroute_escalations_total Requests that escalated at least once.",
            "# TYPE neuroroute_escalations_total counter",
            f"neuroroute_escalations_total {sum(1 for t in traces if t['metrics']['escalations'] > 0)}",
        ]
        if traces:
            n = len(traces)
            lines += [
                "# HELP neuroroute_avg_latency_ms Mean end-to-end latency.",
                "# TYPE neuroroute_avg_latency_ms gauge",
                f"neuroroute_avg_latency_ms {sum(t['metrics']['latency_ms'] for t in traces) / n:.1f}",
                "# HELP neuroroute_escalation_rate Fraction of requests that escalated.",
                "# TYPE neuroroute_escalation_rate gauge",
                f"neuroroute_escalation_rate {sum(1 for t in traces if t['metrics']['escalations'] > 0) / n:.3f}",
            ]
        return "\n".join(lines) + "\n"

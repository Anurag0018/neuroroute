"""NeuroRoute MCP server — the second interface layer.

The REST API serves applications; this MCP (Model Context
Protocol) server serves AI clients — Claude Desktop, Cursor,
other agents — so they can consume NeuroRoute's capabilities as
standard MCP tools. Both interfaces wrap the SAME pipeline and
tool registry: a request entering through MCP gets the identical
validated, budget-capped execution as one entering through HTTP.
The interface is pluggable; the execution core is shared.

Run (stdio transport):
    .venv/bin/python mcp_server.py

Client config (Claude Desktop / Cursor), see README:
    command: <repo>/.venv/bin/python
    args:    [<repo>/mcp_server.py]
    env:     {"NR_MODEL_MODE": "ollama"}   # optional

Needs the optional `mcp` package (pip install "mcp>=1.2,<2").
Environment variables (NR_MODEL_MODE, NR_DECISION_MODEL, ...)
configure the pipeline exactly as for the API server.
"""
from __future__ import annotations

import os
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from backend.execution.tools import run_tool
from backend.intelligence import complexity_estimator, input_analyzer
from backend.pipeline import NeuroRoutePipeline
from backend.planning.graph_validator import validate_plan
from backend.schemas import Budget

mcp = FastMCP("neuroroute")
_pipeline = NeuroRoutePipeline()
_retrieval = _pipeline.executor.node_executor.retrieval


@mcp.tool()
async def neuroroute_query(query: str,
                           attachments: list[str] | None = None,
                           session_id: str = "") -> dict:
    """Run a request through NeuroRoute's full adaptive pipeline:
    routing decision, validated execution graph, verification and
    confidence scoring. Returns the answer with its strategy,
    confidence and measured cost/latency."""
    result = await _pipeline.process(
        query, attachments or [], Budget(),
        session_id=session_id or None)
    return {
        "request_id": result["request_id"],
        "answer": result["answer"],
        "strategy": result["strategy"],
        "confidence_score": result["confidence"]["confidence_score"],
        "confidence_decision": result["confidence"]["decision"],
        "tokens": result["metrics"]["tokens"],
        "cost_usd": result["metrics"]["cost_usd"],
        "latency_ms": result["metrics"]["latency_ms"],
        "escalations": result["metrics"]["escalations"],
        "execution_graph": [
            {"node": n["node_id"], "kind": n["kind"],
             "tier": n["model_tier"], "latency_ms": n["latency_ms"]}
            for n in result["execution_graph"]],
    }


@mcp.tool()
def neuroroute_route_preview(query: str,
                             attachments: list[str] | None = None) -> dict:
    """Show the execution plan NeuroRoute *would* run for a request
    — strategy, graph nodes and estimated cost — without executing
    it. The plan is produced by the configured decision model and
    authorised by the same validator as live requests."""
    analysis = input_analyzer.analyze(query, attachments or [])
    complexity = complexity_estimator.estimate(analysis)
    plan = validate_plan(
        _pipeline.decision_model.propose_plan(analysis, complexity,
                                              Budget()),
        Budget())
    return {"strategy": plan.strategy, "rationale": plan.rationale,
            "estimated_cost_usd": plan.estimated_cost_usd,
            "nodes": [{"node": n.node_id, "kind": n.kind,
                       "tier": n.model_tier,
                       "depends_on": n.depends_on}
                      for n in plan.nodes]}


@mcp.tool()
def neuroroute_calculator(query: str) -> str:
    """Exact arithmetic and percentages via NeuroRoute's registered
    calculator tool (AST-evaluated, never free-form code)."""
    return run_tool("calculator", query)


@mcp.tool()
def neuroroute_datetime(query: str) -> str:
    """Date arithmetic via NeuroRoute's registered datetime tool:
    weekdays, days between dates, date offsets."""
    return run_tool("datetime", query)


@mcp.tool()
def neuroroute_web_search(query: str) -> str:
    """Web lookup via NeuroRoute's registered web_search tool
    (DuckDuckGo Instant Answers, no API key)."""
    return run_tool("web_search", query)


@mcp.tool()
def neuroroute_metrics() -> dict:
    """Aggregate metrics over all requests this server has routed:
    count, mean latency/cost/tokens, escalation rate, per-strategy
    counts."""
    return _pipeline.tracer.summary()


@mcp.tool()
def neuroroute_health() -> dict:
    """Server configuration: model mode, router/planner modes,
    embeddings, vector store and indexed documents."""
    return {
        "service": "neuroroute", "interface": "mcp",
        "model_mode": os.environ.get("NR_MODEL_MODE", "mock"),
        "decision_model": os.environ.get("NR_DECISION_MODEL", "rules"),
        "plan_model": os.environ.get("NR_PLAN_MODEL", "strategy"),
        "embeddings": _retrieval.embedder.name,
        "vector_store": _retrieval.store.name,
        "documents": _retrieval.documents(),
    }


if __name__ == "__main__":
    mcp.run()  # stdio transport

"""Node Executor — runs one DAG node against the real (mock) resources."""
from __future__ import annotations

from backend.models.llm_adapter import default_adapter
from backend.models.model_registry import NODE_COST_USD, NODE_LATENCY_MS
from backend.retrieval.retrieval_service import RetrievalService
from backend.schemas import NodeResult, PlanNode
from backend.validation.answer_verifier import verify


class NodeExecutor:
    def __init__(self, adapter=None,
                 retrieval: RetrievalService | None = None):
        self.adapter = adapter or default_adapter()
        self.retrieval = retrieval or RetrievalService()

    def execute(self, node: PlanNode, query: str,
                prior: dict[str, NodeResult]) -> NodeResult:
        if node.kind == "retrieve":
            docs = self.retrieval.search(query)
            return NodeResult(node.node_id, node.kind,
                              latency_ms=NODE_LATENCY_MS["retrieve"],
                              cost_usd=NODE_COST_USD["retrieve"], output=docs)

        docs = _docs_from(prior)

        if node.kind == "generate":
            gen = self.adapter.generate(node.model_tier, query, docs or None)
            return NodeResult(node.node_id, node.kind, latency_ms=gen.latency_ms,
                              tokens=gen.tokens, cost_usd=gen.cost_usd,
                              output=gen.text)

        if node.kind == "verify":
            answer = _answer_from(prior)
            result = verify(answer, docs or None)
            # The verifier model call is given the actual answer to check;
            # its cost/latency are counted in the node totals either way.
            gen = self.adapter.generate("verifier", answer or query)
            return NodeResult(node.node_id, node.kind, latency_ms=gen.latency_ms,
                              tokens=gen.tokens, cost_usd=gen.cost_usd,
                              output=result)

        raise ValueError(f"Unknown node kind: {node.kind}")  # validator prevents this


def _docs_from(prior: dict[str, NodeResult]) -> list[dict]:
    for r in prior.values():
        if r.kind == "retrieve" and isinstance(r.output, list):
            return r.output
    return []


def _answer_from(prior: dict[str, NodeResult]) -> str:
    for r in prior.values():
        if r.kind == "generate" and isinstance(r.output, str):
            return r.output
    return ""

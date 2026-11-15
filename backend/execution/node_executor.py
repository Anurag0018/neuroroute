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
                prior: dict[str, NodeResult],
                use_llm_verifier: bool = True) -> NodeResult:
        if node.kind == "retrieve":
            # A decomposed branch retrieves for its own sub-question.
            docs = self.retrieval.search(node.params.get("sub_question", query))
            return NodeResult(node.node_id, node.kind,
                              latency_ms=NODE_LATENCY_MS["retrieve"],
                              cost_usd=NODE_COST_USD["retrieve"], output=docs)

        docs = _docs_for(node, prior)

        if node.kind == "generate":
            if node.params.get("synthesize"):
                # Merge node: the dependency branches' answers become
                # the context a final model synthesises from.
                drafts = [prior[d].output for d in node.depends_on
                          if d in prior and isinstance(prior[d].output, str)]
                context = [{"title": f"Sub-answer {i + 1}", "text": t}
                           for i, t in enumerate(drafts)]
                gen = self.adapter.generate(node.model_tier, query,
                                            context or None)
            else:
                gen = self.adapter.generate(
                    node.model_tier,
                    node.params.get("sub_question", query), docs or None)
            return NodeResult(node.node_id, node.kind, latency_ms=gen.latency_ms,
                              tokens=gen.tokens, cost_usd=gen.cost_usd,
                              output=gen.text)

        if node.kind == "verify":
            answer = _answer_from(prior)
            result = verify(answer, docs or None)
            if not use_llm_verifier:
                # Evidence-only check: the coverage computation above IS
                # the verification (see answer_verifier) — no extra model
                # call is made, so nothing is charged for this node.
                return NodeResult(node.node_id, node.kind, output=result)
            # In-graph verifier (verified_large path): the verifier model
            # call is given the actual answer to check; its cost/latency
            # are counted in the node totals.
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


def _docs_for(node: PlanNode, prior: dict[str, NodeResult]) -> list[dict]:
    """Docs for one node: prefer the retrieve nodes it depends on (a
    decomposed branch must see its OWN retrieval, not a sibling's);
    fall back to any retrieval in the graph (Phase 1/2 templates)."""
    for dep in node.depends_on:
        r = prior.get(dep)
        if r is not None and r.kind == "retrieve" and isinstance(r.output, list):
            return r.output
    return _docs_from(prior)


def _answer_from(prior: dict[str, NodeResult]) -> str:
    # The LAST generate node in execution order is the final answer:
    # for decomposed plans that is the synthesis node, not a branch.
    for r in reversed(list(prior.values())):
        if r.kind == "generate" and isinstance(r.output, str):
            return r.output
    return ""

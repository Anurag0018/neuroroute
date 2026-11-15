"""Graph Generator — turns a strategy into a DAG of PlanNodes."""
from __future__ import annotations

from backend.models.model_registry import estimate_cost_usd
from backend.planning.decomposer import split_subquestions
from backend.schemas import (Budget, ComplexityEstimate, ExecutionPlan,
                             InputAnalysis, PlanNode)

# Rough per-path token estimates used for the pre-execution budget check.
_TIER_TOKENS = {"small": 300, "medium": 900, "large": 1800, "verifier": 500}


def estimate_nodes(nodes: list[PlanNode]) -> tuple[int, float]:
    """Estimated (tokens, cost USD) for a node list — shared by the
    strategy templates here and the Phase 3 LLM DAG planner, so every
    plan is priced the same way before the validator sees it."""
    tokens = sum(_TIER_TOKENS[n.model_tier] for n in nodes if n.model_tier)
    cost = round(sum(estimate_cost_usd(n.model_tier, _TIER_TOKENS[n.model_tier])
                     for n in nodes if n.model_tier), 6)
    return tokens, cost


def build_plan(strategy: str, analysis: InputAnalysis,
               complexity: ComplexityEstimate, budget: Budget) -> ExecutionPlan:
    nodes: list[PlanNode] = []

    if strategy == "direct_small":
        nodes = [PlanNode("generate", "generate", "small")]
        rationale = "Low complexity, no retrieval/reasoning signals: cheapest path."

    elif strategy == "rag_medium":
        nodes = [PlanNode("retrieve", "retrieve"),
                 PlanNode("generate", "generate", "medium", ["retrieve"])]
        rationale = "Moderate complexity or retrieval required: ground a medium model."

    elif strategy == "decomposed_research":
        subs = split_subquestions(analysis.query) or [analysis.query]
        nodes = []
        gen_ids = []
        for i, sq in enumerate(subs, 1):
            nodes.append(PlanNode(f"retrieve_{i}", "retrieve", None, [],
                                  {"sub_question": sq}))
            nodes.append(PlanNode(f"generate_{i}", "generate", "medium",
                                  [f"retrieve_{i}"], {"sub_question": sq}))
            gen_ids.append(f"generate_{i}")
        nodes.append(PlanNode("synthesize", "generate", "large", gen_ids,
                              {"synthesize": True}))
        rationale = (f"Multi-part request decomposed into {len(subs)} "
                     "sub-questions answered in parallel, then synthesised "
                     "by the large model.")

    elif strategy == "verified_large":
        if complexity.retrieval_required or analysis.modality == "document":
            # A document request retrieves from the named documents
            # themselves when they have been ingested (the retrieval
            # service falls back to the full index otherwise).
            retrieve_params = ({"sources": list(analysis.attachments)}
                               if analysis.modality == "document"
                               and analysis.attachments else {})
            nodes = [PlanNode("retrieve", "retrieve", None, [],
                              retrieve_params),
                     PlanNode("generate", "generate", "large", ["retrieve"]),
                     PlanNode("verify", "verify", "verifier", ["generate"])]
        else:
            nodes = [PlanNode("generate", "generate", "large"),
                     PlanNode("verify", "verify", "verifier", ["generate"])]
        rationale = "High complexity or high risk: large model plus independent verification."

    else:  # pragma: no cover - strategies are produced internally
        raise ValueError(f"Unknown strategy: {strategy}")

    est_tokens, est_cost = estimate_nodes(nodes)
    return ExecutionPlan(strategy=strategy, nodes=nodes,
                         estimated_tokens=est_tokens,
                         estimated_cost_usd=est_cost, rationale=rationale)

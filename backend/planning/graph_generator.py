"""Graph Generator — turns a strategy into a DAG of PlanNodes."""
from __future__ import annotations

from backend.models.model_registry import estimate_cost_usd
from backend.schemas import (Budget, ComplexityEstimate, ExecutionPlan,
                             InputAnalysis, PlanNode)

# Rough per-path token estimates used for the pre-execution budget check.
_TIER_TOKENS = {"small": 300, "medium": 900, "large": 1800, "verifier": 500}


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

    elif strategy == "verified_large":
        if complexity.retrieval_required or analysis.modality == "document":
            nodes = [PlanNode("retrieve", "retrieve"),
                     PlanNode("generate", "generate", "large", ["retrieve"]),
                     PlanNode("verify", "verify", "verifier", ["generate"])]
        else:
            nodes = [PlanNode("generate", "generate", "large"),
                     PlanNode("verify", "verify", "verifier", ["generate"])]
        rationale = "High complexity or high risk: large model plus independent verification."

    else:  # pragma: no cover - strategies are produced internally
        raise ValueError(f"Unknown strategy: {strategy}")

    est_tokens = sum(_TIER_TOKENS[n.model_tier] for n in nodes if n.model_tier)
    est_cost = round(sum(estimate_cost_usd(n.model_tier, _TIER_TOKENS[n.model_tier])
                         for n in nodes if n.model_tier), 6)
    return ExecutionPlan(strategy=strategy, nodes=nodes,
                         estimated_tokens=est_tokens,
                         estimated_cost_usd=est_cost, rationale=rationale)

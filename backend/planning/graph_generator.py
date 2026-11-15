"""Graph Generator — turns a strategy into a DAG of PlanNodes."""
from __future__ import annotations

from backend.execution.tools import detect_tools
from backend.models.model_registry import estimate_cost_usd
from backend.planning.decomposer import split_subquestions
from backend.schemas import (Budget, ComplexityEstimate, ExecutionPlan,
                             InputAnalysis, PlanNode)

# Rough per-path token estimates used for the pre-execution budget check.
_TIER_TOKENS = {"small": 300, "medium": 900, "large": 1800, "verifier": 500}

_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _image_attachment(analysis: InputAnalysis) -> str:
    for a in analysis.attachments:
        if a.lower().endswith(_IMG_EXT):
            return a
    return analysis.attachments[0] if analysis.attachments else ""


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

    elif strategy == "tool_assisted":
        tools = detect_tools(analysis.query) or ["calculator"]
        nodes = [PlanNode(f"tool_{t}", "tool", None, [], {"tool": t})
                 for t in tools]
        # The tool result is the answer's substance; the model only
        # phrases it — small suffices, except web snippets, which
        # take a medium model to synthesise safely.
        tier = "medium" if "web_search" in tools else "small"
        nodes.append(PlanNode("generate", "generate", tier,
                              [n.node_id for n in nodes]))
        rationale = (f"Registered tool(s) {', '.join(tools)} settle this "
                     f"exactly; a {tier} model phrases the result.")

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

    elif strategy == "vision_grounded":
        img = _image_attachment(analysis)
        nodes = [PlanNode("perceive", "perceive", None, [],
                          {"attachment": img}),
                 PlanNode("generate", "generate", "medium", ["perceive"])]
        rationale = ("Image input: the vision model perceives the image "
                     "(description + text transcription) and a medium "
                     "model answers grounded in that perception.")

    elif strategy == "verified_large":
        if analysis.modality == "image":
            # Verified path for an image keeps the perceive node —
            # escalating away from the image evidence would defeat
            # the verification.
            img = _image_attachment(analysis)
            nodes = [PlanNode("perceive", "perceive", None, [],
                              {"attachment": img}),
                     PlanNode("generate", "generate", "large", ["perceive"]),
                     PlanNode("verify", "verify", "verifier", ["generate"])]
        elif complexity.retrieval_required or analysis.modality == "document":
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

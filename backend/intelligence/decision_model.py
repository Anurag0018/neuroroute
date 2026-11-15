"""Decision-making model.

Its job is NOT to answer the question — it is to decide HOW the
question should be answered, and to propose an execution plan.

Phase 1 implements that contract with deterministic rules
(RuleBasedDecisionModel). Phase 2 adds LLMDecisionModel: a lightweight
LLM (the "small" tier by default — Jev / Laya / Qwen-class) proposes a
strategy as structured JSON behind the same `propose_plan()`
signature. Either way, the proposal is only a proposal:
planning/graph_validator.py must approve it before anything executes.
An LLM never gets to run arbitrary tools.

Safety design of the LLM path:
- The model may only CHOOSE among the three registered strategies;
  the plan itself is still built deterministically by build_plan().
- Deterministic guardrails outrank the model: high risk or document
  inputs are forced to verified_large no matter what the model says.
- Any failure — unreachable model, non-JSON output, unknown strategy —
  falls back to the rule-based model. The router can never take the
  system down.
- The decision call's tokens/latency are recorded on the model
  (last_usage) so the pipeline charges them to the request: routing
  overhead is part of NeuroRoute's measured cost, by design.

Enable with NR_DECISION_MODEL=llm (router tier: NR_DECISION_TIER,
default "small"). Default remains "rules" so benchmark results stay
reproducible; run the benchmark in both modes to compare routers.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

from backend.intelligence.routing_policy import ESCALATION_LADDER, choose_strategy
from backend.planning.graph_generator import build_plan
from backend.schemas import (Budget, ComplexityEstimate, ExecutionPlan,
                             InputAnalysis)

STRATEGIES = tuple(ESCALATION_LADDER)


class DecisionModel(Protocol):
    def propose_plan(self, analysis: InputAnalysis,
                     complexity: ComplexityEstimate,
                     budget: Budget) -> ExecutionPlan: ...


class RuleBasedDecisionModel:
    def propose_plan(self, analysis: InputAnalysis,
                     complexity: ComplexityEstimate,
                     budget: Budget) -> ExecutionPlan:
        strategy = choose_strategy(analysis, complexity)
        return build_plan(strategy, analysis, complexity, budget)


_DECISION_PROMPT = """You are the routing planner of an AI inference system. You do NOT answer the user's request. You choose how it should be answered.

Strategies:
- direct_small: small model only. For simple factual or short questions needing no external knowledge.
- rag_medium: retrieve relevant documents, then a medium model answers from them. For questions needing specific, external or researched knowledge.
- verified_large: large model plus independent verification. For complex reasoning, analysis, documents, or anything high-risk.

Respond with ONLY a JSON object, no other text:
{{"strategy": "<one of: direct_small, rag_medium, verified_large>", "reason": "<one short sentence>"}}

Examples:
Request: "What is the capital of France?" -> {{"strategy": "direct_small", "reason": "Simple factual question."}}
Request: "Explain how vector databases support semantic search" -> {{"strategy": "rag_medium", "reason": "Needs specific researched knowledge."}}
Request: "Analyse this contract for risky clauses" (attachment: contract.pdf) -> {{"strategy": "verified_large", "reason": "Document analysis with high stakes."}}

Now decide for this request:
Request: "{query}"
Task type: {task_type} | Modality: {modality} | Risk: {risk}
Complexity score: {score} | Retrieval required: {retrieval} | Reasoning required: {reasoning}
JSON:"""


def _parse_strategy(text: str) -> tuple[str, str] | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    strategy = data.get("strategy")
    if strategy not in STRATEGIES:
        return None
    return strategy, str(data.get("reason", ""))[:200]


class LLMDecisionModel:
    """LLM-proposed routing with deterministic guardrails + fallback."""

    def __init__(self, adapter=None, tier: str | None = None,
                 fallback: RuleBasedDecisionModel | None = None):
        if adapter is None:
            from backend.models.llm_adapter import default_adapter
            adapter = default_adapter()
        self.adapter = adapter
        self.tier = tier or os.environ.get("NR_DECISION_TIER", "small")
        self.fallback = fallback or RuleBasedDecisionModel()
        # Set by every propose_plan() call; the pipeline reads these to
        # charge the decision cost and to record how the plan was made.
        self.last_usage = None          # Generation | None
        self.last_source = "rules"      # llm | llm_guardrail_override | llm_fallback_rules
        self.last_reason = ""

    def propose_plan(self, analysis: InputAnalysis,
                     complexity: ComplexityEstimate,
                     budget: Budget) -> ExecutionPlan:
        self.last_usage = None
        forced = (analysis.risk_level == "high"
                  or analysis.modality == "document")
        try:
            prompt = _DECISION_PROMPT.format(
                query=analysis.query, task_type=analysis.task_type,
                modality=analysis.modality, risk=analysis.risk_level,
                score=complexity.complexity_score,
                retrieval=complexity.retrieval_required,
                reasoning=complexity.reasoning_required)
            gen = self.adapter.generate(self.tier, prompt)
            self.last_usage = gen
            parsed = _parse_strategy(gen.text)
        except Exception:
            parsed = None

        if parsed is None:
            self.last_source = "llm_fallback_rules"
            self.last_reason = "Decision model output unusable; rules applied."
            return self.fallback.propose_plan(analysis, complexity, budget)

        strategy, reason = parsed
        self.last_reason = reason
        if forced and strategy != "verified_large":
            self.last_source = "llm_guardrail_override"
            plan = build_plan("verified_large", analysis, complexity, budget)
            plan.rationale = (f"Guardrail override: high risk / document "
                              f"input forces verified_large (model proposed "
                              f"{strategy}: {reason})")
            return plan

        self.last_source = "llm"
        plan = build_plan(strategy, analysis, complexity, budget)
        plan.rationale = f"LLM decision ({self.tier}): {reason or strategy}"
        return plan


def default_decision_model():
    """Rules by default; LLM router when NR_DECISION_MODEL=llm."""
    if os.environ.get("NR_DECISION_MODEL", "rules").lower() == "llm":
        return LLMDecisionModel()
    return RuleBasedDecisionModel()

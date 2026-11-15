"""LLM DAG Planner — Phase 3: the model proposes the graph itself.

Phase 2's LLM router could only pick one of three fixed strategies.
Here a lightweight LLM proposes a full execution DAG as JSON — node
kinds, model tiers, dependencies, parallelism — and the backend
decides whether it may run:

- Deterministic guardrails still outrank the model (high-risk or
  document inputs get the verified_large template, untouched).
- The proposal must pass graph_validator unchanged: allowlisted node
  kinds, registered tiers, acyclic, within the caller's budget.
- Any failure — unreachable model, malformed JSON, a validator
  rejection — falls back to the rule-based strategy plan. Planning
  can never take the system down, and an LLM never executes a graph
  the backend has not authorised.
- The planning call's tokens/latency are exposed as last_usage so
  the pipeline charges them to the request, like the Phase 2 router.

Enable with NR_PLAN_MODEL=llm (planner tier: NR_DECISION_TIER,
default "small"). Default is the strategy templates, so benchmark
results stay reproducible.
"""
from __future__ import annotations

import json
import os
import re

from backend.intelligence.routing_policy import choose_strategy
from backend.models.llm_adapter import default_adapter
from backend.models.model_registry import REGISTRY
from backend.planning.graph_generator import build_plan, estimate_nodes
from backend.planning.graph_validator import (ALLOWED_NODE_KINDS,
                                                PlanRejected, validate_plan)
from backend.schemas import (Budget, ComplexityEstimate, ExecutionPlan,
                             InputAnalysis, PlanNode)

_DAG_PROMPT = """You are the execution planner of an AI inference system. You do NOT answer the request. You design the execution graph that will answer it, as cheaply as safely possible.

Node kinds:
- retrieve: fetch relevant documents (no model tier).
- generate: a model writes text. Tiers: small (cheap, simple), medium (grounded answers), large (deep reasoning, synthesis).
- verify: check a generate node's answer. Tier: verifier. Must depend on that generate node.
- tool: run one registered tool (no model tier). Its params must name the tool: {{"tool": "calculator"}} for arithmetic/percentages, {{"tool": "datetime"}} for date/weekday questions, {{"tool": "web_search"}} for explicit web lookups. A generate node depending on it phrases its exact result. Use a tool ONLY for a pure calculation, a date question, or an explicit search request.

Rules:
- Every graph needs at least one generate node, and exactly one FINAL generate node that no other generate depends on (that is the answer).
- Node ids are unique short strings. depends_on lists ids that must finish first. No cycles.
- At most 4 model calls (generate + verify nodes) and at most 6 nodes total.
- Parallel branches are free in wall-clock terms: independent sub-questions should be parallel generate nodes merged by a final large generate node.
- Use the cheapest graph that safely answers the request; retrieval before generation when external knowledge is needed.

Respond with ONLY a JSON object, no other text:
{{"nodes": [{{"id": "...", "kind": "retrieve|generate|verify|tool", "tier": "small|medium|large|verifier or null", "depends_on": ["..."], "params": {{}}}}], "reason": "<one short sentence>"}}

Examples:
Request: "What is the capital of France?" -> {{"nodes": [{{"id": "generate", "kind": "generate", "tier": "small", "depends_on": []}}], "reason": "Single simple fact."}}
Request: "What is 247 * 19 + 305?" -> {{"nodes": [{{"id": "calc", "kind": "tool", "tier": null, "depends_on": [], "params": {{"tool": "calculator"}}}}, {{"id": "generate", "kind": "generate", "tier": "small", "depends_on": ["calc"]}}], "reason": "Pure arithmetic: the calculator settles it exactly."}}
Request: "Explain how vector databases support semantic search" -> {{"nodes": [{{"id": "retrieve", "kind": "retrieve", "tier": null, "depends_on": []}}, {{"id": "generate", "kind": "generate", "tier": "medium", "depends_on": ["retrieve"]}}], "reason": "Grounded medium answer."}}
Request: "Compare embeddings and sparse retrieval, covering how each works" -> {{"nodes": [{{"id": "retrieve_1", "kind": "retrieve", "tier": null, "depends_on": []}}, {{"id": "generate_1", "kind": "generate", "tier": "medium", "depends_on": ["retrieve_1"]}}, {{"id": "retrieve_2", "kind": "retrieve", "tier": null, "depends_on": []}}, {{"id": "generate_2", "kind": "generate", "tier": "medium", "depends_on": ["retrieve_2"]}}, {{"id": "synthesize", "kind": "generate", "tier": "large", "depends_on": ["generate_1", "generate_2"]}}], "reason": "Two sub-topics in parallel, merged by synthesis."}}

Now design the graph for this request:
Request: "{query}"
Task type: {task_type} | Modality: {modality} | Risk: {risk}
Complexity score: {score} | Retrieval required: {retrieval} | Reasoning required: {reasoning}
JSON:"""


def _parse_nodes(text: str) -> list[PlanNode] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        raw = data["nodes"]
        if not isinstance(raw, list) or not (1 <= len(raw) <= 6):
            return None
        nodes = []
        for item in raw:
            kind = item["kind"]
            if kind not in ALLOWED_NODE_KINDS:
                return None
            tier = item.get("tier")
            if kind in ("generate", "verify") and tier not in REGISTRY:
                return None
            deps = item.get("depends_on") or []
            if not isinstance(deps, list):
                return None
            params = item.get("params") or {}
            if not isinstance(params, dict):
                return None
            nodes.append(PlanNode(str(item["id"]), kind, tier,
                                  [str(d) for d in deps], params))
        if not any(n.kind == "generate" for n in nodes):
            return None
        return nodes
    except Exception:
        return None


class LLMDagPlanner:
    """Same propose_plan() protocol as the decision models, so the
    pipeline can use it interchangeably (see default_decision_model)."""

    def __init__(self, adapter=None, tier: str | None = None):
        self.adapter = adapter or default_adapter()
        self.tier = tier or os.environ.get("NR_DECISION_TIER", "small")
        self.last_usage = None
        self.last_source = "llm_dag"
        self.last_reason = ""

    def propose_plan(self, analysis: InputAnalysis,
                     complexity: ComplexityEstimate,
                     budget: Budget) -> ExecutionPlan:
        self.last_usage = None
        # Guardrails outrank the planner, exactly as in Phase 2.
        if analysis.risk_level == "high" or analysis.modality == "document":
            self.last_source = "llm_dag_guardrail"
            return build_plan("verified_large", analysis, complexity, budget)

        prompt = _DAG_PROMPT.format(
            query=analysis.query, task_type=analysis.task_type,
            modality=analysis.modality, risk=analysis.risk_level,
            score=complexity.complexity_score,
            retrieval=complexity.retrieval_required,
            reasoning=complexity.reasoning_required)
        try:
            gen = self.adapter.generate(self.tier, prompt)
        except Exception:
            return self._fallback(analysis, complexity, budget)
        self.last_usage = gen

        nodes = _parse_nodes(gen.text)
        if nodes is None:
            return self._fallback(analysis, complexity, budget)
        tokens, cost = estimate_nodes(nodes)
        plan = ExecutionPlan(
            strategy="llm_dag", nodes=nodes, estimated_tokens=tokens,
            estimated_cost_usd=cost,
            rationale="LLM-proposed execution graph (validator-authorised).")
        try:
            return validate_plan(plan, budget)
        except PlanRejected:
            return self._fallback(analysis, complexity, budget)

    def _fallback(self, analysis: InputAnalysis,
                  complexity: ComplexityEstimate,
                  budget: Budget) -> ExecutionPlan:
        self.last_source = "llm_dag_fallback_rules"
        return build_plan(choose_strategy(analysis, complexity),
                          analysis, complexity, budget)

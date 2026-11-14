"""Graph Validator — the backend authorises every plan before it runs.

This is the safety boundary of NeuroRoute: a decision model (rule-based
today, an LLM in Phase 2) can only ever propose graphs that use
allowlisted node kinds, registered model tiers, an acyclic dependency
structure, and the caller's budget.
"""
from __future__ import annotations

from backend.models.model_registry import REGISTRY
from backend.schemas import Budget, ExecutionPlan

ALLOWED_NODE_KINDS = {"retrieve", "generate", "verify"}


class PlanRejected(Exception):
    pass


def validate_plan(plan: ExecutionPlan, budget: Budget) -> ExecutionPlan:
    ids = [n.node_id for n in plan.nodes]
    if len(ids) != len(set(ids)):
        raise PlanRejected("Duplicate node ids in plan")
    id_set = set(ids)

    model_calls = 0
    for node in plan.nodes:
        if node.kind not in ALLOWED_NODE_KINDS:
            raise PlanRejected(f"Node kind not allowlisted: {node.kind}")
        if node.kind in ("generate", "verify"):
            model_calls += 1
            if node.model_tier not in REGISTRY:
                raise PlanRejected(f"Unknown model tier: {node.model_tier}")
        for dep in node.depends_on:
            if dep not in id_set:
                raise PlanRejected(f"Node {node.node_id} depends on missing node {dep}")

    _assert_acyclic(plan)

    if model_calls > budget.max_model_calls:
        raise PlanRejected(
            f"Plan needs {model_calls} model calls, budget allows {budget.max_model_calls}")
    if plan.estimated_tokens > budget.max_total_tokens:
        raise PlanRejected(
            f"Plan estimates {plan.estimated_tokens} tokens, budget allows "
            f"{budget.max_total_tokens}")
    if plan.estimated_cost_usd > budget.max_api_cost_usd:
        raise PlanRejected(
            f"Plan estimates ${plan.estimated_cost_usd}, budget allows "
            f"${budget.max_api_cost_usd}")
    return plan


def _assert_acyclic(plan: ExecutionPlan) -> None:
    deps = {n.node_id: list(n.depends_on) for n in plan.nodes}
    state: dict[str, int] = {}  # 0=unvisited 1=in-stack 2=done

    def visit(node_id: str) -> None:
        state[node_id] = 1
        for dep in deps[node_id]:
            if state.get(dep, 0) == 1:
                raise PlanRejected(f"Cycle detected involving node {dep}")
            if state.get(dep, 0) == 0:
                visit(dep)
        state[node_id] = 2

    for node_id in deps:
        if state.get(node_id, 0) == 0:
            visit(node_id)

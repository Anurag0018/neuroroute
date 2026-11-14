"""Escalation Controller — manages insufficient confidence, with
hard limits: max escalations, remaining budget, and never re-running
the same failed strategy without a stronger path.
"""
from __future__ import annotations

from backend.intelligence.routing_policy import next_strategy
from backend.optimization.budget_manager import BudgetExceeded, check_can_afford
from backend.schemas import Budget, BudgetUsage, ConfidenceReport


def escalation_target(confidence: ConfidenceReport, current_strategy: str,
                      usage: BudgetUsage, budget: Budget) -> str | None:
    if confidence.decision != "escalate":
        return None
    if usage.escalations >= budget.max_escalations:
        return None
    target = next_strategy(current_strategy)
    if target is None:
        return None
    try:
        check_can_afford(usage, budget, extra_model_calls=1)
    except BudgetExceeded:
        return None
    return target

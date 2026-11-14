"""Resource & Budget Manager — budgets are assigned before execution
and enforced after every attempt. Estimated cost (plan time) and
actual cost (telemetry) are kept separate throughout NeuroRoute.
"""
from __future__ import annotations

from backend.schemas import Budget, BudgetUsage


class BudgetExceeded(Exception):
    pass


def check_can_afford(usage: BudgetUsage, budget: Budget,
                     extra_tokens: int = 0, extra_cost_usd: float = 0.0,
                     extra_model_calls: int = 0) -> None:
    if usage.tokens + extra_tokens > budget.max_total_tokens:
        raise BudgetExceeded("Token budget exhausted")
    if usage.cost_usd + extra_cost_usd > budget.max_api_cost_usd:
        raise BudgetExceeded("Cost budget exhausted")
    if usage.model_calls + extra_model_calls > budget.max_model_calls:
        raise BudgetExceeded("Model-call budget exhausted")
    if usage.latency_ms > budget.max_latency_ms:
        raise BudgetExceeded("Latency budget exhausted")


def remaining(usage: BudgetUsage, budget: Budget) -> dict:
    return {
        "tokens": budget.max_total_tokens - usage.tokens,
        "cost_usd": round(budget.max_api_cost_usd - usage.cost_usd, 6),
        "latency_ms": budget.max_latency_ms - usage.latency_ms,
        "model_calls": budget.max_model_calls - usage.model_calls,
        "escalations": budget.max_escalations - usage.escalations,
    }

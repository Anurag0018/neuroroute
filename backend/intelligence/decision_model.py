"""Decision-making model.

Its job is NOT to answer the question — it is to decide HOW the
question should be answered, and to propose an execution plan.

Phase 1 implements that contract with deterministic rules
(RuleBasedDecisionModel). Phase 2 swaps in Jev / Laya / a small LLM
behind the same `propose_plan()` signature. Either way, the proposal
is only a proposal: planning/graph_validator.py must approve it
before anything executes. An LLM never gets to run arbitrary tools.
"""
from __future__ import annotations

from typing import Protocol

from backend.intelligence.routing_policy import choose_strategy
from backend.planning.graph_generator import build_plan
from backend.schemas import (Budget, ComplexityEstimate, ExecutionPlan,
                             InputAnalysis)


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

"""Routing policy — the three Phase 1 execution paths.

Path A  direct_small    : small LLM only
Path B  rag_medium      : retrieval -> medium LLM
Path C  verified_large  : [retrieval] -> large LLM -> verification

Thresholds are starting values; calibrate them from benchmark
correctness data (see README, Benchmarking).
"""
from __future__ import annotations

from backend.schemas import ComplexityEstimate, InputAnalysis

PATH_A_MAX_COMPLEXITY = 0.35
PATH_C_MIN_COMPLEXITY = 0.75


def choose_strategy(analysis: InputAnalysis,
                    complexity: ComplexityEstimate) -> str:
    # Document inputs imply parsing + long context + grounding checks,
    # so they always take the verified path in Phase 1.
    if (analysis.risk_level == "high" or analysis.modality == "document"
            or complexity.complexity_score >= PATH_C_MIN_COMPLEXITY):
        return "verified_large"
    if complexity.complexity_score < PATH_A_MAX_COMPLEXITY and not (
            complexity.retrieval_required or complexity.reasoning_required):
        return "direct_small"
    return "rag_medium"


# Escalation ladder: each step is a strictly stronger path.
ESCALATION_LADDER = ["direct_small", "rag_medium", "verified_large"]


def next_strategy(strategy: str) -> str | None:
    i = ESCALATION_LADDER.index(strategy)
    return ESCALATION_LADDER[i + 1] if i + 1 < len(ESCALATION_LADDER) else None

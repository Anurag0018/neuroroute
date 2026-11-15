"""Routing policy — the three Phase 1 execution paths.

Path A  direct_small    : small LLM only
Path B  rag_medium      : retrieval -> medium LLM
Path C  verified_large  : [retrieval] -> large LLM -> verification

Thresholds are starting values; calibrate them from benchmark
correctness data (see README, Benchmarking).
"""
from __future__ import annotations

from backend.execution.tools import detect_tools
from backend.planning.decomposer import split_subquestions
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
    # Tool engine: a request a registered tool can settle exactly
    # (arithmetic, date arithmetic, an explicit web lookup) goes to
    # the tool path — deterministic ground truth beats any model's
    # mental arithmetic. Guardrails above still outrank it.
    if detect_tools(analysis.query):
        return "tool_assisted"
    # Phase 3: a genuinely multi-part request is decomposed into
    # parallel sub-question branches plus a synthesis node — unless a
    # guardrail above already claimed it.
    if len(split_subquestions(analysis.query)) >= 2:
        return "decomposed_research"
    if complexity.complexity_score < PATH_A_MAX_COMPLEXITY and not (
            complexity.retrieval_required or complexity.reasoning_required):
        return "direct_small"
    return "rag_medium"


# Escalation ladder: each step is a strictly stronger path.
ESCALATION_LADDER = ["direct_small", "rag_medium", "verified_large"]

# Every registered strategy (Phase 3 adds decomposed_research, which
# is a peer of rag_medium, not a rung above it; the tool engine adds
# tool_assisted, a peer of direct_small whose evidence is a tool
# result rather than the model's own knowledge).
ALL_STRATEGIES = ["direct_small", "rag_medium", "decomposed_research",
                  "verified_large", "tool_assisted"]

# Escalation moves to a strictly stronger path. decomposed_research
# escalates to verified_large (a failed synthesis needs verification,
# not another decomposition); rag_medium does NOT escalate sideways
# into decomposition. A tool path whose result did not ground the
# answer escalates to rag_medium (retrieval may still settle it).
_ESCALATION_NEXT = {"direct_small": "rag_medium",
                    "rag_medium": "verified_large",
                    "decomposed_research": "verified_large",
                    "tool_assisted": "rag_medium"}


def next_strategy(strategy: str) -> str | None:
    return _ESCALATION_NEXT.get(strategy)

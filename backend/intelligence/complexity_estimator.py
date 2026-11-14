"""Complexity Estimator — hybrid, explainable scoring.

score = 0.25*length + 0.25*task_base + 0.20*reasoning + 0.15*retrieval
        + 0.15*risk

Every component is recorded in `signals` so a routing decision can be
explained and, later, the weights can be calibrated against benchmark
outcomes instead of being hand-tuned forever.
"""
from __future__ import annotations

from backend.schemas import ComplexityEstimate, InputAnalysis

_TASK_BASE = {"general_qa": 0.15, "summarization": 0.45, "math": 0.55,
              "coding": 0.60, "research": 0.75}
_REASONING_WORDS = ("why", "how does", "explain", "compare", "analyse", "analyze",
                    "implications", "trade-off", "tradeoff", "step by step")
_RETRIEVAL_WORDS = ("latest", "current", "research", "paper", "study", "source",
                    "cite", "evidence", "news", "according to")
_RISK_SCORE = {"low": 0.1, "medium": 0.5, "high": 1.0}


def estimate(analysis: InputAnalysis) -> ComplexityEstimate:
    q = analysis.query.lower()
    words = max(len(analysis.query.split()), 1)

    length_signal = min(words / 120.0, 1.0)
    task_signal = _TASK_BASE.get(analysis.task_type, 0.2)
    reasoning = any(w in q for w in _REASONING_WORDS) or analysis.task_type in (
        "research", "coding", "math")
    retrieval = (any(w in q for w in _RETRIEVAL_WORDS)
                 or analysis.modality == "document"
                 or analysis.task_type == "research")
    risk_signal = _RISK_SCORE[analysis.risk_level]

    score = (0.25 * length_signal + 0.25 * task_signal
             + 0.20 * (1.0 if reasoning else 0.0)
             + 0.15 * (1.0 if retrieval else 0.0)
             + 0.15 * risk_signal)
    score = round(min(max(score, 0.0), 1.0), 3)

    return ComplexityEstimate(
        complexity_score=score,
        reasoning_required=reasoning,
        retrieval_required=retrieval,
        verification_required=analysis.risk_level == "high" or score >= 0.7,
        signals={
            "length": round(length_signal, 3),
            "task_base": task_signal,
            "reasoning": reasoning,
            "retrieval": retrieval,
            "risk": risk_signal,
        },
    )

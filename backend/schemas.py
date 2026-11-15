"""Shared data structures for NeuroRoute.

Internal modules use dataclasses; the API layer (main.py) converts
to/from these for the HTTP boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Budget:
    max_total_tokens: int = 5000
    max_latency_ms: int = 10_000
    max_api_cost_usd: float = 0.05
    max_model_calls: int = 4
    max_escalations: int = 1


@dataclass
class BudgetUsage:
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    model_calls: int = 0
    escalations: int = 0

    def add(self, tokens: int = 0, cost_usd: float = 0.0, latency_ms: int = 0,
            model_calls: int = 0) -> None:
        self.tokens += tokens
        self.cost_usd += cost_usd
        self.latency_ms += latency_ms
        self.model_calls += model_calls


@dataclass
class InputAnalysis:
    request_id: str
    query: str
    modality: str = "text"          # text | document | image | audio
    task_type: str = "general_qa"   # summarization | coding | research | math | general_qa
    risk_level: str = "low"         # low | medium | high
    attachments: list[str] = field(default_factory=list)


@dataclass
class ComplexityEstimate:
    complexity_score: float         # 0..1, formula in complexity_estimator.py
    reasoning_required: bool = False
    retrieval_required: bool = False
    verification_required: bool = False
    signals: dict = field(default_factory=dict)


@dataclass
class PlanNode:
    node_id: str
    kind: str                       # retrieve | generate | verify  (allowlist)
    model_tier: str | None = None   # small | medium | large | verifier
    depends_on: list[str] = field(default_factory=list)
    # Phase 3: per-node payload, e.g. {"sub_question": "..."} for a
    # decomposed branch, or {"synthesize": True} for a merge node.
    params: dict = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    strategy: str                   # direct_small | rag_medium | verified_large
    nodes: list[PlanNode] = field(default_factory=list)
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    rationale: str = ""


@dataclass
class NodeResult:
    node_id: str
    kind: str
    status: str = "ok"              # ok | failed
    latency_ms: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    output: object = None


@dataclass
class ConfidenceReport:
    confidence_score: float
    evidence_coverage: float
    validation_passed: bool
    retrieval_relevance: float
    model_agreement: float
    decision: str                   # return | verify | escalate

"""Phase 2 — LLM decision model.

The LLM proposes a strategy as JSON; deterministic guardrails and the
rule-based fallback fence it in. These tests use stub adapters, so no
real model is needed.
"""
from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.decision_model import (LLMDecisionModel,
                                                 RuleBasedDecisionModel)
from backend.models.llm_adapter import Generation
from backend.pipeline import NeuroRoutePipeline
from backend.schemas import Budget


class StubAdapter:
    """Returns a canned 'model' reply; records the tier it was asked as."""

    def __init__(self, text: str, tokens: int = 42):
        self.text = text
        self.tokens = tokens
        self.seen_tiers = []

    def generate(self, tier, query, context_docs=None):
        self.seen_tiers.append(tier)
        return Generation(self.text, self.tokens, 0.0, 25)


class BoomAdapter:
    def generate(self, tier, query, context_docs=None):
        raise RuntimeError("model unreachable")


def _inputs(query, attachments=None):
    analysis = input_analyzer.analyze(query, attachments)
    return analysis, complexity_estimator.estimate(analysis)


def test_llm_strategy_is_used_and_plan_built():
    model = LLMDecisionModel(adapter=StubAdapter(
        '{"strategy": "rag_medium", "reason": "Needs researched knowledge."}'))
    analysis, complexity = _inputs("Explain how vector databases work")
    plan = model.propose_plan(analysis, complexity, Budget())
    assert plan.strategy == "rag_medium"
    assert [n.kind for n in plan.nodes] == ["retrieve", "generate"]
    assert model.last_source == "llm"
    assert model.last_usage.tokens == 42


def test_llm_may_upgrade_beyond_rules():
    # Rules would send this trivial question to the small model; a valid
    # LLM choice to escalate the path stands (within the guardrails).
    model = LLMDecisionModel(adapter=StubAdapter(
        '{"strategy": "verified_large", "reason": "Wants certainty."}'))
    analysis, complexity = _inputs("What is Python?")
    plan = model.propose_plan(analysis, complexity, Budget())
    assert plan.strategy == "verified_large"
    assert model.last_source == "llm"


def test_invalid_json_falls_back_to_rules():
    model = LLMDecisionModel(adapter=StubAdapter("I think small is fine!"))
    analysis, complexity = _inputs("What is Python?")
    plan = model.propose_plan(analysis, complexity, Budget())
    expected = RuleBasedDecisionModel().propose_plan(analysis, complexity, Budget())
    assert plan.strategy == expected.strategy == "direct_small"
    assert model.last_source == "llm_fallback_rules"


def test_unknown_strategy_falls_back_to_rules():
    model = LLMDecisionModel(adapter=StubAdapter(
        '{"strategy": "turbo_max", "reason": "?"}'))
    analysis, complexity = _inputs("What is Python?")
    plan = model.propose_plan(analysis, complexity, Budget())
    assert plan.strategy == "direct_small"
    assert model.last_source == "llm_fallback_rules"


def test_unreachable_model_falls_back_to_rules():
    model = LLMDecisionModel(adapter=BoomAdapter())
    analysis, complexity = _inputs("What is Python?")
    plan = model.propose_plan(analysis, complexity, Budget())
    assert plan.strategy == "direct_small"
    assert model.last_source == "llm_fallback_rules"
    assert model.last_usage is None


def test_guardrail_high_risk_overrides_model():
    model = LLMDecisionModel(adapter=StubAdapter(
        '{"strategy": "direct_small", "reason": "Looks simple."}'))
    analysis, complexity = _inputs(
        "I have chest pain, what medicine should I take?")
    assert analysis.risk_level == "high"
    plan = model.propose_plan(analysis, complexity, Budget())
    assert plan.strategy == "verified_large"
    assert model.last_source == "llm_guardrail_override"


def test_guardrail_document_overrides_model():
    model = LLMDecisionModel(adapter=StubAdapter(
        '{"strategy": "rag_medium", "reason": "Research task."}'))
    analysis, complexity = _inputs("Summarise this paper", ["paper.pdf"])
    assert analysis.modality == "document"
    plan = model.propose_plan(analysis, complexity, Budget())
    assert plan.strategy == "verified_large"
    assert model.last_source == "llm_guardrail_override"


def test_pipeline_charges_decision_cost_and_records_source():
    stub = StubAdapter('{"strategy": "direct_small", "reason": "Simple."}',
                       tokens=42)
    pipe = NeuroRoutePipeline(decision_model=LLMDecisionModel(adapter=stub))
    trace = pipe.process_sync("What is Python?")
    assert trace["decision"]["source"] == "llm"
    assert trace["decision"]["tokens"] == 42
    # Decision tokens are inside the request totals, not free.
    assert trace["metrics"]["tokens"] >= 42
    assert trace["metrics"]["model_calls"] >= 2  # decision + generate

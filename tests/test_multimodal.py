"""Multimodal engine: perception modes, validator authorisation,
image routing (incl. the high-risk guardrail keeping its perceive
node), and end-to-end execution in mock mode."""
import asyncio
import json
from pathlib import Path

import pytest

from backend.execution.node_executor import NodeExecutor
from backend.execution.perception import (PerceptionError, VisionPerceiver,
                                          resolve_attachment)
from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.decision_model import RuleBasedDecisionModel
from backend.intelligence.routing_policy import choose_strategy
from backend.pipeline import NeuroRoutePipeline
from backend.planning.graph_validator import PlanRejected, validate_plan
from backend.schemas import Budget, ExecutionPlan, PlanNode

CHART = "benchmark/assets/chart.png"


def _route(query, attachments):
    a = input_analyzer.analyze(query, attachments)
    return choose_strategy(a, complexity_estimator.estimate(a))


def test_image_query_routes_to_vision_grounded():
    assert _route("What does this chart show?", [CHART]) == "vision_grounded"
    assert _route("Transcribe this screenshot", ["shot.png"]) \
        == "vision_grounded"


def test_high_risk_image_stays_verified_and_keeps_perceive():
    a = input_analyzer.analyze("What medicine should I take for this rash?",
                               [CHART])
    plan = RuleBasedDecisionModel().propose_plan(
        a, complexity_estimator.estimate(a), Budget())
    assert plan.strategy == "verified_large"
    kinds = [n.kind for n in plan.nodes]
    assert kinds == ["perceive", "generate", "verify"]


def test_text_routing_unchanged_by_modality_split():
    assert _route("What is Python?", []) == "direct_small"


def test_validator_perceive_rules():
    ok = ExecutionPlan("vision_grounded", [
        PlanNode("perceive", "perceive", None, [], {"attachment": "a.png"}),
        PlanNode("generate", "generate", "medium", ["perceive"])],
        900, 0.001, "t")
    assert validate_plan(ok, Budget()) is ok
    no_attachment = ExecutionPlan("vision_grounded", [
        PlanNode("perceive", "perceive"), 
        PlanNode("generate", "generate", "medium", ["perceive"])],
        900, 0.001, "t")
    with pytest.raises(PlanRejected):
        validate_plan(no_attachment, Budget())
    with_tier = ExecutionPlan("vision_grounded", [
        PlanNode("perceive", "perceive", "medium", [], {"attachment": "a.png"}),
        PlanNode("generate", "generate", "medium", ["perceive"])],
        900, 0.001, "t")
    with pytest.raises(PlanRejected):
        validate_plan(with_tier, Budget())


def test_mock_perceiver_is_labelled_placeholder():
    gen = VisionPerceiver(mode="mock").perceive(CHART, "what is this?")
    assert "Mock perception" in gen.text and gen.tokens > 0


def test_ollama_perceiver_missing_file_fails_cleanly():
    with pytest.raises(PerceptionError):
        VisionPerceiver(mode="ollama").perceive("does-not-exist.png")


def test_resolve_attachment_finds_repo_asset():
    assert resolve_attachment(CHART) is not None
    assert resolve_attachment("nope.png") is None


def test_perceive_node_emits_evidence_docs():
    res = NodeExecutor().execute(
        PlanNode("perceive", "perceive", None, [], {"attachment": CHART}),
        "What does this chart show?", {})
    assert res.status == "ok"
    assert res.output[0]["source"] == f"image:{CHART}"
    assert res.output[0]["relevance"] == 1.0


def test_pipeline_vision_end_to_end():
    result = asyncio.run(NeuroRoutePipeline().process(
        "What does this chart show?", [CHART]))
    assert result["strategy"] == "vision_grounded"
    kinds = [n["kind"] for n in result["execution_graph"]]
    assert kinds == ["perceive", "generate"] or "perceive" in kinds
    assert result["answer"]


def test_vision_dataset_routing():
    data = json.loads(Path("benchmark/queries_vision.json").read_text())
    pipeline = NeuroRoutePipeline()
    for item in data["queries"]:
        result = asyncio.run(
            pipeline.process(item["query"], item.get("attachments", [])))
        history = result.get("escalation_history") or []
        initial = history[0]["from"] if history else result["strategy"]
        assert initial == item["expected_strategy"], item["id"]

"""Audio path: transcriber modes, validator authorisation, audio
routing (incl. the guardrail keeping its transcribe node), mock
end-to-end, and a real faster-whisper round-trip when installed."""
import asyncio
import json
from pathlib import Path

import pytest

from backend.execution.node_executor import NodeExecutor
from backend.execution.transcription import (AudioTranscriber,
                                             TranscriptionError)
from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.decision_model import RuleBasedDecisionModel
from backend.intelligence.routing_policy import choose_strategy
from backend.pipeline import NeuroRoutePipeline
from backend.planning.graph_validator import PlanRejected, validate_plan
from backend.schemas import Budget, ExecutionPlan, PlanNode

WAV = "benchmark/assets/voice_note.wav"


def _route(query, attachments):
    a = input_analyzer.analyze(query, attachments)
    return choose_strategy(a, complexity_estimator.estimate(a))


def test_audio_query_routes_to_audio_grounded():
    assert _route("Transcribe this voice note", [WAV]) == "audio_grounded"
    assert _route("Summarize the meeting", ["call.mp3"]) == "audio_grounded"


def test_high_risk_audio_stays_verified_and_keeps_transcribe():
    a = input_analyzer.analyze("What medicine should I take for this?",
                               [WAV])
    plan = RuleBasedDecisionModel().propose_plan(
        a, complexity_estimator.estimate(a), Budget())
    assert plan.strategy == "verified_large"
    assert [n.kind for n in plan.nodes] == ["transcribe", "generate",
                                            "verify"]


def test_validator_transcribe_rules():
    ok = ExecutionPlan("audio_grounded", [
        PlanNode("transcribe", "transcribe", None, [], {"attachment": "a.wav"}),
        PlanNode("generate", "generate", "medium", ["transcribe"])],
        900, 0.001, "t")
    assert validate_plan(ok, Budget()) is ok
    bad = ExecutionPlan("audio_grounded", [
        PlanNode("transcribe", "transcribe", "small", [],
                 {"attachment": "a.wav"}),
        PlanNode("generate", "generate", "medium", ["transcribe"])],
        900, 0.001, "t")
    with pytest.raises(PlanRejected):
        validate_plan(bad, Budget())


def test_mock_transcriber_is_labelled_placeholder():
    gen = AudioTranscriber(mode="mock").transcribe(WAV)
    assert "Mock transcript" in gen.text and gen.tokens > 0


def test_whisper_mode_missing_file_fails_cleanly():
    with pytest.raises(TranscriptionError):
        AudioTranscriber(mode="whisper").transcribe("does-not-exist.wav")


def test_transcribe_node_emits_evidence_docs():
    res = NodeExecutor().execute(
        PlanNode("transcribe", "transcribe", None, [], {"attachment": WAV}),
        "Transcribe this", {})
    assert res.status == "ok"
    assert res.output[0]["source"] == f"audio:{WAV}"
    assert res.output[0]["relevance"] == 1.0


def test_pipeline_audio_end_to_end():
    result = asyncio.run(NeuroRoutePipeline().process(
        "Transcribe this voice note", [WAV]))
    assert result["strategy"] == "audio_grounded"
    assert "transcribe" in [n["kind"] for n in result["execution_graph"]]
    assert result["answer"]


def test_audio_dataset_routing():
    data = json.loads(Path("benchmark/queries_audio.json").read_text())
    pipeline = NeuroRoutePipeline()
    for item in data["queries"]:
        result = asyncio.run(
            pipeline.process(item["query"], item.get("attachments", [])))
        history = result.get("escalation_history") or []
        initial = history[0]["from"] if history else result["strategy"]
        assert initial == item["expected_strategy"], item["id"]


def test_real_whisper_roundtrip():
    pytest.importorskip("faster_whisper")
    gen = AudioTranscriber(mode="whisper").transcribe(WAV)
    assert "cheapest" in gen.text.lower()
    assert gen.latency_ms > 0

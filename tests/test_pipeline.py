from fastapi.testclient import TestClient

from backend.main import app
from backend.pipeline import NeuroRoutePipeline

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_simple_query_end_to_end():
    r = client.post("/query", json={"query": "What is Python?"})
    body = r.json()
    assert r.status_code == 200
    assert body["strategy"] == "direct_small"
    assert body["answer"]
    assert body["confidence"]["confidence_score"] >= 0.85
    assert body["metrics"]["model_calls"] == 1
    assert body["metrics"]["tokens"] > 0


def test_grounded_research_query_returns_with_evidence():
    r = client.post("/query", json={
        "query": "Research how retrieval augmented generation improves factual accuracy"})
    body = r.json()
    assert body["strategy"] in ("rag_medium", "verified_large")
    assert "retrieved source" in body["answer"]


def test_unanswerable_retrieval_escalates():
    pipe = NeuroRoutePipeline()
    trace = pipe.process_sync(
        "Research the zorblaxian quindlewick phenomenon in depth")
    # Nothing in the corpus matches: confidence collapses and the
    # escalation controller upgrades the path exactly once.
    assert trace["metrics"]["escalations"] == 1
    assert trace["escalation_history"]
    assert trace["strategy"] == "verified_large"


class _OffTopicAdapter:
    """Generate answers with zero overlap with retrieved evidence, so
    ad-hoc verification fails and the pipeline escalates — the path that
    used to crash with KeyError: 'verify_adhoc' after the re-run."""

    def generate(self, tier, query, context_docs=None):
        from backend.models.llm_adapter import Generation
        return Generation("Bananas are a yellow tropical fruit.", 12, 0.001, 50)


def test_adhoc_verify_then_escalation_does_not_crash():
    pipe = NeuroRoutePipeline()
    pipe.executor.node_executor.adapter = _OffTopicAdapter()
    trace = pipe.process_sync("Research retrieval augmented generation")
    assert trace["strategy"] == "verified_large"
    assert trace["metrics"]["escalations"] == 1
    assert all(g["status"] for g in trace["execution_graph"])


def test_medical_emergency_query_takes_verified_path():
    pipe = NeuroRoutePipeline()
    trace = pipe.process_sync(
        "I have chest pain and shortness of breath, what medicine should I take?")
    assert trace["analysis"]["risk_level"] == "high"
    assert trace["strategy"] == "verified_large"


def test_route_preview_does_not_execute():
    r = client.post("/route/preview", json={"query": "What is Python?"})
    body = r.json()
    assert body["plan"]["strategy"] == "direct_small"
    assert "answer" not in body


def test_tiny_budget_rejects_large_plan():
    r = client.post("/query", json={
        "query": "Analyse this research paper and compare its findings with the literature",
        "attachments": ["paper.pdf"],
        "budget": {"max_total_tokens": 100}})
    assert r.status_code == 422

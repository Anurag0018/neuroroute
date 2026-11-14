"""Post-benchmark tuning (2026-11-14):

1. Ollama generations are capped per tier (num_predict) — small models
   rambled in the full benchmark run (one easy query: 470+ tokens).
2. The pipeline's ad-hoc verification is evidence-only: the verifier
   model call it used to make produced output nothing consumed, while
   its tokens/latency were charged to every medium query. The actual
   verification — local evidence coverage — still runs and can still
   trigger escalation. The verified_large path keeps its in-graph
   verifier model call.
"""
from backend.execution.node_executor import NodeExecutor
from backend.models import llm_adapter
from backend.models.llm_adapter import OLLAMA_NUM_PREDICT, OllamaLLMAdapter
from backend.pipeline import NeuroRoutePipeline
from backend.schemas import NodeResult, PlanNode


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "ok", "prompt_eval_count": 5,
                "eval_count": 7, "total_duration": 3_000_000}


class _FakeClient:
    payloads = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):
        _FakeClient.payloads.append(json)
        return _FakeResponse()


def test_ollama_payload_carries_per_tier_cap(monkeypatch):
    _FakeClient.payloads = []
    monkeypatch.setattr(llm_adapter.httpx, "Client", _FakeClient)
    adapter = OllamaLLMAdapter()
    for tier in ("small", "medium", "large", "verifier"):
        adapter.generate(tier, "hello")
    caps = [p["options"]["num_predict"] for p in _FakeClient.payloads]
    assert caps == [OLLAMA_NUM_PREDICT[t] for t in
                    ("small", "medium", "large", "verifier")]
    assert OLLAMA_NUM_PREDICT["small"] < OLLAMA_NUM_PREDICT["medium"] \
        < OLLAMA_NUM_PREDICT["large"]


def _prior(answer="Retrieval augmented generation grounds answers in sources."):
    docs = [{"title": "RAG",
             "text": "Retrieval augmented generation grounds answers in sources.",
             "relevance": 0.6}]
    return {
        "retrieve": NodeResult("retrieve", "retrieve", output=docs),
        "generate": NodeResult("generate", "generate", output=answer),
    }


def test_evidence_only_verify_charges_nothing():
    executor = NodeExecutor()
    node = PlanNode("verify_adhoc", "verify", "verifier")
    res = executor.execute(node, "q", _prior(), use_llm_verifier=False)
    assert res.tokens == 0 and res.cost_usd == 0.0 and res.latency_ms == 0
    assert res.output is not None  # the coverage result still exists


def test_in_graph_verify_still_uses_model_call():
    executor = NodeExecutor()
    node = PlanNode("verify", "verify", "verifier")
    res = executor.execute(node, "q", _prior())
    assert res.tokens > 0  # mock verifier generation is charged


def test_medium_query_still_gets_adhoc_verification():
    pipe = NeuroRoutePipeline()
    trace = pipe.process_sync("Research retrieval augmented generation")
    ids = [g["node_id"] for g in trace["execution_graph"]]
    assert "verify_adhoc" in ids
    assert trace["strategy"] == "rag_medium"

"""Ollama adapter: prompt building, response parsing, adapter selection.
No network — the live server is exercised on the developer's machine.
"""
from backend.models.llm_adapter import (MockLLMAdapter, OllamaLLMAdapter,
                                        build_prompt, default_adapter,
                                        parse_generation)

DOCS = [{"title": "RAG notes", "text": "Retrieval grounds answers in sources.",
         "score": 0.9, "source": "seed"}]


def test_prompt_includes_sources_and_query():
    p = build_prompt("medium", "What is RAG?", DOCS)
    assert "RAG notes" in p and "Retrieval grounds answers" in p
    assert "What is RAG?" in p and "ONLY the sources" in p


def test_verifier_prompt_asks_for_verdict():
    p = build_prompt("verifier", "Some answer text")
    assert p.startswith("Check the following answer")
    assert "Some answer text" in p


def test_parse_generation_uses_provider_counts():
    data = {"response": " hello ", "prompt_eval_count": 11,
            "eval_count": 7, "total_duration": 250_000_000}
    gen = parse_generation(data, "prompt")
    assert gen.text == "hello"
    assert gen.tokens == 18
    assert gen.latency_ms == 250
    assert gen.cost_usd == 0.0  # local inference


def test_default_adapter_is_mock(monkeypatch):
    monkeypatch.delenv("NR_MODEL_MODE", raising=False)
    assert isinstance(default_adapter(), MockLLMAdapter)


def test_default_adapter_switches_to_ollama(monkeypatch):
    monkeypatch.setenv("NR_MODEL_MODE", "ollama")
    assert isinstance(default_adapter(), OllamaLLMAdapter)

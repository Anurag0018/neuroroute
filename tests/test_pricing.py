"""Phase 5 pricing: reference rate card, env overrides, modelled
costs in both adapter modes, and the benchmark's dollar summary."""
import asyncio
import json

from backend.models import model_registry
from backend.models.llm_adapter import MockLLMAdapter, parse_generation
from benchmark import run_benchmark


def test_default_rates_match_registry():
    assert model_registry.price_per_1k("small") == 0.00015
    assert model_registry.price_per_1k("large") == 0.0025
    assert model_registry.estimate_cost_usd("large", 1000) == 0.0025
    assert model_registry.estimate_cost_usd("medium", 500) == 0.0003


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("NR_PRICE_LARGE_PER_1K", "0.01")
    assert model_registry.price_per_1k("large") == 0.01
    assert model_registry.estimate_cost_usd("large", 1000) == 0.01
    assert model_registry.rate_card()["large"] == 0.01


def test_env_override_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("NR_PRICE_LARGE_PER_1K", "not-a-price")
    assert model_registry.price_per_1k("large") == 0.0025
    monkeypatch.setenv("NR_PRICE_LARGE_PER_1K", "-1")
    assert model_registry.price_per_1k("large") == 0.0025


def test_ollama_parse_prices_at_tier_rate():
    data = {"response": "x", "prompt_eval_count": 600,
            "eval_count": 400, "total_duration": 1_000_000}
    gen = parse_generation(data, "p", tier="medium")
    assert gen.tokens == 1000
    assert gen.cost_usd == 0.0006  # medium reference rate


def test_mock_adapter_still_priced():
    gen = MockLLMAdapter().generate("small", "what is python?")
    assert gen.tokens > 0 and gen.cost_usd > 0


def test_benchmark_summary_has_costs(tmp_path, monkeypatch):
    monkeypatch.setenv("NR_MODEL_MODE", "mock")
    out = tmp_path / "r.json"
    monkeypatch.setattr(
        "sys.argv", ["run_benchmark", "--limit", "3", "--output", str(out)])
    asyncio.run(run_benchmark.main())
    summary = json.loads(out.read_text())["summary"]
    for system in ("always_large", "always_medium", "neuroroute"):
        assert summary["avg_cost_usd"][system] > 0
        assert summary["quality_per_dollar"][system] > 0
    # Routing easy queries to small must undercut always-large on cost.
    assert (summary["avg_cost_usd"]["neuroroute"]
            < summary["avg_cost_usd"]["always_large"])
    assert summary["neuroroute_savings_vs_always_large"]["cost_pct"] > 0

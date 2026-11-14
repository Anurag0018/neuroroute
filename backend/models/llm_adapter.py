"""LLM adapters — same `generate()` contract, two implementations:

- MockLLMAdapter: deterministic, no API keys, no network. Default.
- OllamaLLMAdapter: real local models via Ollama. Enable with
  NR_MODEL_MODE=ollama (see README "Using real models").

Token counts are word-based estimates in the mock; the Ollama adapter
uses the provider's own usage counts when it returns them.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from backend.models.model_registry import estimate_cost_usd, get


@dataclass
class Generation:
    text: str
    tokens: int
    cost_usd: float
    latency_ms: int


def _est_tokens(*texts: str) -> int:
    return max(int(sum(len(t.split()) for t in texts) * 1.33), 1)


class MockLLMAdapter:
    def generate(self, tier: str, query: str,
                 context_docs: list[dict] | None = None) -> Generation:
        entry = get(tier)
        if tier == "verifier":
            # The verifier's own "generation" is not used directly;
            # see validation/answer_verifier.py.
            text = "verified"
            tokens = _est_tokens(query, text)
            return Generation(text, tokens, estimate_cost_usd(tier, tokens),
                              entry.avg_latency_ms)

        if context_docs:
            titles = "; ".join(d["title"] for d in context_docs)
            excerpts = " ".join(d["text"][:160] for d in context_docs)
            text = (f"Based on {len(context_docs)} retrieved source(s) "
                    f"({titles}): {excerpts} In summary, this addresses the "
                    f"request: {query.strip()}")
        elif tier == "small":
            text = f"Short answer: {query.strip()} — handled by the small model path."
        else:
            text = (f"Detailed answer ({tier} model): {query.strip()} "
                    f"Reasoning, caveats and supporting detail would be generated "
                    f"here by the {tier} model in production.")

        tokens = _est_tokens(query, text, *(d["text"] for d in context_docs or []))
        return Generation(text, tokens, estimate_cost_usd(tier, tokens),
                          entry.avg_latency_ms)


# ---------------------------------------------------------------------------
# Real local models via Ollama
# ---------------------------------------------------------------------------

OLLAMA_MODELS = {
    "small": os.environ.get("NR_OLLAMA_SMALL", "qwen2.5:1.5b"),
    "medium": os.environ.get("NR_OLLAMA_MEDIUM", "llama3.2:3b"),
    "large": os.environ.get("NR_OLLAMA_LARGE", "qwen2.5:7b"),
    "verifier": os.environ.get("NR_OLLAMA_VERIFIER", "llama3.2:3b"),
}

# Per-tier output caps (max generated tokens). Routing should also bound
# how much each tier may say: small models occasionally ramble (one easy
# benchmark query generated 470+ tokens), and the verifier only needs a
# verdict. Caps apply to every caller of this adapter — the benchmark
# baselines included — so comparisons stay even-handed.
OLLAMA_NUM_PREDICT = {
    "small": 192,
    "medium": 384,
    "large": 768,
    "verifier": 160,
}


def build_prompt(tier: str, query: str,
                 context_docs: list[dict] | None = None) -> str:
    if tier == "verifier":
        return ("Check the following answer for factual errors. Reply "
                "starting with VERIFIED if it is accurate, otherwise reply "
                "with the corrected answer.\n\nAnswer: " + query)
    if context_docs:
        sources = "\n\n".join(
            f"[{d['title']}] {d['text']}" for d in context_docs)
        return ("Answer the question using ONLY the sources below. If the "
                "sources do not contain the answer, say so.\n\nSources:\n"
                f"{sources}\n\nQuestion: {query}\nAnswer:")
    return query


def parse_generation(data: dict, prompt: str) -> Generation:
    text = (data.get("response") or "").strip()
    tokens = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
    if tokens == 0:
        tokens = _est_tokens(prompt, text)
    latency_ms = int((data.get("total_duration") or 0) / 1_000_000)
    # Local inference: no per-token API cost.
    return Generation(text=text, tokens=tokens, cost_usd=0.0,
                      latency_ms=max(latency_ms, 1))


class OllamaLLMAdapter:
    """Real adapter: talks to a local Ollama server (http://localhost:11434)."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = (base_url or os.environ.get(
            "NR_OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, tier: str, query: str,
                 context_docs: list[dict] | None = None) -> Generation:
        prompt = build_prompt(tier, query, context_docs)
        payload = {"model": OLLAMA_MODELS[tier], "prompt": prompt,
                   "stream": False,
                   "options": {"temperature": 0.2,
                               "num_predict": OLLAMA_NUM_PREDICT[tier]}}
        started = time.time()
        try:
            # trust_env=False: never route a localhost call through a proxy.
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Install Ollama, "
                "make sure it is running, and pull the models first "
                "(see README > Using real models).") from exc
        gen = parse_generation(resp.json(), prompt)
        if gen.latency_ms <= 1:
            gen.latency_ms = max(int((time.time() - started) * 1000), 1)
        return gen


def default_adapter():
    """Mock by default; real Ollama models when NR_MODEL_MODE=ollama."""
    if os.environ.get("NR_MODEL_MODE", "mock").lower() == "ollama":
        return OllamaLLMAdapter()
    return MockLLMAdapter()

"""Model Registry — the router selects from measured properties,
not from model size alone. Costs are USD per 1,000 tokens (blended
input+output, illustrative) and latency is a typical value that
should be replaced with rolling measured averages in production.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    tier: str                  # small | medium | large | verifier
    capabilities: tuple[str, ...]
    context_window: int
    cost_per_1k_tokens: float
    avg_latency_ms: int


REGISTRY: dict[str, ModelEntry] = {
    "small": ModelEntry("mock-small", "small",
                        ("general_qa", "summarization"), 8_192, 0.00015, 120),
    "medium": ModelEntry("mock-medium", "medium",
                         ("general_qa", "summarization", "research", "coding",
                          "math"), 32_768, 0.0006, 350),
    "large": ModelEntry("mock-large", "large",
                        ("general_qa", "summarization", "research", "coding",
                         "math", "reasoning"), 128_000, 0.0025, 800),
    "verifier": ModelEntry("mock-verifier", "verifier",
                           ("verification",), 32_768, 0.0006, 300),
}

# Non-model node costs (retrieval, parsing, ...)
NODE_LATENCY_MS = {"retrieve": 60}
NODE_COST_USD = {"retrieve": 0.00005}


def get(tier: str) -> ModelEntry:
    return REGISTRY[tier]


def estimate_cost_usd(tier: str, tokens: int) -> float:
    return round(tokens / 1000.0 * REGISTRY[tier].cost_per_1k_tokens, 6)

"""Model Registry — the router selects from measured properties,
not from model size alone. Costs are USD per 1,000 tokens (blended
input+output, illustrative) and latency is a typical value that
should be replaced with rolling measured averages in production.

Phase 5 pricing: the registry rates are REFERENCE prices. Local
(Ollama) inference has no per-token bill, so generations are priced
at these rates to make routing economics measurable and comparable
to cloud APIs — every dollar figure in NeuroRoute is a modelled
cost under this rate card, never a claim about an actual invoice.
Rates are env-overridable per tier (NR_PRICE_SMALL_PER_1K,
NR_PRICE_MEDIUM_PER_1K, NR_PRICE_LARGE_PER_1K,
NR_PRICE_VERIFIER_PER_1K) so the same code can be priced against a
real provider's card without a code change.
"""
from __future__ import annotations

import os
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


def price_per_1k(tier: str) -> float:
    """Effective USD price per 1K tokens for a tier: the
    NR_PRICE_<TIER>_PER_1K env override when it parses to a
    positive number, else the registry reference rate."""
    raw = os.environ.get(f"NR_PRICE_{tier.upper()}_PER_1K")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return REGISTRY[tier].cost_per_1k_tokens


def rate_card() -> dict:
    """The effective rate card (defaults + any env overrides)."""
    return {tier: price_per_1k(tier) for tier in REGISTRY}


def estimate_cost_usd(tier: str, tokens: int) -> float:
    return round(tokens / 1000.0 * price_per_1k(tier), 6)

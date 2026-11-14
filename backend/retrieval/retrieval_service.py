"""Retrieval service — Phase 1: in-memory corpus scored by token
overlap cosine. The `search()` contract (query -> docs with relevance
scores) is what a Qdrant / embeddings implementation must satisfy in
Phase 2; nothing upstream needs to change.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_SEED_CORPUS = [
    {"id": "doc_neuroroute", "title": "NeuroRoute design notes",
     "text": "NeuroRoute routes each request to the cheapest execution path "
             "that can answer it reliably, using a decision model, dynamic "
             "execution graphs, confidence estimation and escalation."},
    {"id": "doc_gnn", "title": "Graph neural networks overview",
     "text": "Graph neural networks learn representations by message passing "
             "between nodes and edges, and are widely used for molecular "
             "property prediction and recommendation."},
    {"id": "doc_rag", "title": "Retrieval augmented generation",
     "text": "Retrieval augmented generation grounds a language model in "
             "retrieved documents, improving factual accuracy and providing "
             "evidence that a verifier can check answers against."},
    {"id": "doc_python", "title": "Python basics",
     "text": "Python is a programming language known for readable syntax, "
             "functions, lists and dictionaries, used widely for scripting "
             "and data science."},
]


def _tokens(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class RetrievalService:
    def __init__(self, corpus: list[dict] | None = None):
        self.corpus = corpus if corpus is not None else list(_SEED_CORPUS)
        self._vectors = [(d, _tokens(d["title"] + " " + d["text"]))
                         for d in self.corpus]

    def search(self, query: str, k: int = 3, min_relevance: float = 0.08) -> list[dict]:
        q = _tokens(query)
        scored = [{"id": d["id"], "title": d["title"], "text": d["text"],
                   "relevance": round(_cosine(q, vec), 3)}
                  for d, vec in self._vectors]
        scored.sort(key=lambda d: d["relevance"], reverse=True)
        return [d for d in scored[:k] if d["relevance"] >= min_relevance]

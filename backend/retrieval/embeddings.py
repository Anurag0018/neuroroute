"""Embeddings — turn text into dense vectors for retrieval.

Two implementations behind one contract (`embed(texts) -> vectors`):

- HashEmbedder (default): dependency-free. Token counts are hashed
  into a 1024-dim space; cosine there approximates token-overlap
  cosine, so behaviour is deterministic and testable offline. It is
  lexical, not semantic — paraphrases with no shared words score low.
- SentenceTransformerEmbedder: real semantic embeddings
  (all-MiniLM-L6-v2). Used automatically when the optional
  `sentence-transformers` package is installed, or forced with
  NR_EMBEDDINGS=sentence-transformers. NR_EMBEDDINGS=hash forces the
  hashing embedder (the test suite does this for determinism).

This is the seam where a hosted embedding API would also plug in.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter

STOPWORDS = frozenset(
    "the a an in of to and or for with how what is are does do why used use "
    "using over into each other every can could should would this that it "
    "its be by on at from as was were been has have had".split())


def tokenize(text: str) -> Counter:
    # Stopwords removed: without this, unrelated documents match on
    # shared filler words and weak/false retrieval suppresses
    # escalation for genuinely unanswerable queries.
    return Counter(t for t in re.findall(r"[a-z0-9]+", text.lower())
                   if t not in STOPWORDS)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class HashEmbedder:
    name = "hashing-1024"
    dim = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok, count in tokenize(text).items():
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                vec[h % self.dim] += count
            vectors.append(vec)
        return vectors


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers/{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in
                self._model.encode(texts, normalize_embeddings=True)]


def default_embedder():
    mode = os.environ.get("NR_EMBEDDINGS", "auto").lower()
    if mode == "hash":
        return HashEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        # Optional dependency absent (or model not downloaded): the
        # hashing embedder keeps retrieval fully functional offline.
        return HashEmbedder()

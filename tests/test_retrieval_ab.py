"""Embeddings A/B probes: well-formed, and the hashing embedder
aces the keyword-style probes (paraphrase performance is measured
by the benchmark script, not asserted — it is the experiment)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

import retrieval_ab  # noqa: E402
from backend.retrieval.embeddings import HashEmbedder  # noqa: E402
from backend.retrieval.retrieval_service import (  # noqa: E402
    RetrievalService, _SEED_CORPUS)


def test_probes_reference_real_documents():
    ids = {d["id"] for d in _SEED_CORPUS}
    assert len(retrieval_ab.PROBES) >= 6
    for probe in retrieval_ab.PROBES:
        assert probe["expected"] in ids
        assert probe["style"] in ("keyword", "paraphrase")


def test_hashing_embedder_keyword_probes_all_hit():
    rows = retrieval_ab.evaluate(RetrievalService(embedder=HashEmbedder()))
    keyword = [r for r in rows if r["style"] == "keyword"]
    assert keyword and all(r["hit"] for r in keyword)

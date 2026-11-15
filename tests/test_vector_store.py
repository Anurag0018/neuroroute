"""Vector store seam: in-memory default, env selection, and a full
Qdrant round-trip (embedded local mode) when qdrant-client is
installed."""
import pytest

from backend.retrieval.embeddings import HashEmbedder
from backend.retrieval.retrieval_service import RetrievalService
from backend.retrieval.vector_store import (InMemoryVectorStore,
                                            default_vector_store)

TEXT = ("Photosynthesis converts sunlight, water and carbon dioxide "
        "into glucose and oxygen inside plant chloroplasts. " * 3)


def test_default_store_is_memory(monkeypatch):
    monkeypatch.setenv("NR_VECTOR_STORE", "memory")
    assert isinstance(default_vector_store(HashEmbedder()),
                      InMemoryVectorStore)


def test_memory_store_service_roundtrip():
    svc = RetrievalService(embedder=HashEmbedder())
    assert svc.store.name == "memory"
    assert svc.documents().get("seed") == 10
    n = svc.ingest_text("bio.txt", TEXT)
    assert n >= 1 and svc.documents()["bio.txt"] == n
    hits = svc.search("how do plants make food from sunlight",
                      sources=["bio.txt"])
    assert hits and hits[0]["source"] == "bio.txt"
    svc.ingest_text("bio.txt", TEXT)  # re-ingest replaces, not appends
    assert svc.documents()["bio.txt"] == n


def test_qdrant_store_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("NR_QDRANT_PATH", str(tmp_path / "qdrant"))
    from backend.retrieval.vector_store import QdrantVectorStore
    store = QdrantVectorStore(HashEmbedder())
    svc = RetrievalService(embedder=HashEmbedder(), store=store)
    assert svc.store.name == "qdrant"
    assert svc.documents().get("seed") == 10
    hits = svc.search("what is retrieval augmented generation")
    assert hits and hits[0]["id"] == "doc_rag"
    n = svc.ingest_text("bio.txt", TEXT)
    assert svc.documents()["bio.txt"] == n
    scoped = svc.search("how do plants make food from sunlight",
                        sources=["bio.txt"])
    assert scoped and scoped[0]["source"] == "bio.txt"
    svc.ingest_text("bio.txt", TEXT)
    assert svc.documents()["bio.txt"] == n
    # Unknown source falls back to the whole index, as with memory.
    fallback = svc.search("what is retrieval augmented generation",
                          sources=["missing.pdf"])
    assert fallback and fallback[0]["id"] == "doc_rag"


def test_qdrant_selected_by_env(tmp_path, monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("NR_VECTOR_STORE", "qdrant")
    monkeypatch.setenv("NR_QDRANT_PATH", str(tmp_path / "q2"))
    store = default_vector_store(HashEmbedder())
    assert store.name == "qdrant"

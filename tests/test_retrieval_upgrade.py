"""Retrieval upgrade tests: embeddings, chunking, real document
ingestion (txt + a generated PDF), source-scoped search, and the
/documents API. Uses the deterministic hashing embedder (conftest
pins NR_EMBEDDINGS=hash), so no model downloads are needed.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.retrieval.document_loader import chunk_text
from backend.retrieval.embeddings import HashEmbedder, cosine
from backend.retrieval.retrieval_service import RetrievalService

client = TestClient(app)

_REPORT = (
    "Quarterly revenue grew 18 percent as subscription income expanded "
    "across all regions. Operating costs fell because inference spend "
    "was routed to cheaper models. The Zephyr protocol cut p95 latency "
    "substantially during the trial period.\n\n"
    "Risks remain around supplier concentration and currency exposure, "
    "and the board recommends diversifying vendors next quarter.")


def test_chunking_splits_long_text():
    text = "\n\n".join(f"Paragraph {i} " + "content " * 60
                        for i in range(6))
    chunks = chunk_text(text)
    assert len(chunks) >= 3
    assert all(len(c) <= 1200 for c in chunks)
    assert "Paragraph 0" in chunks[0]


def test_hash_embedder_ranks_related_text_higher():
    emb = HashEmbedder()
    q, related, unrelated = emb.embed([
        "vector databases semantic search",
        "embeddings and vector similarity search",
        "python lists dictionaries and readable syntax"])
    assert cosine(q, related) > cosine(q, unrelated)


def test_ingest_and_scoped_search():
    svc = RetrievalService(embedder=HashEmbedder())
    n = svc.ingest_text("q3report.txt", _REPORT)
    assert n >= 1
    hits = svc.search("quarterly revenue growth", sources=["q3report.txt"])
    assert hits and all(h["source"] == "q3report.txt" for h in hits)
    assert "revenue" in hits[0]["text"]


def test_unknown_source_falls_back_to_full_index():
    svc = RetrievalService(embedder=HashEmbedder())
    scoped = svc.search("retrieval augmented generation",
                        sources=["never_uploaded.pdf"])
    plain = svc.search("retrieval augmented generation")
    assert [h["id"] for h in scoped] == [h["id"] for h in plain]
    assert plain  # seed corpus still answers


def test_pdf_ingestion(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "zephyr.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72),
                     "The Zephyr protocol reduces inference latency by "
                     "routing requests across specialised model tiers.")
    doc.save(pdf)
    doc.close()

    svc = RetrievalService(embedder=HashEmbedder())
    assert svc.ingest_file(pdf) == 1
    hits = svc.search("zephyr protocol latency", sources=["zephyr.pdf"])
    assert hits and "Zephyr" in hits[0]["text"]


def test_documents_endpoint_and_scoped_query():
    up = client.post("/documents", files={
        "file": ("boardnotes.txt", _REPORT.encode(), "text/plain")})
    assert up.status_code == 200
    body = up.json()
    assert body["source"] == "boardnotes.txt"
    assert body["chunks_ingested"] >= 1
    assert client.get("/documents").json()["documents"]["boardnotes.txt"] >= 1

    r = client.post("/query", json={
        "query": "Analyse this report and summarise the revenue findings",
        "attachments": ["boardnotes.txt"]})
    assert r.status_code == 200
    trace = r.json()
    assert trace["strategy"] == "verified_large"
    assert "retrieved source" in trace["answer"]

"""Retrieval service — embedding-based search over a document index.

History: Phase 1 scored a fixed in-memory corpus by token-overlap
cosine behind the `search()` contract. This upgrade keeps the
contract and replaces the internals:

- Documents (seed corpus + anything ingested at runtime) are chunked
  and embedded (backend/retrieval/embeddings.py) and held in a
  vector store behind this seam (backend/retrieval/vector_store.py):
  in-memory by default, Qdrant with NR_VECTOR_STORE=qdrant. The
  scoping/threshold policy below is store-independent.
- Real files enter via ingest_file()/ingest_text() (PDF via PyMuPDF,
  .txt/.md natively), chunked by document_loader.
- search(..., sources=[...]) scopes retrieval to named ingested
  documents — that is how a query about an uploaded report reads
  *that report*. If none of the named sources has been ingested,
  the filter is ignored and search falls back to the whole index
  (so filename-only attachments, as in the benchmark, behave as
  before).
"""
from __future__ import annotations

from pathlib import Path

from backend.retrieval.document_loader import chunk_text, extract_text
from backend.retrieval.embeddings import default_embedder
from backend.retrieval.vector_store import default_vector_store

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
    {"id": "doc_embeddings", "title": "Embeddings and semantic search",
     "text": "Embeddings are dense vector representations of text that "
             "capture semantic meaning, so texts with similar meaning have "
             "similar vectors and semantic search can find relevant "
             "documents by vector similarity."},
    {"id": "doc_vectordb", "title": "Vector databases",
     "text": "Vector databases store embeddings and support fast "
             "approximate nearest neighbour search, enabling semantic "
             "search and retrieval augmented generation over large "
             "document collections."},
    {"id": "doc_transformer", "title": "Transformer attention",
     "text": "Transformers use self-attention so every token can attend "
             "to every other token in the sequence, capturing long-range "
             "context in parallel and forming the basis of modern "
             "language models."},
    {"id": "doc_quantization", "title": "Model quantization",
     "text": "Quantization reduces model weights from 16 or 32 bit to "
             "8 or 4 bit, shrinking memory size and inference cost and "
             "increasing speed, with a small trade-off in accuracy."},
    {"id": "doc_finetuning", "title": "Fine-tuning versus prompting",
     "text": "Fine-tuning updates model weights on task data for "
             "consistent behaviour, while prompting and retrieval leave "
             "weights unchanged and steer the model with instructions "
             "and context; the trade-off is cost and flexibility."},
    {"id": "doc_learning_types", "title": "Supervised and unsupervised learning",
     "text": "Supervised learning trains on labelled examples to "
             "predict outputs, while unsupervised learning finds "
             "structure such as clusters in unlabelled data."},
]


class RetrievalService:
    def __init__(self, corpus: list[dict] | None = None, embedder=None,
                 store=None):
        self.embedder = embedder or default_embedder()
        self.store = store or default_vector_store(self.embedder)
        seed = corpus if corpus is not None else _SEED_CORPUS
        docs = [{**d, "source": d.get("source", "seed")} for d in seed]
        vectors = self.embedder.embed(
            [d["title"] + " " + d["text"] for d in docs]) if docs else []
        self.store.reset_with(docs, vectors)

    def documents(self) -> dict:
        """source name -> chunk count, for the API/dashboard."""
        return self.store.documents()

    # ---------- ingestion ----------

    def ingest_text(self, source: str, text: str) -> int:
        """Chunk + embed a document under `source` (re-ingesting the
        same source replaces its previous chunks). Returns chunks."""
        chunks = chunk_text(text)
        docs = [{"id": f"{source}::chunk{i}",
                 "title": f"{source} (part {i + 1})",
                 "text": chunk, "source": source}
                for i, chunk in enumerate(chunks)]
        vectors = self.embedder.embed(
            [d["title"] + " " + d["text"] for d in docs]) if docs else []
        self.store.replace_source(source, docs, vectors)
        return len(chunks)

    def ingest_file(self, path: str | Path) -> int:
        p = Path(path)
        return self.ingest_text(p.name, extract_text(p))

    # ---------- search (the contract everything upstream uses) ----------

    def search(self, query: str, k: int = 3, min_relevance: float = 0.08,
               sources: list[str] | None = None) -> list[dict]:
        scoped_active = bool(sources) and any(
            self.store.has_source(s) for s in sources)
        qv = self.embedder.embed([query])[0]
        scored = self.store.scored(
            qv, sources if scoped_active else None, limit=max(50, k))
        hits = [d for d in scored[:k] if d["relevance"] >= min_relevance]
        if scoped_active and not hits and scored and scored[0]["relevance"] > 0:
            # The user pointed at this document explicitly: return its
            # best-matching chunk even below the global threshold (a
            # long chunk dilutes lexical cosine). Unscoped search keeps
            # the strict threshold — that is what lets unanswerable
            # queries escalate instead of hallucinating grounding.
            hits = scored[:1]
        return hits

"""Vector stores behind the RetrievalService seam.

The service's search contract (scoping, thresholds, fallback) is
store-independent policy; a store only has to (a) hold documents
with their vectors, (b) say which sources it has, and (c) return
documents scored against a query vector, best first.

- InMemoryVectorStore (default): plain cosine over lists — zero
  dependencies, deterministic, what the test suite runs on.
- QdrantVectorStore: selected with NR_VECTOR_STORE=qdrant.
  Embedded local mode (data/qdrant, no server) unless NR_QDRANT_URL
  points at a Qdrant server. Needs the optional qdrant-client
  package; cosine distance, so scores are the same cosine the
  in-memory store computes.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from backend.retrieval.embeddings import cosine

_REPO_DATA = Path(__file__).resolve().parent.parent.parent / "data"


class InMemoryVectorStore:
    name = "memory"

    def __init__(self):
        self._docs: list[dict] = []
        self._vectors: list[list[float]] = []

    def reset_with(self, docs: list[dict], vectors: list[list[float]]):
        self._docs, self._vectors = list(docs), list(vectors)

    def replace_source(self, source: str, docs: list[dict],
                       vectors: list[list[float]]):
        keep = [(d, v) for d, v in zip(self._docs, self._vectors)
                if d["source"] != source]
        self._docs = [d for d, _ in keep] + list(docs)
        self._vectors = [v for _, v in keep] + list(vectors)

    def documents(self) -> dict:
        counts: dict[str, int] = {}
        for d in self._docs:
            counts[d["source"]] = counts.get(d["source"], 0) + 1
        return counts

    def has_source(self, source: str) -> bool:
        return any(d["source"] == source for d in self._docs)

    def scored(self, qv: list[float], sources: list[str] | None = None,
               limit: int = 50) -> list[dict]:
        out = [{"id": d["id"], "title": d["title"], "text": d["text"],
                "source": d["source"],
                "relevance": round(cosine(qv, vec), 3)}
               for d, vec in zip(self._docs, self._vectors)
               if not sources or d["source"] in sources]
        out.sort(key=lambda d: d["relevance"], reverse=True)
        return out[:limit]


class QdrantVectorStore:
    name = "qdrant"
    COLLECTION = "neuroroute_chunks"

    def __init__(self, embedder):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client import models as qm
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "NR_VECTOR_STORE=qdrant needs the optional package: "
                "pip install qdrant-client") from exc
        self._qm = qm
        url = os.environ.get("NR_QDRANT_URL")
        if url:
            self._client = QdrantClient(url=url)
        else:
            path = os.environ.get("NR_QDRANT_PATH",
                                  str(_REPO_DATA / "qdrant"))
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=path)
        self._dim = embedder.dim

    # ---------- collection ----------

    def _ensure_collection(self, recreate: bool = False):
        from qdrant_client import models as qm
        exists = self._client.collection_exists(self.COLLECTION)
        if exists and not recreate:
            info = self._client.get_collection(self.COLLECTION)
            size = info.config.params.vectors.size
            if size == self._dim:
                return
        self._client.recreate_collection(
            collection_name=self.COLLECTION,
            vectors_config=qm.VectorParams(
                size=self._dim, distance=qm.Distance.COSINE))

    def _point(self, doc: dict, vector: list[float]):
        from qdrant_client import models as qm
        return qm.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc["id"])),
            vector=vector,
            payload={"doc_id": doc["id"], "title": doc["title"],
                     "text": doc["text"], "source": doc["source"]})

    def _upsert(self, docs, vectors):
        if docs:
            self._client.upsert(
                collection_name=self.COLLECTION,
                points=[self._point(d, v) for d, v in zip(docs, vectors)])

    # ---------- store contract ----------

    def reset_with(self, docs: list[dict], vectors: list[list[float]]):
        self._ensure_collection(recreate=True)
        self._upsert(docs, vectors)

    def replace_source(self, source: str, docs: list[dict],
                       vectors: list[list[float]]):
        from qdrant_client import models as qm
        self._ensure_collection()
        self._client.delete(
            collection_name=self.COLLECTION,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                qm.FieldCondition(key="source",
                                  match=qm.MatchValue(value=source))])))
        self._upsert(docs, vectors)

    def documents(self) -> dict:
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.COLLECTION, limit=1000, offset=offset,
                with_payload=True, with_vectors=False)
            for p in points:
                src = p.payload.get("source", "?")
                counts[src] = counts.get(src, 0) + 1
            if offset is None:
                return counts

    def has_source(self, source: str) -> bool:
        points, _ = self._client.scroll(
            collection_name=self.COLLECTION, limit=1,
            scroll_filter=self._qm.Filter(must=[
                self._qm.FieldCondition(
                    key="source",
                    match=self._qm.MatchValue(value=source))]),
            with_payload=False, with_vectors=False)
        return bool(points)

    def scored(self, qv: list[float], sources: list[str] | None = None,
               limit: int = 50) -> list[dict]:
        query_filter = None
        if sources:
            query_filter = self._qm.Filter(must=[
                self._qm.FieldCondition(
                    key="source", match=self._qm.MatchAny(any=list(sources)))])
        hits = self._client.query_points(
            collection_name=self.COLLECTION, query=qv, limit=limit,
            query_filter=query_filter, with_payload=True).points
        return [{"id": h.payload["doc_id"], "title": h.payload["title"],
                 "text": h.payload["text"], "source": h.payload["source"],
                 "relevance": round(float(h.score), 3)} for h in hits]


def default_vector_store(embedder):
    if os.environ.get("NR_VECTOR_STORE", "memory").lower() == "qdrant":
        return QdrantVectorStore(embedder)
    return InMemoryVectorStore()

"""Embeddings A/B: hashing (lexical) vs sentence-transformers (semantic).

Runs a fixed probe set against the seed corpus twice — once with the
dependency-free HashEmbedder, once with SentenceTransformerEmbedder
(all-MiniLM-L6-v2, only if sentence-transformers is installed). Probes
come in two styles:

- keyword: the query shares vocabulary with the target document —
  both embedders should find it.
- paraphrase: the query means the same thing in different words —
  lexical hashing is expected to struggle; semantic embeddings are
  expected to win. This is the honest test of whether the heavy
  dependency earns its place.

Metric: top-1 hit (the expected document ranked first) and the
expected document's relevance score. Usage:

    python benchmark/retrieval_ab.py
    # writes benchmark/retrieval_ab_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.retrieval.embeddings import (  # noqa: E402
    HashEmbedder, SentenceTransformerEmbedder)
from backend.retrieval.retrieval_service import RetrievalService  # noqa: E402

PROBES = [
    # keyword-style: shared vocabulary with the target document
    {"query": "how does quantization reduce model memory size and "
              "inference cost",
     "expected": "doc_quantization", "style": "keyword"},
    {"query": "how do vector databases support nearest neighbour search",
     "expected": "doc_vectordb", "style": "keyword"},
    {"query": "supervised and unsupervised learning difference",
     "expected": "doc_learning_types", "style": "keyword"},
    # paraphrase-style: same meaning, different words
    {"query": "how can a chatbot stop making up facts",
     "expected": "doc_rag", "style": "paraphrase"},
    {"query": "making a huge neural network fit on a small laptop",
     "expected": "doc_quantization", "style": "paraphrase"},
    {"query": "finding documents that mean the same thing even with "
              "different words",
     "expected": "doc_embeddings", "style": "paraphrase"},
    {"query": "teaching a model new behaviour without retraining it",
     "expected": "doc_finetuning", "style": "paraphrase"},
    {"query": "how are molecular properties predicted from graph structure",
     "expected": "doc_gnn", "style": "paraphrase"},
]


def evaluate(service: RetrievalService) -> list[dict]:
    rows = []
    for probe in PROBES:
        results = service.search(probe["query"], k=3, min_relevance=0.0)
        top1 = results[0]["id"] if results else None
        expected_rel = next(
            (r["relevance"] for r in results
             if r["id"] == probe["expected"]), 0.0)
        rows.append({**probe, "top1": top1,
                     "hit": top1 == probe["expected"],
                     "expected_relevance": round(expected_rel, 3)})
    return rows


def summarise(name: str, rows: list[dict]) -> dict:
    kw = [r for r in rows if r["style"] == "keyword"]
    para = [r for r in rows if r["style"] == "paraphrase"]
    return {"embedder": name,
            "keyword_hits": f"{sum(r['hit'] for r in kw)}/{len(kw)}",
            "paraphrase_hits": f"{sum(r['hit'] for r in para)}/{len(para)}",
            "rows": rows}


def main() -> int:
    reports = [summarise("hashing-1024",
                         evaluate(RetrievalService(embedder=HashEmbedder())))]
    try:
        import sentence_transformers  # noqa: F401
        reports.append(summarise(
            "sentence-transformers/all-MiniLM-L6-v2",
            evaluate(RetrievalService(
                embedder=SentenceTransformerEmbedder()))))
    except ImportError:
        print("sentence-transformers not installed — hashing only. "
              "Install with: pip install sentence-transformers")
    for rep in reports:
        print(f"\n== {rep['embedder']} ==")
        print(f"keyword hits: {rep['keyword_hits']}   "
              f"paraphrase hits: {rep['paraphrase_hits']}")
        for r in rep["rows"]:
            mark = "HIT " if r["hit"] else "miss"
            print(f"  [{mark}] ({r['style']:>10}) {r['query'][:52]:<52} "
                  f"-> {r['top1']} (expected {r['expected']}, "
                  f"rel {r['expected_relevance']})")
    out = Path(__file__).with_name("retrieval_ab_results.json")
    out.write_text(json.dumps(reports, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

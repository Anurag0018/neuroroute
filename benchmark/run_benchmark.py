"""NeuroRoute benchmark — fixed pipelines vs adaptive routing.

Systems compared on the SAME query set (benchmark/queries.json):
  A. always_large   — every query goes to the large model, no retrieval
  B. always_medium  — every query: retrieval + medium model (fixed RAG)
  C. neuroroute     — the full adaptive pipeline (routing + verification
                      + escalation, overhead included in its totals)

Metrics per query: wall-clock latency, total tokens, and a transparent
quality proxy (fraction of expected keyword groups present in the
answer). With local Ollama models the API cost is $0, so tokens are
the cost proxy; with a paid provider, tokens map directly to cost.

Run (from the project root):
  python -m benchmark.run_benchmark                 # mock models, fast
  NR_MODEL_MODE=ollama python -m benchmark.run_benchmark --limit 9

Results are written to benchmark/results.json and printed as a table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from backend.models.llm_adapter import default_adapter
from backend.pipeline import NeuroRoutePipeline
from backend.retrieval.retrieval_service import RetrievalService

_HERE = Path(__file__).parent


def quality(answer: str, groups: list[list[str]]) -> float:
    if not groups:
        return 1.0
    text = answer.lower()
    covered = sum(1 for g in groups if any(w.lower() in text for w in g))
    return round(covered / len(groups), 3)


def _timed(fn):
    started = time.perf_counter()
    out = fn()
    return out, int((time.perf_counter() - started) * 1000)


def run_baselines(adapter, retrieval, item) -> dict:
    query = item["query"]
    groups = item["expected_keywords"]

    gen_a, ms_a = _timed(lambda: adapter.generate("large", query))
    docs = retrieval.search(query)
    gen_b, ms_b = _timed(lambda: adapter.generate("medium", query, docs or None))
    return {
        "always_large": {"latency_ms": ms_a, "tokens": gen_a.tokens,
                         "quality": quality(gen_a.text, groups)},
        "always_medium": {"latency_ms": ms_b, "tokens": gen_b.tokens,
                          "quality": quality(gen_b.text, groups)},
    }


async def run_neuroroute(pipeline, item) -> dict:
    started = time.perf_counter()
    result = await pipeline.process(item["query"], item.get("attachments", []))
    ms = int((time.perf_counter() - started) * 1000)
    # Routing accuracy judges the DECISION (the first strategy chosen).
    # A later escalation is the safety net working, not a routing miss,
    # so compare against the initial strategy from the escalation history.
    history = result.get("escalation_history") or []
    initial = history[0]["from"] if history else result["strategy"]
    return {"latency_ms": ms, "tokens": result["metrics"]["tokens"],
            "quality": quality(result["answer"], item["expected_keywords"]),
            "strategy": result["strategy"], "initial_strategy": initial,
            "expected_strategy": item["expected_strategy"],
            "routing_correct": initial == item["expected_strategy"]}


def _avg(rows, system, key):
    return round(sum(r[system][key] for r in rows) / len(rows), 2)


def _savings(rows, system, key):
    base = _avg(rows, system, key)
    nr = _avg(rows, "neuroroute", key)
    return round((1 - nr / base) * 100, 1) if base else 0.0


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N queries (0 = all)")
    parser.add_argument("--dataset", default=str(_HERE / "queries.json"))
    parser.add_argument("--output", default=str(_HERE / "results.json"))
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text())
    items = data["queries"][: args.limit or None]
    adapter, retrieval, pipeline = default_adapter(), RetrievalService(), NeuroRoutePipeline()

    rows = []
    for item in items:
        row = {"id": item["id"], "level": item["level"],
               **run_baselines(adapter, retrieval, item),
               "neuroroute": await run_neuroroute(pipeline, item)}
        rows.append(row)
        nr = row["neuroroute"]
        print(f"{item['id']:<8} {item['level']:<7} -> {nr['initial_strategy']:<15} "
              f"{'OK ' if nr['routing_correct'] else 'MISS'} "
              f"NR {nr['latency_ms']:>6}ms {nr['tokens']:>5}tok q={nr['quality']:.2f} | "
              f"large {_avg([row], 'always_large', 'latency_ms'):>6.0f}ms "
              f"{_avg([row], 'always_large', 'tokens'):>5.0f}tok | "
              f"medium {_avg([row], 'always_medium', 'latency_ms'):>6.0f}ms "
              f"{_avg([row], 'always_medium', 'tokens'):>5.0f}tok")

    correct = sum(1 for r in rows if r["neuroroute"]["routing_correct"])
    summary = {
        "queries": len(rows),
        "routing_accuracy": f"{correct}/{len(rows)}",
        "avg_latency_ms": {s: _avg(rows, s, "latency_ms")
                           for s in ("always_large", "always_medium", "neuroroute")},
        "avg_tokens": {s: _avg(rows, s, "tokens")
                       for s in ("always_large", "always_medium", "neuroroute")},
        "avg_quality": {s: _avg(rows, s, "quality")
                        for s in ("always_large", "always_medium", "neuroroute")},
        "neuroroute_savings_vs_always_large": {
            "latency_pct": _savings(rows, "always_large", "latency_ms"),
            "tokens_pct": _savings(rows, "always_large", "tokens")},
        "neuroroute_savings_vs_always_medium": {
            "latency_pct": _savings(rows, "always_medium", "latency_ms"),
            "tokens_pct": _savings(rows, "always_medium", "tokens")},
    }
    Path(args.output).write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=1))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=1))
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

# NeuroRoute — Phase 1 MVP

**Adaptive AI Inference & Orchestration Platform**

A lightweight decision model decides *how* a request should be answered —
which models, which tools, how much budget — then a validated execution
graph runs that plan, a confidence engine checks the result, and an
escalation controller upgrades the path when the result isn't good enough.
The goal is **quality per unit cost**, proved against fixed-pipeline
baselines, not just speed.

> Phase 1 scope (this scaffold): three execution paths, rule-based decision
> model behind the real decision-model interface, custom DAG executor,
> confidence + verification + escalation, budget enforcement, and full
> per-request tracing. Everything runs **offline with mock models** — no
> API keys needed. Later phases swap the insides, not the contracts.

## The three Phase 1 paths

| Path | Strategy | Graph | Used when |
| --- | --- | --- | --- |
| A | `direct_small` | small LLM | Simple, low-risk queries (e.g. "What is Python?") |
| B | `rag_medium` | retrieve → medium LLM | Retrieval or moderate reasoning needed |
| C | `verified_large` | [retrieve →] large LLM → verifier | High complexity or high risk |

## Quick start

> Requires Python 3.10+ (recommended). Python 3.9 also works — the
> `eval_type_backport` dependency in requirements.txt covers the older
> typing syntax — but recreate the venv with `python3.11` / `python3.12`
> if you have it.

```bash
cd neuroroute
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

- API docs: http://localhost:8000/docs
- Live dashboard: http://localhost:8000/dashboard
- Try it:

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "What is Python?"}'

curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "Research how retrieval augmented generation improves factual accuracy"}'

# Decision only, nothing executes:
curl -X POST localhost:8000/route/preview -H 'Content-Type: application/json' \
  -d '{"query": "Analyse this research paper", "attachments": ["paper.pdf"]}'
```

## Using real models (Ollama)

By default NeuroRoute uses a deterministic mock adapter (no keys, no
network). To run real local models:

1. Install Ollama from https://ollama.com and make sure it is running.
2. Pull the three tier models (one-time download, ~8 GB total):

```bash
ollama pull qwen2.5:1.5b    # small
ollama pull llama3.2:3b     # medium + verifier
ollama pull qwen2.5:7b      # large (slow on a MacBook Air — see note)
```

3. Start NeuroRoute in Ollama mode, in the same terminal:

```bash
export NR_MODEL_MODE=ollama
uvicorn backend.main:app --reload
```

Check `GET /health` — it should report `"model_mode": "ollama"`.
Model names are overridable: `NR_OLLAMA_SMALL`, `NR_OLLAMA_MEDIUM`,
`NR_OLLAMA_LARGE`, `NR_OLLAMA_VERIFIER`, `NR_OLLAMA_URL`.

> MacBook Air note: the 7B large model works but is slow. For day-to-day
> development set `NR_OLLAMA_LARGE=llama3.2:3b` and switch back to the
> 7B for the final demo/benchmark. Local inference reports cost $0 —
> the routing still saves real time and tokens, which the traces show.

## Benchmark

Fixed dataset of 24 queries (8 easy / 8 medium / 8 hard) in
`benchmark/queries.json`, compared against two fixed pipelines on the
same queries: always-large, and always-medium+RAG. NeuroRoute's totals
include its own routing/verification overhead.

```bash
python -m benchmark.run_benchmark                  # mock models: checks the harness only
NR_MODEL_MODE=ollama python -m benchmark.run_benchmark --limit 9   # real numbers
```

Output: per-query routing/latency/tokens/quality plus a summary with
routing accuracy and % savings vs each baseline, written to
`benchmark/results.json`. Quality is a transparent keyword-coverage
proxy (see the notes in queries.json), not a judge model. Mock-mode
numbers are meaningless by design — only the Ollama run produces
reportable results.

Every `/query` response includes the answer **plus** the routing rationale,
the execution graph with per-node latency, the confidence breakdown,
escalation history, and actual tokens / cost / latency. Traces are kept at
`/traces`, aggregate stats at `/metrics/summary`.

## Tests

```bash
pytest -q
```

Covers routing decisions, graph validation (allowlist, cycles, missing
dependencies, budget rejection), the end-to-end pipeline, and escalation
when retrieval finds no evidence.

## How it maps to the full architecture

| Architecture subsystem | Phase 1 module |
| --- | --- |
| Input Analyzer | `backend/intelligence/input_analyzer.py` |
| Complexity Estimator | `backend/intelligence/complexity_estimator.py` |
| Decision-Making Model | `backend/intelligence/decision_model.py` (rules now, Jev/Laya later — same `propose_plan()` contract) |
| Routing Policy | `backend/intelligence/routing_policy.py` |
| Graph Generator / Validator | `backend/planning/` — the validator is the safety boundary: allowlisted node kinds, registered tiers, acyclic, within budget. An LLM proposes; it never executes unchecked. |
| Graph Orchestrator | `backend/execution/graph_executor.py` (topological levels, concurrent within a level) |
| Model Registry / Adapters | `backend/models/` (mock adapter now; real adapters implement `generate()`) |
| Retrieval Engine | `backend/retrieval/` (in-memory overlap scoring now; Qdrant behind the same `search()` contract) |
| Confidence Estimator | `backend/validation/confidence_estimator.py` |
| Verification Engine | `backend/validation/answer_verifier.py` (evidence coverage, not a second opinion) |
| Escalation Controller | `backend/validation/escalation_controller.py` (max escalations, budget-checked, never repeats a failed strategy) |
| Budget Manager | `backend/optimization/budget_manager.py` (estimated vs actual kept separate) |
| Observability | `backend/observability/execution_tracer.py` + dashboard |

This is a **modular monolith** — one FastAPI app, modules matching the
service boundaries in the full architecture. Split into services only when
a module actually needs independent scaling.

## Key design decisions (kept from the architecture)

1. **The decision model is lightweight.** Routing must cost far less than
   the inference it saves — and its cost is counted in every benchmark.
2. **Execution is deterministic wherever possible.** Plans are validated
   against an allowlist and a budget before anything runs.
3. **Confidence is measured, not self-reported.** Evidence coverage,
   validation, retrieval relevance and verifier agreement are combined;
   thresholds (return ≥ 0.85, verify 0.60–0.84, escalate < 0.60) are
   starting values to calibrate against measured correctness.
4. **Escalation has hard limits** — max retries, max model calls, max
   tokens, max cost, and no repeating the same failed strategy.

## Phase 2 hooks

- Replace `RuleBasedDecisionModel` with Jev / Laya structured output
  (`backend/intelligence/decision_model.py`).
- Implement `generate()` in `backend/models/llm_adapter.py` against Ollama /
  vLLM / cloud APIs and put real measured costs + latencies in the registry.
- Swap the retrieval corpus for embeddings + Qdrant behind `search()`.
- Add the benchmark harness: Baseline A (always large), Baseline B (always
  medium) vs NeuroRoute on one fixed mixed-difficulty dataset, reporting
  accuracy / latency / cost distributions — including decision + verifier
  overhead in NeuroRoute's totals.

## Post-benchmark tuning (v1.1, 2026-11-14)

Two overhead leaks found in the full 24-query run, fixed without
changing any routing decision:

1. **Ad-hoc verification is evidence-only.** The middle-band check used
   to make a verifier model call whose output nothing consumed, while
   charging its tokens/latency to every medium query. Verification is
   the local evidence-coverage measurement (per the architecture:
   evidence-based, not a second opinion), so the check still runs and
   can still escalate — it just no longer pays for a discarded call.
   The verified_large path keeps its in-graph verifier model call.
2. **Per-tier output caps** (`num_predict`: small 192, medium 384,
   large 768, verifier 160). Small models occasionally rambled (one
   easy query generated 470+ tokens). Caps apply to the baselines too
   (same adapter), keeping comparisons even-handed.

Tests: 27 passing (4 new in tests/test_tuning.py). Re-run
`python -m benchmark.run_benchmark` with NR_MODEL_MODE=ollama to
refresh the results table with tuned numbers.

## Phase 2 — LLM decision model (2026-11-14)

The router itself can now be a model. `LLMDecisionModel`
(backend/intelligence/decision_model.py) asks a lightweight LLM —
the small tier by default (NR_DECISION_TIER to change) — to choose a
strategy as structured JSON. It never builds or executes anything
itself: the plan is still constructed deterministically by
build_plan() and must pass the graph validator.

Safety rails, in order:
1. Guardrails outrank the model: high-risk or document inputs are
   forced to verified_large whatever the model proposes.
2. The model may only pick one of the three registered strategies.
3. Non-JSON output, an unknown strategy, or an unreachable model
   falls back to the rule-based router — routing can never take the
   system down.
4. The decision call is not free: its tokens/latency are charged to
   the request totals and recorded in every trace ("decision" field:
   source llm | llm_guardrail_override | llm_fallback_rules | rules).

Enable it (default stays "rules" for reproducible benchmarks):

    export NR_DECISION_MODEL=llm
    export NR_MODEL_MODE=ollama         # counterpart; needs real models

Router A/B experiment: run the benchmark once per mode and compare
routing accuracy and the overhead the LLM router adds:

    NR_DECISION_MODEL=rules python -m benchmark.run_benchmark
    NR_DECISION_MODEL=llm   python -m benchmark.run_benchmark

Tests: 35 passing (8 new in tests/test_decision_model.py, all with
stub adapters — no real model needed).

### Router A/B results (real, local Ollama, 24 queries, 2026-11-15)

| Router | Routing | NR latency | NR tokens | Quality |
|---|---|---|---|---|
| Rules (tuned) | 24/24 | 10,022 ms | 440 | 0.97 |
| LLM v1 (qwen2.5:1.5b) | 19/24 | 16,968 ms | 939 | 0.94 |
| LLM v2 (calibrated prompt) | 24/24 | 12,279 ms | 927 | 0.96 |

Findings:
- The v1 prompt's misses were **all over-escalation**: 5 of 8 medium
  (research/explain) queries were sent to `verified_large` "to be safe".
  No query was under-routed, and guardrails bounded every miss to
  non-critical queries.
- The v2 prompt fixed accuracy (24/24) by stating the cost asymmetry
  (~5x), requiring a concrete reason for `verified_large`, and adding
  a tie-break rule: **choose the cheaper strategy — the escalation
  controller catches weak answers downstream**, so the router should
  bias cheap.
- Even at accuracy parity, the LLM router costs ~2x the tokens of
  rules: the decision call itself (~450–500 tokens/query, mostly the
  prompt) is a structural overhead rules don't pay. Its value is
  generalisation to query types nobody wrote rules for, not beating
  rules on a fixed set. Rules therefore remain the default.

## Phase 3 — Dynamic execution DAGs (2026-11-15)

Plans are no longer only three fixed templates:

1. **Decomposition (`decomposed_research`)**: a genuinely multi-part
   request (multiple questions, semicolon clauses, compare requests,
   conjoined imperatives) is split by a deterministic decomposer
   into sub-questions. Each gets a parallel retrieve -> medium-generate
   branch, and a large-model **synthesis** node merges the branch
   answers. Guardrails still outrank it (a document multipart stays
   verified_large), and it escalates to verified_large, never sideways.
   False-positive splits found against the original benchmark
   ("Compare supervised and unsupervised...", "...and why it matters")
   are excluded by topic-size and anaphora rules, with regression
   tests: the original 24 queries still route 24/24, and a new
   multipart set (benchmark/queries_multipart.json) routes 6/6.
2. **LLM DAG planner (`NR_PLAN_MODEL=llm`)**: a lightweight LLM now
   proposes the execution graph itself as JSON — node kinds, tiers,
   dependencies, parallelism. The proposal runs only if
   graph_validator authorises it unchanged (allowlist, registered
   tiers, acyclic, within budget); guardrail inputs bypass it, and
   any failure falls back to the strategy templates. Its planning
   call is charged to the request like the Phase 2 router's. A
   free-form llm_dag plan has no escalation template, so a
   low-confidence result stands flagged uncertain rather than
   escalating.

Tests: 48 passing (13 new in tests/test_phase3.py, stub adapters).
Multipart benchmark:
    python -m benchmark.run_benchmark --dataset benchmark/queries_multipart.json --output benchmark/results_multipart.json

## Retrieval upgrade — real documents + embeddings (2026-11-15)

Retrieval is no longer a fixed toy corpus scored by hand:

- **Embeddings** (backend/retrieval/embeddings.py): every document
  and query becomes a dense vector; relevance is cosine similarity.
  Default is a dependency-free hashing embedder (deterministic,
  lexical). Install `sentence-transformers` and real semantic
  embeddings (all-MiniLM-L6-v2) are used automatically;
  NR_EMBEDDINGS=hash|sentence-transformers forces a choice.
- **Real ingestion**: POST /documents (or the dashboard's upload
  button) accepts .pdf (PyMuPDF), .txt and .md files. Documents are
  chunked (paragraph-aware, ~1200 chars with overlap), embedded and
  indexed in memory — the in-memory vector index is the seam where
  Qdrant slots in later; nothing upstream changed.
- **Scoped retrieval**: a document plan's retrieve node carries the
  attachment names, and search() restricts itself to those ingested
  sources — a question about an uploaded report now reads *that
  report*. Filename-only attachments (as in the benchmark) fall back
  to the full index, and unscoped search keeps the strict relevance
  threshold so unanswerable queries still escalate.
- GET /documents lists ingested sources and chunk counts; /health
  reports the active embedder.

Tests: 54 passing (6 new in tests/test_retrieval_upgrade.py,
including a generated-PDF ingestion test).

## Dashboard — visual execution graph (2026-11-15)

The dashboard now renders the execution graph as a real DAG (SVG):
nodes are laid out in topological levels — parallel branches appear
side by side — with dependency arrows, per-node kind/tier/status/
latency, and kind-based colours (retrieve teal, generate indigo,
verify green). The routing card also shows the decision source
(rules | llm | llm_dag + planner tier/tokens), and escalations are
annotated above the graph they produced. The ad-hoc verify node now
records its true dependency (the plan's final generate node) in the
trace so it places correctly in decomposed and LLM-planned graphs.

Tests: 56 passing (2 new in tests/test_dashboard.py; the layout
logic is also exercised headlessly against a decomposed trace).

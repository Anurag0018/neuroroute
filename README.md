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

## Embeddings A/B results (2026-11-15, benchmark/retrieval_ab.py)

8 probes over the seed corpus — 3 keyword-style (query uses the
document's own words) and 5 paraphrase-style (same meaning, different
words). Top-1 hit = the expected document ranked first.

| Embedder | Keyword | Paraphrase |
|---|---|---|
| hashing-1024 (default, no deps) | 3/3 | 3/5 |
| all-MiniLM-L6-v2 (sentence-transformers) | 3/3 | 4/5 |

Findings:
- Keyword queries tie — both embedders are perfect, though MiniLM's
  relevance margins are larger (0.876 vs 0.611 on the quantization
  probe).
- Paraphrase is where they separate. Hashing scores the correct
  document at relevance 0.0 on two probes ("chatbot stop making up
  facts" → RAG doc; "huge neural network on a small laptop" →
  quantization doc): zero shared tokens, zero retrieval. MiniLM
  fixes the laptop probe outright (top-1, 0.208) and lifts the
  chatbot probe's correct document from 0.0 to 0.152 — still not
  top-1 (the embeddings document edges it out), an honest near-miss.
- Verdict: hashing is a sound zero-dependency default for
  keyword-shaped traffic; semantic embeddings earn their heavy
  install (torch) exactly when users phrase things their own way.
  Full per-probe data: benchmark/retrieval_ab_results.json.

## Phase 4 — Confidence calibration (2026-11-15)

Method: a fresh 24-query Ollama run whose rows record the final
confidence score next to measured quality (benchmark/calibrate.py
buckets answers by band and sweeps alternative return thresholds).
Full data: benchmark/results-calibration.json.

Band quality under the current policy (return >= 0.85, escalate < 0.60):

| Band | n | Mean quality | Min quality |
|---|---|---|---|
| return | 11 | 0.97 | 0.667 |
| verify | 7 | 1.00 | 1.00 |
| escalate | 6 | 0.945 | 0.667 |

Findings:
- Zero escalations occurred and routing was 24/24 — the pipeline
  behaved; what calibration exposed was a measurement problem.
- The bands do NOT separate quality, because confidence is
  confounded by strategy: the escalate band is mostly
  verified_large rows. Hard queries averaged quality 0.958 at mean
  confidence 0.401; hard_1 produced a perfect answer scored 0.095.
- Why: confidence measures evidence-groundedness against the
  10-document seed corpus; the quality proxy measures keyword
  presence in the answer regardless of grounding. Hard queries
  are largely unanswerable from that corpus, so groundedness is
  low even when the large model's parametric answer scores
  perfectly. The two metrics measure different things, and on
  this benchmark the proxy cannot validate the confidence scale
  for the hard path. (An earlier draft of this section blamed
  embedder relevance scaling; the embeddings A/B data contradicts
  that - MiniLM relevance scored HIGHER than hashing on every
  shared probe - so that explanation was withdrawn.)
- One genuinely over-confident row: med_2 (confidence 0.879,
  quality 0.667).
- The sweep's mechanical recommendation (loosen return to 0.70)
  was NOT adopted: it would mask the relevance-scaling issue
  instead of fixing it. Thresholds stay 0.85 / 0.60 and are now
  env-overridable (NR_RETURN_THRESHOLD / NR_ESCALATE_THRESHOLD)
  so a future, cleaner calibration can be applied without a code
  change. Recorded next steps: log the confidence components
  (evidence coverage, retrieval relevance, validation, agreement)
  per benchmark row - now done in the runner - so the next
  calibration can decompose the score instead of guessing at it;
  evaluate hard-path confidence against a corpus that actually
  covers the hard queries; and log pre-escalation confidence for
  an uncontaminated calibration set.
- Run configuration note: this run executed with the LLM decision
  model active (NR_DECISION_MODEL=llm was still exported in the
  benchmark terminal — found via the run's uniform ~+450
  tokens/query decision overhead and confirmed by env check).
  NeuroRoute averaged 910 tokens vs 440 in the rules-router tuned
  run, at latency 11,233ms (-74.6% vs an always-large baseline
  that ran slow this session, 44,252ms). The confidence/quality
  conclusions are unaffected — the LLM router matched the expected
  strategy 24/24 — but the token figures belong to the LLM-router
  configuration and are not comparable to the rules run.

## Phase 5 — Cost model & budgets (2026-11-15)

Until now the dollar figures only existed in mock mode: the Ollama
adapter returned cost_usd = 0.0 ("local inference is free"), so in
real runs every trace, metric and budget check saw $0. Phase 5
makes cost a first-class, modelled quantity in both modes.

Rate card (USD per 1K tokens, blended input+output, reference
prices in backend/models/model_registry.py):

| Tier | $/1K tokens |
|---|---|
| small | 0.00015 |
| medium | 0.0006 |
| large | 0.0025 |
| verifier | 0.0006 |
| retrieval node | 0.00005 flat |

Local inference has no invoice, so these are MODELLED costs: real
token counts priced at the reference card, which makes routing
economics measurable and comparable to cloud APIs. Any tier's rate
can be replaced with a real provider's price via env override
(NR_PRICE_SMALL_PER_1K, NR_PRICE_MEDIUM_PER_1K,
NR_PRICE_LARGE_PER_1K, NR_PRICE_VERIFIER_PER_1K; invalid values
fall back to the defaults). /health exposes the effective card.

Budget enforcement (budgets existed since Phase 1; now they bind
on real numbers):
- Plan time: the validator rejects any plan whose estimated cost
  exceeds the request budget (default cap $0.05, 5,000 tokens,
  4 model calls, 10s, 1 escalation per request).
- Run time: after every attempt the budget manager re-checks
  actual tokens / dollars / calls / latency and stops escalation
  when a cap is hit. Estimated and actual cost are tracked as
  separate fields end to end.

Costed run (2026-11-15, Ollama, 24 queries, routing 24/24;
benchmark/results-cost.json):

| System | Tokens/query | $/query | $/1,000 queries | Quality | Quality/$ |
|---|---|---|---|---|---|
| always-large | 544 | 0.00136 | 1.36 | 0.98 | 721 |
| always-medium | 197 | 0.000118 | 0.12 | 0.69 | 5,848 |
| NeuroRoute | 891 | 0.000528 | 0.53 | 0.95 | 1,799 |

Findings:
- Tokens are not dollars. In this run NeuroRoute used 64% MORE
  tokens than always-large yet cost 61% LESS ($0.000528 vs
  $0.00136 per query), because its tokens sit mostly in the
  small/medium tiers while always-large pays the large rate for
  every token. Tier mix, not token count, drives cost - the
  reason the project tracks both.
- Quality per dollar: NeuroRoute 1,799 vs always-large 721
  (2.5x). Always-medium posts the highest raw figure (5,848)
  because the proxy rewards its cheap partial answers; it cannot
  reach the 0.95 quality tier at any budget in this setup.
- Configuration note: this run executed with the LLM decision
  model active (NR_DECISION_MODEL=llm was still exported in the
  benchmark terminal - detected from the easy-row token
  fingerprint: 573-759 tokens vs the ~212 maximum for a
  rules-mode small-tier call, and confirmed by the 891-token
  average matching the Phase 2 A/B signature). The rules
  configuration (440 tokens/query in the tuned run, identical
  strategies) will price lower; its exact figure awaits a pinned
  re-run:
    NR_MODEL_MODE=ollama NR_DECISION_MODEL=rules \
      python -m benchmark.run_benchmark \
      --output benchmark/results-cost.json

## Tool engine — subsystem 3 completed (2026-11-15)

The architecture document's Tool Engine ("APIs / Python / Search";
execution layer: "Tool executor — APIs, calculations, web
search") is now built, under the document's own rule: the model
proposes, the backend authorises — never arbitrary tool execution.

- New node kind `tool` in the validator allowlist. A tool node is
  valid only if it names a REGISTERED tool in its params and
  carries no model tier. Registered tools (backend/execution/
  tools.py): `calculator` (arithmetic + percentages, evaluated by
  walking an AST allowlist — no eval(), names/calls/imports are
  rejected), `datetime` (weekdays, days-between, date offsets),
  `web_search` (DuckDuckGo Instant Answers, no API key).
- Tool output enters the graph as evidence documents with
  relevance 1.0 — the architecture's "tool validation" confidence
  signal: a deterministic result is ground truth, and a failed
  tool yields no evidence, so confidence (and escalation) price
  its absence. A failed tool path escalates tool_assisted ->
  rag_medium.
- Routing: deterministic detection (digits around an operator, a
  parseable date + a date question, an explicit search request)
  sends such queries to the new `tool_assisted` strategy: tool
  node(s) first, then a small model phrases the result (medium
  for web snippets). Guardrails (high risk / documents) still
  outrank tools. The LLM strategy router and the LLM DAG planner
  can both select tools too — the planner via tool nodes with
  params, authorised by the same validator.
- The document lists "Python" as a tool surface; that is
  deliberately NOT implemented as free-form code execution
  (architecture decision 2: keep execution deterministic, no
  unrestricted agents). The calculator is the bounded form of it.
- Benchmark: benchmark/queries_tools.json (4 deterministic tool
  queries) routes 4/4 in mock mode; base and multipart routing
  unchanged (24/24, 6/6 — detection has no false positives on
  either set). 89 tests total.

Measured (real Ollama run, rules router, 2026-11-15;
benchmark/results-tools.json):

| System | Latency | Tokens | $/query | Quality | Quality/$ |
|---|---|---|---|---|---|
| always-large | 8,696ms | 254 | 0.000636 | 0.50 | 786 |
| always-medium | 1,341ms | 136 | 0.000082 | 0.25 | 3,049 |
| NeuroRoute (tool) | 464ms | 124 | 0.000039 | 1.00 | 25,641 |

On the task class where LLMs are structurally unreliable —
exact arithmetic and date math — the large model scored 0.50
and the medium 0.25 doing it mentally, while the tool path was
perfect, 94.7% faster and 93.9% cheaper than always-large
(quality-per-dollar 25,641 vs 786, a 33x gap). This is the
architecture document's core claim in miniature: choose the
right computational pathway, don't just buy a bigger model.

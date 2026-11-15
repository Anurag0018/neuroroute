# NeuroRoute

**Adaptive AI Inference & Orchestration Platform**

Most AI applications send every request to one fixed pipeline —
usually the biggest model they have. NeuroRoute treats inference
as a *routing* problem instead: a lightweight decision model
decides **how** each request should be answered (which model
tier, whether to retrieve, which tools, whether an image or
audio file must be perceived first, how much budget it may
spend), a validator authorises the resulting execution graph,
and a confidence engine checks the outcome and escalates when
it isn't good enough. The objective is **quality per unit
cost**, measured against fixed always-large and always-medium
baselines on identical datasets — decision and verification
overhead included in NeuroRoute's own totals.

Everything runs locally (Ollama) with a deterministic mock mode
for tests and demos; no API keys are required.

## Headline results (real local models, 2026-11-14/15)

| Benchmark | Routing | Quality (NR vs large / medium) | Headline |
| --- | --- | --- | --- |
| Base, 24 queries | 24/24 | 0.97 vs 1.00 / 0.88 | **-62.6% latency** vs always-large |
| Base, costed (rules) | 24/24 | 0.97 vs 0.98 / 0.67 | **$0.000513 vs $0.001339 per query (-61.7%)**; quality/$ 1,891 vs 732 |
| Tools, 4 queries | 4/4 | **1.00** vs 0.50 / 0.25 | Quality/$ **25,641 vs 786**; -94.7% latency |
| Multipart research, 6 | 6/6 | 0.92 vs 0.92 / 0.92 | -59.9% latency at equal quality |
| Vision, 3 images | 3/3 | **0.67** vs 0.50 / 0.50 | Capability path: baselines answer blind |
| Audio, 3 voice notes | 3/3 | **0.67** vs 0.00 / 0.00 | Capability path: baselines cannot hear |

Quality is a transparent keyword-coverage proxy (documented in
each dataset), not a judge model. Full methodology, per-phase
results and the failures along the way are in the development
log below — nothing is cherry-picked; runs that executed in
the wrong configuration are labelled as such.

## Architecture

```
                ┌────────────────────────────────────────────┐
  request ─────►│ 1 · INTELLIGENCE & ROUTING                 │
  (text, files) │ input analysis → complexity estimate →     │
                │ decision model (rules or lightweight LLM)  │
                └──────────────────┬─────────────────────────┘
                                   │ proposes a plan (never executes)
                ┌──────────────────▼─────────────────────────┐
                │ 2 · DYNAMIC EXECUTION PLANNING             │
                │ graph validator: allowlisted node kinds &  │
                │ tiers, registered tools only, acyclic,     │
                │ budget-capped — guardrails outrank models  │
                └──────────────────┬─────────────────────────┘
                                   │ authorised DAG
                ┌──────────────────▼─────────────────────────┐
                │ 3 · MODEL & TOOL ORCHESTRATION             │
                │ DAG executor (parallel levels) over:       │
                │ small/medium/large LLMs · retrieval        │
                │ (embeddings + vector store) · tools        │
                │ (calculator / datetime / web search) ·     │
                │ vision perception · audio transcription    │
                └──────────────────┬─────────────────────────┘
                                   │ answer + evidence
                ┌──────────────────▼─────────────────────────┐
                │ 4 · CONFIDENCE & VERIFICATION              │
                │ evidence coverage · validation · retrieval │
                │ relevance · model agreement → score, then  │
                │ return / verify / escalate (hard limits)   │
                └──────────────────┬─────────────────────────┘
                ┌──────────────────▼─────────────────────────┐
                │ 5 · BUDGETS        6 · OBSERVABILITY       │
                │ per-request caps   full trace per request: │
                │ (tokens, latency,  spans, decision source, │
                │ cost, calls),      confidence, cost — via  │
                │ enforced at plan   dashboards, JSON and    │
                │ time and run time  Prometheus metrics      │
                └────────────────────────────────────────────┘
```

**Request lifecycle.** (1) The input analyzer classifies the
request (modality, risk, document/image/audio attachments).
(2) The complexity estimator scores it. (3) The decision model
— deterministic rules by default, or a lightweight LLM behind
the same interface — selects a strategy or proposes a full
graph. (4) The validator authorises or rejects the plan; an
LLM never executes unchecked, and deterministic guardrails
(high-risk or document requests) outrank any model. (5) The
executor runs the DAG level by level, in parallel within a
level. (6) The confidence engine scores the result from
evidence, not self-report; below threshold, the escalation
controller retries on a stronger path — at most once, never
repeating a failed strategy without new evidence, and only
while the budget allows.

**Execution strategies** (the plan templates the decision
model chooses between):

| Strategy | Graph | Chosen when |
| --- | --- | --- |
| `direct_small` | generate (small) | Simple, low-risk queries |
| `rag_medium` | retrieve → generate (medium) | Retrieval or moderate reasoning needed |
| `verified_large` | [retrieve / perceive / transcribe →] generate (large) → verify | High complexity, high risk, or document-grounded |
| `decomposed_research` | parallel (retrieve → generate medium) per sub-question → synthesize (large) | Multi-part / comparison requests |
| `tool_assisted` | tool(s) → generate (small/medium) | Arithmetic, dates, live web facts |
| `vision_grounded` | perceive → generate (medium) | Image attachments |
| `audio_grounded` | transcribe → generate (medium) | Audio attachments |
| `llm_dag` | planner-proposed graph, validator-authorised | `NR_PLAN_MODEL=llm`: the LLM designs the graph itself |

Node kinds the validator allows: `retrieve`, `generate`,
`verify`, `tool`, `perceive`, `transcribe`. Tool nodes must
name a registered tool; perception/transcription nodes must
name an attachment; no node may invent a model tier.

**Confidence & verification.** Score = 0.35·evidence coverage
+ 0.25·validation + 0.20·retrieval relevance + 0.20·model
agreement. Bands: return ≥ 0.85, verify 0.60–0.84, escalate
< 0.60 (env-tunable; calibration findings in the Phase 4 log).
Verification is evidence- and rule-based — never a second
LLM's opinion of the first.

**Safety model.** Plans are validated before anything runs
(allowlist, acyclicity, budget estimate); budgets are
re-checked after every attempt at run time; free-form code
execution is deliberately absent — the calculator evaluates an
AST allowlist instead. Optional API-key auth guards the HTTP
interface.

**Stack (as built).** FastAPI modular monolith · custom async
DAG executor · Ollama tiers (Qwen 2.5 1.5B / Llama 3.2 3B /
Qwen 2.5 7B, llava for vision) · hashing embeddings by default,
sentence-transformers when installed · in-memory vector store
by default, Qdrant when configured · faster-whisper for audio
· Next.js console + built-in observability dashboard ·
Prometheus metrics endpoint · Docker Compose · MCP server
interface. Substitutions vs the original design sketch are
documented in the log (custom DAG for LangGraph, AST
calculator for free-form Python tools, vision-model
transcription for a separate OCR engine).

**Interfaces.** REST API (below) for applications; an **MCP
server** (`mcp_server.py`) exposing the same pipeline and tool
registry to AI clients; two dashboards (built-in at
`/dashboard`, Next.js at `frontend/`).

## Quick start

> Python 3.10+ recommended (3.9 works via the
> `eval_type_backport` dependency in requirements.txt).

```bash
cd neuroroute
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8000/dashboard

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "What is Python?"}'

# Decision only, nothing executes:
curl -X POST localhost:8000/route/preview -H 'Content-Type: application/json' \
  -d '{"query": "Analyse this research paper", "attachments": ["paper.pdf"]}'
```

**Real models (Ollama).** Install Ollama, pull the tiers
(`ollama pull qwen2.5:1.5b llama3.2:3b qwen2.5:7b`, plus
`llava:7b` for vision), then start with `NR_MODEL_MODE=ollama`.
Key configuration (all optional, mock/rules defaults):

| Variable | Effect |
| --- | --- |
| `NR_MODEL_MODE` | `mock` (default) or `ollama` |
| `NR_DECISION_MODEL` | `rules` (default) or `llm` router |
| `NR_PLAN_MODEL` | `strategy` (default) or `llm` DAG planner |
| `NR_EMBEDDINGS` | `hash` forces hashing; MiniLM auto-used when installed |
| `NR_VECTOR_STORE` | `memory` (default) or `qdrant` |
| `NR_VISION_MODEL` | default `llava:7b` |
| `NR_AUDIO_MODE` / `NR_WHISPER_MODEL` | `auto`/`whisper`/`mock`; default `tiny` |
| `NR_API_KEY` | when set, non-public routes require it |
| `NR_PRICE_<TIER>_PER_1K` | override the modelled rate card |

> Environment variables live only in the terminal that sets
> them — the benchmark log below contains three separate runs
> that silently executed in the wrong configuration because of
> this. Pin modes inline in benchmark commands.

## Benchmarks

Datasets in `benchmark/` (base 24, multipart 6, tools 4,
vision 3, audio 3), each run against always-large and
always-medium fixed pipelines on identical inputs:

```bash
NR_MODEL_MODE=ollama NR_DECISION_MODEL=rules \
  python -m benchmark.run_benchmark --output benchmark/results.json
```

Routing accuracy judges the *initial* strategy (an escalation
is the safety net working, not a miss). Mock-mode runs check
the harness only; their numbers are meaningless by design.

## Tests

```bash
pytest -q   # 118 tests
```

Routing, graph validation, the full pipeline, escalation,
retrieval, tools, multimodal paths, auth, the MCP protocol
round-trip, and a real embedded-Qdrant round-trip. Tests that
need optional packages (Qdrant, faster-whisper, MCP SDK) skip
when those aren't installed.

## Project structure

```
backend/
  intelligence/   analysis, complexity, decision models, routing policy
  planning/       graph generator, validator, decomposer, LLM DAG planner
  execution/      DAG executor, node executor, tools, perception, transcription
  models/         tier registry + rate card, mock/Ollama adapters
  retrieval/      embeddings, document loader, vector stores, retrieval service
  validation/     verifier, confidence estimator, escalation controller
  optimization/   budget manager
  observability/  execution tracer (+ Prometheus exposition)
  main.py         FastAPI app (REST, dashboards, metrics, auth)
frontend/         Next.js console (proxies /api to FastAPI)
mcp_server.py     MCP interface over the same pipeline
benchmark/        datasets, runner, calibration + retrieval A/B tools
tests/            pytest suite
```

---

## Development log — measured results, phase by phase

Everything below is the chronological build log with the real
numbers as they were measured, including the tuning, the
calibration study, the wrong-configuration runs and what they
taught. The headline table above is drawn from these sections.

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
  average matching the Phase 2 A/B signature).

Rules configuration, measured (pinned re-run, 2026-11-15;
benchmark/results-cost-rules.json): routing 24/24, NeuroRoute
396 tokens/query (the rules fingerprint - no decision call),
quality 0.97, avg cost $0.000513/query vs always-large
$0.001339 (-61.7%), latency -74.1% vs always-large,
quality-per-dollar 1,891 vs 732. Both configurations are now
measured, and the comparison is the interesting part: the LLM
router uses 2.2x the tokens of rules (891 vs 396) yet costs
only ~3% more per query ($0.000528 vs $0.000513), because the
decision call runs on the cheapest tier. Token overhead is not
cost overhead when the overhead itself is routed cheaply -
which is, in miniature, the platform's whole thesis.

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

## Multimodal engine — subsystem 3 completed (2026-11-15)

The architecture routes multimodal requests "through OCR, vision
or audio preprocessing only when required" (section A). That is
now implemented as a first-class graph node:

- **New node kind `perceive`** (validator allowlist): runs the
  vision model over one image attachment and emits its output as
  evidence documents (description + exact transcription of
  visible text — the OCR duty), which downstream generate nodes
  ground on exactly like retrieved documents. It takes no model
  tier and must name its attachment, or the plan is rejected.
- **Routing**: an image attachment routes to the new
  `vision_grounded` strategy (perceive -> medium generate).
  Perception is a requirement, not a complexity judgement, so
  the modality decision precedes the complexity path; hard image
  questions escalate vision_grounded -> verified_large, and the
  verified template *keeps* the perceive node — escalating away
  from the image evidence would defeat the verification.
  Guardrails still outrank: a high-risk image question ("what
  medicine for this rash?" + photo) goes straight to
  verified_large, perceive node included.
- **Perceiver**: mock by default (a labelled placeholder — no
  visual content is invented); with NR_MODEL_MODE=ollama it calls
  a local vision model (NR_VISION_MODEL, default llava:7b;
  moondream is a lighter alternative), priced at the medium
  tier's reference rate. A missing file or unreachable model
  fails the node with no evidence, like a failed tool.
- Uploads: POST /documents now accepts images (stored for
  per-query perception instead of text indexing); the dashboard
  upload accepts image files and auto-attaches them.
- The LLM DAG planner bypasses image requests to the
  vision_grounded template (it cannot know attachment filenames;
  perception is a fixed preprocessing requirement, not a design
  choice), while the Phase 2 router prompt documents the new
  strategy.
- Benchmark: benchmark/queries_vision.json (3 queries over the
  generated assets in benchmark/assets/ — a labelled bar chart
  and an invoice screenshot) routes 3/3 in mock mode; base,
  multipart and tool routing unchanged. 99 tests total. Real
  runs need `ollama pull llava:7b` once.

Measured (real Ollama run, llava:7b, rules router, 2026-11-15;
benchmark/results-vision.json): routing 3/3, NeuroRoute quality
0.67 vs 0.50 for both baselines — but latency 25,105ms vs
5,957ms and $0.001035 vs $0.000307 per query, so quality-per-
dollar favours the baselines here (647 vs 1,629). Read honestly:
the vision path is a *capability* path, not an optimisation
path. The baselines are cheap because they answer blind — they
cannot see the image at all, and their 0.50 is keyword luck on
inputs that are unanswerable for them — while NeuroRoute pays
for an actual perception pass: llava:7b on a MacBook Air is slow
(~25s/image) and its image tokens inflate the token count
(1,296 avg). The win is answering at all, at the best quality
of the three; a smaller vision model (moondream via
NR_VISION_MODEL) is the obvious lever on the latency.

## Qdrant vector store + Next.js frontend — project complete (2026-11-15)

The last two named technologies in the architecture document are
now in place, both behind seams the design already had:

- **Qdrant** (retrieval store): the RetrievalService policy
  (scoping, thresholds, unknown-source fallback) is unchanged and
  store-independent; the store itself is selected by
  NR_VECTOR_STORE — `memory` (default, zero dependencies) or
  `qdrant`. Qdrant runs in embedded local mode under data/qdrant
  (no server needed) or against NR_QDRANT_URL, with cosine
  distance so scores match the in-memory store's. Needs the
  optional `qdrant-client` package. Verified by a round-trip test
  against a real embedded Qdrant: seed search parity, scoped
  search, replace-on-reingest, and the unknown-source fallback all
  behave identically to the in-memory store. /health reports the
  active store.
- **Next.js frontend** (frontend/): the product UI from the
  architecture's stack section — a console over the same FastAPI
  API: health chips (mode, router, planner, embeddings, vector
  store, vision model), query + document/image upload with
  auto-attach, the answer with strategy/confidence/cost, the
  execution span waterfall, aggregate metrics and a clickable
  recent-traces list. It proxies /api/* to the backend via Next
  rewrites (no CORS), and is verified end-to-end against the
  running API. The built-in /dashboard remains the
  zero-dependency fallback. Run it with:

      cd frontend && npm install && npm run dev   # http://localhost:3000

With this, every subsystem and technology named in the
architecture document is implemented: routing intelligence, LLM
decision model, dynamic DAG planning with a validating backend,
model/tool/vision orchestration, evidence-based confidence with
calibrated thresholds, a priced budget system, HoneyHive-style
observability, and benchmarked quality-per-unit-cost against
always-large and always-medium baselines. 103 tests.

## Audio path — multimodal engine complete (2026-11-15)

The last capability named in the architecture (section E: "Audio
model — speech recognition"; section 1: Text / PDF / Image /
Audio) is implemented exactly like the vision path:

- **New node kind `transcribe`**: turns an audio attachment into
  evidence text (the transcript) that a generate node grounds on.
  Validator rules mirror `perceive`: no model tier, must name its
  attachment.
- **Routing**: audio attachments route to the new
  `audio_grounded` strategy (transcribe -> medium generate),
  escalating to verified_large — whose template keeps the
  transcribe node, the same evidence-preservation rule as images.
  High-risk audio queries go straight to verified_large with the
  transcript included. The DAG planner bypasses audio requests to
  the template, as with images.
- **Transcriber**: faster-whisper running locally (optional
  package; NR_WHISPER_MODEL, default "tiny", int8 CPU), selected
  automatically when installed (NR_AUDIO_MODE=whisper|mock|auto).
  Without it, a labelled mock placeholder — no speech content is
  invented. Missing file / no speech / decode failure = the node
  fails with no evidence. Note the version pin in
  requirements.txt: av>=13 breaks faster-whisper's audio decode
  (found by the round-trip test below).
- **Verified end-to-end with real speech**: benchmark/assets/
  voice_note.wav is synthesized speech (ffmpeg flite) saying
  "NeuroRoute routes each request to the cheapest reliable path";
  the tiny model transcribed it in ~4s as "Narrowed Roots each
  request to the cheapest reliable path" — content words intact,
  proper noun misheard, which is exactly the tiny model's known
  accuracy class. A round-trip test asserts on it.
- Uploads accept audio files in both dashboards; transcribe
  nodes render red in the execution graph. Benchmark:
  benchmark/queries_audio.json routes 3/3 in mock mode. 112 tests.

Measured (real run, faster-whisper tiny, rules router,
2026-11-15; benchmark/results-audio.json): routing 3/3,
NeuroRoute quality 0.67 vs 0.00 for both baselines - the
baselines score literally zero because they cannot hear the
audio at all; like the vision path, this is a capability path,
the only one of the three that can answer. Latency 3,544ms
(faster than always-large's 3,939ms), $0.000278/query. The
0.67 rather than 1.00 is the tiny whisper model's ceiling: it
transcribes the content words faithfully ("...cheapest
reliable path") but mishears the product name ("Narrowed
Roots"), so answers quoting it miss the routing keyword group;
NR_WHISPER_MODEL=base is the accuracy lever. (Two earlier
attempts at this run scored 1/3 for an environment reason, not
a code one: NR_DECISION_MODEL=llm was still exported in the
benchmark terminal and the run command did not pin it, so the
1.5B router sent both voice-note queries to rag_medium.
Pinning the mode inline - as the vision and cost runs did -
fixed it. The export trap, third occurrence.)

With this, every row of the architecture's execution table —
small/medium/large LLMs, embeddings, vector DB, vision, OCR,
audio, tools, verification — is implemented and tested.

## Infrastructure pack (2026-11-15)

The deployment rows of the architecture's stack table:

- **API-key authentication** (section 1, API gateway): set
  NR_API_KEY and every non-public route requires it via the
  X-API-Key header or a Bearer token; unset = open (the local
  default). Public routes stay public: /, /health, /dashboard,
  /docs. The key is read per request (rotation without restart),
  /health reports `auth_enabled`, and the built-in dashboard has
  a key field (stored in localStorage) that sends it on every
  call. **Session management**: /query accepts a session_id,
  traces record it, and GET /traces?session_id=... filters by it.
- **Prometheus metrics**: GET /metrics/prometheus serves the
  exposition format — per-strategy request counters, token and
  cost totals, escalation counter/rate, mean latency — derived
  from the same traces the dashboards use, ready for Prometheus
  scraping and Grafana charting.
- **Docker**: Dockerfile for the API, frontend/Dockerfile for
  the Next.js console, and docker-compose.yml wiring api + web +
  a Qdrant service (the API is preconfigured with
  NR_VECTOR_STORE=qdrant / NR_QDRANT_URL). Ollama stays on the
  host (models are large and Metal-bound on a Mac) and is reached
  via host.docker.internal. `docker compose up --build`, then
  localhost:3000 (console) or localhost:8000/dashboard.
  (The compose file is validated; the images were not built in
  the sandbox, which has no Docker — first build happens on the
  developer's machine.)
- Deliberately still swappable seams, not wired services:
  PostgreSQL/Redis. Nothing in the request path needs shared
  state yet — traces sit behind the ExecutionTracer class and
  uploads on the filesystem — so they are documented here as the
  production swap rather than deployed unused. 117 tests.

## MCP server — the agent interface (2026-11-15)

NeuroRoute now speaks two protocols over one execution core:

- **REST** (FastAPI) for applications — dashboards, benchmarks,
  scripts.
- **MCP** (Model Context Protocol, mcp_server.py) for AI clients
  — Claude Desktop, Cursor, other agents — which discover and
  call NeuroRoute's capabilities as standard MCP tools over
  stdio: `neuroroute_query` (the full routed pipeline),
  `neuroroute_route_preview` (the authorised plan, unexecuted),
  `neuroroute_calculator` / `neuroroute_datetime` /
  `neuroroute_web_search` (the registered tools directly),
  `neuroroute_metrics`, `neuroroute_health`.

Both interfaces wrap the same NeuroRoutePipeline and tool
registry, so a request entering via MCP receives the identical
validated, budget-capped execution as one entering via HTTP —
the interface is pluggable, the safety properties are shared.
Verified by a protocol round-trip test: an MCP client spawns the
server, lists its tools and calls them (calculator returns
52,986 for 1,247 * 38 + 5,600 through the protocol).

Setup: `.venv/bin/pip install "mcp>=1.2,<2"`, then register in
the client's MCP config:

```json
{
  "mcpServers": {
    "neuroroute": {
      "command": "/path/to/neuroroute/.venv/bin/python",
      "args": ["/path/to/neuroroute/mcp_server.py"],
      "env": {"NR_MODEL_MODE": "ollama"}
    }
  }
}
```

The usual NR_* variables configure the MCP server's pipeline
exactly as they configure the API server. 118 tests.

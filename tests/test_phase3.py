"""Phase 3 tests: query decomposition, the decomposed_research
strategy (parallel branches + synthesis), and the LLM DAG planner
behind NR_PLAN_MODEL=llm — all with stub adapters, no live models.
"""
import asyncio

from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.routing_policy import choose_strategy
from backend.models.llm_adapter import Generation
from backend.pipeline import NeuroRoutePipeline
from backend.planning.dag_planner import LLMDagPlanner
from backend.planning.decomposer import split_subquestions
from backend.planning.graph_generator import build_plan
from backend.planning.graph_validator import validate_plan
from backend.schemas import Budget
from backend.validation.escalation_controller import escalation_target
from backend.schemas import BudgetUsage, ConfidenceReport


def _route(query, attachments=None):
    a = input_analyzer.analyze(query, attachments)
    return choose_strategy(a, complexity_estimator.estimate(a)), a


# ---------- decomposer ----------

def test_split_two_questions():
    parts = split_subquestions(
        "What is retrieval augmented generation? "
        "How does chunking affect retrieval quality?")
    assert len(parts) == 2 and all(p.endswith("?") for p in parts)


def test_split_compare():
    parts = split_subquestions(
        "Compare how vector databases and traditional relational "
        "databases handle search")
    assert len(parts) == 2


def test_split_conjoined_imperatives():
    parts = split_subquestions(
        "Explain how embeddings work and describe how quantization "
        "reduces model size")
    assert len(parts) == 2


def test_single_question_not_decomposed():
    assert split_subquestions("What is Python?") == []
    # "and" inside one question must not split it
    assert split_subquestions(
        "What is the difference between supervised and unsupervised "
        "learning?") == []
    # A short compare pair is one grounded answer, not two branches
    assert split_subquestions(
        "Compare supervised and unsupervised learning with examples") == []
    # A follow-up clause that refers back ("why IT matters") is not
    # an independent sub-question
    assert split_subquestions(
        "Explain the transformer attention mechanism and why it "
        "matters for language models") == []


# ---------- routing + plan shape ----------

def test_multipart_routes_to_decomposed():
    strategy, _ = _route(
        "What is retrieval augmented generation? "
        "How does chunking affect retrieval quality?")
    assert strategy == "decomposed_research"


def test_document_multipart_stays_verified():
    strategy, _ = _route(
        "Analyse the attached report and compare its cost findings "
        "with its risk findings", ["migration_assessment.pdf"])
    assert strategy == "verified_large"


def test_decomposed_plan_shape_and_budget():
    _, a = _route("Explain how embeddings work and describe how "
                  "quantization reduces model size")
    c = complexity_estimator.estimate(a)
    plan = validate_plan(build_plan("decomposed_research", a, c, Budget()),
                         Budget())
    by_id = {n.node_id: n for n in plan.nodes}
    assert by_id["synthesize"].depends_on == ["generate_1", "generate_2"]
    assert by_id["generate_1"].depends_on == ["retrieve_1"]
    assert by_id["generate_1"].params["sub_question"].startswith("Explain")
    model_calls = sum(1 for n in plan.nodes if n.kind in ("generate", "verify"))
    assert model_calls == 3  # 2 branch mediums + 1 synthesis


# ---------- end-to-end (mock) ----------

def test_pipeline_decomposed_end_to_end():
    pipe = NeuroRoutePipeline()
    trace = asyncio.run(pipe.process(
        "What is retrieval augmented generation? "
        "How does chunking affect retrieval quality?"))
    assert trace["strategy"] == "decomposed_research"
    graph_ids = [n["node_id"] for n in trace["execution_graph"]
                 if n["node_id"] != "verify_adhoc"]
    assert graph_ids[-1] == "synthesize"  # synthesis is the final plan node
    assert trace["answer"]
    assert trace["metrics"]["model_calls"] == 3


# ---------- LLM DAG planner ----------

class StubAdapter:
    def __init__(self, text):
        self.text = text

    def generate(self, tier, prompt, context_docs=None):
        return Generation(text=self.text, tokens=64, cost_usd=0.0,
                          latency_ms=3)


def _analysis(query, attachments=None):
    a = input_analyzer.analyze(query, attachments)
    return a, complexity_estimator.estimate(a)


def test_dag_planner_accepts_valid_graph():
    dag = ('{"nodes": [{"id": "retrieve", "kind": "retrieve", "tier": null,'
           ' "depends_on": []}, {"id": "generate", "kind": "generate",'
           ' "tier": "medium", "depends_on": ["retrieve"]}],'
           ' "reason": "grounded"}')
    planner = LLMDagPlanner(adapter=StubAdapter(dag))
    a, c = _analysis("Explain how vector databases support semantic search")
    plan = planner.propose_plan(a, c, Budget())
    assert plan.strategy == "llm_dag"
    assert [n.node_id for n in plan.nodes] == ["retrieve", "generate"]
    assert planner.last_source == "llm_dag"
    assert planner.last_usage.tokens == 64


def test_dag_planner_falls_back_on_garbage():
    planner = LLMDagPlanner(adapter=StubAdapter("no json here"))
    a, c = _analysis("Explain how vector databases support semantic search")
    plan = planner.propose_plan(a, c, Budget())
    assert plan.strategy == "rag_medium"
    assert planner.last_source == "llm_dag_fallback_rules"


def test_dag_planner_falls_back_when_validator_rejects():
    # 5 large generate nodes: 5 model calls > default budget of 4.
    nodes = ", ".join(
        f'{{"id": "g{i}", "kind": "generate", "tier": "large",'
        f' "depends_on": []}}' for i in range(5))
    planner = LLMDagPlanner(adapter=StubAdapter('{"nodes": [' + nodes + "]}"))
    a, c = _analysis("Explain how vector databases support semantic search")
    plan = planner.propose_plan(a, c, Budget())
    assert plan.strategy == "rag_medium"
    assert planner.last_source == "llm_dag_fallback_rules"


def test_dag_planner_guardrail_forces_verified():
    planner = LLMDagPlanner(adapter=StubAdapter("{}"))
    a, c = _analysis("I have chest pain and shortness of breath, "
                     "what medicine should I take?")
    plan = planner.propose_plan(a, c, Budget())
    assert plan.strategy == "verified_large"
    assert planner.last_source == "llm_dag_guardrail"


def test_llm_dag_strategy_cannot_escalate_but_does_not_crash():
    conf = ConfidenceReport(confidence_score=0.1, evidence_coverage=0.0,
                            validation_passed=False, retrieval_relevance=0.0,
                            model_agreement=0.0, decision="escalate")
    assert escalation_target(conf, "llm_dag", BudgetUsage(), Budget()) is None

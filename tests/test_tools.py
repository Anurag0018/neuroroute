"""Tool engine: registry safety, detection precision, validator
authorisation, routing, and end-to-end execution in mock mode."""
import asyncio

import pytest

from backend.execution import tools
from backend.execution.node_executor import NodeExecutor
from backend.execution.tools import ToolError, detect_tools, run_tool
from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.routing_policy import choose_strategy
from backend.pipeline import NeuroRoutePipeline
from backend.planning.graph_validator import PlanRejected, validate_plan
from backend.schemas import Budget, ExecutionPlan, PlanNode


# --- calculator ----------------------------------------------------

def test_calculator_arithmetic_with_commas():
    out = run_tool("calculator", "What is 1,247 * 38 + 5,600?")
    assert "52,986" in out


def test_calculator_percent_of():
    assert "12,600" in run_tool("calculator", "Calculate 15% of 84,000")


def test_calculator_precedence():
    assert "= 14" in run_tool("calculator", "what is 2 + 3 * 4?")


def test_calculator_rejects_code():
    # AST allowlist: names, calls and imports are not arithmetic.
    with pytest.raises(ToolError):
        run_tool("calculator", "calculate __import__('os') * 2")
    with pytest.raises(ToolError):
        run_tool("calculator", "what is x + 2?")


def test_calculator_division_by_zero():
    with pytest.raises(ToolError):
        run_tool("calculator", "what is 10 / 0?")


# --- datetime ------------------------------------------------------

def test_datetime_days_between():
    out = run_tool("datetime",
                   "How many days between 15 November 2026 and "
                   "1 December 2026?")
    assert "16 days" in out


def test_datetime_weekday():
    out = run_tool("datetime", "What day of the week is 25 December 2026?")
    assert "Friday" in out


def test_datetime_add_days():
    out = run_tool("datetime", "What date is 30 days from 1 November 2026?")
    assert "01 December 2026" in out


# --- web search ----------------------------------------------------

def test_web_search_parses_instant_answers(monkeypatch):
    monkeypatch.setattr(tools, "_fetch_ddg", lambda q: {
        "AbstractText": "Anthropic is an AI company.",
        "AbstractSource": "Wikipedia",
        "RelatedTopics": [{"Text": "Founded in 2021."}, {"NoText": 1}]})
    out = run_tool("web_search", "search the web for Anthropic")
    assert "AI company" in out and "Founded in 2021" in out


def test_web_search_failure_is_tool_error(monkeypatch):
    def boom(q):
        raise RuntimeError("offline")
    monkeypatch.setattr(tools, "_fetch_ddg", boom)
    with pytest.raises(ToolError):
        run_tool("web_search", "search the web for anything")


# --- detection precision -------------------------------------------

def test_detect_tools_positives():
    assert detect_tools("What is 12 * 9 + 5?") == ["calculator"]
    assert detect_tools("What day of the week is 2026-12-25?") == ["datetime"]
    assert detect_tools("search the web for fusion news") == ["web_search"]


def test_detect_tools_no_false_positives():
    for q in ("Explain how vector databases support semantic search",
              "Compare supervised and unsupervised learning",
              "What is the probability of rain tomorrow?",
              "Summarize the 2026-11-15 release notes",  # a date, not subtraction
              "How does retrieval augmented generation work?"):
        assert detect_tools(q) == [], q


# --- validator -----------------------------------------------------

def _plan(nodes):
    return ExecutionPlan("tool_assisted", nodes, 300, 0.001, "test")


def test_validator_accepts_registered_tool():
    plan = _plan([PlanNode("tool_calculator", "tool", None, [],
                           {"tool": "calculator"}),
                  PlanNode("generate", "generate", "small",
                           ["tool_calculator"])])
    assert validate_plan(plan, Budget()) is plan


def test_validator_rejects_unknown_tool():
    plan = _plan([PlanNode("t", "tool", None, [], {"tool": "shell"}),
                  PlanNode("generate", "generate", "small", ["t"])])
    with pytest.raises(PlanRejected):
        validate_plan(plan, Budget())


def test_validator_rejects_tool_with_tier():
    plan = _plan([PlanNode("t", "tool", "small", [], {"tool": "calculator"}),
                  PlanNode("generate", "generate", "small", ["t"])])
    with pytest.raises(PlanRejected):
        validate_plan(plan, Budget())


# --- routing + execution -------------------------------------------

def test_tool_queries_route_to_tool_assisted():
    for q in ("What is 1,247 * 38 + 5,600?",
              "How many days are there between 15 November 2026 and "
              "1 December 2026?"):
        a = input_analyzer.analyze(q)
        assert choose_strategy(a, complexity_estimator.estimate(a)) \
            == "tool_assisted"


def test_tool_node_executor_emits_evidence():
    ex = NodeExecutor()
    node = PlanNode("tool_calculator", "tool", None, [],
                    {"tool": "calculator"})
    res = ex.execute(node, "What is 6 * 7?", {})
    assert res.status == "ok"
    assert res.output[0]["source"] == "tool:calculator"
    assert res.output[0]["relevance"] == 1.0
    assert "42" in res.output[0]["text"]


def test_pipeline_tool_path_end_to_end():
    result = asyncio.run(
        NeuroRoutePipeline().process("What is 1,247 * 38 + 5,600?"))
    assert result["strategy"] == "tool_assisted"
    kinds = [n["kind"] for n in result["execution_graph"]]
    assert "tool" in kinds and result["answer"]

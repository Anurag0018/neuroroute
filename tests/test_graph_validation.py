import pytest

from backend.planning.graph_validator import PlanRejected, validate_plan
from backend.schemas import Budget, ExecutionPlan, PlanNode


def plan(nodes, est_tokens=100, est_cost=0.001):
    return ExecutionPlan(strategy="test", nodes=nodes,
                         estimated_tokens=est_tokens, estimated_cost_usd=est_cost)


def test_valid_plan_passes():
    p = plan([PlanNode("generate", "generate", "small")])
    assert validate_plan(p, Budget()) is p


def test_unknown_node_kind_rejected():
    p = plan([PlanNode("hack", "shell_exec")])
    with pytest.raises(PlanRejected):
        validate_plan(p, Budget())


def test_unknown_model_tier_rejected():
    p = plan([PlanNode("generate", "generate", "gigantic")])
    with pytest.raises(PlanRejected):
        validate_plan(p, Budget())


def test_cycle_rejected():
    p = plan([PlanNode("a", "generate", "small", ["b"]),
              PlanNode("b", "generate", "small", ["a"])])
    with pytest.raises(PlanRejected):
        validate_plan(p, Budget())


def test_missing_dependency_rejected():
    p = plan([PlanNode("generate", "generate", "small", ["nope"])])
    with pytest.raises(PlanRejected):
        validate_plan(p, Budget())


def test_over_budget_plan_rejected():
    p = plan([PlanNode("generate", "generate", "large")], est_tokens=999_999)
    with pytest.raises(PlanRejected):
        validate_plan(p, Budget(max_total_tokens=5000))

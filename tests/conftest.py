"""Make the test suite hermetic: tests always run against the mock
adapter and the rule-based router, whatever the developer's shell has
exported. Without this, running pytest in a terminal with
NR_MODEL_MODE=ollama makes the API tests call real models — slow, and
their free-text answers break assertions about mock behaviour.
(conftest is imported before any test module, hence before backend
code reads the environment.)
"""
import os

os.environ["NR_MODEL_MODE"] = "mock"
os.environ["NR_DECISION_MODEL"] = "rules"
os.environ["NR_PLAN_MODEL"] = "strategy"

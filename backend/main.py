"""NeuroRoute API — FastAPI modular monolith (Phase 1).

Run:  uvicorn backend.main:app --reload
Docs: http://localhost:8000/docs
Dash: http://localhost:8000/dashboard
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.decision_model import RuleBasedDecisionModel
from backend.models.model_registry import REGISTRY
from backend.pipeline import NeuroRoutePipeline
from backend.planning.graph_validator import PlanRejected, validate_plan
from backend.schemas import Budget

app = FastAPI(title="NeuroRoute", version="0.1.0",
              description="Adaptive AI inference & orchestration platform — Phase 1 MVP")
pipeline = NeuroRoutePipeline()
_decision_model = RuleBasedDecisionModel()
_STATIC = Path(__file__).parent / "static"


class BudgetIn(BaseModel):
    max_total_tokens: int = 5000
    max_latency_ms: int = 10_000
    max_api_cost_usd: float = 0.05
    max_model_calls: int = 4
    max_escalations: int = 1


class QueryIn(BaseModel):
    query: str = Field(min_length=1)
    attachments: list[str] = []
    budget: Optional[BudgetIn] = None


def _budget(b: Optional[BudgetIn]) -> Budget:
    return Budget(**b.model_dump()) if b else Budget()


@app.get("/health")
def health():
    return {"status": "ok", "service": "neuroroute", "phase": 1,
            "model_mode": os.environ.get("NR_MODEL_MODE", "mock")}


@app.get("/models")
def models():
    return {tier: asdict(entry) | {"capabilities": list(entry.capabilities)}
            for tier, entry in REGISTRY.items()}


@app.post("/route/preview")
def route_preview(body: QueryIn):
    """Decision only — show the plan NeuroRoute *would* run, without running it."""
    analysis = input_analyzer.analyze(body.query, body.attachments)
    complexity = complexity_estimator.estimate(analysis)
    budget = _budget(body.budget)
    try:
        plan = validate_plan(
            _decision_model.propose_plan(analysis, complexity, budget), budget)
    except PlanRejected as e:
        raise HTTPException(status_code=422, detail=f"Plan rejected: {e}")
    return {"analysis": asdict(analysis), "complexity": asdict(complexity),
            "plan": asdict(plan)}


@app.post("/query")
async def query(body: QueryIn):
    try:
        return await pipeline.process(body.query, body.attachments,
                                      _budget(body.budget))
    except PlanRejected as e:
        raise HTTPException(status_code=422, detail=f"Plan rejected: {e}")


@app.get("/traces")
def traces():
    return pipeline.tracer.list()


@app.get("/traces/{request_id}")
def trace(request_id: str):
    found = pipeline.tracer.get(request_id)
    if not found:
        raise HTTPException(status_code=404, detail="trace not found")
    return found


@app.get("/metrics/summary")
def metrics_summary():
    return pipeline.tracer.summary()


@app.get("/dashboard")
def dashboard():
    return FileResponse(_STATIC / "dashboard.html")


@app.get("/")
def root():
    return {"service": "NeuroRoute", "docs": "/docs", "dashboard": "/dashboard"}

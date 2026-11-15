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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.decision_model import RuleBasedDecisionModel
from backend.models.model_registry import REGISTRY, rate_card
from backend.pipeline import NeuroRoutePipeline
from backend.planning.graph_validator import PlanRejected, validate_plan
from backend.schemas import Budget

app = FastAPI(title="NeuroRoute", version="0.1.0",
              description="Adaptive AI inference & orchestration platform — Phase 1 MVP")

# API-gateway authentication: when NR_API_KEY is set, every
# non-public route requires it (X-API-Key header or Bearer token).
# Unset = open, which is the local-development default. The key is
# read per request so it can be rotated without a restart.
_PUBLIC_PATHS = {"/", "/health", "/dashboard", "/docs", "/openapi.json"}


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    key = os.environ.get("NR_API_KEY")
    if key and request.url.path not in _PUBLIC_PATHS:
        supplied = request.headers.get("x-api-key", "")
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
        if supplied != key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"})
    return await call_next(request)
pipeline = NeuroRoutePipeline()
_decision_model = RuleBasedDecisionModel()
_STATIC = Path(__file__).parent / "static"
_UPLOADS = Path(__file__).resolve().parent.parent / "data" / "uploads"
# The retrieval service the pipeline actually queries; uploads are
# ingested into this same index.
retrieval = pipeline.executor.node_executor.retrieval


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
    session_id: Optional[str] = None


def _budget(b: Optional[BudgetIn]) -> Budget:
    return Budget(**b.model_dump()) if b else Budget()


@app.get("/health")
def health():
    return {"status": "ok", "service": "neuroroute", "phase": 5,
            "model_mode": os.environ.get("NR_MODEL_MODE", "mock"),
            "decision_model": os.environ.get("NR_DECISION_MODEL", "rules"),
            "plan_model": os.environ.get("NR_PLAN_MODEL", "strategy"),
            "embeddings": retrieval.embedder.name,
            "vector_store": retrieval.store.name,
            "vision_model": os.environ.get("NR_VISION_MODEL", "llava:7b"),
            "whisper_model": os.environ.get("NR_WHISPER_MODEL", "tiny"),
            "auth_enabled": bool(os.environ.get("NR_API_KEY")),
            "prices_per_1k_usd": rate_card(),
            "documents": retrieval.documents()}


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


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Ingest a real document (PDF/TXT/MD): it is chunked, embedded
    and added to the retrieval index under its filename. Queries
    that name the file in `attachments` then retrieve from it."""
    _UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = _UPLOADS / Path(file.filename).name
    dest.write_bytes(await file.read())
    if dest.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        # Images are not text-indexed: they are perceived per query
        # by the multimodal engine when named in `attachments`.
        return {"source": dest.name, "kind": "image",
                "chunks_ingested": 0, "documents": retrieval.documents()}
    if dest.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg",
                               ".flac", ".webm"):
        # Audio is transcribed per query, not text-indexed.
        return {"source": dest.name, "kind": "audio",
                "chunks_ingested": 0, "documents": retrieval.documents()}
    try:
        chunks = retrieval.ingest_file(dest)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"source": dest.name, "chunks_ingested": chunks,
            "documents": retrieval.documents()}


@app.get("/documents")
def list_documents():
    return {"embeddings": retrieval.embedder.name,
            "documents": retrieval.documents()}


@app.post("/query")
async def query(body: QueryIn):
    try:
        return await pipeline.process(body.query, body.attachments,
                                      _budget(body.budget),
                                      session_id=body.session_id)
    except PlanRejected as e:
        raise HTTPException(status_code=422, detail=f"Plan rejected: {e}")


@app.get("/traces")
def traces(session_id: Optional[str] = None):
    all_traces = pipeline.tracer.list()
    if session_id is not None:
        return [t for t in all_traces if t.get("session_id") == session_id]
    return all_traces


@app.get("/traces/{request_id}")
def trace(request_id: str):
    found = pipeline.tracer.get(request_id)
    if not found:
        raise HTTPException(status_code=404, detail="trace not found")
    return found


@app.get("/metrics/summary")
def metrics_summary():
    return pipeline.tracer.summary()


@app.get("/metrics/prometheus")
def metrics_prometheus():
    """Prometheus exposition format for scraping (monitoring row of
    the architecture's stack table; Grafana can chart these)."""
    return PlainTextResponse(pipeline.tracer.prometheus_text(),
                             media_type="text/plain; version=0.0.4")


@app.get("/dashboard")
def dashboard():
    return FileResponse(_STATIC / "dashboard.html")


@app.get("/")
def root():
    return {"service": "NeuroRoute", "docs": "/docs", "dashboard": "/dashboard"}

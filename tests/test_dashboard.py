"""Dashboard + health surface: the visual graph renderer ships in
the page, and /health reports the active planner/embedder/index."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_dashboard_serves_visual_graph_renderer():
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "renderGraph" in r.text
    assert "levelMap" in r.text
    assert "renderWaterfall" in r.text
    assert "loadTraces" in r.text


def test_health_reports_modes_and_index():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["plan_model"] in ("strategy", "llm")
    assert "embeddings" in body
    assert body["documents"].get("seed") == 10

"""Infrastructure pack: API-key auth (env-gated), session
management on traces, and the Prometheus metrics endpoint."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_key_by_default(monkeypatch):
    monkeypatch.delenv("NR_API_KEY", raising=False)


def test_open_when_no_key_configured():
    r = client.post("/query", json={"query": "What is Python?"})
    assert r.status_code == 200


def test_auth_required_when_key_set(monkeypatch):
    monkeypatch.setenv("NR_API_KEY", "test-secret")
    assert client.post("/query",
                       json={"query": "What is Python?"}).status_code == 401
    assert client.post(
        "/query", json={"query": "What is Python?"},
        headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post(
        "/query", json={"query": "What is Python?"},
        headers={"X-API-Key": "test-secret"}).status_code == 200
    assert client.post(
        "/query", json={"query": "What is Python?"},
        headers={"Authorization": "Bearer test-secret"}).status_code == 200


def test_health_stays_public_with_key(monkeypatch):
    monkeypatch.setenv("NR_API_KEY", "test-secret")
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["auth_enabled"] is True


def test_session_id_recorded_and_filterable():
    r = client.post("/query", json={"query": "What is Python?",
                                    "session_id": "sess-infra"})
    assert r.status_code == 200
    assert r.json()["session_id"] == "sess-infra"
    mine = client.get("/traces", params={"session_id": "sess-infra"}).json()
    assert mine and all(t["session_id"] == "sess-infra" for t in mine)
    assert client.get(
        "/traces", params={"session_id": "no-such"}).json() == []


def test_prometheus_endpoint_format():
    client.post("/query", json={"query": "What is Python?"})
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    assert "# TYPE neuroroute_requests_total counter" in body
    assert 'neuroroute_requests_total{strategy="direct_small"}' in body
    assert "neuroroute_tokens_total" in body
    assert "neuroroute_cost_usd_total" in body
    assert "neuroroute_avg_latency_ms" in body

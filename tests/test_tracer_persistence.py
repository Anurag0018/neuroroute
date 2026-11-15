"""Trace persistence: metrics and history must survive a backend
restart (a fresh ExecutionTracer over the same JSONL file), and
request ids minted after a restart must not overwrite history.
"""
from backend.observability.execution_tracer import ExecutionTracer


def _trace(rid="req_0001", strategy="direct_small"):
    return {
        "request_id": rid,
        "strategy": strategy,
        "metrics": {"latency_ms": 100, "cost_usd": 0.001, "tokens": 50,
                    "escalations": 0},
    }


def test_traces_survive_restart(tmp_path):
    path = tmp_path / "traces.jsonl"
    first = ExecutionTracer(persist_path=path)
    first.record(_trace())
    first.record(_trace("req_0002", "rag_medium"))
    assert first.summary()["requests"] == 2

    second = ExecutionTracer(persist_path=path)  # simulated restart
    assert second.summary()["requests"] == 2
    assert second.get("req_0001")["strategy"] == "direct_small"
    assert second.summary()["strategies"] == {
        "direct_small": 1, "rag_medium": 1}


def test_duplicate_request_id_after_restart_is_kept(tmp_path):
    path = tmp_path / "traces.jsonl"
    ExecutionTracer(persist_path=path).record(_trace())
    restarted = ExecutionTracer(persist_path=path)
    restarted.record(_trace())  # same req_0001 minted by the new process
    assert restarted.summary()["requests"] == 2
    ids = {t["request_id"] for t in restarted.list()}
    assert ids == {"req_0001", "req_0001-2"}


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text('not json\n{"request_id": "req_0009", "strategy": '
                    '"direct_small", "metrics": {"latency_ms": 1, '
                    '"cost_usd": 0.0, "tokens": 1, "escalations": 0}}\n')
    tracer = ExecutionTracer(persist_path=path)
    assert tracer.summary()["requests"] == 1

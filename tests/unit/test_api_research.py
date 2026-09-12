"""Route layer only — the pipeline is always stubbed, no agent ever runs here."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from amaris.api.main import create_app
from amaris.api.routes import research
from amaris.graph.state import GraphState, new_state
from amaris.memory import redis_memory

TERMINAL = ("done", "failed")


def _finished_state(report: str = "# Report\nbody") -> GraphState:
    state = new_state("what is langgraph")
    state["final_report"] = report
    state["citations"] = [{"url": "https://x.com/1", "title": "X"}]
    state["critic_scores"] = {"faithfulness": 0.9, "coherence": 0.8}
    state["evaluation_scores"] = {"faithfulness": 0.7, "overall": 0.75}
    state["agent_path"] = ["planner", "researcher", "writer", "critic"]
    return state


def _fake_stream(steps: list[tuple[str, dict[str, Any]]], final: GraphState):
    async def stream(query: str, session_id: str | None = None) -> AsyncIterator[tuple]:
        for node, delta in steps:
            yield node, delta, final

    return stream


@pytest.fixture
def client() -> Iterator[TestClient]:
    redis_memory._store = redis_memory.InMemoryJobStore()
    with TestClient(create_app()) as test_client:
        yield test_client
    redis_memory._store = None


def _await_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    """Poll until terminal — the run lives on the app's loop, not this thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/research/{job_id}").json()
        if body["status"] in TERMINAL:
            return body
        time.sleep(0.01)
    pytest.fail(f"job {job_id} never finished")


def test_post_returns_immediately_and_the_run_completes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = _finished_state()
    steps = [("planner", {"research_plan": [1, 2, 3]}), ("writer", {}), ("critic", {})]
    monkeypatch.setattr(research, "stream_research", _fake_stream(steps, final))

    accepted = client.post("/research", json={"query": "what is langgraph"})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    assert accepted.json()["status"] == "queued"

    body = _await_job(client, job_id)
    assert body["status"] == "done"
    assert body["progress_pct"] == 100
    assert body["result"]["report"].startswith("# Report")
    assert body["result"]["agent_path"] == final["agent_path"]


def test_a_supplied_session_id_is_echoed_back(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(research, "stream_research", _fake_stream([], _finished_state()))
    body = client.post(
        "/research", json={"query": "what is langgraph", "session_id": "abc123"}
    ).json()
    assert body["session_id"] == "abc123"


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/research/nope").status_code == 404


def test_short_query_is_rejected_before_a_job_exists(client: TestClient) -> None:
    assert client.post("/research", json={"query": "hi"}).status_code == 422


def test_a_raising_pipeline_marks_the_job_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explodes(query: str, session_id: str | None = None) -> AsyncIterator[tuple]:
        yield "planner", {}, new_state(query)
        raise RuntimeError("groq fell over")

    monkeypatch.setattr(research, "stream_research", explodes)
    job_id = client.post("/research", json={"query": "what is langgraph"}).json()["job_id"]

    body = _await_job(client, job_id)
    assert body["status"] == "failed"
    assert "groq fell over" in body["error"]


def test_a_run_that_errored_but_still_wrote_a_report_is_a_degraded_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = _finished_state()
    final["error"] = "critic failed: rate limited"
    monkeypatch.setattr(research, "stream_research", _fake_stream([("writer", {})], final))
    job_id = client.post("/research", json={"query": "what is langgraph"}).json()["job_id"]

    body = _await_job(client, job_id)
    assert body["status"] == "done"
    assert body["error"] == "critic failed: rate limited"


def test_websocket_streams_to_a_terminal_event(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = [("planner", {}), ("researcher", {}), ("writer", {}), ("critic", {})]
    monkeypatch.setattr(research, "stream_research", _fake_stream(steps, _finished_state()))
    job_id = client.post("/research", json={"query": "what is langgraph"}).json()["job_id"]

    seen: list[dict[str, Any]] = []
    with client.websocket_connect(f"/research/{job_id}/stream") as socket:
        while True:
            event = socket.receive_json()
            seen.append(event)
            if event["status"] in TERMINAL:
                break

    percentages = [e["progress_pct"] for e in seen]
    assert percentages == sorted(percentages), "progress must never go backwards"
    assert seen[-1]["progress_pct"] == 100


def test_websocket_rejects_an_unknown_job(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/research/nope/stream") as socket,
    ):
        socket.receive_json()


def test_eval_scores_are_prefixed_so_they_cannot_shadow_the_critic() -> None:
    """Both the critic and the evaluator emit a 'faithfulness' — a plain merge loses one."""
    result = research._result_from(_finished_state())
    assert result.scores["faithfulness"] == 0.9
    assert result.scores["eval_faithfulness"] == 0.7
    assert result.scores["eval_overall"] == 0.75


def test_report_falls_back_to_the_draft_when_the_evaluator_never_ran() -> None:
    state = _finished_state()
    state["final_report"] = ""
    state["draft_report"] = "# Draft"
    assert research._result_from(state).report == "# Draft"

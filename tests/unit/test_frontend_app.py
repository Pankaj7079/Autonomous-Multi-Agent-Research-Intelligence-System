"""Cloud mode drives the pipeline directly. Local mode is covered by the live API check."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from amaris.api.schemas import ProgressEvent
from amaris.graph import pipeline
from amaris.graph.state import GraphState
from frontend import app


@pytest.fixture
def finished_state(sample_state: GraphState) -> GraphState:
    """A run that reached the evaluator, built on the shared sample_state fixture."""
    state = sample_state
    state["final_report"] = "# Report"
    state["critic_scores"] = {"faithfulness": 0.9}
    state["evaluation_scores"] = {"overall": 0.8}
    state["agent_path"] = ["planner", "writer"]
    return state


@pytest.fixture
def stubbed_pipeline(monkeypatch: pytest.MonkeyPatch):
    def _apply(steps: list[tuple[str, dict]], final: GraphState) -> None:
        async def stream(query: str, session_id: str | None = None) -> AsyncIterator[tuple]:
            for node, delta in steps:
                yield node, delta, final

        monkeypatch.setattr(pipeline, "stream_research", stream)

    return _apply


async def test_cloud_mode_emits_monotonic_progress_and_a_terminal_event(
    stubbed_pipeline, finished_state: GraphState
) -> None:
    final = finished_state
    stubbed_pipeline(
        [("planner", {}), ("critic", {}), ("supervisor", {}), ("researcher", {})], final
    )

    seen: list[ProgressEvent] = []
    result, session_id, error = await app._run_cloud("what is langgraph", seen.append)

    percentages = [e.progress_pct for e in seen]
    assert percentages == sorted(percentages), "a re-route must not rewind the bar"
    assert seen[-1].status == "done"
    assert seen[-1].progress_pct == 100
    assert session_id == final["session_id"]
    assert error is None
    assert result is not None
    assert result.scores == {"faithfulness": 0.9, "eval_overall": 0.8}


async def test_cloud_mode_reports_a_pipeline_that_produced_nothing(
    stubbed_pipeline, finished_state: GraphState
) -> None:
    stubbed_pipeline([], finished_state)
    result, _session, error = await app._run_cloud("q", lambda _: None)
    assert result is None
    assert error

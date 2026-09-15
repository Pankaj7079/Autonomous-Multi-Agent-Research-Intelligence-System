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
        async def stream(
            query: str, session_id: str | None = None, *, seed: dict | None = None
        ) -> AsyncIterator[tuple]:
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
    result, session_id, error = await app._run_cloud({"query": "what is langgraph"}, seen.append)

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
    result, _session, error = await app._run_cloud({"query": "qqq"}, lambda _: None)
    assert result is None
    assert error


class _SessionOnly:
    """app._toggle_sidebar only touches session_state, and a plain dict is one."""

    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}


def test_hiding_the_sidebar_is_reversible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit's own collapse reopens from the app header this UI removes, so the toggle
    has to round-trip on our flag or the sidebar is gone for the rest of the session."""
    fake = _SessionOnly()
    monkeypatch.setattr(app, "st", fake)

    app._toggle_sidebar()
    assert fake.session_state[app.SIDEBAR_HIDDEN] is True

    app._toggle_sidebar()
    assert fake.session_state[app.SIDEBAR_HIDDEN] is False


def test_the_sidebar_starts_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _SessionOnly()
    monkeypatch.setattr(app, "st", fake)
    assert not fake.session_state.get(app.SIDEBAR_HIDDEN)


def test_the_collapsed_sheet_hides_by_width_not_display() -> None:
    """display:none would unmount the widgets inside and lose their state on reopen."""
    from frontend.styles import collapsed_css

    sheet = collapsed_css()
    assert "width: 0 !important" in sheet
    assert "display: none" not in sheet

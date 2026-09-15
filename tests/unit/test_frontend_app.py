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


def test_each_example_names_a_real_depth_and_locks_the_run_to_it() -> None:
    """The examples advertise a research level and a budget. If the depth did not reach the
    request they would be describing a run the system was not about to make."""
    from amaris.agents.triage import DEPTH_BUDGETS
    from frontend import thread

    assert len(app.EXAMPLES) == 2
    for question, depth in app.EXAMPLES:
        assert depth in DEPTH_BUDGETS, depth
        assert thread.ask_request(question, depth)["depth"] == depth
    # one shallow and one deep, or the pair demonstrates nothing about what depth buys
    assert {depth for _q, depth in app.EXAMPLES} == {"brief", "deep"}


def test_every_advertised_depth_has_a_measured_runtime() -> None:
    """A budget with no time next to it is the one number a reader actually wants."""
    for _question, depth in app.EXAMPLES:
        assert app.DEPTH_ETA.get(depth)
    for depth in app.DEPTH_CHOICES:
        if depth != "auto":
            assert app.DEPTH_ETA.get(depth), depth


def test_a_supplied_key_travels_with_the_request_in_local_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local mode runs the pipeline in the API process. Without this the panel would look like
    it worked, show its badge, and the run would quietly spend the deployment's own key."""
    import streamlit as st

    monkeypatch.setitem(st.session_state, app.BYO_KEYS, {"anthropic_api_key": "sk-ant-fake"})
    body = app._payload({"query": "what is MCP?"})
    assert body["api_keys"] == {"anthropic_api_key": "sk-ant-fake"}


def test_no_key_means_no_key_field_on_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamlit as st

    st.session_state.pop(app.BYO_KEYS, None)
    assert "api_keys" not in app._payload({"query": "what is MCP?"})


def test_every_provider_offered_in_the_panel_is_one_the_router_can_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider in the dropdown the router has no builder for accepts a key and then fails
    the run with "no API key configured", which reads as the key being wrong."""
    from amaris.llm.router import _BUILDERS, _KEY_FIELDS

    for provider, field in app.KEY_FIELDS.items():
        assert provider in _BUILDERS, provider
        assert _KEY_FIELDS[provider] == field

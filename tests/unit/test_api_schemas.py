"""The progress contract. Monotonicity matters more than the exact numbers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from amaris.api.schemas import (
    AGENT_PROGRESS,
    ResearchRequest,
    build_progress_event,
    progress_for,
)


def test_weights_increase_along_the_happy_path() -> None:
    order = ["planner", "researcher", "analyst", "writer", "critic", "evaluator"]
    values = [AGENT_PROGRESS[a] for a in order]
    assert values == sorted(values)


def test_a_reroute_never_walks_the_bar_backwards() -> None:
    """The critic sending work back to the researcher is the system's whole selling point,
    and a fixed weight table would show 92% dropping to 45%."""
    assert progress_for("researcher", 92) == 92


def test_supervisor_does_not_move_the_bar() -> None:
    assert progress_for("supervisor", 45) == 45


def test_unknown_node_holds_position() -> None:
    assert progress_for("something_new", 65) == 65


def test_event_reports_what_the_node_actually_did() -> None:
    event = build_progress_event("researcher", {"raw_research": [{}, {}, {}]}, 15)
    assert event.progress_pct == 45
    assert "3 sources" in event.message
    assert event.status == "running"


def test_event_falls_back_to_a_generic_message() -> None:
    event = build_progress_event("analyst", {}, 45)
    assert event.message
    assert event.progress_pct == 65


def test_supervisor_event_names_the_route() -> None:
    event = build_progress_event("supervisor", {"next_agent": "writer"}, 45)
    assert "writer" in event.message


def test_query_is_length_bounded() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(query="hi")
    with pytest.raises(ValidationError):
        ResearchRequest(query="x" * 501)
    assert ResearchRequest(query="what is langgraph").session_id is None

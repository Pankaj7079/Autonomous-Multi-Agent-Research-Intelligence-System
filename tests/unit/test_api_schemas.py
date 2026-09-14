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


def test_an_ordinary_question_seeds_nothing() -> None:
    from amaris.api.schemas import ResearchRequest

    assert ResearchRequest(query="what is langgraph").seed() == {}


def test_an_expand_request_locks_the_depth_and_keeps_the_sources() -> None:
    """The user's click already decided the depth, so triage must not spend a call re-deciding."""
    from amaris.api.schemas import ResearchRequest

    seed = ResearchRequest(
        query="tell about god rama",
        expand_from_depth="brief",
        prior_sources=[{"url": "https://a.com", "title": "a"}],
    ).seed()

    assert seed["query_depth"] == "brief"
    assert seed["depth_locked"] is True
    assert seed["raw_research"] == [{"url": "https://a.com", "title": "a", "content": ""}]


def test_a_reused_source_keeps_its_text_under_the_key_the_researcher_reads() -> None:
    """trace.sources trims content into "snippet"; handing that back unmapped loses every body."""
    from amaris.api.schemas import ResearchRequest

    seed = ResearchRequest(
        query="tell about god rama",
        expand_from_depth="brief",
        prior_sources=[{"url": "https://a.com", "snippet": "Rama was born in Ayodhya."}],
    ).seed()
    assert seed["raw_research"][0]["content"] == "Rama was born in Ayodhya."


def test_a_bogus_expand_depth_is_ignored_rather_than_trusted() -> None:
    from amaris.api.schemas import ResearchRequest

    assert ResearchRequest(query="q about things", expand_from_depth="enormous").seed() == {}


def test_history_reaches_the_seed_as_plain_dicts() -> None:
    from amaris.api.schemas import ResearchRequest

    seed = ResearchRequest(
        query="what about his brother?",
        history=[{"query": "tell about god rama", "answer": "Rama was born in Ayodhya."}],
    ).seed()
    assert seed["history"] == [
        {"query": "tell about god rama", "answer": "Rama was born in Ayodhya."}
    ]

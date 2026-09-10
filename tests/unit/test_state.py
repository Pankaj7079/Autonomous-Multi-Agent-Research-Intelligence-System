"""GraphState is the contract every agent writes through — it must start complete."""

from __future__ import annotations

from amaris.graph.state import AGENTS, FINISH, GraphState, new_state, source_count


def test_new_state_populates_every_declared_field() -> None:
    """A missing key becomes a KeyError deep inside an agent, so check them all here."""
    state = new_state("What is LangGraph?")
    assert set(state) == set(GraphState.__annotations__)


def test_agentic_fields_start_neutral() -> None:
    state = new_state("q")
    assert state["research_quality"] == 0.0
    assert state["routing_hint"] == ""
    assert state["next_agent"] == ""
    assert state["revision_count"] == 0


def test_query_is_stripped_and_session_id_generated() -> None:
    state = new_state("  spaced query  ")
    assert state["original_query"] == "spaced query"
    assert len(state["session_id"]) == 8


def test_explicit_session_id_is_kept() -> None:
    assert new_state("q", session_id="abc12345")["session_id"] == "abc12345"


def test_source_count_dedupes_by_url() -> None:
    """The researcher can return the same page from two tasks — the supervisor must not double count."""
    state = new_state("q")
    state["raw_research"] = [
        {"url": "https://a.com", "title": "a"},
        {"url": "https://a.com", "title": "a again"},
        {"url": "https://b.com", "title": "b"},
        {"title": "no url at all"},
    ]
    assert source_count(state) == 2


def test_finish_is_not_an_agent_name() -> None:
    """The conditional edge maps FINISH to the evaluator, so it must not collide."""
    assert FINISH not in AGENTS

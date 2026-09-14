"""GraphState is the contract every agent writes through — it must start complete."""

from __future__ import annotations

from amaris.graph.state import AGENTS, FINISH, GraphState, new_state, source_count, subject


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


def test_a_seed_carries_the_previous_turn_forward_and_blanks_everything_else() -> None:
    """An expand click reuses the sources it already paid for; it does not reuse the draft."""
    seeded = new_state(
        "explain in detail",
        seed={
            "raw_research": [{"url": "https://a.com", "title": "a"}],
            "query_depth": "brief",
            "depth_locked": True,
            "history": [{"query": "q", "answer": "a"}],
        },
    )
    assert seeded["raw_research"] == [{"url": "https://a.com", "title": "a"}]
    assert seeded["query_depth"] == "brief"
    assert seeded["depth_locked"] is True
    assert seeded["history"] == [{"query": "q", "answer": "a"}]
    # the previous run's output must not survive, or the critic reviews a stale draft
    assert seeded["draft_report"] == ""
    assert seeded["agent_path"] == []
    assert seeded["quality_score"] == 0.0


def test_no_seed_is_the_same_cold_start_as_before() -> None:
    # started_at is second-resolution, so comparing it would flake across a tick
    plain = {k: v for k, v in new_state("q", session_id="s").items() if k != "started_at"}
    seeded = {
        k: v for k, v in new_state("q", session_id="s", seed=None).items() if k != "started_at"
    }
    assert plain == seeded


def test_subject_prefers_the_resolved_question_for_finding_sources() -> None:
    """The answer still addresses what was typed; only the search uses the rewritten form."""
    state = new_state("what about his brother?")
    assert subject(state) == "what about his brother?"
    state["resolved_query"] = "Who was Rama's brother Lakshmana?"
    assert subject(state) == "Who was Rama's brother Lakshmana?"
    assert state["original_query"] == "what about his brother?"

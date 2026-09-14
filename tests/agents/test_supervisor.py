"""Routing behaviour. These are the most important tests in the project."""

from __future__ import annotations

import pytest

from amaris.agents.supervisor import MAX_RESEARCH_VISITS, SupervisorAgent
from amaris.graph.state import (
    APPROVE,
    FINISH,
    FIX_WRITING,
    NEED_MORE_RESEARCH,
    WRONG_TOPIC,
    GraphState,
)
from tests.helpers import ScriptedLLM, patch_invoke


def at_research_gate(state: GraphState) -> GraphState:
    """State as the graph presents it after a researcher run."""
    state["agent_path"] = [*state["agent_path"], "researcher"]
    return state


def at_review_gate(state: GraphState) -> GraphState:
    """State as the graph presents it after a critic run."""
    state["agent_path"] = [*state["agent_path"], "critic"]
    return state


async def route(
    monkeypatch: pytest.MonkeyPatch, state: GraphState, reply: str = "researcher"
) -> str:
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(reply))
    return (await agent.run(state))["next_agent"]


# ── the research gate ─────────────────────────────────────────────────────


async def test_thin_research_is_a_real_decision_the_model_makes(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """Sources exist but quality is under the floor — that is genuinely arguable, so it asks."""
    researched_state["research_quality"] = 0.45
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    assert (await agent.run(at_research_gate(researched_state)))["next_agent"] == "researcher"
    assert len(llm.prompts) == 1


async def test_the_model_may_judge_thin_research_good_enough(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """The gate exists so this can differ from the source count, not merely confirm it."""
    researched_state["research_quality"] = 0.45
    assert await route(monkeypatch, at_research_gate(researched_state), "analyst") == "analyst"


async def test_good_research_moves_on_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """Quality above the floor with real sources leaves nothing to weigh, so nothing is spent."""
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    update = await agent.run(at_research_gate(researched_state))
    assert update["next_agent"] == "analyst"
    assert llm.prompts == []
    assert update["decision_log"][0]["llm_decided"] is False


async def test_no_sources_goes_back_to_research_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("analyst"))

    assert (await agent.run(at_research_gate(state)))["next_agent"] == "researcher"
    assert llm.prompts == []


async def test_a_shallow_query_skips_the_analyst_entirely(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """A six-source direct answer needs no synthesis pass, so that reasoning call is not made."""
    researched_state["query_depth"] = "direct"
    assert await route(monkeypatch, at_research_gate(researched_state), "analyst") == "writer"


async def test_research_visits_are_capped(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """A gate that can always ask for more research is an unbounded bill."""
    researched_state["research_quality"] = 0.1
    researched_state["agent_path"] = ["researcher"] * MAX_RESEARCH_VISITS
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    assert (await agent.run(researched_state))["next_agent"] == "analyst"
    assert llm.prompts == []


async def test_the_research_gate_shows_the_model_what_was_found(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """Judging relevance needs the titles — a bare count is what kept 69 off-topic pages."""
    researched_state["research_quality"] = 0.45
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("analyst"))
    await agent.run(at_research_gate(researched_state))

    prompt = llm.prompts[0]
    assert researched_state["original_query"] in prompt
    assert "Source 1" in prompt
    assert str(agent.settings.research_quality_threshold) in prompt


# ── the review gate ───────────────────────────────────────────────────────


async def test_an_approved_draft_finishes_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """The critic approved and the score agrees — there is no decision left to pay for."""
    drafted_state["routing_hint"] = APPROVE
    drafted_state["quality_score"] = 0.85
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("writer"))

    assert (await agent.run(at_review_gate(drafted_state)))["next_agent"] == FINISH
    assert llm.prompts == []


async def test_a_failing_draft_is_a_real_four_way_choice(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """The behaviour the whole architecture exists for: a quality failure reaches the researcher."""
    drafted_state["routing_hint"] = NEED_MORE_RESEARCH
    drafted_state["quality_score"] = 0.4
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    assert (await agent.run(at_review_gate(drafted_state)))["next_agent"] == "researcher"
    assert len(llm.prompts) == 1


async def test_a_wrong_topic_report_can_be_sent_back_to_the_planner(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """Re-searching a bad plan is what produced 2000 words about weather APIs."""
    drafted_state["routing_hint"] = WRONG_TOPIC
    drafted_state["quality_score"] = 0.3
    assert await route(monkeypatch, at_review_gate(drafted_state), "planner") == "planner"


async def test_the_review_gate_shows_the_model_the_answer_fit(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    drafted_state["routing_hint"] = FIX_WRITING
    drafted_state["quality_score"] = 0.5
    drafted_state["critic_scores"] = {"answer_fit": 0.3, "coherence": 0.9}
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("writer"))
    await agent.run(at_review_gate(drafted_state))

    assert "answer_fit 0.30" in llm.prompts[0]
    assert str(agent.settings.quality_approve_threshold) in llm.prompts[0]


# ── code-decided stops ────────────────────────────────────────────────────


async def test_error_finishes_without_calling_the_llm(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """A broken run must not spend a routing call to discover it is broken."""
    state["error"] = "researcher exploded"
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    assert (await agent.run(state))["next_agent"] == FINISH
    assert llm.prompts == []


async def test_revision_cap_finishes_deterministically(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """An LLM must not be the only thing stopping an infinite revision loop."""
    drafted_state["revision_count"] = 2
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("writer"))

    assert (await agent.run(at_review_gate(drafted_state)))["next_agent"] == FINISH
    assert llm.prompts == []


async def test_step_cap_finishes_even_if_the_llm_keeps_routing(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """Guards against an LLM ping-ponging between agents without incrementing revisions."""
    state["agent_path"] = ["researcher"] * 15
    assert await route(monkeypatch, state, "researcher") == FINISH


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("researcher", "researcher"),
        ("  RESEARCHER  ", "researcher"),
        ("`analyst`", "analyst"),
        ("I think the analyst should run next.", "analyst"),
        ("banana", "analyst"),
        ("", "analyst"),
    ],
)
async def test_unexpected_research_gate_replies_never_break_routing(
    monkeypatch: pytest.MonkeyPatch, researched_state, reply: str, expected: str
) -> None:
    """An unroutable word must fall forward, not crash the graph with a bad edge key."""
    researched_state["research_quality"] = 0.45
    assert await route(monkeypatch, at_research_gate(researched_state), reply) == expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("writer", "writer"),
        ("FINISH", FINISH),
        ("planner", "planner"),
        ("banana", FINISH),
        ("", FINISH),
    ],
)
async def test_unexpected_review_gate_replies_never_break_routing(
    monkeypatch: pytest.MonkeyPatch, drafted_state, reply: str, expected: str
) -> None:
    drafted_state["routing_hint"] = FIX_WRITING
    drafted_state["quality_score"] = 0.5
    assert await route(monkeypatch, at_review_gate(drafted_state), reply) == expected


async def test_the_supervisor_does_not_write_the_agent_path(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """agent_path is who actually ran; each node appends itself, or direct edges vanish from it."""
    researched_state["research_quality"] = 0.45
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("analyst"))

    assert "agent_path" not in await agent.run(at_research_gate(researched_state))


# ── the decision log ──────────────────────────────────────────────────────


async def test_decision_log_records_the_gate_and_the_invariant(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """This is the only input trajectory_eval.py's Layer 3 has."""
    researched_state["research_quality"] = 0.45
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    entry = (await agent.run(at_research_gate(researched_state)))["decision_log"][0]
    assert entry["gate"] == "research_gate"
    assert entry["to_agent"] == "researcher"
    assert entry["expected_agent"] == "researcher"
    assert entry["matched_rule"] == "below_quality_floor"
    assert entry["llm_decided"] is True


async def test_decision_log_flags_a_mismatch_between_model_and_invariant(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """Disagreement is informative, so the log must make it visible rather than hide it."""
    researched_state["research_quality"] = 0.45
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("analyst"))

    entry = (await agent.run(at_research_gate(researched_state)))["decision_log"][0]
    assert entry["to_agent"] == "analyst"
    assert entry["expected_agent"] == "researcher"


async def test_decision_log_marks_settled_steps_as_not_llm_decided(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """Scoring a code-decided stop as a routing choice would be meaningless."""
    state["error"] = "boom"
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("planner"))

    entry = (await agent.run(state))["decision_log"][0]
    assert entry["llm_decided"] is False
    assert entry["to_agent"] == FINISH
    assert entry["matched_rule"] == "error_set"


async def test_decision_log_appends_rather_than_replaces(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    researched_state["decision_log"] = [{"step": 1, "to_agent": "planner"}]
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    log = (await agent.run(at_research_gate(researched_state)))["decision_log"]
    assert len(log) == 2
    assert log[1]["step"] == 2

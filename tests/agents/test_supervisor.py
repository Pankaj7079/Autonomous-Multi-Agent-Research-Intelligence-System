"""Routing behaviour. These are the most important tests in the project."""

from __future__ import annotations

import pytest

from amaris.agents.supervisor import SupervisorAgent
from amaris.graph.state import FINISH, NEED_MORE_RESEARCH, GraphState
from tests.helpers import ScriptedLLM, patch_invoke


async def route(
    monkeypatch: pytest.MonkeyPatch, state: GraphState, reply: str = "researcher"
) -> str:
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(reply))
    return (await agent.run(state))["next_agent"]


async def test_routes_to_what_the_llm_chose(monkeypatch: pytest.MonkeyPatch, state) -> None:
    assert await route(monkeypatch, state, "planner") == "planner"


async def test_llm_can_send_work_back_to_research(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """The behaviour the whole architecture exists for: a quality failure reaches the researcher."""
    drafted_state["routing_hint"] = NEED_MORE_RESEARCH
    drafted_state["quality_score"] = 0.4
    assert await route(monkeypatch, drafted_state, "researcher") == "researcher"


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

    assert (await agent.run(drafted_state))["next_agent"] == FINISH
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
        ("`writer`", "writer"),
        ("FINISH", FINISH),
        ("I think the analyst should run next.", "analyst"),
        ("banana", FINISH),
        ("", FINISH),
    ],
)
async def test_unexpected_replies_never_break_routing(
    monkeypatch: pytest.MonkeyPatch, state, reply: str, expected: str
) -> None:
    """An unroutable word must end the run, not crash the graph with a bad edge key."""
    assert await route(monkeypatch, state, reply) == expected


async def test_agent_path_records_every_hop(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """The path is how a run gets explained afterwards."""
    state["agent_path"] = ["planner"]
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    assert (await agent.run(state))["agent_path"] == ["planner", "researcher"]


async def test_prompt_carries_the_state_the_rules_need(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """The supervisor can only reason over what the prompt actually contains."""
    drafted_state["routing_hint"] = NEED_MORE_RESEARCH
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))
    await agent.run(drafted_state)

    prompt = llm.prompts[0]
    assert "sources found: 5" in prompt
    assert NEED_MORE_RESEARCH in prompt
    assert "plan exists: True" in prompt


async def test_thresholds_in_the_prompt_come_from_settings(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """Hard-coding 0.72 in the prompt would let it drift from the code that routes on it."""
    agent = SupervisorAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("planner"))
    await agent.run(state)

    assert str(agent.settings.quality_approve_threshold) in llm.prompts[0]
    assert str(agent.settings.research_quality_threshold) in llm.prompts[0]


async def test_decision_log_records_what_the_rule_expected(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """This is the only input trajectory_eval.py's routing_accuracy has."""
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("planner"))

    entry = (await agent.run(state))["decision_log"][0]
    assert entry["to_agent"] == "planner"
    assert entry["expected_agent"] == "planner"
    assert entry["matched_rule"] == "rule_3_no_plan"
    assert entry["llm_decided"] is True


async def test_decision_log_flags_a_mismatch_between_llm_and_rule(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """When the LLM ignores the documented rule, the log must make that visible, not hide it."""
    state["research_plan"] = [{"task_id": "t1", "description": "d"}]
    agent = SupervisorAgent()
    # rule 6 says researcher (no sources yet); the scripted reply deliberately disagrees
    patch_invoke(monkeypatch, agent, ScriptedLLM("analyst"))

    entry = (await agent.run(state))["decision_log"][0]
    assert entry["to_agent"] == "analyst"
    assert entry["expected_agent"] == "researcher"
    assert entry["matched_rule"] == "rule_6_thin_research"


async def test_decision_log_marks_terminal_steps_as_not_llm_decided(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """Rules 1-2 are code-decided — scoring them as a routing choice would be meaningless."""
    state["error"] = "boom"
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("planner"))

    entry = (await agent.run(state))["decision_log"][0]
    assert entry["llm_decided"] is False
    assert entry["to_agent"] == FINISH
    assert entry["matched_rule"] == "error_set"


async def test_decision_log_appends_rather_than_replaces(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    state["decision_log"] = [{"step": 1, "to_agent": "planner"}]
    agent = SupervisorAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("researcher"))

    log = (await agent.run(state))["decision_log"]
    assert len(log) == 2
    assert log[1]["step"] == 2

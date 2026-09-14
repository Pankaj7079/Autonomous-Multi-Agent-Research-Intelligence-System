"""Triage sets every budget downstream, so a wrong answer here is paid for by the whole run."""

from __future__ import annotations

import json

import pytest

from amaris.agents.triage import DEPTH_BUDGETS, TriageAgent, budget_for
from amaris.graph.state import DEFAULT_DEPTH, DEPTHS
from tests.helpers import ScriptedLLM, patch_invoke


def verdict(
    depth: str = "standard",
    answerable: bool = True,
    clarifying_question: str = "",
    sections: list[str] | None = None,
    word_target: int = 700,
    reason: str = "needs a few angles weighed",
    resolved_query: str = "",
) -> str:
    return json.dumps(
        {
            "depth": depth,
            "answerable": answerable,
            "resolved_query": resolved_query,
            "clarifying_question": clarifying_question,
            "sections": sections if sections is not None else ["Answer", "Key Findings"],
            "word_target": word_target,
            "reason": reason,
        }
    )


async def triage(monkeypatch: pytest.MonkeyPatch, state, reply: str) -> dict:
    agent = TriageAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(reply))
    return await agent.run(state)


async def test_depth_and_shape_reach_the_state(monkeypatch: pytest.MonkeyPatch, state) -> None:
    update = await triage(monkeypatch, state, verdict(depth="deep", word_target=1100))
    assert update["query_depth"] == "deep"
    assert update["word_target"] == 1100
    assert update["answerable"] is True


async def test_an_unanswerable_question_carries_the_question_to_ask_back(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """The 92-second weather run had no location — searching harder could never fix that."""
    update = await triage(
        monkeypatch,
        state,
        verdict(answerable=False, clarifying_question="Which city?", depth="direct"),
    )
    assert update["answerable"] is False
    assert update["clarifying_question"] == "Which city?"


async def test_the_answer_always_leads_whatever_headings_were_asked_for(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    update = await triage(
        monkeypatch, state, verdict(sections=["Background", "Answer", "Conclusion"])
    )
    assert update["report_sections"][0] == "Answer"
    assert update["report_sections"].count("Answer") == 1


async def test_missing_sections_fall_back_to_the_depth_template(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    update = await triage(monkeypatch, state, verdict(depth="brief", sections=[]))
    assert update["report_sections"] == list(budget_for("brief").sections)


@pytest.mark.parametrize("bad", ["exhaustive", "", "DEEP DIVE", "1"])
async def test_an_unknown_depth_falls_back_to_standard(
    monkeypatch: pytest.MonkeyPatch, state, bad: str
) -> None:
    update = await triage(monkeypatch, state, verdict(depth=bad))
    assert update["query_depth"] == DEFAULT_DEPTH


@pytest.mark.parametrize("bad", [0, 5, 99999, -10])
async def test_an_implausible_word_target_falls_back_to_the_budget(
    monkeypatch: pytest.MonkeyPatch, state, bad: int
) -> None:
    update = await triage(monkeypatch, state, verdict(depth="brief", word_target=bad))
    assert update["word_target"] == budget_for("brief").word_target


async def test_malformed_output_still_runs_a_normal_research_pass(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """A triage failure must never silently suppress a real question."""
    update = await triage(monkeypatch, state, "the model wrote prose instead of json")
    assert update["query_depth"] == DEFAULT_DEPTH
    assert update["answerable"] is True
    assert update["clarifying_question"] == ""


async def test_the_prompt_asks_about_the_question_not_about_routing(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """A query-to-route lookup table here would rebuild the decorative-LLM bug one layer up."""
    agent = TriageAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM(verdict()))
    await agent.run(state)

    prompt = llm.prompts[0].lower()
    assert state["original_query"] in llm.prompts[0]
    for agent_name in ("planner", "researcher", "analyst", "writer", "critic", "supervisor"):
        assert agent_name not in prompt


def test_every_depth_has_a_budget_and_none_gathers_more_than_it_reads() -> None:
    """Gathering past what the writer reads is exactly the waste this replaced.

    Read from the setting rather than a literal: the researcher clamps with
    min(budget.max_sources, max_sources_in_prompt), so a hard-coded ceiling here passed
    happily while that clamp silently cut deep's budget back down to standard's.
    """
    from amaris.config.settings import get_settings

    ceiling = get_settings().max_sources_in_prompt
    assert set(DEPTH_BUDGETS) == set(DEPTHS)
    for depth, budget in DEPTH_BUDGETS.items():
        assert budget.max_sources <= ceiling, depth
        assert budget.sections[0] == "Answer", depth
        assert budget.tasks >= 1 and budget.react_iterations >= 1, depth


def test_deep_actually_reads_more_than_standard() -> None:
    """Otherwise the depth picker's "deep" is a longer report written from the same evidence."""
    assert budget_for("deep").max_sources > budget_for("standard").max_sources


def test_shallow_depths_skip_the_analyst() -> None:
    """A synthesis pass over six sources costs a reasoning call and tells the writer nothing."""
    assert budget_for("direct").analysis is False
    assert budget_for("brief").analysis is False
    assert budget_for("standard").analysis is True


def test_an_unknown_depth_still_returns_a_usable_budget() -> None:
    assert budget_for("nonsense") == DEPTH_BUDGETS[DEFAULT_DEPTH]


def test_a_shallow_depth_does_not_earn_a_rewrite_round() -> None:
    """The live 96.9s run spent 43.7s re-researching a brief answer to gain twelve words."""
    assert budget_for("direct").max_revisions == 1
    assert budget_for("brief").max_revisions == 1
    assert budget_for("standard").max_revisions == 2


# ── a locked depth: the user already decided, so triage must spend nothing ─────────


async def test_a_locked_depth_costs_no_model_call(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """A confirmation call here would be ADR-030's decorative-LLM bug rebuilt one layer up."""
    state["query_depth"] = "deep"
    state["depth_locked"] = True

    agent = TriageAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM(verdict()))
    update = await agent.run(state)

    assert llm.prompts == []
    assert update["query_depth"] == "deep"
    assert update["word_target"] == budget_for("deep").word_target


async def test_a_locked_depth_is_obeyed_not_bumped(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """The caller resolves the depth — "explain in detail" sends the bumped one already."""
    state["query_depth"] = "standard"
    state["depth_locked"] = True
    update = await triage(monkeypatch, state, verdict())

    assert update["query_depth"] == "standard"
    assert update["depth_locked"] is False


async def test_a_nonsense_locked_depth_falls_back_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    state["query_depth"] = "enormous"
    state["depth_locked"] = True
    assert (await triage(monkeypatch, state, verdict()))["query_depth"] == DEFAULT_DEPTH


# ── follow-ups ────────────────────────────────────────────────────────────


async def test_prior_turns_reach_the_prompt_so_a_pronoun_can_be_resolved(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """Without this, "what about his brother?" is triaged as a question missing its subject."""
    state["original_query"] = "what about his brother?"
    state["history"] = [{"query": "tell about god rama", "answer": "Rama was born in Ayodhya."}]

    agent = TriageAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM(verdict()))
    await agent.run(state)

    assert "tell about god rama" in llm.prompts[0]
    assert "Rama was born in Ayodhya." in llm.prompts[0]


async def test_a_first_question_carries_no_history_block(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    agent = TriageAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM(verdict()))
    await agent.run(state)

    assert "Earlier in this conversation" not in llm.prompts[0]


async def test_a_follow_up_is_rewritten_into_a_standalone_search_string(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """ "what about his brother?" searched verbatim finds nothing — it must carry its subject."""
    state["original_query"] = "what about his brother?"
    update = await triage(
        monkeypatch, state, verdict(resolved_query="Who was Rama's brother Lakshmana?")
    )
    assert update["resolved_query"] == "Who was Rama's brother Lakshmana?"


async def test_a_standalone_question_needs_no_rewrite(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    update = await triage(monkeypatch, state, verdict())
    assert update["resolved_query"] == ""

"""The critic's routing_hint is how a quality failure reaches the researcher instead of the writer."""

from __future__ import annotations

import json

import pytest

from amaris.agents.critic import CriticAgent
from amaris.graph.state import APPROVE, FIX_WRITING, NEED_MORE_RESEARCH
from tests.agents.helpers import ScriptedLLM, patch_invoke


def verdict(
    faithfulness: float = 0.8,
    completeness: float = 0.8,
    coherence: float = 0.8,
    citation_quality: float = 0.8,
    overall: float = 0.8,
    hint: str = APPROVE,
) -> str:
    return json.dumps(
        {
            "scores": {
                "faithfulness": faithfulness,
                "completeness": completeness,
                "coherence": coherence,
                "citation_quality": citation_quality,
            },
            "overall": overall,
            "feedback": "section 2 lacks a source",
            "top_issue": "uncited failure rate",
            "routing_hint": hint,
        }
    )


async def critique(monkeypatch: pytest.MonkeyPatch, state, reply: str) -> dict:
    agent = CriticAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(reply))
    return await agent.run(state)


async def test_scores_and_hint_are_passed_through(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    update = await critique(monkeypatch, drafted_state, verdict())
    assert update["quality_score"] == 0.8
    assert update["routing_hint"] == APPROVE
    assert update["critic_scores"]["faithfulness"] == 0.8
    assert update["top_issue"] == "uncited failure rate"


async def test_low_faithfulness_can_route_back_to_research(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """This single field is what makes the architecture agentic rather than a chain."""
    update = await critique(
        monkeypatch, drafted_state, verdict(faithfulness=0.3, overall=0.4, hint=NEED_MORE_RESEARCH)
    )
    assert update["routing_hint"] == NEED_MORE_RESEARCH


async def test_revision_count_increments(monkeypatch: pytest.MonkeyPatch, drafted_state) -> None:
    """The critic counts revisions; the supervisor enforces the cap."""
    drafted_state["revision_count"] = 1
    update = await critique(monkeypatch, drafted_state, verdict())
    assert update["revision_count"] == 2


async def test_unknown_hint_is_derived_from_the_scores(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """An unrecognised hint would strand the supervisor, so it must be replaced, not passed on."""
    update = await critique(
        monkeypatch, drafted_state, verdict(faithfulness=0.2, hint="send_it_back_please")
    )
    assert update["routing_hint"] == NEED_MORE_RESEARCH


async def test_unknown_hint_with_good_faithfulness_becomes_fix_writing(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    update = await critique(monkeypatch, drafted_state, verdict(faithfulness=0.9, hint="???"))
    assert update["routing_hint"] == FIX_WRITING


async def test_missing_overall_is_recomputed_from_the_dimensions(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    reply = json.dumps(
        {
            "scores": {
                "faithfulness": 0.6,
                "completeness": 0.8,
                "coherence": 0.8,
                "citation_quality": 0.6,
            },
            "routing_hint": FIX_WRITING,
        }
    )
    update = await critique(monkeypatch, drafted_state, reply)
    assert update["quality_score"] == 0.7


@pytest.mark.parametrize("bad", ["1.9", "-3", "null", '"high"'])
async def test_out_of_range_scores_are_clamped(
    monkeypatch: pytest.MonkeyPatch, drafted_state, bad: str
) -> None:
    """A score above 1.0 would let a bad report pass the approve threshold."""
    reply = f'{{"scores": {{"faithfulness": {bad}}}, "overall": {bad}, "routing_hint": "approve"}}'
    update = await critique(monkeypatch, drafted_state, reply)
    assert 0.0 <= update["quality_score"] <= 1.0
    assert 0.0 <= update["critic_scores"]["faithfulness"] <= 1.0


async def test_json_inside_prose_is_still_parsed(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    reply = f"Here is my review:\n{verdict()}\nHope that helps."
    assert (await critique(monkeypatch, drafted_state, reply))["quality_score"] == 0.8


async def test_fenced_json_is_still_parsed(monkeypatch: pytest.MonkeyPatch, drafted_state) -> None:
    assert (await critique(monkeypatch, drafted_state, f"```json\n{verdict()}\n```"))[
        "quality_score"
    ] == 0.8

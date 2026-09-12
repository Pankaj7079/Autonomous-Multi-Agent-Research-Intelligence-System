"""ReAct loop behaviour: self-termination, the iteration cap, and dedupe."""

from __future__ import annotations

import json
from typing import Any

import pytest

from amaris.agents import researcher as researcher_module
from amaris.agents.researcher import ResearcherAgent
from tests.agents.helpers import ScriptedLLM, patch_invoke


def decision(action: str, action_input: Any = "langgraph agents", sufficient: bool = False) -> str:
    return json.dumps(
        {
            "thought": "need more",
            "action": action,
            "action_input": action_input,
            "reasoning": "because",
            "sufficient": sufficient,
        }
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every tool and memory call is stubbed — these tests must never hit the network."""

    async def fake_search(query: str, max_results: int = 8) -> list[dict[str, Any]]:
        return [
            {
                "title": f"Result for {query}",
                "url": f"https://example.com/{query.replace(' ', '-')}",
                "content": "x" * 500,
                "source": "duckduckgo",
            }
        ]

    async def no_scrape(url: str, **kwargs: Any) -> str:
        return ""

    async def no_recall(query: str, limit: int = 3) -> list[str]:
        return []

    async def no_store(query: str, finding: str, url: str = "") -> bool:
        return False

    monkeypatch.setattr(researcher_module, "smart_search", fake_search)
    monkeypatch.setattr(researcher_module, "scrape_url", no_scrape)
    monkeypatch.setattr(researcher_module, "recall_related", no_recall)
    monkeypatch.setattr(researcher_module, "add_research_finding", no_store)


async def test_loop_stops_when_the_agent_says_sufficient(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """The agent decides when it has enough — this is the autonomy claim."""
    state["research_plan"] = [{"task_id": "t1", "description": "find things"}]
    agent = ResearcherAgent()
    llm = patch_invoke(
        monkeypatch,
        agent,
        ScriptedLLM(decision("web_search"), decision("stop", None, sufficient=True), "0.8"),
    )

    await agent.run(state)
    # two react calls then one self-assessment, not the full four iterations
    assert len(llm.prompts) == 3


async def test_loop_never_exceeds_the_iteration_cap(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """An agent that never says stop must still be cut off."""
    state["research_plan"] = [{"task_id": "t1", "description": "find things"}]
    agent = ResearcherAgent()
    cap = agent.settings.max_react_iterations
    # one more search decision than the cap allows, so only the cap can stop it
    llm = patch_invoke(
        monkeypatch, agent, ScriptedLLM(*[decision("web_search")] * (cap + 1), "0.7")
    )

    await agent.run(state)
    react_calls = [p for p in llm.prompts if "ReAct research agent" in p]
    assert len(react_calls) == cap


async def test_sources_are_deduped_by_url(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """Two tasks searching the same thing must not double count — the supervisor routes on the count."""
    state["research_plan"] = [
        {"task_id": "t1", "description": "same angle"},
        {"task_id": "t2", "description": "same angle"},
    ]
    agent = ResearcherAgent()
    patch_invoke(
        monkeypatch,
        agent,
        ScriptedLLM(decision("web_search"), decision("stop", None, True), "0.8"),
    )

    update = await agent.run(state)
    urls = [s["url"] for s in update["raw_research"]]
    assert len(urls) == len(set(urls))


async def test_a_second_visit_keeps_earlier_sources(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """The critic can route back here, so existing research must survive a re-run."""
    agent = ResearcherAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(decision("stop", None, True), "0.9"))

    update = await agent.run(researched_state)
    assert len(update["raw_research"]) >= len(researched_state["raw_research"])


async def test_quality_is_parsed_from_a_messy_float(monkeypatch: pytest.MonkeyPatch, state) -> None:
    state["research_plan"] = [{"task_id": "t1", "description": "d"}]
    agent = ResearcherAgent()
    patch_invoke(
        monkeypatch,
        agent,
        ScriptedLLM(
            decision("web_search"), decision("stop", None, True), "I rate this 0.85 overall"
        ),
    )

    assert (await agent.run(state))["research_quality"] == 0.85


async def test_zero_sources_scores_zero(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """No sources must score 0.0 so the supervisor sends it straight back to research."""
    state["research_plan"] = [{"task_id": "t1", "description": "d"}]
    agent = ResearcherAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(decision("stop", None, True)))

    update = await agent.run(state)
    assert update["raw_research"] == []
    assert update["research_quality"] == 0.0


async def test_one_failing_task_does_not_lose_the_others(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """asyncio.gather with return_exceptions — a crashed task must not void the whole pass."""
    state["research_plan"] = [
        {"task_id": "t1", "description": "good"},
        {"task_id": "t2", "description": "bad"},
    ]
    agent = ResearcherAgent()

    async def selective(self: Any, prompt: str, task_type: str | None = None) -> str:
        if "bad" in prompt:
            raise RuntimeError("this task blew up")
        if "Rate overall research quality" in prompt:
            return "0.6"
        return (
            decision("stop", None, True)
            if "Sources found so far: 1" in prompt
            else decision("web_search")
        )

    monkeypatch.setattr(ResearcherAgent, "_invoke", selective)
    update = await agent.run(state)
    assert len(update["raw_research"]) >= 1


async def test_malformed_react_json_ends_that_task_quietly(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    state["research_plan"] = [{"task_id": "t1", "description": "d"}]
    agent = ResearcherAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("not json at all", "0.3"))

    update = await agent.run(state)
    assert update["research_quality"] == 0.0


async def test_missing_plan_falls_back_to_the_raw_query(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """The researcher can be routed to before the planner has run."""
    agent = ResearcherAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM(decision("stop", None, True), "0.5"))

    await agent.run(state)
    assert state["original_query"] in llm.prompts[0]

"""ReAct loop behaviour: self-termination, the iteration cap, and dedupe."""

from __future__ import annotations

import json
from typing import Any

import pytest

from amaris.agents import researcher as researcher_module
from amaris.agents.researcher import ResearcherAgent
from amaris.agents.triage import budget_for
from tests.helpers import ScriptedLLM, patch_invoke


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
                "content": "Multi-agent system routing notes. " + "x" * 400,
                "source": "duckduckgo",
            }
        ]

    async def no_scrape(url: str, **kwargs: Any) -> str:
        return ""

    monkeypatch.setattr(researcher_module, "smart_search", fake_search)
    monkeypatch.setattr(researcher_module, "scrape_url", no_scrape)
    # _mcp_sources imports this lazily from its source module, not from researcher_module —
    # without this, a dev whose own .env has MCP_CLIENT_ENABLED=true hits the real network here
    monkeypatch.setattr("amaris.tools.mcp_client.configured_servers", lambda: [])


async def test_loop_stops_when_the_agent_says_sufficient(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """The agent decides when it has enough — this is the autonomy claim."""
    state["research_plan"] = [{"task_id": "t1", "description": "find things"}]
    agent = ResearcherAgent()
    llm = patch_invoke(
        monkeypatch,
        agent,
        ScriptedLLM(decision("search"), decision("stop", None, sufficient=True), "0.8"),
    )

    await agent.run(state)
    # two react calls then one self-assessment, not the full four iterations
    assert len(llm.prompts) == 3


async def test_loop_never_exceeds_the_iteration_cap(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """An agent that never says stop must still be cut off."""
    state["research_plan"] = [{"task_id": "t1", "description": "find things"}]
    agent = ResearcherAgent()
    cap = min(
        budget_for(state["query_depth"]).react_iterations, agent.settings.max_react_iterations
    )
    # one more search decision than the cap allows, so only the cap can stop it
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM(*[decision("search")] * (cap + 1), "0.7"))

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
        ScriptedLLM(decision("search"), decision("stop", None, True), "0.8"),
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
        ScriptedLLM(decision("search"), decision("stop", None, True), "I rate this 0.85 overall"),
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

    async def selective(
        self: Any, prompt: str, task_type: str | None = None, json_mode: bool = False
    ) -> str:
        if "bad" in prompt:
            raise RuntimeError("this task blew up")
        if "Rate how well these sources answer" in prompt:
            return "0.6"
        return (
            decision("stop", None, True)
            if "sources found so far: 1" in prompt.lower()
            else decision("search")
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


async def test_researcher_consumes_the_routing_hint(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """need_more_research must be cleared once acted on, or routing sticks here forever."""
    researched_state["routing_hint"] = "need_more_research"
    agent = ResearcherAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(decision("stop", None, True), "0.8"))

    assert (await agent.run(researched_state))["routing_hint"] == ""


async def test_react_stats_records_self_termination(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """react_discipline (Layer 3) needs to tell a clean stop from hitting the iteration cap."""
    state["research_plan"] = [{"task_id": "t1", "description": "d"}]
    agent = ResearcherAgent()
    patch_invoke(
        monkeypatch, agent, ScriptedLLM(decision("search"), decision("stop", None, True), "0.8")
    )

    update = await agent.run(state)
    assert update["react_stats"]["t1"] == {"iterations_used": 2, "self_terminated": True}


async def test_react_stats_records_hitting_the_cap(monkeypatch: pytest.MonkeyPatch, state) -> None:
    state["research_plan"] = [{"task_id": "t1", "description": "d"}]
    agent = ResearcherAgent()
    cap = min(
        budget_for(state["query_depth"]).react_iterations, agent.settings.max_react_iterations
    )
    patch_invoke(monkeypatch, agent, ScriptedLLM(*[decision("search")] * (cap + 1), "0.7"))

    update = await agent.run(state)
    stats = update["react_stats"]["t1"]
    assert stats["iterations_used"] == cap
    assert stats["self_terminated"] is False


async def test_react_stats_merge_across_tasks_and_visits(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """A re-visited task_id overwrites its own stats; other tasks' stats must survive."""
    researched_state["react_stats"] = {"t0": {"iterations_used": 4, "self_terminated": False}}
    agent = ResearcherAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(decision("stop", None, True), "0.9"))

    update = await agent.run(researched_state)
    assert "t0" in update["react_stats"]
    assert update["react_stats"]["t1"]["self_terminated"] is True


async def test_an_unambiguous_relevance_signal_skips_the_self_assessment(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """Sources that clearly do or don't match the question need no judge — the call can't move it."""

    async def off_topic(query: str, max_results: int = 8) -> list[dict[str, Any]]:
        return [
            {
                "title": "AccuWeather API pricing plans",
                "url": "https://example.com/api-pricing",
                "content": "Developer tiers, request quotas and billing for our data feeds.",
            }
        ]

    monkeypatch.setattr(researcher_module, "smart_search", off_topic)
    state["research_plan"] = [{"task_id": "t1", "description": "d"}]
    agent = ResearcherAgent()
    llm = patch_invoke(
        monkeypatch, agent, ScriptedLLM(decision("search"), decision("stop", None, True), "0.9")
    )

    update = await agent.run(state)
    assert [p for p in llm.prompts if "Rate how well" in p] == []
    assert update["research_quality"] < 0.2


async def test_a_one_iteration_budget_spends_no_reasoning_call(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """With no loop to reason about, the only sensible first move is to search the task itself."""
    state["query_depth"] = "direct"
    state["research_plan"] = [{"task_id": "t1", "description": "what is langgraph"}]
    agent = ResearcherAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM(decision("search"), "0.8"))

    update = await agent.run(state)
    assert [p for p in llm.prompts if "ReAct research agent" in p] == []
    assert len(update["raw_research"]) == 1


async def test_off_topic_sources_are_dropped_before_anything_reads_them(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """69 gathered, 12 read, 57 paid for and binned — the cut belongs here, once."""

    async def mixed(query: str, max_results: int = 8) -> list[dict[str, Any]]:
        return [
            {
                "title": "Multi-agent system design and agentic routing",
                "url": "https://example.com/good",
                "content": "How to make an agentic multi-agent system route work.",
            },
            {
                "title": "Best pasta recipes for winter",
                "url": "https://example.com/pasta",
                "content": "Boil water, add salt, cook for eleven minutes.",
            },
        ]

    monkeypatch.setattr(researcher_module, "smart_search", mixed)
    state["research_plan"] = [{"task_id": "t1", "description": "d"}]
    agent = ResearcherAgent()
    patch_invoke(
        monkeypatch, agent, ScriptedLLM(decision("search"), decision("stop", None, True), "0.8")
    )

    urls = [s["url"] for s in (await agent.run(state))["raw_research"]]
    assert urls == ["https://example.com/good"]


async def test_an_attached_file_reaches_the_report_as_a_citable_source(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """The whole point of an upload: its chunks are retrieved per task and cited like any source."""
    state["research_plan"] = [{"task_id": "t1", "description": "what does the spec say"}]
    state["attachments"] = [{"name": "spec.pdf", "url": "file://spec.pdf", "chunks": 3}]

    async def fake_kb(
        query: str, limit: int = 5, session_id: str = "", urls: list[str] | None = None
    ) -> list[dict[str, Any]]:
        # by url, not session: ingestion uses the conversation's id and the run has its own,
        # so filtering by session_id here matched nothing and the file was silently ignored
        assert urls == ["file://spec.pdf"], f"must filter by the attached file, got {urls}"
        return [
            {
                "text": "The supervisor decides at two gates.",
                "url": "file://spec.pdf",
                "title": "spec.pdf",
                "score": 0.9,
            },
            {
                "text": "The revision cap is two.",
                "url": "file://spec.pdf",
                "title": "spec.pdf",
                "score": 0.8,
            },
        ]

    monkeypatch.setattr("amaris.tools.vector_tool.search_knowledge_base", fake_kb)
    agent = ResearcherAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(decision("stop", None, sufficient=True), "0.8"))

    sources = (await agent.run(state))["raw_research"]
    attached = [s for s in sources if s["url"] == "file://spec.pdf"]

    # one source per file, not one per chunk, or a single pdf fills the whole reference list
    assert len(attached) == 1
    assert "two gates" in attached[0]["content"]
    assert "revision cap" in attached[0]["content"]
    assert attached[0]["title"] == "spec.pdf"


async def test_an_attached_file_survives_the_relevance_floor(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """The floor exists to drop junk search results. A file the user chose is not junk, even
    when it shares no words with the question."""
    state["research_plan"] = [{"task_id": "t1", "description": "find things"}]
    state["attachments"] = [{"name": "notes.pdf", "url": "file://notes.pdf", "chunks": 1}]

    async def fake_kb(
        query: str, limit: int = 5, session_id: str = "", urls: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return [
            {
                "text": "Completely unrelated prose about gardening.",
                "url": "file://notes.pdf",
                "title": "notes.pdf",
                "score": 0.1,
            }
        ]

    monkeypatch.setattr("amaris.tools.vector_tool.search_knowledge_base", fake_kb)
    agent = ResearcherAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(decision("stop", None, sufficient=True), "0.8"))

    sources = (await agent.run(state))["raw_research"]
    assert any(s["url"] == "file://notes.pdf" for s in sources)


async def test_no_attachment_means_no_vector_lookup(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """An ordinary question must not pay a Qdrant round trip it has no use for."""
    state["research_plan"] = [{"task_id": "t1", "description": "find things"}]
    called = False

    async def fake_kb(
        query: str, limit: int = 5, session_id: str = "", urls: list[str] | None = None
    ) -> list[dict[str, Any]]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr("amaris.tools.vector_tool.search_knowledge_base", fake_kb)
    agent = ResearcherAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(decision("stop", None, sufficient=True), "0.8"))

    await agent.run(state)
    assert not called

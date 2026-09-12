"""Planner, analyst and writer: shape of what they produce, not the prose they produce."""

from __future__ import annotations

import json
from typing import Any

import pytest

from amaris.agents import analyst as analyst_module
from amaris.agents.analyst import AnalystAgent
from amaris.agents.planner import PlannerAgent
from amaris.agents.writer import WriterAgent
from tests.agents.helpers import ScriptedLLM, patch_invoke

# ── planner ───────────────────────────────────────────────────────────────


def plan(*descriptions: str) -> str:
    return json.dumps(
        {
            "tasks": [
                {"task_id": f"x{i}", "description": d, "assigned_to": "researcher"}
                for i, d in enumerate(descriptions, start=1)
            ],
            "research_strategy": "cover evidence then counterpoints",
        }
    )


async def test_planner_normalises_task_ids(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """The researcher logs against task_id, so they must be sequential regardless of the model."""
    agent = PlannerAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(plan("find evidence", "find counterpoints")))

    update = await agent.run(state)
    assert [t["task_id"] for t in update["research_plan"]] == ["t1", "t2"]
    assert update["research_strategy"] == "cover evidence then counterpoints"


async def test_planner_caps_the_task_count(monkeypatch: pytest.MonkeyPatch, state) -> None:
    """Each task is a full ReAct loop, so an over-eager plan is a runtime and quota problem."""
    agent = PlannerAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(plan(*[f"task {n}" for n in range(12)])))

    assert len((await agent.run(state))["research_plan"]) <= 5


async def test_planner_always_produces_at_least_one_task(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    """An empty plan would stall the pipeline, so it falls back to the raw query."""
    agent = PlannerAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM('{"tasks": [], "research_strategy": ""}'))

    plan_out = (await agent.run(state))["research_plan"]
    assert len(plan_out) == 1
    assert plan_out[0]["description"] == state["original_query"]


async def test_planner_skips_blank_descriptions(monkeypatch: pytest.MonkeyPatch, state) -> None:
    agent = PlannerAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM(plan("real task", "   ")))

    assert len((await agent.run(state))["research_plan"]) == 1


# ── analyst ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_kb(monkeypatch: pytest.MonkeyPatch) -> None:
    async def empty(query: str, limit: int = 5) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(analyst_module, "search_knowledge_base", empty)


async def test_analyst_skips_code_when_it_emits_none(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """Conditional tool use is the agentic part — conceptual work must not run code."""
    agent = AnalystAgent()
    patch_invoke(
        monkeypatch, agent, ScriptedLLM("## Analysis Type\nconceptual, no numbers present")
    )

    called = False

    async def spy(code: str, timeout_s: int = 10) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"success": True, "stdout": "", "stderr": "", "error": ""}

    monkeypatch.setattr(analyst_module, "execute_python", spy)

    update = await agent.run(researched_state)
    assert update["code_outputs"] == []
    assert called is False


async def test_analyst_runs_code_when_it_emits_some(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    agent = AnalystAgent()
    patch_invoke(
        monkeypatch,
        agent,
        ScriptedLLM("## Analysis Type\nstatistical\n```python\nprint(2 + 2)\n```"),
    )

    update = await agent.run(researched_state)
    assert update["code_outputs"][0]["stdout"].strip() == "4"
    assert "## Computed Results" in update["analyzed_data"]


async def test_analyst_ignores_failed_code(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """A broken snippet must not put a traceback into the report the writer reads."""
    agent = AnalystAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("## Analysis\n```python\nimport os\n```"))

    update = await agent.run(researched_state)
    assert update["code_outputs"] == []
    assert "## Computed Results" not in update["analyzed_data"]


# ── writer ────────────────────────────────────────────────────────────────


async def test_writer_numbers_citations_from_the_sources(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """The writer and the UI must agree on what [3] refers to."""
    agent = WriterAgent()
    patch_invoke(monkeypatch, agent, ScriptedLLM("## Executive Summary\nA claim [1]."))

    citations = (await agent.run(researched_state))["citations"]
    assert [c["index"] for c in citations] == [1, 2, 3, 4, 5]
    assert citations[0]["url"] == "https://example.com/1"


async def test_writer_prompt_includes_source_content_not_just_titles(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    """A writer given only titles invents the rest — that is how hallucinated reports happen."""
    agent = WriterAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("report"))
    await agent.run(researched_state)

    assert "Supervisor routing lets a critic" in llm.prompts[0]


async def test_first_draft_has_no_revision_block(
    monkeypatch: pytest.MonkeyPatch, researched_state
) -> None:
    agent = WriterAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("report"))
    await agent.run(researched_state)

    assert "This is revision" not in llm.prompts[0]


async def test_rewrite_carries_the_critic_feedback(
    monkeypatch: pytest.MonkeyPatch, drafted_state
) -> None:
    """Without this the writer rewrites blind and the critic scores it the same way twice."""
    drafted_state["revision_count"] = 1
    drafted_state["critic_feedback"] = "section 2 has no source"
    drafted_state["top_issue"] = "uncited failure rate"

    agent = WriterAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("better report"))
    await agent.run(drafted_state)

    assert "This is revision 1" in llm.prompts[0]
    assert "section 2 has no source" in llm.prompts[0]
    assert "uncited failure rate" in llm.prompts[0]


async def test_writer_with_no_sources_is_told_not_to_invent(
    monkeypatch: pytest.MonkeyPatch, state
) -> None:
    agent = WriterAgent()
    llm = patch_invoke(monkeypatch, agent, ScriptedLLM("report"))

    update = await agent.run(state)
    assert update["citations"] == []
    assert "instead of inventing citations" in llm.prompts[0]

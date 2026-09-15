"""Whole-graph runs with every agent mocked. Proves the rewiring terminates and what it costs."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.graph import nodes as nodes_module
from amaris.graph.pipeline import build_graph
from amaris.graph.state import APPROVE, NEED_MORE_RESEARCH, new_state


def stub(name: str, update: dict[str, Any]):
    """An agent that returns a fixed update, so only the graph's own wiring is under test."""

    class _Stub:
        pass

    _Stub.name = name
    _Stub.run = lambda self, state: _return(update)
    return _Stub


async def _return(update: dict[str, Any]) -> dict[str, Any]:
    return dict(update)


def _sources(n: int) -> list[dict[str, Any]]:
    return [
        {
            "title": f"Agentic multi-agent system source {i}",
            "url": f"https://example.com/{i}",
            "content": "An agentic multi-agent system routes work between agents.",
        }
        for i in range(n)
    ]


def wire(monkeypatch: pytest.MonkeyPatch, depth: str, critic_update: dict[str, Any]) -> None:
    monkeypatch.setattr(
        nodes_module,
        "TriageAgent",
        stub("triage", {"query_depth": depth, "answerable": True, "word_target": 300}),
    )
    monkeypatch.setattr(
        nodes_module,
        "PlannerAgent",
        stub("planner", {"research_plan": [{"task_id": "t1", "description": "d"}]}),
    )
    monkeypatch.setattr(
        nodes_module,
        "ResearcherAgent",
        stub("researcher", {"raw_research": _sources(6), "research_quality": 0.8}),
    )
    monkeypatch.setattr(nodes_module, "AnalystAgent", stub("analyst", {"analyzed_data": "## A"}))
    monkeypatch.setattr(
        nodes_module, "WriterAgent", stub("writer", {"draft_report": "## Answer\nyes"})
    )
    monkeypatch.setattr(nodes_module, "CriticAgent", stub("critic", critic_update))

    async def no_memory(*args: Any, **kwargs: Any) -> None:
        return None

    # no judge to stub out any more — the evaluator's citation audit is pure and free (ADR-039)
    monkeypatch.setattr(nodes_module, "add_session_summary", no_memory)


APPROVED = {
    "critic_scores": {"answer_fit": 0.9},
    "quality_score": 0.85,
    "routing_hint": APPROVE,
    "revision_count": 1,
}


async def test_a_clean_standard_run_costs_no_routing_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Good research plus an agreed approval leaves the supervisor nothing to decide."""
    wire(monkeypatch, "standard", APPROVED)
    final = await build_graph().compile().ainvoke(new_state("what makes a system agentic"))

    assert final["agent_path"] == [
        "triage",
        "planner",
        "researcher",
        "analyst",
        "writer",
        "critic",
        "evaluator",
    ]
    assert final["final_report"].startswith("## Answer")
    assert [d["llm_decided"] for d in final["decision_log"]] == [False, False]


async def test_a_shallow_run_skips_the_analyst(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(monkeypatch, "direct", APPROVED)
    final = await build_graph().compile().ainvoke(new_state("who created python"))

    assert "analyst" not in final["agent_path"]
    assert final["agent_path"][-2:] == ["critic", "evaluator"]


async def test_the_revision_cap_still_ends_a_run_that_keeps_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The critic can no longer be silenced by a high score, so the cap has to hold the line."""
    wire(
        monkeypatch,
        "standard",
        {
            "critic_scores": {"answer_fit": 0.2},
            "quality_score": 0.2,
            "routing_hint": NEED_MORE_RESEARCH,
            "revision_count": 2,
        },
    )
    final = await build_graph().compile().ainvoke(new_state("what makes a system agentic"))

    assert final["agent_path"][-1] == "evaluator"
    assert final["revision_count"] == 2

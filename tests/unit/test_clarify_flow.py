"""An unanswerable question must cost a question back, not a full research run."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.api.schemas import result_from_state
from amaris.graph import nodes as nodes_module
from amaris.graph.pipeline import build_graph
from amaris.graph.state import new_state


class _Triage:
    """Stands in for the triage agent, which is the only LLM call this flow should make."""

    name = "triage"

    async def run(self, state: Any) -> dict[str, Any]:
        return {
            "query_depth": "direct",
            "answerable": False,
            "clarifying_question": "Which city's weather do you want?",
            "triage_reason": "no location was given and today's conditions are local",
            "report_sections": ["Answer"],
            "word_target": 120,
        }


class _MustNotRun:
    name = "researcher"

    async def run(self, state: Any) -> dict[str, Any]:
        raise AssertionError("a clarification run must never reach a research agent")


async def test_a_clarification_run_asks_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    for attr in ("PlannerAgent", "ResearcherAgent", "AnalystAgent", "WriterAgent", "CriticAgent"):
        monkeypatch.setattr(nodes_module, attr, _MustNotRun)
    monkeypatch.setattr(nodes_module, "TriageAgent", _Triage)

    graph = build_graph().compile()
    final = await graph.ainvoke(new_state("tell about today weather"))

    assert final["agent_path"] == ["triage", "clarify"]
    assert "Which city" in final["final_report"]
    assert final["raw_research"] == []
    # the evaluator scores answers; there is no answer here to score
    assert final["evaluation_scores"] == {}


async def test_the_api_marks_a_clarification_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    """The UI has to tell 'here is your answer' apart from 'I need one more thing'."""
    for attr in ("PlannerAgent", "ResearcherAgent", "AnalystAgent", "WriterAgent", "CriticAgent"):
        monkeypatch.setattr(nodes_module, attr, _MustNotRun)
    monkeypatch.setattr(nodes_module, "TriageAgent", _Triage)

    graph = build_graph().compile()
    result = result_from_state(await graph.ainvoke(new_state("tell about today weather")))

    assert result.awaiting_clarification is True
    assert result.trace is not None
    assert result.trace.triage["clarifying_question"] == "Which city's weather do you want?"

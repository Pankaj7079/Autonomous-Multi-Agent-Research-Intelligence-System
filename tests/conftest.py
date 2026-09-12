"""Shared fixtures. Phase 9 adds sample_state and mock_llm here."""

from __future__ import annotations

from pathlib import Path

import pytest

from amaris.graph.state import GraphState, new_state
from amaris.llm.router import reset_cooldowns
from amaris.observability.logging import configure_logging
from tests.helpers import ScriptedLLM


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point sinks at a temp dir so tests never write into the real logs/."""
    log_dir: Path = tmp_path_factory.mktemp("logs")
    configure_logging(level="DEBUG", log_dir=log_dir, json_enabled=False, force=True)


@pytest.fixture(autouse=True)
def _no_parked_providers() -> None:
    """Provider cooldowns are module state, so one rate-limit test would skew every later one."""
    reset_cooldowns()


@pytest.fixture
def state() -> GraphState:
    return new_state("What makes a multi-agent system agentic?", session_id="test1234")


@pytest.fixture
def researched_state(state: GraphState) -> GraphState:
    """State as it looks after a successful research pass."""
    state["research_plan"] = [
        {"task_id": "t1", "description": "find case studies", "assigned_to": "researcher"}
    ]
    state["raw_research"] = [
        {
            "title": f"Source {n}",
            "url": f"https://example.com/{n}",
            "content": "Supervisor routing lets a critic send work back to research. " * 5,
            "task_id": "t1",
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        }
        for n in range(1, 6)
    ]
    state["research_quality"] = 0.8
    return state


@pytest.fixture
def drafted_state(researched_state: GraphState) -> GraphState:
    """State as it looks once the writer has produced something for the critic."""
    researched_state["analyzed_data"] = "## Analysis Type\ncomparative"
    researched_state["draft_report"] = "## Executive Summary\nAgentic systems route via an LLM [1]."
    researched_state["citations"] = [
        {"index": 1, "title": "Source 1", "url": "https://example.com/1"}
    ]
    return researched_state


@pytest.fixture
def sample_state(state: GraphState) -> GraphState:
    """Canonical name for a fresh run state. `state` is the older alias agent tests use."""
    return state


@pytest.fixture
def mock_llm() -> type[ScriptedLLM]:
    """Queue replies in order: mock_llm("first", "second"). The last reply repeats."""
    return ScriptedLLM

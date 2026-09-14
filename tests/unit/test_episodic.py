"""Long-term memory: what survives a run, what is refused, and what is not stored twice."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.memory import episodic
from amaris.memory.episodic import (
    COLLECTION,
    add_research_finding,
    add_session_summary,
    recall_related,
)

FINDING = "Supervisor-based routing lets a critic send work back to research, not just writing."


def _capture(monkeypatch: pytest.MonkeyPatch, existing_score: float = 0.0) -> dict[str, Any]:
    """Records what was stored, and controls what the duplicate probe finds."""
    seen: dict[str, Any] = {"stored": []}

    async def fake_search(
        query: str,
        limit: int = 5,
        session_id: str = "",
        urls: list[str] | None = None,
        collection: str = "",
    ) -> list[dict[str, Any]]:
        seen["searched_collection"] = collection
        if not existing_score:
            return []
        return [{"text": "already known", "url": "", "title": "", "score": existing_score}]

    async def fake_upsert(
        documents: list[dict[str, Any]], session_id: str = "", collection: str = ""
    ) -> int:
        seen["stored"].extend(documents)
        seen["stored_collection"] = collection
        return len(documents)

    monkeypatch.setattr(episodic, "search_knowledge_base", fake_search)
    monkeypatch.setattr(episodic, "upsert_documents", fake_upsert)
    return seen


async def test_a_finding_is_stored_for_future_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)

    assert await add_research_finding("what makes a system agentic", FINDING, "https://a.com")

    stored = seen["stored"][0]
    assert stored["text"] == FINDING
    assert stored["kind"] == "finding"
    # memories live in their own collection: mixing them with attachment chunks would let a
    # past run's conclusion surface as if it were a document the user uploaded
    assert seen["stored_collection"] == COLLECTION


async def test_the_same_finding_is_not_stored_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dedup by vector distance rather than by an LLM call — mem0 spent a model call here."""
    seen = _capture(monkeypatch, existing_score=0.99)

    assert not await add_research_finding("q", FINDING)
    assert seen["stored"] == []


async def test_a_near_miss_is_still_worth_storing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, existing_score=0.60)

    assert await add_research_finding("q", FINDING)
    assert len(seen["stored"]) == 1


async def test_a_trivially_short_finding_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "It depends." is not worth recalling in six months."""
    seen = _capture(monkeypatch)

    assert not await add_research_finding("q", "It depends.")
    assert seen["stored"] == []


async def test_a_session_summary_keeps_the_question_with_the_conclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recall matches on the question, so storing the answer alone would rarely be found."""
    seen = _capture(monkeypatch)

    assert await add_session_summary(
        "what is MCP?", "MCP is a tool protocol. " * 5, {"overall": 0.8}
    )

    stored = seen["stored"][0]
    assert stored["text"].startswith("what is MCP?")
    assert stored["kind"] == "session_summary"


async def test_an_empty_report_is_not_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)

    assert not await add_session_summary("q", "   ", {})
    assert seen["stored"] == []


async def test_recall_reads_only_the_memory_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, existing_score=0.7)

    assert await recall_related("agentic routing") == ["already known"]
    assert seen["searched_collection"] == COLLECTION

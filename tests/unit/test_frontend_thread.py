"""What the sidebar derives from a conversation, and what the transcript export carries."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.api.schemas import ResearchResult, RunTrace
from frontend import thread


class FakeStreamlit:
    """thread.py only ever touches session_state, and a plain dict is one."""

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeStreamlit:
    recorder = FakeStreamlit()
    monkeypatch.setattr(thread, "st", recorder)
    return recorder


def _turn(query: str, report: str, score: float, urls: list[str], elapsed: float) -> dict[str, Any]:
    return {
        "query": query,
        "result": ResearchResult(
            report=report,
            citations=[],
            scores={},
            agent_path=["planner", "researcher", "writer", "critic"],
            trace=RunTrace(
                decisions=[{"step": 1}, {"step": 2}],
                quality_score=score,
                source_count=len(urls),
                sources=[{"url": url, "title": url} for url in urls],
                triage={"depth": "brief"},
            ),
        ),
        "events": [],
        "session_id": "abc12345",
        "error": None,
        "elapsed": elapsed,
    }


def test_no_totals_before_anything_has_run(fake: FakeStreamlit) -> None:
    """An empty strip of zeroes reads as a broken session rather than a new one."""
    assert thread.session_totals() == {}


def test_sources_are_deduped_across_turns(fake: FakeStreamlit) -> None:
    """A follow-up re-reads the same pages, so summing per-run counts would claim the
    conversation saw twice the evidence it actually did."""
    fake.session_state[thread.TURNS] = [
        _turn("q1", "a1", 0.90, ["https://a.com", "https://b.com"], 40.0),
        _turn("q2", "a2", 0.86, ["https://b.com", "https://c.com"], 20.0),
    ]

    totals = thread.session_totals()
    assert totals["Sources"] == "3"
    assert totals["Avg score"] == "0.88"
    assert totals["Routing"] == "4"


def test_elapsed_switches_to_minutes_once_a_conversation_is_long(fake: FakeStreamlit) -> None:
    fake.session_state[thread.TURNS] = [_turn("q", "a", 0.9, ["https://a.com"], 45.0)]
    assert thread.session_totals()["Elapsed"] == "45s"

    fake.session_state[thread.TURNS] = [_turn("q", "a", 0.9, ["https://a.com"], 200.0)]
    assert thread.session_totals()["Elapsed"] == "3.3m"


def test_a_failed_turn_is_counted_in_time_but_not_in_the_average(fake: FakeStreamlit) -> None:
    """It has no score to average, and dropping its seconds would understate the session."""
    failed = {
        "query": "q0",
        "result": None,
        "events": [],
        "session_id": "",
        "error": "provider down",
        "elapsed": 10.0,
    }
    fake.session_state[thread.TURNS] = [failed, _turn("q1", "a1", 0.80, ["https://a.com"], 50.0)]

    totals = thread.session_totals()
    assert totals["Avg score"] == "0.80"
    assert totals["Elapsed"] == "1.0m"


def test_the_transcript_keeps_every_turn_in_order_with_its_question(fake: FakeStreamlit) -> None:
    fake.session_state[thread.TURNS] = [
        _turn("what is MCP?", "MCP is a tool protocol.", 0.9, ["https://a.com"], 30.0),
        _turn("and CrewAI?", "CrewAI is a framework.", 0.8, ["https://b.com"], 30.0),
    ]

    text = thread.transcript_markdown()
    assert text.index("what is MCP?") < text.index("and CrewAI?")
    # headings, so the exported document gets real headings rather than bold paragraphs
    assert "# 1. what is MCP?" in text
    assert "MCP is a tool protocol." in text
    assert "brief research" in text


def test_a_turn_that_produced_nothing_still_appears_in_the_transcript(
    fake: FakeStreamlit,
) -> None:
    """Silently dropping it would make the exported conversation disagree with the screen."""
    fake.session_state[thread.TURNS] = [
        {
            "query": "a question that failed",
            "result": None,
            "events": [],
            "session_id": "",
            "error": "provider down",
            "elapsed": 1.0,
        }
    ]

    text = thread.transcript_markdown()
    assert "a question that failed" in text
    assert "no report was produced" in text

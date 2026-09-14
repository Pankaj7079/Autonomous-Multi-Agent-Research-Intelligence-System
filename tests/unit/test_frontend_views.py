"""Renders a finished run through the real script, so every tab is proven to draw."""

from __future__ import annotations

from pathlib import Path

import pytest

from amaris.api.schemas import ProgressEvent, ResearchResult, RunTrace

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")


def _events() -> list[ProgressEvent]:
    raw = [
        ("planner", "running", 15, 3.0),
        ("researcher", "running", 45, 22.0),
        ("analyst", "running", 65, 31.0),
        ("writer", "running", 85, 40.0),
        ("critic", "running", 92, 46.0),
        ("researcher", "running", 92, 61.0),
        ("evaluator", "running", 96, 70.0),
        ("", "done", 100, 72.0),
    ]
    return [
        ProgressEvent(agent=a, status=s, message=f"{a or 'run'} step", progress_pct=p, elapsed_s=e)
        for a, s, p, e in raw
    ]


def _result() -> ResearchResult:
    return ResearchResult(
        report="# Findings\n\nThe protocol standardises tool access [1].",
        citations=[{"index": 1, "title": "Spec", "url": "https://example.test/spec"}],
        scores={
            "answer_fit": 0.84,
            "faithfulness": 0.9,
            "completeness": 0.7,
            "coherence": 0.85,
            "citation_quality": 0.8,
            "eval_context_precision": 0.77,
        },
        agent_path=["planner", "researcher", "analyst", "writer", "critic", "researcher"],
        trace=RunTrace(
            decisions=[
                {
                    "step": 1,
                    "from_agent": "critic",
                    "to_agent": "researcher",
                    "expected_agent": "writer",
                    "matched_rule": "fix_writing",
                    "llm_decided": True,
                    "research_quality": 0.6,
                    "quality_score": 0.65,
                }
            ],
            react_stats={"t1": {"iterations_used": 3, "self_terminated": True}},
            research_quality=0.78,
            revision_count=1,
            source_count=9,
            critic_feedback="well sourced overall",
            top_issue="one claim is thinly supported",
            research_plan=[
                {
                    "task_id": "t1",
                    "description": "find the specification",
                    "assigned_to": "researcher",
                }
            ],
            research_strategy="breadth first, then depth on the spec",
            analysis="Three findings emerged from the sources.",
            routing_hint="approve",
            quality_score=0.81,
            triage={
                "depth": "brief",
                "answerable": True,
                "sections": ["Answer", "Key Points"],
                "word_target": 300,
                "reason": "a couple of related points",
            },
            sources=[
                {
                    "title": "Spec",
                    "url": "https://example.test/spec",
                    "task_id": "t1",
                    "snippet": "a",
                },
                {
                    "title": "Blog",
                    "url": "https://example.test/blog",
                    "task_id": "t1",
                    "snippet": "b",
                },
            ],
        ),
    )


def _turn(result: ResearchResult | None, error: str | None = None, events=None) -> dict:
    return {
        "query": "What is MCP?",
        "result": result,
        "events": events if events is not None else _events(),
        "session_id": "5e700386",
        "error": error,
        "elapsed": 72.0,
    }


def _finished_app():
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30)
    app.session_state["turns"] = [_turn(_result())]
    return app.run()


def test_a_finished_run_renders_every_tab_without_raising() -> None:
    app = _finished_app()
    assert not app.exception, [e.value for e in app.exception]


def test_the_report_and_the_session_id_are_shown() -> None:
    app = _finished_app()
    drawn = "".join(b.value for b in app.markdown)
    assert "The protocol standardises tool access" in drawn
    assert any("5e700386" in c.value for c in app.caption)


def test_the_planner_and_analyst_output_reach_the_screen() -> None:
    """Both existed only in graph state before — an agent with nothing on screen is a black box."""
    app = _finished_app()
    drawn = "".join(b.value for b in app.markdown)
    assert "find the specification" in drawn
    assert "Three findings emerged" in drawn
    assert "breadth first" in drawn


def test_the_rerouted_researcher_is_visible_as_a_repeat_visit() -> None:
    app = _finished_app()
    drawn = "".join(b.value for b in app.markdown)
    assert "&times;2" in drawn, "the flow strip must show the researcher ran twice"
    assert "diverged" in drawn, "the critic overruling the rule table is the headline behaviour"


def test_an_uncited_source_is_still_reachable() -> None:
    app = _finished_app()
    drawn = "".join(b.value for b in app.markdown)
    assert "example.test/blog" in drawn


def test_the_system_tab_reports_the_live_configuration() -> None:
    app = _finished_app()
    drawn = "".join(b.value for b in app.markdown)
    assert "approve floor" in drawn
    assert "supervisor cap" in drawn


def test_a_failed_run_shows_the_error_and_the_events_not_a_blank_page() -> None:
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30)
    app.session_state["turns"] = [
        _turn(None, error="provider chain exhausted", events=_events()[:3])
    ]
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    drawn = "".join(b.value for b in app.markdown) + "".join(e.value for e in app.error)
    assert "provider chain exhausted" in drawn
    assert 'class="log"' in drawn, "the event stream must survive a failed run"


def test_a_turn_offers_the_deeper_rerun_and_names_the_depth_it_moves_to() -> None:
    """Short by default is only reasonable if more is one click away and says what it costs."""
    app = _finished_app()
    labels = [b.label for b in app.button]
    assert any("explain in detail" in label and "standard" in label for label in labels), labels


def test_the_deepest_depth_has_nothing_left_to_expand_into() -> None:
    result = _result()
    result.trace.triage = {**result.trace.triage, "depth": "deep"}
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30)
    app.session_state["turns"] = [_turn(result)]
    app.run()
    assert not any("explain in detail" in b.label for b in app.button)


def test_a_second_turn_renders_both_questions() -> None:
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30)
    first = _turn(_result())
    second = {**_turn(_result()), "query": "what about its transport?"}
    app.session_state["turns"] = [first, second]
    app.run()

    assert not app.exception, [e.value for e in app.exception]
    drawn = "".join(b.value for b in app.markdown)
    assert "What is MCP?" in drawn
    assert "what about its transport?" in drawn

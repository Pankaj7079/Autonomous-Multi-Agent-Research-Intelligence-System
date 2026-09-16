"""Harness plumbing: runs the golden set through an injected pipeline, no real LLM calls here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from amaris.evaluation import _ragas_shared as shared
from amaris.evaluation.base import EvalReport, EvalResult
from amaris.evaluation.golden_set import GoldenQuery
from amaris.evaluation.harness import (
    _golden_checks,
    _load_report,
    _write_report,
    compare_reports,
    run_evaluation,
)
from amaris.graph.state import new_state


def golden(**overrides: Any) -> GoldenQuery:
    defaults = {
        "id": "g1",
        "query": "what is X",
        "category": "factual",
        "expected_min_sources": 2,
        "expected_agents_involved": ["planner", "researcher"],
        "should_require_revision": False,
        "notes": "",
        "reference_answer": None,
    }
    return GoldenQuery(**{**defaults, **overrides})


def finished_state(
    *, sources: int = 3, agents: list[str] | None = None, revisions: int = 0
) -> dict:
    state = new_state("what is X")
    state["raw_research"] = [{"url": f"https://x.com/{n}", "content": "x"} for n in range(sources)]
    state["agent_path"] = agents or ["planner", "researcher", "writer", "critic"]
    state["revision_count"] = revisions
    state["decision_log"] = [
        {
            "step": 1,
            "from_agent": "START",
            "to_agent": "planner",
            "llm_decided": True,
            "expected_agent": "planner",
            "matched_rule": "rule_3_no_plan",
            "research_quality": 0.0,
            "source_count": 0,
            "quality_score": 0.0,
            "analysis_done": False,
            "draft_exists": False,
            "revision_count": 0,
            "routing_hint": "none",
        }
    ]
    state["react_stats"] = {"t1": {"iterations_used": 2, "self_terminated": True}}
    state["draft_report"] = "the report"
    state["final_report"] = "the report"
    return state


# ── golden_checks ────────────────────────────────────────────────────────────


def test_golden_checks_pass_when_expectations_are_met() -> None:
    results = _golden_checks(golden(), finished_state(sources=3, agents=["planner", "researcher"]))
    assert all(r.passed for r in results)


def test_golden_checks_flag_too_few_sources() -> None:
    results = _golden_checks(golden(expected_min_sources=10), finished_state(sources=3))
    check = next(r for r in results if r.metric == "min_sources")
    assert check.passed is False
    assert "[g1]" in check.detail


def test_golden_checks_flag_a_missing_expected_agent() -> None:
    results = _golden_checks(
        golden(expected_agents_involved=["planner", "critic"]), finished_state(agents=["planner"])
    )
    check = next(r for r in results if r.metric == "agents_involved")
    assert check.passed is False
    assert "critic" in check.detail


def test_golden_checks_flag_an_agent_that_should_have_been_skipped() -> None:
    """The direct/clarify/live point is that a whole run was avoided, not merely shortened."""
    results = _golden_checks(
        golden(forbidden_agents=["analyst"]),
        finished_state(agents=["planner", "researcher", "analyst"]),
    )
    check = next(r for r in results if r.metric == "agents_skipped")
    assert check.passed is False
    assert "analyst" in check.detail


def test_golden_checks_pass_when_the_forbidden_agent_stayed_out() -> None:
    results = _golden_checks(
        golden(forbidden_agents=["analyst"]), finished_state(agents=["planner", "researcher"])
    )
    check = next(r for r in results if r.metric == "agents_skipped")
    assert check.passed is True


def test_golden_checks_flag_a_wrongly_sized_run() -> None:
    state = finished_state()
    state["query_depth"] = "deep"
    results = _golden_checks(golden(expected_depth=["direct"]), state)
    check = next(r for r in results if r.metric == "depth_sized")
    assert check.passed is False
    assert "deep" in check.detail


def test_depth_is_not_checked_when_the_golden_query_does_not_care() -> None:
    results = _golden_checks(golden(expected_depth=[]), finished_state())
    assert not [r for r in results if r.metric == "depth_sized"]


def test_golden_checks_verify_revision_expectation() -> None:
    results = _golden_checks(golden(should_require_revision=True), finished_state(revisions=0))
    check = next(r for r in results if r.metric == "revision_expected")
    assert check.passed is False


def test_golden_checks_skip_revision_check_when_not_required() -> None:
    results = _golden_checks(golden(should_require_revision=False), finished_state(revisions=0))
    assert not any(r.metric == "revision_expected" for r in results)


# ── run_evaluation ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_ragas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every query in these tests has real sources but scoring is not what's under test."""
    monkeypatch.setattr(shared, "is_available", lambda: False)
    monkeypatch.setattr(shared, "unavailable_reason", lambda: "not under test")


async def test_run_evaluation_scores_every_query(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_pipeline(query: str) -> dict:
        return finished_state()

    report = await run_evaluation([golden(id="a"), golden(id="b")], pipeline=fake_pipeline)
    ids = {r.detail.split("]")[0].strip("[") for r in report.results}
    assert ids == {"a", "b"}


async def test_a_crashed_query_does_not_lose_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    """One query blowing up must not void the whole golden-set run."""

    async def flaky_pipeline(query: str) -> dict:
        if query == "bad query":
            raise RuntimeError("pipeline exploded")
        return finished_state()

    report = await run_evaluation(
        [golden(id="ok", query="fine"), golden(id="bad", query="bad query")],
        pipeline=flaky_pipeline,
    )
    assert any(r.metric == "pipeline_error" and "[bad]" in r.detail for r in report.results)
    assert any("[ok]" in r.detail for r in report.results)


async def test_a_layer_crash_still_returns_the_other_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amaris.evaluation import trajectory_eval

    async def boom(self: Any, state: Any) -> list[EvalResult]:
        raise RuntimeError("layer 3 blew up")

    monkeypatch.setattr(trajectory_eval.TrajectoryEvaluator, "evaluate", boom)

    async def fake_pipeline(query: str) -> dict:
        return finished_state()

    report = await run_evaluation([golden()], pipeline=fake_pipeline)
    assert any(r.layer == "golden_set" for r in report.results)
    assert any(r.metric == "error" and r.layer == "trajectory" for r in report.results)


# ── compare_reports ────────────────────────────────────────────────────────


def test_compare_reports_flags_a_real_regression() -> None:
    old = EvalReport(results=[EvalResult("trajectory", "routing_accuracy", 0.9, True, "x")])
    new = EvalReport(results=[EvalResult("trajectory", "routing_accuracy", 0.8, True, "x")])
    flags = compare_reports(old, new)
    assert len(flags) == 1
    assert "routing_accuracy" in flags[0]


def test_compare_reports_ignores_a_small_drop() -> None:
    old = EvalReport(results=[EvalResult("trajectory", "routing_accuracy", 0.90, True, "x")])
    new = EvalReport(results=[EvalResult("trajectory", "routing_accuracy", 0.87, True, "x")])
    assert compare_reports(old, new) == []


def test_compare_reports_flags_a_metric_that_disappeared() -> None:
    old = EvalReport(results=[EvalResult("report", "faithfulness", 0.9, True, "x")])
    new = EvalReport(results=[EvalResult("retrieval", "context_precision", 0.9, True, "x")])
    flags = compare_reports(old, new)
    assert any("missing" in f for f in flags)


# ── report round-trip ────────────────────────────────────────────────────────


def test_write_and_load_report_round_trips(tmp_path: Path) -> None:
    report = EvalReport(
        results=[EvalResult("trajectory", "termination_quality", 1.0, True, "approved")],
        query="q",
        session_id="s1",
    )
    md_path, json_path = _write_report(report, tmp_path)
    assert md_path.exists()
    assert json_path.exists()
    assert json.loads(json_path.read_text())["overall"] == 1.0

    loaded = _load_report(json_path)
    assert loaded.overall() == report.overall()
    assert loaded.results[0].metric == "termination_quality"

"""EvalResult/EvalReport shapes — every layer's output rides on these being right."""

from __future__ import annotations

from amaris.evaluation.base import EvalReport, EvalResult, zero_result


def test_zero_result_never_passes() -> None:
    result = zero_result("retrieval", "context_precision", "no sources")
    assert result.score == 0.0
    assert result.passed is False
    assert result.detail == "no sources"


def test_overall_is_empty_safe() -> None:
    assert EvalReport(results=[]).overall() == 0.0
    assert EvalReport(results=[]).pass_rate() == 0.0


def test_overall_averages_across_layers() -> None:
    report = EvalReport(
        results=[
            EvalResult("retrieval", "context_precision", 1.0, True, "ok"),
            EvalResult("report", "faithfulness", 0.5, False, "half grounded"),
        ]
    )
    assert report.overall() == 0.75


def test_pass_rate_counts_passed_flag_not_score() -> None:
    report = EvalReport(
        results=[
            EvalResult("trajectory", "routing_accuracy", 0.9, True, "ok"),
            EvalResult("trajectory", "loop_efficiency", 0.9, False, "borderline"),
        ]
    )
    assert report.pass_rate() == 0.5


def test_by_layer_groups_without_losing_order() -> None:
    report = EvalReport(
        results=[
            EvalResult("retrieval", "context_precision", 1.0, True, "a"),
            EvalResult("report", "faithfulness", 1.0, True, "b"),
            EvalResult("retrieval", "context_recall", 0.5, True, "c"),
        ]
    )
    grouped = report.by_layer()
    assert [r.metric for r in grouped["retrieval"]] == ["context_precision", "context_recall"]
    assert [r.metric for r in grouped["report"]] == ["faithfulness"]


def test_markdown_renders_every_layer_as_a_table() -> None:
    report = EvalReport(
        results=[EvalResult("trajectory", "termination_quality", 1.0, True, "approved")],
        query="what is X",
    )
    md = report.to_markdown()
    assert "trajectory" in md
    assert "termination_quality" in md
    assert "what is X" in md


def test_to_dict_rounds_scores_for_json() -> None:
    result = EvalResult("report", "faithfulness", 0.123456789, True, "ok")
    assert result.to_dict()["score"] == 0.1235

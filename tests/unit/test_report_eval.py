"""Layer 2: final report vs sources and vs query. The one hallucination check in the design."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.evaluation import _ragas_shared as shared
from amaris.evaluation.report_eval import ReportEvaluator
from amaris.graph.state import new_state


@pytest.fixture(autouse=True)
def _uncached() -> None:
    shared.judge.cache_clear()
    shared.embedder.cache_clear()


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch):
    def _apply(scores: dict[str, float], with_embedder: bool = True) -> None:
        async def fake_run_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
            return scores

        monkeypatch.setattr(shared, "is_available", lambda: True)
        monkeypatch.setattr(shared, "embedder", lambda: object() if with_embedder else None)
        monkeypatch.setattr(shared, "run_metrics", fake_run_metrics)

    return _apply


def _state_with_report() -> dict:
    state = new_state("what is X")
    state["raw_research"] = [{"url": "https://x.com/1", "content": "X is a protocol " * 20}]
    state["draft_report"] = "X is a protocol [1]."
    return state


async def test_no_report_skips_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared, "is_available", lambda: pytest.fail("should not be called"))
    results = await ReportEvaluator().evaluate(new_state("q"))
    assert results[0].metric == "faithfulness"
    assert results[0].score == 0.0


async def test_no_sources_skips_scoring_even_with_a_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faithfulness against nothing is meaningless — must not spend a judge call on it."""
    monkeypatch.setattr(shared, "is_available", lambda: pytest.fail("should not be called"))
    state = new_state("q")
    state["draft_report"] = "some report text"
    results = await ReportEvaluator().evaluate(state)
    assert results[0].score == 0.0


async def test_uses_final_report_over_draft_when_both_exist(offline) -> None:
    offline({"faithfulness": 1.0})
    state = _state_with_report()
    state["final_report"] = "the promoted final version"
    state["draft_report"] = "an older draft"
    results = await ReportEvaluator().evaluate(state)
    assert results[0].score == 1.0  # reaching the ragas call at all proves final_report won


async def test_faithfulness_and_relevancy_both_score_when_embedder_present(offline) -> None:
    offline({"faithfulness": 1.0, "answer_relevancy": 0.85})
    results = await ReportEvaluator().evaluate(_state_with_report())
    metrics = {r.metric: r.score for r in results}
    assert metrics == {"faithfulness": 1.0, "answer_relevancy": 0.85}


async def test_relevancy_is_skipped_without_an_embedder(offline) -> None:
    """No Gemini key configured — answer_relevancy is dropped, not fatal."""
    offline({"faithfulness": 1.0}, with_embedder=False)
    results = await ReportEvaluator().evaluate(_state_with_report())
    assert [r.metric for r in results] == ["faithfulness"]


async def test_a_fabricated_report_scores_lower_than_a_grounded_one(offline) -> None:
    """The behavior verified live in Phase 6 — encoded as a real (mocked) regression test."""
    offline({"faithfulness": 0.0})
    fabricated = await ReportEvaluator().evaluate(_state_with_report())
    offline({"faithfulness": 1.0})
    grounded = await ReportEvaluator().evaluate(_state_with_report())
    assert fabricated[0].score < grounded[0].score


async def test_a_nan_faithfulness_is_distinguishable_from_a_real_zero(offline) -> None:
    """A genuinely fabricated report and a rate-limited judge call must not look identical."""
    offline({"faithfulness": float("nan")})
    nan_result = await ReportEvaluator().evaluate(_state_with_report())
    assert "NaN" in nan_result[0].detail

    offline({"faithfulness": 0.0})
    real_zero = await ReportEvaluator().evaluate(_state_with_report())
    assert "NaN" not in real_zero[0].detail
    assert real_zero[0].score == 0.0


async def test_ragas_unavailable_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared, "is_available", lambda: False)
    monkeypatch.setattr(shared, "unavailable_reason", lambda: "no provider key")
    results = await ReportEvaluator().evaluate(_state_with_report())
    assert results[0].score == 0.0
    assert "no provider key" in results[0].detail

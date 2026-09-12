"""Layer 1: scoped to sources vs the query. Must never reach into the final report."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.evaluation import _ragas_shared as shared
from amaris.evaluation.retrieval_eval import RetrievalEvaluator, _response_proxy
from amaris.graph.state import new_state


@pytest.fixture(autouse=True)
def _uncached() -> None:
    shared.judge.cache_clear()
    shared.embedder.cache_clear()


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch):
    def _apply(scores: dict[str, float]) -> None:
        async def fake_run_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
            return scores

        monkeypatch.setattr(shared, "is_available", lambda: True)
        monkeypatch.setattr(shared, "run_metrics", fake_run_metrics)

    return _apply


def _state_with_sources(n: int = 3) -> dict:
    state = new_state("what is X")
    state["raw_research"] = [
        {"title": f"Source {i}", "url": f"https://x.com/{i}", "content": f"content {i}" * 20}
        for i in range(n)
    ]
    return state


async def test_no_sources_skips_scoring_without_calling_ragas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared, "is_available", lambda: pytest.fail("should not be called"))
    results = await RetrievalEvaluator().evaluate(new_state("q"))
    assert results[0].metric == "context_precision"
    assert results[0].score == 0.0


async def test_ragas_unavailable_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared, "is_available", lambda: False)
    monkeypatch.setattr(shared, "unavailable_reason", lambda: "ragas not installed")
    results = await RetrievalEvaluator().evaluate(_state_with_sources())
    assert results[0].score == 0.0
    assert "not installed" in results[0].detail


async def test_precision_only_by_default(offline) -> None:
    offline({"llm_context_precision_without_reference": 0.8})
    results = await RetrievalEvaluator().evaluate(_state_with_sources())
    assert [r.metric for r in results] == ["context_precision"]
    assert results[0].score == 0.8


async def test_recall_is_added_when_a_reference_is_given(offline) -> None:
    offline({"llm_context_precision_without_reference": 0.8, "context_recall": 0.6})
    results = await RetrievalEvaluator().evaluate(
        _state_with_sources(), reference="the real answer"
    )
    metrics = {r.metric: r.score for r in results}
    assert metrics == {"context_precision": 0.8, "context_recall": 0.6}


async def test_scores_are_clamped(offline) -> None:
    offline({"llm_context_precision_without_reference": 1.4})
    results = await RetrievalEvaluator().evaluate(_state_with_sources())
    assert results[0].score == 1.0


async def test_a_nan_score_is_not_reported_as_a_confirmed_zero(offline) -> None:
    """Found live: a rate-limited judge made ragas return NaN, and min(1.0, max(0.0, nan)) == 0.0
    silently turned "the judge couldn't score this" into "confirmed bad retrieval"."""
    offline({"llm_context_precision_without_reference": float("nan")})
    results = await RetrievalEvaluator().evaluate(_state_with_sources())
    assert results[0].score == 0.0
    assert "NaN" in results[0].detail
    assert "rate limited" in results[0].detail


async def test_a_ragas_failure_returns_zero_not_a_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared, "is_available", lambda: True)

    async def boom(*args: Any, **kwargs: Any) -> None:
        return None  # this is how run_metrics signals timeout/failure

    monkeypatch.setattr(shared, "run_metrics", boom)
    results = await RetrievalEvaluator().evaluate(_state_with_sources())
    assert results[0].score == 0.0


def test_response_proxy_uses_analysis_not_the_report() -> None:
    """Layer 1 must never see the writer's report — that line is the whole point of the split."""
    state = _state_with_sources()
    state["analyzed_data"] = "## Analysis\nsome synthesis"
    state["draft_report"] = "## Executive Summary\nthe polished report"
    assert _response_proxy(state) == "## Analysis\nsome synthesis"
    assert "polished report" not in _response_proxy(state)


def test_response_proxy_falls_back_to_content_not_bare_titles() -> None:
    """Found live: titles alone give the precision judge nothing to assess and score near 0."""
    state = _state_with_sources(2)
    proxy = _response_proxy(state)
    assert "content 0" in proxy
    assert "content 1" in proxy

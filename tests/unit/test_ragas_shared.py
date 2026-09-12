"""_ragas_shared.run_metrics: the RunConfig passed to ragas, not ragas itself."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.evaluation import _ragas_shared as shared


@pytest.fixture(autouse=True)
def _uncached() -> None:
    shared.judge.cache_clear()
    shared.embedder.cache_clear()


def test_the_vertexai_shim_installs_a_real_class_not_object() -> None:
    """ragas isinstance-checks this symbol — object would make every model match the wrong branch."""
    import sys

    sys.modules.pop("langchain_community.chat_models.vertexai", None)
    shared._shim_sunset_vertexai()
    installed = sys.modules["langchain_community.chat_models.vertexai"]
    assert installed.ChatVertexAI is not object
    assert not isinstance("anything", installed.ChatVertexAI)


def test_trim_contexts_caps_count_and_length() -> None:
    trimmed = shared.trim_contexts(["x" * 5000] * (shared.MAX_CONTEXTS + 5))
    assert len(trimmed) == shared.MAX_CONTEXTS
    assert all(len(t) == shared.CONTEXT_CHAR_LIMIT for t in trimmed)


def test_trim_contexts_drops_blanks() -> None:
    assert shared.trim_contexts(["real", "", "   ", "more"]) == ["real", "more"]


def test_clamp_or_none_returns_none_for_nan() -> None:
    """min(1.0, max(0.0, nan)) silently evaluates to 0.0 in plain Python — this must not."""
    assert shared.clamp_or_none(float("nan")) is None


@pytest.mark.parametrize(("raw", "expected"), [(0.5, 0.5), (1.4, 1.0), (-0.3, 0.0), (0.0, 0.0)])
def test_clamp_or_none_clamps_real_values(raw: float, expected: float) -> None:
    assert shared.clamp_or_none(raw) == expected


async def test_ragas_runs_with_a_short_retry_budget_not_its_own_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found live: ragas's RunConfig defaults to max_retries=10/max_wait=60, blind to our
    cooldown-aware router — a single rate-limited call spent 120s retrying inside ragas alone."""
    captured: dict[str, Any] = {}

    class FakeResult:
        def __init__(self) -> None:
            self.scores = [{"faithfulness": 1.0}]

    def fake_evaluate(**kwargs: Any) -> FakeResult:
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr("ragas.evaluate", fake_evaluate)
    monkeypatch.setattr(shared, "judge", lambda: object())
    monkeypatch.setattr(shared, "embedder", lambda: None)

    await shared.run_metrics([object()], "q", "a", ["context"], label="test")

    run_config = captured["run_config"]
    assert run_config.max_retries == shared._RAGAS_MAX_RETRIES
    assert run_config.max_wait == shared._RAGAS_MAX_WAIT_SECONDS


async def test_timeout_logs_which_layer_via_label(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def never_returns(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError

    monkeypatch.setattr(shared.asyncio, "wait_for", never_returns)
    monkeypatch.setattr(shared, "judge", lambda: object())

    result = await shared.run_metrics([object()], "q", "a", ["c"], label="retrieval")
    assert result is None

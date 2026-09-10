"""Fallback behaviour, not provider internals — no test here makes a real API call."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.config.settings import Settings
from amaris.llm import router
from amaris.llm.router import LLMConfigError, configured_chain, get_llm, invoke_with_fallback


class FakeLLM:
    """Minimal stand-in: either returns a canned reply or raises what the test wants."""

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.calls = 0

    async def ainvoke(self, messages: Any, **kwargs: Any) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return f"reply from {self.name}"


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch):
    """Swap in settings built with explicit keys, ignoring any real .env on disk."""

    def _apply(**provider_keys: str) -> Settings:
        settings = Settings(_env_file=None, **provider_keys)
        monkeypatch.setattr(router, "get_settings", lambda: settings)
        return settings

    return _apply


@pytest.fixture
def fake_providers(monkeypatch: pytest.MonkeyPatch):
    """Replace the model cache so no provider SDK is constructed."""

    def _apply(**by_provider: FakeLLM) -> dict[str, FakeLLM]:
        monkeypatch.setattr(router, "_cached_llm", lambda provider, task: by_provider[provider])
        return by_provider

    return _apply


def test_chain_follows_fallback_order_and_skips_missing_keys(keys) -> None:
    keys(gemini_api_key="g", groq_api_key="k")
    assert configured_chain() == ["groq", "gemini"]


def test_get_llm_raises_with_actionable_message_when_nothing_configured(keys) -> None:
    keys()
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        get_llm("planning")


def test_get_llm_rejects_a_provider_without_a_key(keys) -> None:
    keys(groq_api_key="k")
    with pytest.raises(LLMConfigError, match="cerebras"):
        get_llm("planning", provider="cerebras")


@pytest.mark.parametrize(
    ("task", "expected_model_field"),
    [("planning", "groq_model_reasoning"), ("research", "groq_model_fast")],
)
def test_task_type_picks_the_right_groq_model(keys, task: str, expected_model_field: str) -> None:
    """The researcher's many short calls must not burn the 70b quota."""
    settings = keys(groq_api_key="gsk_test")
    router._cached_llm.cache_clear()
    llm = get_llm(task)
    assert llm.model_name == getattr(settings, expected_model_field)


def test_supervisor_runs_at_zero_temperature() -> None:
    """Routing has to be reproducible, so the supervisor never samples."""
    assert router._temperature("supervisor") == 0.0


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (RuntimeError("Error code: 429 - rate limit reached"), True),
        (TimeoutError("request timed out"), True),
        (ConnectionError("connection refused"), True),
        (ValueError("invalid prompt template"), False),
    ],
)
def test_is_retryable_matches_transient_failures(exc: Exception, retryable: bool) -> None:
    assert router.is_retryable(exc) is retryable


async def test_falls_through_to_the_next_provider_on_a_rate_limit(keys, fake_providers) -> None:
    keys(groq_api_key="k", gemini_api_key="g")
    llms = fake_providers(
        groq=FakeLLM("groq", RuntimeError("Error code: 429 rate limit")),
        gemini=FakeLLM("gemini"),
    )

    assert await invoke_with_fallback("hello") == "reply from gemini"
    assert llms["groq"].calls == 1


async def test_does_not_fall_through_on_a_real_bug(keys, fake_providers) -> None:
    """A malformed request fails the same way everywhere — hopping just wastes quota."""
    keys(groq_api_key="k", gemini_api_key="g")
    llms = fake_providers(groq=FakeLLM("groq", ValueError("bad input")), gemini=FakeLLM("gemini"))

    with pytest.raises(ValueError, match="bad input"):
        await invoke_with_fallback("hello")
    assert llms["gemini"].calls == 0


async def test_raises_when_every_provider_rate_limits(keys, fake_providers) -> None:
    keys(groq_api_key="k", gemini_api_key="g")
    fake_providers(
        groq=FakeLLM("groq", RuntimeError("429")),
        gemini=FakeLLM("gemini", RuntimeError("429 quota exceeded")),
    )

    with pytest.raises(RuntimeError, match="quota"):
        await invoke_with_fallback("hello")


async def test_invoke_with_no_keys_explains_how_to_fix_it(keys) -> None:
    keys()
    with pytest.raises(LLMConfigError, match=r"console\.groq\.com"):
        await invoke_with_fallback("hello")

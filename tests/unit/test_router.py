"""Fallback behaviour, not provider internals — no test here makes a real API call."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.config.settings import Settings
from amaris.llm import router
from amaris.llm.router import (
    LLMConfigError,
    ProvidersCoolingDown,
    chain_wait_seconds,
    configured_chain,
    cooldown_remaining,
    get_llm,
    invoke_with_fallback,
    retry_after_seconds,
)


class FakeMessage:
    """Mimics BaseMessage.text — the real bug this project hit was content as a block list."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeLLM:
    """Minimal stand-in: either returns a canned reply or raises what the test wants."""

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.calls = 0

    async def ainvoke(self, messages: Any, **kwargs: Any) -> FakeMessage:
        self.calls += 1
        if self.error:
            raise self.error
        return FakeMessage(f"reply from {self.name}")


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


def test_anthropic_is_last_since_it_is_not_free_tier(keys) -> None:
    """Free providers are tried first — anthropic only runs when both of them fail."""
    keys(groq_api_key="k", gemini_api_key="g", anthropic_api_key="a")
    assert configured_chain() == ["groq", "gemini", "anthropic"]


def test_get_llm_raises_with_actionable_message_when_nothing_configured(keys) -> None:
    keys()
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        get_llm("planning")


def test_get_llm_rejects_a_provider_without_a_key(keys) -> None:
    keys(groq_api_key="k")
    with pytest.raises(LLMConfigError, match="anthropic"):
        get_llm("planning", provider="anthropic")


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

    assert (await invoke_with_fallback("hello")).text == "reply from gemini"
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


def test_text_property_normalises_gemini_3x_block_content() -> None:
    """Gemini 3.x returns content as a block list, not a string — .content.strip() breaks on it."""
    from langchain_core.messages import AIMessage

    plain = AIMessage(content="researcher")
    blocks = AIMessage(content=[{"type": "text", "text": "researcher", "extras": {}}])
    assert plain.text == blocks.text == "researcher"


async def test_invoke_with_no_keys_explains_how_to_fix_it(keys) -> None:
    keys()
    with pytest.raises(LLMConfigError, match=r"console\.groq\.com"):
        await invoke_with_fallback("hello")


# ── rate-limit cooldown ───────────────────────────────────────────────────
# found live: the writer's big call exhausts groq's tpm window, and every later
# call in the run then burned a request on an already-exhausted gemini


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Rate limit reached. Please try again in 58.64s.", 58.64),
        ("Limit 6000, used 5994. Please try again in 1m12.5s.", 72.5),
        ("RESOURCE_EXHAUSTED ... retryDelay: '34s'", 34.0),
        ("429 Too Many Requests, retry-after: 60", 60.0),
        # gemini says "retry in", groq says "try again in" — both have to parse
        ("429 You exceeded your quota. Please retry in 34.9s.", 34.9),
        ("RESOURCE_EXHAUSTED retry_delay { seconds: 27 }", 27.0),
        ("Error code: 429 - rate limit reached", None),
        ("connection reset by peer", None),
    ],
)
def test_retry_after_is_read_from_the_provider_message(
    message: str, expected: float | None
) -> None:
    """Backoff guessed 0.5s/1s/2s against a 60s window — the provider already says the number."""
    assert retry_after_seconds(RuntimeError(message)) == expected


async def test_a_parked_provider_is_skipped_on_the_next_call(keys, fake_providers) -> None:
    """Asking a provider again inside the window it just named wastes a request and a round trip."""
    keys(groq_api_key="k", gemini_api_key="g")
    llms = fake_providers(
        groq=FakeLLM("groq", RuntimeError("429 rate limit, please try again in 58.6s")),
        gemini=FakeLLM("gemini"),
    )

    assert (await invoke_with_fallback("one")).text == "reply from gemini"
    assert (await invoke_with_fallback("two")).text == "reply from gemini"
    assert llms["groq"].calls == 1


async def test_a_transient_error_does_not_park_a_provider(keys, fake_providers) -> None:
    """No stated delay means no quota claim, so a healthy primary must not be sidelined."""
    keys(groq_api_key="k", gemini_api_key="g")
    llms = fake_providers(
        groq=FakeLLM("groq", ConnectionError("connection reset")), gemini=FakeLLM("gemini")
    )

    await invoke_with_fallback("one")
    await invoke_with_fallback("two")
    assert llms["groq"].calls == 2
    assert cooldown_remaining() == 0.0


async def test_a_per_day_quota_parks_the_provider_for_the_whole_run(keys, fake_providers) -> None:
    """Gemini's free tier is 20 requests/day; its 34s retryDelay is useless once that is spent."""
    keys(groq_api_key="k", gemini_api_key="g")
    fake_providers(
        groq=FakeLLM("groq", RuntimeError("429 try again in 5.0s")),
        gemini=FakeLLM(
            "gemini", RuntimeError("RESOURCE_EXHAUSTED GenerateRequestsPerDay limit: 20")
        ),
    )

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await invoke_with_fallback("one")
    # groq frees up in 5s, gemini not for an hour — the wait reported is the sooner one
    assert 5.0 < cooldown_remaining() <= 6.0


async def test_every_provider_parked_raises_with_the_wait(keys, fake_providers) -> None:
    """The agent needs the number to sleep on, not a generic failure it can only guess at."""
    keys(groq_api_key="k", gemini_api_key="g")
    fake_providers(
        groq=FakeLLM("groq", RuntimeError("429 try again in 30s")),
        gemini=FakeLLM("gemini", RuntimeError("429 try again in 45s")),
    )

    with pytest.raises(RuntimeError):
        await invoke_with_fallback("one")
    with pytest.raises(ProvidersCoolingDown) as caught:
        await invoke_with_fallback("two")
    assert 29.0 < caught.value.seconds <= 31.0


def test_cooling_down_counts_as_retryable() -> None:
    """Otherwise the agent raises straight through instead of waiting out the window."""
    assert router.is_retryable(ProvidersCoolingDown(30.0)) is True


async def test_chain_wait_is_zero_while_any_provider_is_free(keys, fake_providers) -> None:
    """The primary blipping must not inherit the fallback's hour-long quota wait."""
    keys(groq_api_key="k", gemini_api_key="g")
    fake_providers(
        groq=FakeLLM("groq", ConnectionError("connection reset")),
        gemini=FakeLLM("gemini", RuntimeError("429 GenerateRequestsPerDay limit: 20")),
    )

    with pytest.raises(RuntimeError):
        await invoke_with_fallback("one")
    # gemini is parked for an hour, groq was never parked, so there is still something to call
    assert cooldown_remaining() > 0.0
    assert chain_wait_seconds() == 0.0


# ── patch 1: glm as the third free hop ──────────────────────────────────────


def test_glm_sits_before_anthropic_so_the_paid_key_stays_last() -> None:
    assert router.FALLBACK_ORDER.index("glm") < router.FALLBACK_ORDER.index("anthropic")


def test_glm_is_skipped_when_it_has_no_key(keys) -> None:
    keys(groq_api_key="gsk_test")
    assert "glm" not in configured_chain()


def test_glm_joins_the_chain_once_a_key_exists(keys) -> None:
    keys(groq_api_key="gsk_test", glm_api_key="zai_test")
    assert configured_chain() == ["groq", "glm"]


def test_glm_is_built_through_the_openai_protocol(keys) -> None:
    """z.ai is openai-compatible, so it needs the base_url override to reach the right host."""
    settings = keys(glm_api_key="zai_test")
    llm = router._build_glm(settings, "planning")
    assert llm.model_name == settings.glm_model
    assert "z.ai" in str(llm.openai_api_base)


def test_a_groq_tool_call_400_falls_back_instead_of_failing_the_step() -> None:
    """Found live: gpt-oss emits a tool call from the react prompt's tool-shaped names and
    groq 400s it. json mode does not prevent it, so the chain must treat it as hoppable."""
    error = Exception(
        "Error code: 400 - {'error': {'message': 'Tool choice is none, but model called a "
        "tool', 'code': 'tool_use_failed'}}"
    )
    assert router.is_retryable(error)


def test_a_plain_400_is_still_not_retryable() -> None:
    """A bad key or bad model id fails identically everywhere — hopping would just waste quota."""
    assert not router.is_retryable(Exception("Error code: 400 - invalid model id"))

"""Retry and JSON-parsing behaviour shared by every agent."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.agents import base_agent as base_module
from amaris.agents.base_agent import AgentError, BaseAgent
from amaris.llm.router import ProvidersCoolingDown
from amaris.observability.context import get_agent


class Probe(BaseAgent):
    """Minimal concrete agent — records the agent name visible to logs during the run."""

    name = "probe"

    async def _run(self, state: Any) -> dict[str, Any]:
        return {"seen_agent": get_agent()}


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


@pytest.fixture
def probe() -> Probe:
    return Probe()


async def test_run_binds_the_agent_name_for_logging(probe: Probe, state) -> None:
    """Every log line inside an agent must carry which agent produced it."""
    assert (await probe.run(state))["seen_agent"] == "probe"
    assert get_agent() == "-"


async def test_retries_an_empty_response(monkeypatch: pytest.MonkeyPatch, probe: Probe) -> None:
    """groq returns empty content on a first call sometimes — the retry is load-bearing."""
    replies = ["", "  ", "finally real"]

    async def flaky(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        return FakeResponse(replies.pop(0))

    monkeypatch.setattr(base_module, "invoke_with_fallback", flaky)
    monkeypatch.setattr(base_module.asyncio, "sleep", _no_sleep)

    assert await probe._invoke("prompt") == "finally real"


async def test_gives_up_after_the_attempt_cap(
    monkeypatch: pytest.MonkeyPatch, probe: Probe
) -> None:
    calls = 0

    async def always_empty(
        prompt: Any, task_type: str = "reasoning", **kwargs: Any
    ) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse("")

    monkeypatch.setattr(base_module, "invoke_with_fallback", always_empty)
    monkeypatch.setattr(base_module.asyncio, "sleep", _no_sleep)

    with pytest.raises(AgentError, match="no usable response"):
        await probe._invoke("prompt")
    assert calls == base_module.MAX_ATTEMPTS


async def test_retries_a_rate_limit_but_not_a_bug(
    monkeypatch: pytest.MonkeyPatch, probe: Probe
) -> None:
    """A malformed request fails identically on a retry, so retrying just wastes quota."""
    calls = 0

    async def bad_request(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise ValueError("invalid prompt template")

    monkeypatch.setattr(base_module, "invoke_with_fallback", bad_request)
    monkeypatch.setattr(base_module.asyncio, "sleep", _no_sleep)

    with pytest.raises(ValueError, match="invalid prompt"):
        await probe._invoke("prompt")
    assert calls == 1


async def test_rate_limits_are_retried(monkeypatch: pytest.MonkeyPatch, probe: Probe) -> None:
    attempts = 0

    async def throttled(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Error code: 429 rate limit reached")
        return FakeResponse("ok")

    monkeypatch.setattr(base_module, "invoke_with_fallback", throttled)
    monkeypatch.setattr(base_module.asyncio, "sleep", _no_sleep)

    assert await probe._invoke("prompt") == "ok"
    assert attempts == 3


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure, here you go:\n{"a": 1}\nLet me know!',
    ],
)
def test_json_is_extracted_from_every_wrapper_models_use(probe: Probe, raw: str) -> None:
    assert probe._parse_json(raw) == {"a": 1}


@pytest.mark.parametrize("raw", ["no json here", "", "{broken", "[1, 2"])
def test_unparsable_json_raises_agent_error(probe: Probe, raw: str) -> None:
    """Node wrappers catch AgentError and turn it into state["error"], so this must not be a crash."""
    with pytest.raises(AgentError):
        probe._parse_json(raw)


async def test_retry_waits_until_the_chain_is_free(
    monkeypatch: pytest.MonkeyPatch, probe: Probe
) -> None:
    """0.5s of backoff against a 60s tokens-per-minute window is a guaranteed second failure."""
    slept: list[float] = []
    replies: list[Any] = [RuntimeError("429 rate limit"), FakeResponse("ok")]

    async def flaky(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        nxt = replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(base_module, "invoke_with_fallback", flaky)
    monkeypatch.setattr(base_module, "chain_wait_seconds", lambda: 42.0)
    monkeypatch.setattr(base_module.asyncio, "sleep", _record(slept))

    assert await probe._invoke("prompt") == "ok"
    assert slept == [42.0]


async def test_a_transient_failure_retries_promptly_even_behind_a_long_hint(
    monkeypatch: pytest.MonkeyPatch, probe: Probe
) -> None:
    """Found live: groq blipped, the error surfaced was gemini's daily quota, and one attempt ended the run."""
    slept: list[float] = []
    gemini_daily = RuntimeError("429 RESOURCE_EXHAUSTED ... retryDelay: '3600s'")
    replies: list[Any] = [gemini_daily, FakeResponse("ok")]

    async def flaky(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        nxt = replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(base_module, "invoke_with_fallback", flaky)
    # groq is healthy and unparked, so the failing provider's hint is not our wait
    monkeypatch.setattr(base_module, "chain_wait_seconds", lambda: 0.0)
    monkeypatch.setattr(base_module.asyncio, "sleep", _record(slept))

    assert await probe._invoke("prompt") == "ok"
    assert slept == [base_module.BACKOFF_BASE_SECONDS]


async def test_a_wait_longer_than_the_budget_fails_fast(
    monkeypatch: pytest.MonkeyPatch, probe: Probe
) -> None:
    """A daily quota cannot be waited out, so the node should write a partial report now."""
    slept: list[float] = []

    async def exhausted(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        raise RuntimeError("429 quota exceeded")

    monkeypatch.setattr(base_module, "invoke_with_fallback", exhausted)
    monkeypatch.setattr(base_module, "chain_wait_seconds", lambda: 3600.0)
    monkeypatch.setattr(base_module.asyncio, "sleep", _record(slept))

    with pytest.raises(AgentError, match="retry budget"):
        await probe._invoke("prompt")
    assert slept == []


async def test_retry_waits_stay_inside_the_budget(
    monkeypatch: pytest.MonkeyPatch, probe: Probe
) -> None:
    """Three 60s waits would stall a run far past the 90s target for no extra chance of success."""
    slept: list[float] = []

    async def limited(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(base_module, "invoke_with_fallback", limited)
    monkeypatch.setattr(base_module, "chain_wait_seconds", lambda: 60.0)
    monkeypatch.setattr(base_module.asyncio, "sleep", _record(slept))

    with pytest.raises(AgentError):
        await probe._invoke("prompt")
    assert sum(slept) <= probe.settings.llm_retry_budget_seconds


async def test_a_parked_chain_does_not_burn_an_attempt(
    monkeypatch: pytest.MonkeyPatch, probe: Probe
) -> None:
    """Found live: the run died 0.4s short of a free provider because the wait spent attempt 3."""
    slept: list[float] = []
    replies: list[Any] = [
        RuntimeError("429 rate limit"),
        ProvidersCoolingDown(0.4),
        ProvidersCoolingDown(0.4),
        FakeResponse("ok"),
    ]

    async def flaky(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        nxt = replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(base_module, "invoke_with_fallback", flaky)
    monkeypatch.setattr(base_module, "chain_wait_seconds", lambda: 0.4)
    monkeypatch.setattr(base_module.asyncio, "sleep", _record(slept))

    # four trips through the loop but only two providers were ever called
    assert await probe._invoke("prompt") == "ok"
    assert len(slept) == 3


async def test_a_near_zero_chain_wait_cannot_spin_the_loop(
    monkeypatch: pytest.MonkeyPatch, probe: Probe
) -> None:
    """A cooldown a microsecond from expiring must still advance the waiting budget."""
    slept: list[float] = []

    async def parked(prompt: Any, task_type: str = "reasoning", **kwargs: Any) -> FakeResponse:
        raise ProvidersCoolingDown(1e-9)

    monkeypatch.setattr(base_module, "invoke_with_fallback", parked)
    monkeypatch.setattr(base_module, "chain_wait_seconds", lambda: 1e-9)
    monkeypatch.setattr(base_module.asyncio, "sleep", _record(slept))

    with pytest.raises(AgentError):
        await probe._invoke("prompt")
    assert all(delay >= base_module.MIN_RETRY_SLEEP_SECONDS for delay in slept)
    assert sum(slept) <= probe.settings.llm_retry_budget_seconds


def _record(into: list[float]):
    """Capture the requested backoff instead of actually waiting it out."""

    async def _sleep(seconds: float) -> None:
        into.append(seconds)

    return _sleep


async def _no_sleep(seconds: float) -> None:
    """Backoff is real in production but would just make the suite slow."""
    return None

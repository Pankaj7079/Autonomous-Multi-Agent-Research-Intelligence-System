"""Picks the model for a task and walks the provider chain when one rate-limits."""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

    from amaris.config.settings import Settings

# which model size a task needs: the researcher makes many short calls, the rest reason
_TASK_TIER: dict[str, str] = {
    # routing and triage are short structured decisions, so the 120b model buys nothing
    # but latency — it was ~4s per hop for a one-word answer
    "supervisor": "fast",
    "triage": "fast",
    "planning": "reasoning",
    "analysis": "reasoning",
    "writing": "reasoning",
    "critique": "reasoning",
    "evaluation": "reasoning",
    "research": "fast",
    "react": "fast",
    "memory": "fast",
}

# routing must be stable across runs, prose can breathe
_TASK_TEMPERATURE: dict[str, float] = {
    "supervisor": 0.0,
    "triage": 0.0,
    "critique": 0.1,
    "research": 0.2,
    "react": 0.2,
    "planning": 0.3,
    "analysis": 0.3,
    "writing": 0.5,
}

_DEFAULT_TEMPERATURE = 0.3
_REQUEST_TIMEOUT = 60
# glm measured ~17s on a short prompt, so judge-sized prompts blow the shared 60s budget.
# agent work gets a shorter ceiling: a 180s hang mid-run looks identical to a crash, and
# there is nothing useful to wait for when the reply is that late.
_GLM_REQUEST_TIMEOUT = 90
_GLM_EVAL_REQUEST_TIMEOUT = 180

# free-tier providers first, anthropic last since it needs a paid key
FALLBACK_ORDER = ("groq", "gemini", "glm", "anthropic")

_KEY_FIELDS = {
    "groq": "groq_api_key",
    "gemini": "gemini_api_key",
    "glm": "glm_api_key",
    "anthropic": "anthropic_api_key",
}

# a 429 from any provider looks different, so match on text rather than exception type
_RETRYABLE_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota",
    "resource_exhausted",
    "resource exhausted",
    "timeout",
    "timed out",
    "connection",
    "503",
    "overloaded",
    # gpt-oss emits a native tool call from a tool-shaped name in the react prompt and groq
    # 400s it. json mode does not actually prevent it, so hop instead of losing the step.
    "tool_use_failed",
    # same shape: gpt-oss returns an empty generation and groq rejects it as invalid json
    "json_validate_failed",
)


class LLMConfigError(RuntimeError):
    """No usable provider key. Raised loudly — a silent no-LLM run is worse."""


class ProvidersCoolingDown(RuntimeError):
    """Every provider is inside a rate limit window. Carries the seconds until one frees up."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"all providers hit a rate limit, retry after {seconds:.1f}s")
        self.seconds = seconds


# providers word it differently: groq "try again in 1m12.5s", gemini "Please retry in 34.9s"
_MESSAGE_DELAY = re.compile(r"(?:try again|retry) in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)
# and again in structured form: "retryDelay: '34s'", "retry_delay { seconds: 34 }", "retry-after: 60"
_FIELD_DELAY = re.compile(r"retry[-_ ]?(?:delay|after)\D{0,15}?(\d+(?:\.\d+)?)", re.IGNORECASE)

# a per-day quota will not clear inside a run, unlike a per-minute one
_DAILY_QUOTA_MARKERS = ("perday", "per day", "requests per day")
# errors that mean "you are over a limit" rather than "the network hiccuped"
_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota",
    "resource_exhausted",
    "resource exhausted",
)
# not every provider states a delay — glm 429s with code 1302 and no hint at all, and
# treating that as unparked burned all three attempts in 2s of a 75s budget
_UNHINTED_RATE_LIMIT_COOLDOWN = 30.0
_DAILY_COOLDOWN_SECONDS = 3600.0
# sleeping the exact hint can land a hair early and 429 again
_COOLDOWN_PAD_SECONDS = 1.0

_cooldown_until: dict[str, float] = {}


def retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait, read from its own error text. None if it didn't say."""
    text = str(exc)
    stated = _MESSAGE_DELAY.search(text)
    if stated:
        return float(stated.group(1) or 0) * 60 + float(stated.group(2))
    field = _FIELD_DELAY.search(text)
    return float(field.group(1)) if field else None


def _park(provider: str, exc: BaseException) -> float | None:
    """Stop asking a rate-limited provider until it said to come back. Returns seconds parked."""
    text = str(exc).lower()
    if any(marker in text for marker in _DAILY_QUOTA_MARKERS):
        seconds = _DAILY_COOLDOWN_SECONDS
    else:
        hinted = retry_after_seconds(exc)
        if hinted is not None:
            seconds = hinted + _COOLDOWN_PAD_SECONDS
        elif any(marker in text for marker in _RATE_LIMIT_MARKERS):
            # it said we are over a limit but not for how long, so assume a per-minute window
            seconds = _UNHINTED_RATE_LIMIT_COOLDOWN
        else:
            return None  # a connection blip is not a quota, don't sideline a healthy provider
    _cooldown_until[provider] = time.monotonic() + seconds
    return seconds


def _available(chain: list[str]) -> list[str]:
    now = time.monotonic()
    return [name for name in chain if _cooldown_until.get(name, 0.0) <= now]


def cooldown_remaining() -> float:
    """Seconds until the first parked provider frees up. 0.0 when none are parked."""
    if not _cooldown_until:
        return 0.0
    return max(0.0, min(_cooldown_until.values()) - time.monotonic())


def chain_wait_seconds() -> float:
    """Seconds until any configured provider can be called. 0.0 when one is free right now."""
    chain = configured_chain()
    if not chain or _available(chain):
        return 0.0
    return max(0.0, min(_cooldown_until[name] for name in chain) - time.monotonic())


def provider_status() -> dict[str, float]:
    """Configured provider -> seconds until it is usable. 0.0 means ready right now."""
    now = time.monotonic()
    return {name: max(0.0, _cooldown_until.get(name, 0.0) - now) for name in configured_chain()}


def reset_cooldowns() -> None:
    """Forget every parked provider. Used by tests and on a fresh process."""
    _cooldown_until.clear()


def _tier(task_type: str) -> str:
    return _TASK_TIER.get(task_type, "reasoning")


def _temperature(task_type: str) -> float:
    return _TASK_TEMPERATURE.get(task_type, _DEFAULT_TEMPERATURE)


def _build_groq(settings: Settings, task_type: str) -> BaseChatModel:
    from langchain_groq import ChatGroq

    model = (
        settings.groq_model_reasoning
        if _tier(task_type) == "reasoning"
        else settings.groq_model_fast
    )
    return ChatGroq(
        model=model,
        api_key=settings.key("groq_api_key"),
        temperature=_temperature(task_type),
        timeout=_REQUEST_TIMEOUT,
        max_retries=0,  # the chain handles retries, don't double up
    )


def _build_gemini(settings: Settings, task_type: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model_fallback,
        google_api_key=settings.key("gemini_api_key"),
        temperature=_temperature(task_type),
        timeout=_REQUEST_TIMEOUT,
        max_retries=0,
    )


def _build_glm(settings: Settings, task_type: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    # langchain-openai, not langchain-community's ChatZhipuAI — that package sunset a
    # per-provider class on us once already (ADR-018) and langchain-zhipuai is at 0.0.1
    return ChatOpenAI(
        model=settings.glm_model,
        api_key=settings.key("glm_api_key"),
        base_url=settings.glm_base_url,
        temperature=_temperature(task_type),
        timeout=(_GLM_EVAL_REQUEST_TIMEOUT if task_type == "evaluation" else _GLM_REQUEST_TIMEOUT),
        max_retries=0,
    )


def _build_anthropic(settings: Settings, task_type: str) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # single model regardless of tier — this fallback rarely runs, so it isn't worth tiering
    return ChatAnthropic(
        model=settings.anthropic_model,
        anthropic_api_key=settings.key("anthropic_api_key"),
        temperature=_temperature(task_type),
        default_request_timeout=_REQUEST_TIMEOUT,
        max_retries=0,
    )


_BUILDERS = {
    "groq": _build_groq,
    "gemini": _build_gemini,
    "glm": _build_glm,
    "anthropic": _build_anthropic,
}

# json mode asks for plain json text. it reduces but does not eliminate gpt-oss emitting a
# native tool call, so tool_use_failed is also in _RETRYABLE_MARKERS as the real backstop.
_JSON_MODE_BINDINGS: dict[str, dict[str, Any]] = {
    "groq": {"response_format": {"type": "json_object"}},
    # z.ai speaks the openai protocol and accepts json_object — verified against the live api
    "glm": {"response_format": {"type": "json_object"}},
}


def configured_chain() -> list[str]:
    """Providers with a key present, in fallback order. Empty means nothing is configured."""
    settings = get_settings()
    chain = [name for name in FALLBACK_ORDER if settings.key(_KEY_FIELDS[name])]
    # PRIMARY_PROVIDER only promotes; the rest keep their documented order behind it
    lead = settings.primary_provider.strip().lower()
    if lead in chain and chain[0] != lead:
        chain.remove(lead)
        chain.insert(0, lead)
    return chain


@lru_cache(maxsize=32)
def _cached_llm(provider: str, task_type: str) -> BaseChatModel:
    return _BUILDERS[provider](get_settings(), task_type)


def get_llm(task_type: str = "reasoning", provider: str | None = None) -> BaseChatModel:
    """Model for this task. Groq unless it has no key, then the next configured provider."""
    chain = configured_chain()
    if not chain:
        raise LLMConfigError(
            "No LLM provider key found. Set GROQ_API_KEY in .env "
            "(free at console.groq.com), or GEMINI_API_KEY / ANTHROPIC_API_KEY."
        )

    chosen = provider or chain[0]
    if chosen not in chain:
        raise LLMConfigError(f"{chosen} has no API key configured — available: {', '.join(chain)}")

    if chosen != "groq":
        logger.bind(provider=chosen, task=task_type).debug("llm.primary_not_groq")
    return _cached_llm(chosen, task_type)


def get_fallback_llm(task_type: str = "reasoning") -> BaseChatModel | None:
    """Last provider in the chain, or None when only one is configured."""
    chain = configured_chain()
    return _cached_llm(chain[-1], task_type) if len(chain) > 1 else None


def is_rate_limited(exc: BaseException) -> bool:
    """True when a provider refused because we are over a limit, not because of a network blip."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def is_retryable(exc: BaseException) -> bool:
    """True for rate limits, timeouts and transient network errors — worth another provider."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


async def invoke_with_fallback(
    messages: str | list[BaseMessage],
    task_type: str = "reasoning",
    json_mode: bool = False,
    **kwargs: Any,
) -> BaseMessage:
    """Invoke down the provider chain, hopping on rate limits. Raises if all providers fail."""
    chain = configured_chain()
    if not chain:
        raise LLMConfigError(
            "No LLM provider key found. Set GROQ_API_KEY in .env (free at console.groq.com)."
        )

    # a provider that just said "try again in 58s" will say it again, so skip it until then
    usable = _available(chain)
    if not usable:
        raise ProvidersCoolingDown(cooldown_remaining())

    for position, provider in enumerate(usable):
        llm = _cached_llm(provider, task_type)
        if json_mode and provider in _JSON_MODE_BINDINGS:
            llm = llm.bind(**_JSON_MODE_BINDINGS[provider])
        started = time.perf_counter()
        try:
            response = await llm.ainvoke(messages, **kwargs)
        except Exception as exc:
            retryable = is_retryable(exc)
            parked = _park(provider, exc) if retryable else None
            if position == len(usable) - 1 or not retryable:
                logger.bind(provider=provider, task=task_type, error=str(exc)[:200]).error(
                    "llm.failed"
                )
                raise
            logger.bind(
                provider=provider,
                to=usable[position + 1],
                reason=type(exc).__name__,
                # the class name alone hides why a provider was dropped, and only the last
                # provider in the chain ever reaches llm.failed where the text is logged
                error=str(exc)[:200],
                parked_for=parked,
                task=task_type,
            ).warning("llm.fallback")
            continue

        logger.bind(
            provider=provider,
            model=getattr(llm, "model_name", None) or getattr(llm, "model", "?"),
            task=task_type,
            chars=len(response.text),  # gemini 3.x content is a block list, not a string
            ms=round((time.perf_counter() - started) * 1000, 1),
        ).debug("llm.call")
        return response

    raise LLMConfigError("provider chain exhausted without a response")  # unreachable in practice

"""Picks the model for a task and walks the provider chain when one rate-limits."""

from __future__ import annotations

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
    "supervisor": "reasoning",
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
    "critique": 0.1,
    "research": 0.2,
    "react": 0.2,
    "planning": 0.3,
    "analysis": 0.3,
    "writing": 0.5,
}

_DEFAULT_TEMPERATURE = 0.3
_REQUEST_TIMEOUT = 60

# free-tier providers first, anthropic last since it needs a paid key
FALLBACK_ORDER = ("groq", "gemini", "anthropic")

_KEY_FIELDS = {
    "groq": "groq_api_key",
    "gemini": "gemini_api_key",
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
)


class LLMConfigError(RuntimeError):
    """No usable provider key. Raised loudly — a silent no-LLM run is worse."""


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


_BUILDERS = {"groq": _build_groq, "gemini": _build_gemini, "anthropic": _build_anthropic}


def configured_chain() -> list[str]:
    """Providers with a key present, in fallback order. Empty means nothing is configured."""
    settings = get_settings()
    return [name for name in FALLBACK_ORDER if settings.key(_KEY_FIELDS[name])]


@lru_cache(maxsize=32)
def _cached_llm(provider: str, task_type: str) -> BaseChatModel:
    return _BUILDERS[provider](get_settings(), task_type)


def get_llm(task_type: str = "reasoning", provider: str | None = None) -> BaseChatModel:
    """Model for this task. Groq unless it has no key, then the next configured provider."""
    chain = configured_chain()
    if not chain:
        raise LLMConfigError(
            "No LLM provider key found. Set GROQ_API_KEY in .env "
            "(free at console.groq.com), or GEMINI_API_KEY / CEREBRAS_API_KEY."
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


def is_retryable(exc: BaseException) -> bool:
    """True for rate limits, timeouts and transient network errors — worth another provider."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


async def invoke_with_fallback(
    messages: str | list[BaseMessage],
    task_type: str = "reasoning",
    **kwargs: Any,
) -> BaseMessage:
    """Invoke down the provider chain, hopping on rate limits. Raises if all providers fail."""
    chain = configured_chain()
    if not chain:
        raise LLMConfigError(
            "No LLM provider key found. Set GROQ_API_KEY in .env (free at console.groq.com)."
        )

    for position, provider in enumerate(chain):
        llm = _cached_llm(provider, task_type)
        started = time.perf_counter()
        try:
            response = await llm.ainvoke(messages, **kwargs)
        except Exception as exc:
            is_last = position == len(chain) - 1
            if is_last or not is_retryable(exc):
                logger.bind(provider=provider, task=task_type, error=str(exc)[:200]).error(
                    "llm.failed"
                )
                raise
            logger.bind(
                provider=provider,
                to=chain[position + 1],
                reason=type(exc).__name__,
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

"""RAGAS plumbing shared by retrieval_eval.py (Layer 1) and report_eval.py (Layer 2).

Not a layer itself — no Evaluator here, just the judge/embedder/shim both layers need so the
sunset-import workaround and the Groq quirks are fixed in exactly one place (ADR-017).
"""

from __future__ import annotations

import asyncio
import sys
import types
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

# every context costs the judge another llm call, which is what exhausts groq's per-minute limit
MAX_CONTEXTS = 4
CONTEXT_CHAR_LIMIT = 800

# ragas asks the judge for 3 question variants; groq returns 1 generation and ragas then crashes
RELEVANCY_STRICTNESS = 1
# groq has no embeddings endpoint, and gemini's embed quota is separate from its 20 prompts/day
_EMBED_MODEL = "models/gemini-embedding-001"

_unavailable_reason: str | None = None


def unavailable_reason() -> str | None:
    """Why ragas is off, for the health endpoint in Phase 7."""
    return _unavailable_reason


def _shim_sunset_vertexai() -> None:
    """ragas 0.4 imports a chat model that langchain-community deleted when it was sunset."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    shim = types.ModuleType(name)
    # a throwaway class, never object: ragas isinstance-checks this and everything matches object
    shim.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[name] = shim


# installed at import time, not lazily inside judge(): retrieval_eval.py and report_eval.py both
# import `ragas.metrics` directly (to build metric objects), and that import alone triggers the
# broken chain — waiting for judge() to be called first would leave that path unprotected
_shim_sunset_vertexai()


@contextmanager
def quiet_deprecations() -> Iterator[None]:
    """ragas v1 moves these to an instructor-based llm we cannot route through; warnings until then."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def clamp_or_none(value: float) -> float | None:
    """Clamp to 0-1, or None if ragas couldn't actually score it.

    ragas returns NaN, not an exception, when a metric's internal judge calls fail (a rate limit
    mid-computation is the common case) — found live, where a fast-failing judge under load
    produced NaN on every metric in one run. `min(1.0, max(0.0, nan))` silently evaluates to 0.0
    in Python, which would misreport "the judge couldn't score this" as "confirmed hallucination".
    Callers must treat None as "no result", never as a real 0.0.
    """
    if value != value:
        return None
    return round(min(1.0, max(0.0, value)), 4)


def trim_contexts(contexts: list[str]) -> list[str]:
    """Cap count and length — the judge re-reads every context on every metric."""
    trimmed = [text.strip()[:CONTEXT_CHAR_LIMIT] for text in contexts if text and text.strip()]
    return trimmed[:MAX_CONTEXTS]


@lru_cache(maxsize=1)
def judge() -> Any | None:
    """The Groq judge ragas/deepeval-style metrics call, or None if no provider is configured."""
    global _unavailable_reason
    _shim_sunset_vertexai()
    try:
        with quiet_deprecations():
            from ragas.llms import LangchainLLMWrapper

            from amaris.llm.router import configured_chain, get_fallback_llm, get_llm

            # the evaluator runs last, after the pipeline has spent the primary's rate window,
            # so scoring must not go to the primary or it NaNs on a rate limit every run
            preferred = get_settings().eval_provider.strip().lower()
            if preferred in configured_chain():
                return LangchainLLMWrapper(get_llm("evaluation", provider=preferred))
            return LangchainLLMWrapper(get_fallback_llm("evaluation") or get_llm("evaluation"))
    except Exception as exc:
        _unavailable_reason = f"{type(exc).__name__}: {exc}"[:200]
        logger.bind(reason=_unavailable_reason).warning("ragas.unavailable")
        return None


@lru_cache(maxsize=1)
def embedder() -> Any | None:
    """Gemini embeddings for relevancy-style metrics, or None so that metric is simply skipped."""
    settings = get_settings()
    if not settings.key("gemini_api_key"):
        return None
    try:
        with quiet_deprecations():
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper

            return LangchainEmbeddingsWrapper(
                GoogleGenerativeAIEmbeddings(
                    model=_EMBED_MODEL, google_api_key=settings.key("gemini_api_key")
                )
            )
    except Exception as exc:
        logger.bind(error=str(exc)[:150]).warning("ragas.no_embedder")
        return None


def is_available() -> bool:
    """True when a judge can actually be built — both layers gate on this."""
    return judge() is not None


# ragas's own RunConfig defaults to max_retries=10, max_wait=60, timeout=180 — a retry loop
# with zero awareness of our router's cooldown parking (ADR-002). Found live: Layer 1 ran right
# after a groq rate limit and spent the full 120s of our own timeout inside ragas's internal
# backoff, never reaching our asyncio.wait_for at all. A judge call is one metric on one sample —
# it should fail in a few seconds, not retry for two minutes.
_RAGAS_MAX_RETRIES = 1
_RAGAS_MAX_WAIT_SECONDS = 15
# ragas defaults to 16 parallel judge calls, which both times out and 429s a slow free-tier
# provider. the evaluator is not latency critical, so it goes narrow instead of wide.
_RAGAS_MAX_WORKERS = 2


async def run_metrics(
    metrics: list[Any],
    query: str,
    answer: str,
    contexts: list[str],
    reference: str | None = None,
    label: str = "ragas",
) -> dict[str, float] | None:
    """Run ragas metrics against one sample, off the event loop, bounded by ragas_timeout_seconds.

    Returns None on timeout or failure so callers log their own zero-result per metric. `label`
    only affects logging — it says which layer's call this was, since two layers share this call.
    """
    with quiet_deprecations():
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.run_config import RunConfig

    sample_kwargs: dict[str, Any] = {
        "user_input": query,
        "response": answer,
        "retrieved_contexts": contexts,
    }
    if reference is not None:
        sample_kwargs["reference"] = reference

    timeout = get_settings().ragas_timeout_seconds
    run_config = RunConfig(
        timeout=int(timeout),
        max_retries=_RAGAS_MAX_RETRIES,
        max_wait=_RAGAS_MAX_WAIT_SECONDS,
        max_workers=_RAGAS_MAX_WORKERS,
    )

    def _blocking() -> dict[str, float]:
        dataset = EvaluationDataset(samples=[SingleTurnSample(**sample_kwargs)])
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge(),
            embeddings=embedder(),
            run_config=run_config,
            show_progress=False,
        )
        return {key: float(value) for key, value in result.scores[0].items()}

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, _blocking), timeout=timeout)
    except TimeoutError:
        logger.bind(label=label, timeout=timeout).warning("ragas.timed_out")
        return None
    except Exception as exc:
        logger.bind(label=label, error=str(exc)[:200]).warning("ragas.failed")
        return None

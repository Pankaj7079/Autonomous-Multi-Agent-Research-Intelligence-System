"""LLM tracing. Loguru says what the graph did; a tracer says what the models were asked.

The JSONL log already reconstructs a run's agent path from a session_id (docs/OBSERVABILITY.md).
What it cannot show is the prompt, the completion and the token cost of each call, which is what
a tracer adds — and what an interviewer asks to see.

Both backends are optional and neither is required for a run. Keys were configured long before
anything read them, so a key in .env bought nothing until this module existed (ADR-043).
"""

from __future__ import annotations

import os
from typing import Any

from amaris.observability.logging import logger

# what a trace is filed under in either backend
DEFAULT_PROJECT = "amaris"

_handler: Any = None
_backend = ""


def configure_tracing() -> str:
    """Turn on whichever tracer is configured and return its name, or "" for none.

    LangSmith needs no callback: LangChain reads these variables itself, so every model call
    in the graph is traced once they are set. Langfuse needs a callback handler, which
    `run_config` attaches to the graph invocation.
    """
    global _handler, _backend
    from amaris.config.settings import get_settings

    settings = get_settings()
    _handler, _backend = None, ""

    langsmith = settings.key("langsmith_api_key")
    if langsmith:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = langsmith
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project or DEFAULT_PROJECT
        _backend = "langsmith"
        logger.bind(project=os.environ["LANGSMITH_PROJECT"]).info("tracing.enabled")
        return _backend

    public = settings.key("langfuse_public_key")
    secret = settings.key("langfuse_secret_key")
    if not (public and secret):
        logger.debug("tracing.disabled")
        return ""

    os.environ["LANGFUSE_PUBLIC_KEY"] = public
    os.environ["LANGFUSE_SECRET_KEY"] = secret
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    try:
        from langfuse.langchain import CallbackHandler

        _handler = CallbackHandler()
    except Exception as exc:
        # a tracer that cannot start must never stop a run — same rule as every other extra
        logger.bind(error=str(exc)[:200]).warning("tracing.langfuse_failed")
        return ""

    _backend = "langfuse"
    logger.bind(host=settings.langfuse_host).info("tracing.enabled")
    return _backend


def callbacks() -> list[Any]:
    """Callbacks to attach to a graph run. Empty for LangSmith, which needs none."""
    return [_handler] if _handler is not None else []


def active_backend() -> str:
    """Which tracer is on: "langsmith", "langfuse", or "" — shown in the UI system panel."""
    return _backend

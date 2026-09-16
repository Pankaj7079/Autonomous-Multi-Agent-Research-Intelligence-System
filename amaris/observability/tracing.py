"""LLM tracing. Loguru says what the graph did; LangSmith says what the models were asked.

The JSONL log already reconstructs a run's agent path from a session_id (docs/OBSERVABILITY.md).
What it cannot show is the prompt, the completion and the token cost of each call, which is what
a tracer adds — and what an interviewer asks to see.

Tracing is optional and is never required for a run. LangSmith is the only backend: it needs no
callback and no self-hosted server, so there is nothing to keep working when it is switched off
(ADR-046).
"""

from __future__ import annotations

import os

from amaris.observability.logging import logger

# what a trace is filed under
DEFAULT_PROJECT = "amaris"

_backend = ""


def configure_tracing() -> str:
    """Turn LangSmith on when a key is configured and return its name, or "" for none.

    No callback is attached anywhere: langchain reads these three variables itself, so every
    model call in the graph is traced once they are set.
    """
    global _backend
    from amaris.config.settings import get_settings

    settings = get_settings()
    _backend = ""

    key = settings.key("langsmith_api_key")
    if not key:
        logger.debug("tracing.disabled")
        return ""

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project or DEFAULT_PROJECT
    _backend = "langsmith"
    logger.bind(project=os.environ["LANGSMITH_PROJECT"]).info("tracing.enabled")
    return _backend


def active_backend() -> str:
    """ "langsmith" when tracing is on, "" when it is not — shown in the UI system panel."""
    return _backend

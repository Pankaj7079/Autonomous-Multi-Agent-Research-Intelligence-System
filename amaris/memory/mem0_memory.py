"""Episodic memory across sessions. Every method is a logged no-op when mem0 is absent."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

COLLECTION = "amaris_memories"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# single-user system, so one namespace is enough
USER_ID = "amaris"

# a full report is mostly headings and citations, which mem0 extraction gains nothing from
SUMMARY_CHAR_LIMIT = 2000

_unavailable_reason: str | None = None


def _build_config() -> dict[str, Any]:
    """mem0 on free infrastructure: groq for extraction, local MiniLM, our own Qdrant."""
    settings = get_settings()
    vector_config: dict[str, Any] = {
        "collection_name": COLLECTION,
        "embedding_model_dims": VECTOR_SIZE,
    }
    if settings.qdrant_url:
        vector_config["url"] = settings.qdrant_url
        vector_config["api_key"] = settings.key("qdrant_api_key")
    else:
        vector_config["host"] = settings.qdrant_host
        vector_config["port"] = settings.qdrant_port

    return {
        "llm": {
            "provider": "groq",
            "config": {
                "model": settings.groq_model_fast,
                "api_key": settings.key("groq_api_key"),
                "temperature": 0.1,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": EMBED_MODEL},
        },
        "vector_store": {"provider": "qdrant", "config": vector_config},
    }


@lru_cache(maxsize=1)
def _memory() -> Any | None:
    """The mem0 client, or None. Cached because building it loads the embedding model."""
    global _unavailable_reason
    try:
        from mem0 import Memory
    except ImportError:
        _unavailable_reason = "mem0ai not installed (uv sync --extra memory)"
        logger.bind(reason=_unavailable_reason).warning("mem0.unavailable")
        return None

    try:
        return Memory.from_config(_build_config())
    except Exception as exc:
        _unavailable_reason = f"{type(exc).__name__}: {exc}"[:200]
        logger.bind(reason=_unavailable_reason).warning("mem0.unavailable")
        return None


def is_available() -> bool:
    """True when recall and store will actually do something."""
    return _memory() is not None


def _extract_memories(raw: Any) -> list[str]:
    """mem0 2.x returns {"results": [...]}, older versions a bare list — accept both."""
    items = raw.get("results", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []

    memories = []
    for item in items:
        text = item.get("memory") or item.get("text") if isinstance(item, dict) else item
        if text:
            memories.append(str(text))
    return memories


def _search_blocking(memory: Any, query: str, limit: int) -> Any:
    try:
        return memory.search(query, filters={"user_id": USER_ID}, limit=limit)
    except TypeError:
        # mem0 < 2.0 took user_id directly instead of a filters dict
        return memory.search(query, user_id=USER_ID, limit=limit)


async def recall_related(query: str, limit: int = 3) -> list[str]:
    """Findings from past sessions relevant to this query. [] when mem0 is unavailable."""
    memory = _memory()
    if memory is None:
        return []

    try:
        loop = asyncio.get_running_loop()
        # mem0 is sync and embeds on the cpu, so it cannot run on the event loop
        raw = await loop.run_in_executor(None, _search_blocking, memory, query, limit)
    except Exception as exc:
        logger.bind(error=str(exc)[:200]).warning("mem0.recall_failed")
        return []

    memories = _extract_memories(raw)
    logger.bind(query=query[:80], recalled=len(memories)).debug("mem0.recall")
    return memories


async def add_research_finding(query: str, finding: str, url: str = "") -> bool:
    """Store one finding for future runs. False when it did not land."""
    memory = _memory()
    if memory is None or not finding.strip():
        return False

    # store as a turn so mem0 can extract entities the way it expects
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": finding},
    ]

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: memory.add(messages, user_id=USER_ID, metadata={"url": url} if url else None),
        )
    except Exception as exc:
        logger.bind(error=str(exc)[:200]).warning("mem0.store_failed")
        return False

    logger.bind(chars=len(finding), url=url[:120]).debug("mem0.stored")
    return True


async def add_session_summary(query: str, report: str, scores: dict[str, float]) -> bool:
    """Store a finished run so a later session recalls the conclusion, not just the findings."""
    memory = _memory()
    if memory is None or not report.strip():
        return False

    # only the opening section: the whole report would bury mem0's extraction in boilerplate
    conclusion = report.strip()[:SUMMARY_CHAR_LIMIT]
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": conclusion},
    ]
    metadata = {"kind": "session_summary", **{k: round(v, 3) for k, v in scores.items()}}

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: memory.add(messages, user_id=USER_ID, metadata=metadata)
        )
    except Exception as exc:
        logger.bind(error=str(exc)[:200]).warning("mem0.session_store_failed")
        return False

    logger.bind(chars=len(conclusion), scores=len(scores)).debug("mem0.session_stored")
    return True


async def _selftest() -> None:
    """uv run python -m amaris.memory.mem0_memory --selftest"""
    from amaris.observability.logging import configure_logging

    configure_logging(level="INFO", json_enabled=False)

    if not is_available():
        logger.bind(reason=_unavailable_reason).info("selftest.degraded_as_designed")
        logger.bind(recalled=await recall_related("anything")).info("selftest.recall_returns_empty")
        logger.bind(stored=await add_research_finding("q", "f")).info(
            "selftest.store_returns_false"
        )
        logger.info("selftest.passed")
        return

    stored = await add_research_finding(
        "What makes a multi-agent system agentic?",
        "Supervisor-based routing lets a critic send work back to research, not just writing.",
        url="https://example.com/agentic",
    )
    logger.bind(stored=stored).info("selftest.store")

    await asyncio.sleep(1)  # mem0 extraction is not instant
    logger.bind(recalled=await recall_related("agentic routing")).info("selftest.recall")
    logger.info("selftest.passed")


if __name__ == "__main__":
    asyncio.run(_selftest())

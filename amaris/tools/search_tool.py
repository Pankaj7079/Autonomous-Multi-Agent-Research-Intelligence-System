"""Web search. DuckDuckGo always, Tavily as a bonus when a key exists."""

from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict
from urllib.parse import urlparse, urlunparse

from amaris.config.settings import get_settings
from amaris.observability.logging import logger
from amaris.observability.tool_trace import record

DEFAULT_MAX_RESULTS = 8


class SearchResult(TypedDict):
    """What every search backend normalises to. The researcher adds task_id later."""

    title: str
    url: str
    content: str
    source: str


def _normalize_url(url: str) -> str:
    """Key for deduping. Same page from two engines must collapse to one entry."""
    try:
        parts = urlparse(url.strip())
    except ValueError:
        return url.strip().lower()
    # drop query and fragment: tracking params make identical pages look different
    return urlunparse(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "", "")
    )


def _ddg_blocking(query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    return DDGS().text(query, max_results=max_results)


async def ddg_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
    """DuckDuckGo, no key needed. Returns [] on any failure — callers must not crash on search."""
    started = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        # ddgs is sync and does network I/O, so it cannot run on the event loop
        raw = await loop.run_in_executor(None, _ddg_blocking, query, max_results)
    except Exception as exc:
        logger.bind(tool="ddg_search", error=str(exc)[:200]).warning("tool.failed")
        return []

    results: list[SearchResult] = [
        SearchResult(
            title=str(item.get("title", "")),
            url=str(item.get("href", "")),
            content=str(item.get("body", "")),
            source="duckduckgo",
        )
        for item in raw
        if item.get("href")
    ]
    logger.bind(
        tool="ddg_search", ms=round((time.perf_counter() - started) * 1000, 1), results=len(results)
    ).debug("tool.call")
    return results


async def tavily_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
    """Tavily when TAVILY_API_KEY is set, otherwise a silent no-op. Returns [] on failure."""
    api_key = get_settings().key("tavily_api_key")
    if not api_key:
        return []

    started = time.perf_counter()
    try:
        # optional extra — a missing package must degrade, never crash (ADR-009)
        from tavily import AsyncTavilyClient

        response = await AsyncTavilyClient(api_key=api_key).search(
            query, max_results=max_results, search_depth="basic"
        )
    except ImportError:
        logger.bind(tool="tavily_search").debug("tool.extra_missing")
        return []
    except Exception as exc:
        logger.bind(tool="tavily_search", error=str(exc)[:200]).warning("tool.failed")
        return []

    results: list[SearchResult] = [
        SearchResult(
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            content=str(item.get("content", "")),
            source="tavily",
        )
        for item in response.get("results", [])
        if item.get("url")
    ]
    logger.bind(
        tool="tavily_search",
        ms=round((time.perf_counter() - started) * 1000, 1),
        results=len(results),
    ).debug("tool.call")
    return results


async def smart_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
    """Both engines in parallel, deduped by URL, Tavily first. Returns [] if everything fails."""
    started = time.perf_counter()
    tavily, ddg = await asyncio.gather(
        tavily_search(query, max_results),
        ddg_search(query, max_results),
        return_exceptions=False,
    )

    merged: list[SearchResult] = []
    seen: set[str] = set()
    # tavily first because its snippets are richer, so it wins any duplicate
    for result in [*tavily, *ddg]:
        key = _normalize_url(result["url"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= max_results:
            break

    elapsed = round((time.perf_counter() - started) * 1000, 1)
    engines = [name for name, hits in (("tavily", tavily), ("duckduckgo", ddg)) if hits]
    logger.bind(
        tool="smart_search",
        query=query[:80],
        ms=elapsed,
        results=len(merged),
        engines=len(engines),
    ).info("tool.call")
    record(
        "web_search",
        target=query,
        ok=bool(merged),
        ms=elapsed,
        detail=f"{len(merged)} results · {', '.join(engines) or 'no engine answered'}",
    )
    return merged

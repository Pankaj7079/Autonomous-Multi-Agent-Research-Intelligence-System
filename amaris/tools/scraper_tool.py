"""Fetch a page as clean markdown. Degrades to nothing when crawl4ai isn't installed."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from amaris.observability.logging import logger

SCRAPE_TIMEOUT = 15
MAX_CHARS = 8000

# only scrape when the search snippet is too thin to be useful on its own
THIN_SNIPPET_CHARS = 300

_warned_missing = False


def _extract_markdown(result: Any) -> str:
    """crawl4ai moved markdown between attributes across versions, so try both shapes."""
    markdown = getattr(result, "markdown", "") or ""
    return str(getattr(markdown, "raw_markdown", markdown) or "")


async def _crawl(url: str) -> str:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)
    if not getattr(result, "success", True):
        return ""
    return _extract_markdown(result)


async def scrape_url(url: str, timeout_s: int = SCRAPE_TIMEOUT, max_chars: int = MAX_CHARS) -> str:
    """Page as markdown, capped. Returns "" on timeout, failure or a missing extra."""
    global _warned_missing
    started = time.perf_counter()

    try:
        text = await asyncio.wait_for(_crawl(url), timeout=timeout_s)
    except ImportError:
        # optional `scraping` extra — the researcher falls back to search snippets
        if not _warned_missing:
            logger.bind(tool="scrape_url").warning("tool.extra_missing")
            _warned_missing = True
        return ""
    except TimeoutError:
        logger.bind(tool="scrape_url", url=url[:120], timeout=timeout_s).warning("tool.timeout")
        return ""
    except Exception as exc:
        logger.bind(tool="scrape_url", url=url[:120], error=str(exc)[:200]).warning("tool.failed")
        return ""

    clipped = text.strip()[:max_chars]
    logger.bind(
        tool="scrape_url",
        url=url[:120],
        ms=round((time.perf_counter() - started) * 1000, 1),
        chars=len(clipped),
    ).debug("tool.call")
    return clipped


def needs_scraping(snippet: str) -> bool:
    """True when a search snippet is too thin to answer anything on its own."""
    return len(snippet.strip()) < THIN_SNIPPET_CHARS

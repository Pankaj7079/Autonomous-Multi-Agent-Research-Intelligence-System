"""Fetch a page as clean markdown. Degrades to nothing when crawl4ai isn't installed."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from amaris.observability.logging import logger

SCRAPE_TIMEOUT = 25
MAX_CHARS = 8000

# only scrape when the search snippet is too thin to be useful on its own
THIN_SNIPPET_CHARS = 300

_warned_missing = False

# one browser for the whole process. the old code opened a fresh AsyncWebCrawler per url, so
# every scrape paid 2-5s of chromium boot first and wikipedia timed out before it fetched anything
_crawler: Any = None
_start_lock = asyncio.Lock()

# tabs in one browser now, not separate browsers, so this can be wider than the old limit of 2
_PAGES = asyncio.Semaphore(4)
# a chromium kept alive all day grows; let crawl4ai recycle it instead of us restarting per call
_PAGES_BEFORE_RECYCLE = 40


def _browser_config() -> Any:
    """Stealth is off by default, and the default UA says HeadlessChrome — which is the tell."""
    from crawl4ai import BrowserConfig

    return BrowserConfig(
        headless=True,
        # patches navigator.webdriver and the other properties a bot check reads first
        enable_stealth=True,
        user_agent_mode="random",
        # no text_mode: it looked like free speed but measured 12% less text on js-rendered docs
        light_mode=True,
        max_pages_before_recycle=_PAGES_BEFORE_RECYCLE,
        viewport_width=1280,
        viewport_height=800,
    )


def _run_config(timeout_s: int) -> Any:
    from crawl4ai import CacheMode, CrawlerRunConfig

    return CrawlerRunConfig(
        # a research answer must reflect the page as it is now, not as it was on an earlier run
        cache_mode=CacheMode.BYPASS,
        # a real visitor scrolls and moves a mouse; these defeat the cheap behavioural checks
        simulate_user=True,
        override_navigator=True,
        # measured a GAIN from this one — cookie walls were being counted as the page
        remove_overlay_elements=True,
        # no excluded_tags: stripping nav/header/aside cost groq.com 1177 of its 1790 chars,
        # because modern sites put real content inside those landmarks
        page_timeout=timeout_s * 1000,
    )


async def _get_crawler() -> Any:
    """The one browser, started on first use. Concurrent first calls must not race it open."""
    global _crawler
    if _crawler is not None:
        return _crawler
    async with _start_lock:
        if _crawler is None:
            from crawl4ai import AsyncWebCrawler

            crawler = AsyncWebCrawler(config=_browser_config())
            await crawler.start()
            _crawler = crawler
            logger.bind(tool="scrape_url").info("scraper.browser_started")
    return _crawler


async def close_scraper() -> None:
    """Shut the shared browser down. Called from reset_pipeline, so a run never leaks chromium."""
    global _crawler
    if _crawler is None:
        return
    try:
        await _crawler.close()
    except Exception as exc:
        logger.bind(tool="scrape_url", error=str(exc)[:200]).warning("scraper.close_failed")
    finally:
        _crawler = None
        logger.bind(tool="scrape_url").debug("scraper.browser_closed")


def _extract_markdown(result: Any) -> str:
    """crawl4ai moved markdown between attributes across versions, so try both shapes."""
    markdown = getattr(result, "markdown", "") or ""
    return str(getattr(markdown, "raw_markdown", markdown) or "")


async def _crawl(url: str, timeout_s: int = SCRAPE_TIMEOUT) -> str:
    crawler = await _get_crawler()
    result = await crawler.arun(url=url, config=_run_config(timeout_s))
    if not getattr(result, "success", True):
        return ""
    return _extract_markdown(result)


async def scrape_url(url: str, timeout_s: int = SCRAPE_TIMEOUT, max_chars: int = MAX_CHARS) -> str:
    """Page as markdown, capped. Returns "" on timeout, failure or a missing extra."""
    global _warned_missing
    started = time.perf_counter()

    try:
        async with _PAGES:
            text = await asyncio.wait_for(_crawl(url, timeout_s), timeout=timeout_s)
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
        # a browser that died takes the singleton with it, or every later scrape reuses a corpse
        await close_scraper()
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

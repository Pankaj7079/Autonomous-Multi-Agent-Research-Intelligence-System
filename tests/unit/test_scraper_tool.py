"""Scraping is optional infrastructure — every failure path must return "", not raise."""

from __future__ import annotations

import asyncio

import pytest

from amaris.tools import scraper_tool
from amaris.tools.scraper_tool import needs_scraping, scrape_url


async def test_returns_markdown_capped_at_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_crawl(url: str) -> str:
        return "x" * 20_000

    monkeypatch.setattr(scraper_tool, "_crawl", fake_crawl)
    assert len(await scrape_url("https://a.com", max_chars=8000)) == 8000


async def test_missing_extra_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """crawl4ai lives behind an extra — without it the researcher uses search snippets."""

    async def missing(url: str) -> str:
        raise ImportError("No module named 'crawl4ai'")

    monkeypatch.setattr(scraper_tool, "_crawl", missing)
    assert await scrape_url("https://a.com") == ""


async def test_slow_page_times_out_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(url: str) -> str:
        await asyncio.sleep(5)
        return "never gets here"

    monkeypatch.setattr(scraper_tool, "_crawl", slow)
    assert await scrape_url("https://a.com", timeout_s=1) == ""


async def test_crawler_exception_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(url: str) -> str:
        raise RuntimeError("browser crashed")

    monkeypatch.setattr(scraper_tool, "_crawl", boom)
    assert await scrape_url("https://a.com") == ""


@pytest.mark.parametrize(
    ("snippet", "thin"),
    [("short text", True), ("", True), ("y" * 400, False)],
)
def test_needs_scraping_only_for_thin_snippets(snippet: str, thin: bool) -> None:
    """Scraping everything tripled runtime for barely better context, so it is conditional."""
    assert needs_scraping(snippet) is thin

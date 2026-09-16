"""Scraping is optional infrastructure — every failure path must return "", not raise."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from amaris.tools import scraper_tool
from amaris.tools.scraper_tool import close_scraper, needs_scraping, scrape_url


@pytest.fixture(autouse=True)
async def _no_leftover_browser():
    """The crawler is module state now, so a test must never inherit another test's browser."""
    scraper_tool._crawler = None
    yield
    scraper_tool._crawler = None


async def test_returns_markdown_capped_at_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_crawl(url: str, timeout_s: int = 25) -> str:
        return "x" * 20_000

    monkeypatch.setattr(scraper_tool, "_crawl", fake_crawl)
    assert len(await scrape_url("https://a.com", max_chars=8000)) == 8000


async def test_missing_extra_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """crawl4ai lives behind an extra — without it the researcher uses search snippets."""

    async def missing(url: str, timeout_s: int = 25) -> str:
        raise ImportError("No module named 'crawl4ai'")

    monkeypatch.setattr(scraper_tool, "_crawl", missing)
    assert await scrape_url("https://a.com") == ""


async def test_slow_page_times_out_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(url: str, timeout_s: int = 25) -> str:
        await asyncio.sleep(5)
        return "never gets here"

    monkeypatch.setattr(scraper_tool, "_crawl", slow)
    assert await scrape_url("https://a.com", timeout_s=1) == ""


async def test_crawler_exception_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(url: str, timeout_s: int = 25) -> str:
        raise RuntimeError("browser crashed")

    monkeypatch.setattr(scraper_tool, "_crawl", boom)
    assert await scrape_url("https://a.com") == ""


# ── the shared browser ───────────────────────────────────────────────────────


class _FakeCrawler:
    """Stands in for AsyncWebCrawler so the singleton can be tested without chromium."""

    started = 0

    def __init__(self) -> None:
        self.closed = False
        _FakeCrawler.started += 1

    async def start(self) -> None: ...

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def stub_crawl4ai(monkeypatch: pytest.MonkeyPatch):
    """A stub module, not the real import: importing crawl4ai pulls litellm, which loads the
    developer's .env into os.environ and made unrelated settings tests fail."""
    _FakeCrawler.started = 0
    module = types.ModuleType("crawl4ai")
    module.AsyncWebCrawler = lambda config=None: _FakeCrawler()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crawl4ai", module)
    monkeypatch.setattr(scraper_tool, "_browser_config", lambda: None)
    return module


async def test_one_browser_is_reused_across_scrapes(stub_crawl4ai) -> None:
    """The whole point of the fix: opening chromium per url is what made wikipedia time out."""
    first = await scraper_tool._get_crawler()
    second = await scraper_tool._get_crawler()

    assert first is second
    assert _FakeCrawler.started == 1


async def test_concurrent_first_calls_do_not_open_two_browsers(stub_crawl4ai) -> None:
    """Four research tasks can reach the scraper at once; only one of them may start a browser."""
    crawlers = await asyncio.gather(*(scraper_tool._get_crawler() for _ in range(4)))

    assert len({id(c) for c in crawlers}) == 1
    assert _FakeCrawler.started == 1


async def test_a_dead_browser_is_dropped_rather_than_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reusing a crashed browser would turn one bad page into every later scrape failing."""
    scraper_tool._crawler = _FakeCrawler()

    async def boom(url: str, timeout_s: int = 25) -> str:
        raise RuntimeError("browser crashed")

    monkeypatch.setattr(scraper_tool, "_crawl", boom)
    assert await scrape_url("https://a.com") == ""
    assert scraper_tool._crawler is None


async def test_close_is_safe_when_nothing_was_started() -> None:
    scraper_tool._crawler = None
    await close_scraper()


async def test_close_survives_a_browser_that_will_not_shut_down() -> None:
    """Shutdown must not raise, or the API lifespan fails on the way out."""

    class _Stuck(_FakeCrawler):
        async def close(self) -> None:
            raise RuntimeError("already gone")

    scraper_tool._crawler = _Stuck()
    await close_scraper()
    assert scraper_tool._crawler is None


@pytest.mark.parametrize(
    ("snippet", "thin"),
    [("short text", True), ("", True), ("y" * 400, False)],
)
def test_needs_scraping_only_for_thin_snippets(snippet: str, thin: bool) -> None:
    """Scraping everything tripled runtime for barely better context, so it is conditional."""
    assert needs_scraping(snippet) is thin

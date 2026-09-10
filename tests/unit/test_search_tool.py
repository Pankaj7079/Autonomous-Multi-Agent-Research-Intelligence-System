"""Search must dedupe, merge and never raise. No test here hits the network."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.config.settings import Settings
from amaris.tools import search_tool
from amaris.tools.search_tool import _normalize_url, ddg_search, smart_search


def _ddg_rows(*urls: str) -> list[dict[str, Any]]:
    return [{"title": f"title {u}", "href": u, "body": f"body {u}"} for u in urls]


@pytest.fixture
def no_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests run the DuckDuckGo-only path, which is the no-key default."""
    monkeypatch.setattr(search_tool, "get_settings", lambda: Settings(_env_file=None))


async def test_ddg_results_are_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_tool, "_ddg_blocking", lambda q, n: _ddg_rows("https://a.com/x"))
    results = await ddg_search("q")
    assert results == [
        {
            "title": "title https://a.com/x",
            "url": "https://a.com/x",
            "content": "body https://a.com/x",
            "source": "duckduckgo",
        }
    ]


async def test_ddg_failure_returns_empty_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead search engine must degrade the run, not end it."""

    def boom(query: str, max_results: int) -> list[dict[str, Any]]:
        raise RuntimeError("ddg is down")

    monkeypatch.setattr(search_tool, "_ddg_blocking", boom)
    assert await ddg_search("q") == []


async def test_rows_without_a_url_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_tool, "_ddg_blocking", lambda q, n: [{"title": "t", "body": "b"}])
    assert await ddg_search("q") == []


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("https://a.com/page", "https://a.com/page/"),
        ("https://A.com/page", "https://a.com/page"),
        ("https://a.com/page", "https://a.com/page?utm_source=x"),
        ("https://a.com/page", "https://a.com/page#section"),
    ],
)
def test_urls_that_mean_the_same_page_collapse(left: str, right: str) -> None:
    assert _normalize_url(left) == _normalize_url(right)


async def test_smart_search_dedupes_across_engines(
    monkeypatch: pytest.MonkeyPatch, no_tavily: None
) -> None:
    """The same page from two engines must count once, or the supervisor overcounts sources."""
    monkeypatch.setattr(
        search_tool,
        "_ddg_blocking",
        lambda q, n: _ddg_rows("https://a.com/x", "https://a.com/x/", "https://b.com/y"),
    )
    results = await smart_search("q")
    assert [r["url"] for r in results] == ["https://a.com/x", "https://b.com/y"]


async def test_tavily_wins_duplicates_because_its_snippets_are_richer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_tool, "get_settings", lambda: Settings(_env_file=None, tavily_api_key="tvly")
    )
    monkeypatch.setattr(search_tool, "_ddg_blocking", lambda q, n: _ddg_rows("https://a.com/x"))

    async def fake_tavily(query: str, max_results: int = 8) -> list[dict[str, Any]]:
        return [
            {
                "title": "tavily title",
                "url": "https://a.com/x",
                "content": "much longer tavily content",
                "source": "tavily",
            }
        ]

    monkeypatch.setattr(search_tool, "tavily_search", fake_tavily)
    results = await smart_search("q")
    assert len(results) == 1
    assert results[0]["source"] == "tavily"


async def test_smart_search_respects_max_results(
    monkeypatch: pytest.MonkeyPatch, no_tavily: None
) -> None:
    monkeypatch.setattr(
        search_tool,
        "_ddg_blocking",
        lambda q, n: _ddg_rows(*[f"https://a.com/{i}" for i in range(20)]),
    )
    assert len(await smart_search("q", max_results=3)) == 3


async def test_tavily_is_skipped_without_a_key(no_tavily: None) -> None:
    assert await search_tool.tavily_search("q") == []

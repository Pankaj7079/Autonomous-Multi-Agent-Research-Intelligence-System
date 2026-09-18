"""Consuming external MCP servers. The other half of the MCP story (ADR-042)."""

from __future__ import annotations

import pytest

from amaris.config.settings import Settings, get_settings
from amaris.tools import mcp_client
from amaris.tools.mcp_client import ServerSpec, configured_servers, search

ARXIV = ServerSpec(
    name="arxiv", transport="stdio", command="uvx", args=("x",), tools=("search_papers",)
)


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    # _env_file=None: a developer's own .env (this one has MCP_CLIENT_ENABLED=true) must never
    # leak into a test asserting what happens with nothing configured — deleting the env var
    # alone is not enough, since pydantic-settings falls back to the .env file underneath it
    monkeypatch.setattr(mcp_client, "get_settings", lambda: Settings(_env_file=None))
    yield
    get_settings.cache_clear()


def test_no_servers_are_started_unless_the_client_is_turned_on(monkeypatch) -> None:
    """Found by the test suite hanging: a research run was spawning real npx and uvx processes
    just because those binaries existed on the machine."""
    monkeypatch.delenv("MCP_CLIENT_ENABLED", raising=False)
    assert configured_servers() == []


def test_the_github_server_needs_a_token(monkeypatch) -> None:
    monkeypatch.setenv("MCP_CLIENT_ENABLED", "true")
    # empty, not deleted: a developer with a real token in .env would otherwise fail this test,
    # because pydantic reads the file when the variable is absent
    monkeypatch.setenv("GITHUB_TOKEN", "")
    assert "github" not in {spec.name for spec in configured_servers()}


def test_a_configured_token_adds_the_remote_github_server(monkeypatch) -> None:
    """The npm server is deprecated, so this talks to GitHub's hosted endpoint over HTTP —
    which is also the only transport that could work where no node runtime exists."""
    monkeypatch.setenv("MCP_CLIENT_ENABLED", "true")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_example")
    github = next(spec for spec in configured_servers() if spec.name == "github")
    assert github.transport == "http"
    assert github.url.startswith("https://")
    assert github.env["Authorization"].endswith("ghp_example")


def test_a_tool_outside_the_shortlist_is_refused(monkeypatch) -> None:
    """The GitHub server publishes dozens of tools, including writes. Only the shortlist runs."""

    async def unreachable(*args, **kwargs):
        raise AssertionError("no session should be opened for a refused tool")

    monkeypatch.setattr(mcp_client, "_session", unreachable)

    async def run():
        return await mcp_client.call_tool(ARXIV, "create_issue", {})

    import asyncio

    assert asyncio.run(run()) == ""


async def test_a_dead_server_is_a_missing_source_not_a_failed_run(monkeypatch) -> None:
    """A research run must survive an MCP server that is down, rate-limited or not installed —
    arxiv returned nothing but HTTP 429 on the first live probe."""

    async def boom(spec, tool, arguments):
        return ""

    monkeypatch.setattr(mcp_client, "call_tool", boom)
    assert await search(ARXIV, "anything") is None


async def test_a_result_becomes_a_source_shaped_like_every_other(monkeypatch) -> None:
    """MCP output joins raw_research, so it reaches the writer through the same path as a
    scraped page — including the untrusted-content wrapper."""

    async def text(spec, tool, arguments):
        assert tool == "search_papers"
        return "Paper: agentic retrieval"

    monkeypatch.setattr(mcp_client, "call_tool", text)
    source = await search(ARXIV, "agentic retrieval augmented generation")

    assert source is not None
    assert source["content"] == "Paper: agentic retrieval"
    assert source["url"] == "mcp://arxiv/search_papers"
    assert source["task_id"] == "mcp"


def test_the_filesystem_server_never_starts_in_cloud_mode(monkeypatch) -> None:
    """A public Space answers questions from strangers, and a query-derived glob against the
    container filesystem is an enumeration primitive rather than a research source."""
    monkeypatch.setenv("MCP_CLIENT_ENABLED", "true")
    monkeypatch.setenv("MCP_FILESYSTEM_ROOT", "/app/docs")
    monkeypatch.setenv("DEPLOYMENT_MODE", "cloud")
    assert "filesystem" not in {spec.name for spec in configured_servers()}

    # and it is still available locally, where the operator chose the directory themselves
    monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    get_settings.cache_clear()
    names = {spec.name for spec in configured_servers(include_disabled=True)}
    assert "filesystem" in names

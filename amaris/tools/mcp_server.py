"""The four tools, exposed two ways: plain callables for agents, MCP for outside clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amaris.observability.logging import logger
from amaris.tools.code_executor import execute_python
from amaris.tools.scraper_tool import scrape_url
from amaris.tools.search_tool import smart_search
from amaris.tools.vector_tool import search_knowledge_base

if TYPE_CHECKING:
    from collections.abc import Callable

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "web_search": smart_search,
    "scrape_webpage": scrape_url,
    "execute_python": execute_python,
    "search_knowledge_base": search_knowledge_base,
}

# who is allowed what — the researcher must not run code, the analyst must not scrape
AGENT_TOOLS: dict[str, tuple[str, ...]] = {
    "researcher": ("web_search", "scrape_webpage"),
    "analyst": ("execute_python", "search_knowledge_base"),
    "planner": (),
    "writer": (),
    "critic": (),
    "supervisor": (),
}


def get_tools_for_agent(agent: str) -> dict[str, Callable[..., Any]]:
    """Callables this agent may use. Plain functions, so agents work without the mcp extra."""
    names = AGENT_TOOLS.get(agent, ())
    return {name: TOOL_REGISTRY[name] for name in names}


def build_server() -> Any:
    """FastMCP server exposing all four tools. Raises ImportError without the mcp extra."""
    from fastmcp import FastMCP

    server = FastMCP("amaris-tools")

    @server.tool()
    async def web_search(query: str, max_results: int = 8) -> list[dict[str, Any]]:
        """Search the web and return deduped {title, url, content} results."""
        return list(await smart_search(query, max_results))

    @server.tool()
    async def scrape_webpage(url: str) -> str:
        """Fetch a page as clean markdown, capped at 8000 characters."""
        return await scrape_url(url)

    @server.tool()
    async def run_python(code: str) -> dict[str, Any]:
        """Run Python in a sandboxed subprocess and return stdout, stderr and success."""
        return dict(await execute_python(code))

    @server.tool()
    async def knowledge_base(query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Semantic search over research stored from earlier sessions."""
        return [dict(hit) for hit in await search_knowledge_base(query, limit)]

    return server


def main() -> None:
    """Run the MCP server over stdio: uv run python -m amaris.tools.mcp_server."""
    try:
        server = build_server()
    except ImportError:
        logger.bind(extra="mcp").error("mcp.extra_missing")
        raise SystemExit("fastmcp is not installed — run: uv sync --extra mcp") from None
    logger.bind(tools=list(TOOL_REGISTRY)).info("mcp.serving")
    server.run()


if __name__ == "__main__":
    main()

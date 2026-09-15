"""Consume external MCP servers as evidence sources. The other half of `mcp_server.py`.

AMARIS already exposes its own tools over MCP; this connects outward, so a question about a
library can be answered from the repository and the papers rather than from blog posts.

Every result is untrusted input — a README or an abstract is attacker-controllable in exactly
the way a scraped page is — so callers must wrap what comes back before it reaches a prompt
(ADR-015, ADR-042).
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

# a cold `npx` or `uvx` start is seconds, and a server that cannot answer in this long is not
# worth making a research run wait for
CONNECT_TIMEOUT = 45.0
# arxiv answers in seconds or 429s and retries internally for a minute; a research run
# should not wait out someone else's backoff
CALL_TIMEOUT = 30.0
# enough of a tool result to be evidence, without pasting a whole repository into a prompt
RESULT_CHARS = 4000
# a server saying it found nothing, in the words each one happens to use
EMPTY_ANSWERS = frozenset(["no matches found", "no results found", "[]", "none"])


@dataclass(frozen=True)
class ServerSpec:
    """One MCP server we can talk to, and the few of its tools we actually expose.

    `tools` is deliberately a shortlist. The GitHub server alone publishes dozens, and handing
    an agent every one of them turns the tool list into most of the prompt.
    """

    name: str
    transport: str
    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)
    tools: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        """False when the server cannot start here — no node, no uv, or no token."""
        if self.transport == "http":
            return bool(self.url)
        return bool(self.command) and shutil.which(self.command) is not None


def configured_servers(include_disabled: bool = False) -> list[ServerSpec]:
    """The servers this deployment can actually reach, in preference order.

    Empty is a normal state: cloud mode has no node runtime, so the research path simply runs
    without them rather than failing (ADR-009).
    """
    settings = get_settings()
    if not settings.mcp_client_enabled and not include_disabled:
        return []
    specs: list[ServerSpec] = []

    root = (settings.mcp_filesystem_root or "").strip()
    # never in cloud mode: a public Space answers questions from strangers, and a query-derived
    # glob against the container filesystem is an enumeration primitive, not a research source
    if root and settings.is_cloud:
        logger.warning("mcp_client.filesystem_refused_in_cloud")
        root = ""
    if root:
        specs.append(
            ServerSpec(
                name="filesystem",
                transport="stdio",
                command="npx",
                # pinned: an unpinned npx pulls whatever is newest mid-run
                args=("-y", f"@modelcontextprotocol/server-filesystem@{FILESYSTEM_VERSION}", root),
                tools=("search_files", "read_text_file", "list_directory"),
            )
        )

    specs.append(
        ServerSpec(
            name="arxiv",
            transport="stdio",
            command="uvx",
            args=(f"arxiv-mcp-server@{ARXIV_VERSION}",),
            tools=("search_papers", "download_paper", "read_paper"),
        )
    )

    token = settings.key("github_token")
    if token:
        # the npm server was deprecated; GitHub's own server is a Go binary or this remote
        # endpoint, and the endpoint needs nothing installed (ADR-042)
        specs.append(
            ServerSpec(
                name="github",
                transport="http",
                url=GITHUB_MCP_URL,
                env={"Authorization": f"Bearer {token}"},
                tools=("search_repositories", "search_code", "list_releases", "get_file_contents"),
            )
        )

    return [spec for spec in specs if spec.runnable or include_disabled]


FILESYSTEM_VERSION = "2026.8.31"
ARXIV_VERSION = "0.7.2"
GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"


async def _session(spec: ServerSpec):
    """Open a session to one server. Per call on purpose: a cached stdio process outlives the
    run that needed it, and a leaked node process per question is worse than a slow start."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if spec.transport == "http":
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        # mcp 2.x takes a prepared http client, not a headers kwarg — auth goes on the client
        http = create_mcp_http_client(headers=spec.env)
        return streamable_http_client(spec.url, http_client=http), ClientSession

    params = StdioServerParameters(command=spec.command, args=list(spec.args), env=spec.env or None)
    return stdio_client(params), ClientSession


async def list_tools(spec: ServerSpec) -> list[dict[str, Any]]:
    """What this server offers, filtered to the shortlist. [] on any failure."""
    try:
        transport, session_cls = await _session(spec)
        async with transport as streams, session_cls(*streams[:2]) as session:
            await asyncio.wait_for(session.initialize(), timeout=CONNECT_TIMEOUT)
            found = await asyncio.wait_for(session.list_tools(), timeout=CONNECT_TIMEOUT)
    except Exception as exc:
        logger.bind(server=spec.name, error=str(exc)[:200]).warning("mcp_client.unreachable")
        return []

    tools = [
        {"name": tool.name, "description": (tool.description or "")[:300]}
        for tool in found.tools
        if not spec.tools or tool.name in spec.tools
    ]
    logger.bind(server=spec.name, tools=len(tools)).info("mcp_client.listed")
    return tools


async def call_tool(spec: ServerSpec, tool: str, arguments: dict[str, Any]) -> str:
    """Run one tool and return its text. "" on failure — a dead server is a missing source,
    never a failed run."""
    if spec.tools and tool not in spec.tools:
        logger.bind(server=spec.name, tool=tool).warning("mcp_client.tool_not_allowed")
        return ""

    started = time.perf_counter()
    try:
        transport, session_cls = await _session(spec)
        async with transport as streams, session_cls(*streams[:2]) as session:
            await asyncio.wait_for(session.initialize(), timeout=CONNECT_TIMEOUT)
            result = await asyncio.wait_for(
                session.call_tool(tool, arguments), timeout=CALL_TIMEOUT
            )
    except Exception as exc:
        logger.bind(server=spec.name, tool=tool, error=str(exc)[:200]).warning("mcp_client.failed")
        return ""

    parts = [str(getattr(item, "text", "")) for item in (result.content or [])]
    text = "\n".join(part for part in parts if part)[:RESULT_CHARS]
    logger.bind(
        server=spec.name,
        tool=tool,
        chars=len(text),
        ms=round((time.perf_counter() - started) * 1000, 1),
    ).info("mcp_client.called")
    return text


def _any_case(word: str) -> str:
    """A glob that ignores case. The filesystem server matches case-sensitively, so a lowercase
    query never found DECISIONS.md until each letter was expanded to a character class."""
    letters = "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in word)
    return f"*{letters}*"


# each server's search tool takes a different argument shape, so the call is built per server
def _search_call(spec: ServerSpec, query: str) -> tuple[str, dict[str, Any]]:
    if spec.name == "arxiv":
        return "search_papers", {"query": query, "max_results": 3}
    if spec.name == "github":
        return "search_repositories", {"query": query}
    # the filesystem server matches globs, not prose, so the longest word stands in for the
    # query — weak by design, and it is why filesystem is off unless a root is configured
    words = sorted((w for w in query.split() if len(w) > 4), key=len, reverse=True)
    # the root is the last launch argument; "." resolves outside the server's allowed directory
    root = spec.args[-1] if spec.args else "."
    return "search_files", {"path": root, "pattern": _any_case(words[0] if words else "")}


async def search(spec: ServerSpec, query: str) -> dict[str, Any] | None:
    """One server's answer to one research task, shaped like every other source, or None.

    The url is a pseudo-url: an MCP result has no address of its own, and a citation has to
    point somewhere the reader can identify.
    """
    tool, arguments = _search_call(spec, query)
    text = await call_tool(spec, tool, arguments)
    # "No matches found" is a sentence, not evidence, and it would occupy a citation slot
    if not text.strip() or text.strip().lower() in EMPTY_ANSWERS:
        return None
    return {
        "title": f"{spec.name} · {tool}",
        "url": f"mcp://{spec.name}/{tool}",
        "content": text,
        "task_id": "mcp",
        "source": f"mcp:{spec.name}",
    }


async def selftest() -> bool:
    """Probe every server that could be configured and print what worked. True if any answered.

    Matches the `--selftest` idiom the memory module already uses: the point is to tell you
    which server is reachable before a research run quietly runs without it.
    """
    settings = get_settings()
    specs = configured_servers(include_disabled=True)
    print(f"mcp client enabled : {settings.mcp_client_enabled}")
    print(f"filesystem root    : {settings.mcp_filesystem_root or '(unset — server disabled)'}")
    print(f"github token       : {'set' if settings.key('github_token') else '(unset)'}")
    print("")

    answered = False
    for spec in specs:
        launcher = spec.command or spec.url
        if not spec.runnable:
            print(f"  {spec.name:<11} SKIPPED   {launcher} not found on PATH")
            continue

        tools = await list_tools(spec)
        if not tools:
            print(f"  {spec.name:<11} NO ANSWER {launcher} — see the warning above for why")
            continue

        names = ", ".join(tool["name"] for tool in tools)
        source = await search(spec, "agent")
        got = f"{len(source['content'])} chars" if source else "no result for the probe query"
        print(f"  {spec.name:<11} OK        tools: {names}")
        print(f"  {'':<11}           probe: {got}")
        answered = True

    if not answered:
        print("\nno server answered. that is survivable — a run just has fewer sources.")
    return answered


if __name__ == "__main__":
    import sys

    sys.exit(0 if asyncio.run(selftest()) else 1)

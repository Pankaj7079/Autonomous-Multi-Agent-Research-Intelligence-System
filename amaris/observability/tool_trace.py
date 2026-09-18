"""Every tool and MCP call a run made, kept so the UI can show what actually ran.

Tools are called deep inside agents, so threading state down into each one would change every
tool signature. A contextvar carries the buffer instead — the same trick `context.py` already
uses for session_id — and the node wrapper drains it into state after each agent finishes.

Only the shape of a call is kept: what ran, against what, how long, and whether it worked.
Arguments and results are deliberately left out — an argument dict carries the user's query and
an MCP server's auth header, and this list is rendered in the UI and ships in the JSON export.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from amaris.observability.context import get_agent

# a runaway loop must not grow the checkpointed state without bound
MAX_CALLS = 200
TARGET_CHARS = 120
DETAIL_CHARS = 80

# kinds, so the UI can tell an MCP server apart from a local tool at a glance
TOOL = "tool"
MCP = "mcp"

_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("tool_calls", default=None)


def start_recording() -> None:
    """Open a fresh buffer for the node about to run."""
    _calls.set([])


def record(
    tool: str,
    *,
    kind: str = TOOL,
    target: str = "",
    ok: bool = True,
    ms: float = 0.0,
    detail: str = "",
) -> None:
    """Note one call. A no-op when nothing is recording, so tools still work standalone."""
    buffer = _calls.get()
    # dropping past the cap rather than trimming: the first 200 calls are the interesting ones
    if buffer is None or len(buffer) >= MAX_CALLS:
        return
    buffer.append(
        {
            "tool": tool,
            "kind": kind,
            "agent": get_agent(),
            "target": target[:TARGET_CHARS],
            "ok": ok,
            "ms": round(ms, 1),
            "detail": detail[:DETAIL_CHARS],
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )


def drain() -> list[dict[str, Any]]:
    """Take everything recorded since the last start and empty the buffer.

    Cleared in place, not replaced — parallel gather() tasks hold a reference to this same list.
    """
    buffer = _calls.get()
    if not buffer:
        return []
    collected = list(buffer)
    buffer.clear()
    return collected

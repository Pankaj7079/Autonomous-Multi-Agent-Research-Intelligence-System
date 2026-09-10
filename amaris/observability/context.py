"""session_id and current agent carried in contextvars, so every log line gets them."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_session_id: ContextVar[str] = ContextVar("session_id", default="-")
_agent: ContextVar[str] = ContextVar("agent", default="-")


def new_session_id() -> str:
    """Short run id. 8 hex chars is enough to grep one run out of the log."""
    return uuid.uuid4().hex[:8]


def bind_session(session_id: str | None = None) -> str:
    """Attach a session id to everything logged from here on. Returns the id."""
    resolved = session_id or new_session_id()
    _session_id.set(resolved)
    return resolved


def get_session_id() -> str:
    """Current session id, or '-' if nothing is bound."""
    return _session_id.get()


def get_agent() -> str:
    """Current agent name, or '-' outside an agent block."""
    return _agent.get()


@contextmanager
def agent_context(name: str) -> Iterator[None]:
    """Tag every log record emitted inside the block with the running agent."""
    # reset by token, not by remembering the old value — gather() runs these in parallel
    token = _agent.set(name)
    try:
        yield
    finally:
        _agent.reset(token)


def context_patcher(record: dict) -> None:
    """Loguru patcher that injects the current context into record["extra"]."""
    extra = record["extra"]
    extra.setdefault("session_id", _session_id.get())
    extra.setdefault("agent", _agent.get())

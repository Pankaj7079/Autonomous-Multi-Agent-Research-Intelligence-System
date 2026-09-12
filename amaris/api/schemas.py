"""Request/response contracts and the one progress event shape both transports use."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobState = Literal["queued", "running", "done", "failed"]

# progress is derived from which agent is active, not a real fraction — the path is
# decided at runtime so an exact percentage would be a lie (docs/DESIGN.md)
AGENT_PROGRESS: dict[str, int] = {
    "planner": 15,
    "researcher": 45,
    "analyst": 65,
    "writer": 85,
    "critic": 92,
    "evaluator": 96,
}
DONE_PROGRESS = 100

# the supervisor is a router, not work, so it never moves the bar
_AGENT_MESSAGES: dict[str, str] = {
    "supervisor": "deciding what runs next",
    "planner": "breaking the question into research tasks",
    "researcher": "searching and gathering sources",
    "analyst": "analysing the gathered sources",
    "writer": "drafting the report with citations",
    "critic": "reviewing the draft",
    "evaluator": "scoring the finished run",
}


class ResearchRequest(BaseModel):
    """POST /research body."""

    query: str = Field(min_length=3, max_length=500)
    session_id: str | None = None


class ResearchAccepted(BaseModel):
    """POST /research response — the run has not started yet."""

    job_id: str
    session_id: str
    status: JobState = "queued"


class ResearchResult(BaseModel):
    """The finished artefact, only present once status is done."""

    report: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    agent_path: list[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    """GET /research/{job_id} response."""

    job_id: str
    session_id: str = ""
    query: str = ""
    status: JobState = "queued"
    current_agent: str = ""
    progress_pct: int = 0
    result: ResearchResult | None = None
    error: str | None = None


class ProgressEvent(BaseModel):
    """One node transition. Published to pubsub, forwarded verbatim over the WebSocket."""

    agent: str
    status: JobState
    message: str
    progress_pct: int
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class HealthStatus(BaseModel):
    """/health and /ready. checks is empty on the liveness probe."""

    status: str
    mode: str
    checks: dict[str, bool] = Field(default_factory=dict)


def progress_for(agent: str, current_pct: int) -> int:
    """Monotonic progress. A critic -> researcher re-route must not walk the bar backwards."""
    return max(current_pct, AGENT_PROGRESS.get(agent, 0))


def _detail(agent: str, delta: dict[str, Any]) -> str:
    """Turn a node's state delta into something a human can read mid-run."""
    if agent == "planner" and delta.get("research_plan"):
        return f"planned {len(delta['research_plan'])} research tasks"
    if agent == "researcher" and delta.get("raw_research"):
        return f"gathered {len(delta['raw_research'])} sources"
    if agent == "critic" and delta.get("quality_score") is not None:
        return f"scored the draft {delta['quality_score']:.2f}"
    if agent == "supervisor" and delta.get("next_agent"):
        return f"routing to {delta['next_agent']}"
    return _AGENT_MESSAGES.get(agent, agent)


def build_progress_event(
    agent: str, delta: dict[str, Any], current_pct: int, status: JobState = "running"
) -> ProgressEvent:
    """Build the event a node transition emits. Same shape for WebSocket and in-process."""
    return ProgressEvent(
        agent=agent,
        status=status,
        message=_detail(agent, delta),
        progress_pct=progress_for(agent, current_pct),
    )

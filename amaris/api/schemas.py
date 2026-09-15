"""Request/response contracts and the one progress event shape both transports use."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from amaris.agents.triage import HISTORY_TURNS
from amaris.graph.state import DEPTHS

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

JobState = Literal["queued", "running", "done", "failed"]

# an expand reuses what the last turn gathered; 20 is the deepest budget's max_sources
MAX_PRIOR_SOURCES = 20
# each attached file becomes one citable source, so this is also a cap on how much of the
# reference list one upload can occupy
MAX_ATTACHMENTS = 5

# progress is derived from which agent is active, not a real fraction — the path is
# decided at runtime so an exact percentage would be a lie (docs/DESIGN.md)
AGENT_PROGRESS: dict[str, int] = {
    "triage": 5,
    "clarify": 99,
    "planner": 15,
    "researcher": 45,
    "analyst": 65,
    "writer": 85,
    "critic": 92,
    "evaluator": 96,
}
DONE_PROGRESS = 100

# enough of a source to judge whether it was worth citing, without shipping whole pages
SOURCE_SNIPPET_CHARS = 400

# the supervisor is a router, not work, so it never moves the bar
_AGENT_MESSAGES: dict[str, str] = {
    "triage": "sizing the question before spending anything on it",
    "clarify": "asking for the missing detail",
    "supervisor": "deciding what runs next",
    "planner": "breaking the question into research tasks",
    "researcher": "searching and gathering sources",
    "analyst": "analysing the gathered sources",
    "writer": "drafting the report with citations",
    "critic": "reviewing the draft",
    "evaluator": "scoring the finished run",
}


class PriorTurn(BaseModel):
    """One earlier question and what it was answered with."""

    query: str = Field(max_length=500)
    answer: str = Field(default="", max_length=4000)


class ResearchRequest(BaseModel):
    """POST /research body. The last three fields carry a conversation, not a cold start."""

    query: str = Field(min_length=3, max_length=500)
    session_id: str | None = None
    history: list[PriorTurn] = Field(default_factory=list, max_length=HISTORY_TURNS)
    # the depth this run must use, already resolved by the caller — the depth picker sends its
    # choice and "explain in detail" sends the bumped one. None means triage decides as usual.
    depth: str | None = None
    prior_sources: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_PRIOR_SOURCES)
    # files already ingested into Qdrant for this session: [{name, url, chunks}]. The text is
    # not carried here — the researcher retrieves what it needs by session_id.
    attachments: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_ATTACHMENTS)
    # the caller's own provider keys, bound for this job only (ADR-037). deliberately kept out
    # of seed(): a key has no business in GraphState, which is checkpointed and replayed
    api_keys: dict[str, str] = Field(default_factory=dict, exclude=True)

    def seed(self) -> dict[str, Any]:
        """What new_state() should carry forward. Empty dict for an ordinary first question."""
        seed: dict[str, Any] = {}
        if self.history:
            seed["history"] = [turn.model_dump() for turn in self.history]
        if self.attachments:
            seed["attachments"] = list(self.attachments)
        if self.depth in DEPTHS:
            seed["query_depth"] = self.depth
            seed["depth_locked"] = True
        if self.prior_sources:
            # the trace trims content down to "snippet", so accept either key or the reused
            # sources reach the writer with no text at all
            seed["raw_research"] = [
                {**item, "content": item.get("content") or item.get("snippet", "")}
                for item in self.prior_sources
            ]
        return seed


class ResearchAccepted(BaseModel):
    """POST /research response — the run has not started yet."""

    job_id: str
    session_id: str
    status: JobState = "queued"


class RunTrace(BaseModel):
    """Why the run went the way it did. Without this the UI is a box that returns prose."""

    decisions: list[dict[str, Any]] = Field(default_factory=list)
    react_stats: dict[str, Any] = Field(default_factory=dict)
    research_quality: float = 0.0
    revision_count: int = 0
    source_count: int = 0
    critic_feedback: str = ""
    top_issue: str = ""
    # the planner's tasks and the analyst's synthesis existed only in graph state until now,
    # so the two agents in the middle of the run had nothing to show for themselves
    research_plan: list[dict[str, Any]] = Field(default_factory=list)
    research_strategy: str = ""
    analysis: str = ""
    code_outputs: list[dict[str, Any]] = Field(default_factory=list)
    routing_hint: str = ""
    quality_score: float = 0.0
    sources: list[dict[str, Any]] = Field(default_factory=list)
    # what triage decided, and therefore what every budget downstream was set from
    triage: dict[str, Any] = Field(default_factory=dict)


class ResearchResult(BaseModel):
    """The finished artefact, only present once status is done."""

    report: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    agent_path: list[str] = Field(default_factory=list)
    trace: RunTrace | None = None
    # true when the run stopped to ask a question rather than answer one
    awaiting_clarification: bool = False


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
    # seconds since the run started, set by the producer — a ui cannot derive this from ts,
    # which is second-resolution and says nothing about when the run itself began
    elapsed_s: float = 0.0
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
    if agent == "triage" and delta.get("query_depth"):
        return f"triaged as {delta['query_depth']}"
    if agent == "supervisor" and delta.get("next_agent"):
        return f"routing to {delta['next_agent']}"
    return _AGENT_MESSAGES.get(agent, agent)


def build_progress_event(
    agent: str,
    delta: dict[str, Any],
    current_pct: int,
    status: JobState = "running",
    elapsed_s: float = 0.0,
) -> ProgressEvent:
    """Build the event a node transition emits. Same shape for WebSocket and in-process."""
    return ProgressEvent(
        agent=agent,
        status=status,
        message=_detail(agent, delta),
        progress_pct=progress_for(agent, current_pct),
        elapsed_s=round(elapsed_s, 2),
    )


def _trim_source(item: dict[str, Any]) -> dict[str, Any]:
    """A source row for the UI. Content is cut because 60 full pages would bloat the payload."""
    return {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "task_id": item.get("task_id", ""),
        "snippet": str(item.get("content", ""))[:SOURCE_SNIPPET_CHARS],
    }


def result_from_state(state: GraphState) -> ResearchResult:
    """Final state to the shape both the API and the in-process frontend return."""
    # critic and evaluator both emit a "faithfulness", so the eval scores get prefixed
    scores = {
        **state["critic_scores"],
        **{f"eval_{k}": v for k, v in state["evaluation_scores"].items()},
    }
    return ResearchResult(
        report=state["final_report"] or state["draft_report"],
        citations=state["citations"],
        scores=scores,
        agent_path=state["agent_path"],
        awaiting_clarification=not state.get("answerable", True),
        trace=RunTrace(
            decisions=state["decision_log"],
            react_stats=state["react_stats"],
            research_quality=state["research_quality"],
            revision_count=state["revision_count"],
            source_count=len({s.get("url") for s in state["raw_research"] if s.get("url")}),
            critic_feedback=state["critic_feedback"],
            top_issue=state["top_issue"],
            research_plan=state["research_plan"],
            research_strategy=state["research_strategy"],
            analysis=state["analyzed_data"],
            code_outputs=state["code_outputs"],
            routing_hint=state["routing_hint"],
            quality_score=state["quality_score"],
            sources=[_trim_source(item) for item in state["raw_research"]],
            triage={
                "depth": state.get("query_depth", ""),
                "answerable": state.get("answerable", True),
                "clarifying_question": state.get("clarifying_question", ""),
                "sections": state.get("report_sections", []),
                "word_target": state.get("word_target", 0),
                "reason": state.get("triage_reason", ""),
            },
        ),
    )

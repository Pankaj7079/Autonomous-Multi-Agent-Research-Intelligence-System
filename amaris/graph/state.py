"""GraphState — the only data contract between agents. LangGraph requires a TypedDict."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict

from amaris.observability.context import new_session_id

# routing_hint values the critic may write, read by the supervisor
RoutingHint = str
NEED_MORE_RESEARCH = "need_more_research"
FIX_WRITING = "fix_writing"
APPROVE = "approve"
# the report answers a different question than the one asked — re-plan, don't re-search
WRONG_TOPIC = "wrong_topic"

# named individually so trajectory_eval.py can re-derive supervisor routing without
# hard-coding agent name strings a second time
PLANNER = "planner"
RESEARCHER = "researcher"
ANALYST = "analyst"
WRITER = "writer"
CRITIC = "critic"

# nodes, not routing choices — the supervisor is never asked to pick either of these
TRIAGE = "triage"
CLARIFY = "clarify"

# every value state["next_agent"] is allowed to take
AGENTS = (PLANNER, RESEARCHER, ANALYST, WRITER, CRITIC)
FINISH = "FINISH"

# how deeply a query is worth researching; triage picks one, budgets key off it
DEPTHS = ("direct", "brief", "standard", "deep")
DEFAULT_DEPTH = "standard"


class GraphState(TypedDict):
    """Full run state. No reducers needed — the supervisor routes one agent at a time."""

    session_id: str
    original_query: str
    started_at: str

    # triage — set once, before anything is spent, and read by every agent downstream
    query_depth: str
    answerable: bool
    clarifying_question: str
    report_sections: list[str]
    word_target: int
    triage_reason: str

    # planner
    research_plan: list[dict[str, Any]]
    research_strategy: str

    # researcher — research_quality is its own 0-1 self-assessment
    raw_research: list[dict[str, Any]]
    research_quality: float

    # analyst
    analyzed_data: str
    code_outputs: list[dict[str, Any]]

    # writer
    draft_report: str
    citations: list[dict[str, Any]]

    # critic — routing_hint is how it talks to the supervisor
    critic_scores: dict[str, float]
    quality_score: float
    critic_feedback: str
    top_issue: str
    routing_hint: RoutingHint
    revision_count: int

    # supervisor writes next_agent, the conditional edge routes on it
    next_agent: str
    agent_path: list[str]
    # one entry per supervisor call — the trajectory evaluator's only input (ADR-017)
    decision_log: list[dict[str, Any]]

    # researcher — per task_id: {iterations_used, self_terminated}. Powers react_discipline;
    # loguru sees every ReAct step, but only this survives to be scored after the run ends
    react_stats: dict[str, dict[str, Any]]

    # evaluator
    final_report: str
    evaluation_scores: dict[str, float]

    # set by any node that failed; the supervisor sees it and routes to FINISH
    error: str | None


def new_state(query: str, session_id: str | None = None) -> GraphState:
    """Build a complete initial state. Never hand-build this dict elsewhere."""
    return GraphState(
        session_id=session_id or new_session_id(),
        original_query=query.strip(),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        query_depth=DEFAULT_DEPTH,
        answerable=True,
        clarifying_question="",
        report_sections=[],
        word_target=0,
        triage_reason="",
        research_plan=[],
        research_strategy="",
        raw_research=[],
        research_quality=0.0,
        analyzed_data="",
        code_outputs=[],
        draft_report="",
        citations=[],
        critic_scores={},
        quality_score=0.0,
        critic_feedback="",
        top_issue="",
        routing_hint="",
        revision_count=0,
        next_agent="",
        agent_path=[],
        decision_log=[],
        react_stats={},
        final_report="",
        evaluation_scores={},
        error=None,
    )


def source_count(state: GraphState) -> int:
    """Unique sources gathered so far. The supervisor routes on this number."""
    return len({item.get("url") for item in state["raw_research"] if item.get("url")})

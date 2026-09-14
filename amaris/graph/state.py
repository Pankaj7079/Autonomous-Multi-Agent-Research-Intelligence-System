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
    # true when the user asked to go deeper, so the depth is already decided and triage skips
    depth_locked: bool
    answerable: bool
    clarifying_question: str
    report_sections: list[str]
    word_target: int
    triage_reason: str

    # prior turns in this conversation, oldest first: {"query": ..., "answer": ...}
    history: list[dict[str, str]]
    # the question as a standalone sentence — "what about his brother?" is useless as a
    # search string, so triage resolves it against history and everything that hunts for
    # sources uses this instead. The answer still addresses original_query.
    resolved_query: str

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


SEEDABLE = ("raw_research", "query_depth", "depth_locked", "history")


def new_state(
    query: str, session_id: str | None = None, *, seed: dict[str, Any] | None = None
) -> GraphState:
    """Build a complete initial state. Never hand-build this dict elsewhere.

    `seed` carries a few fields forward from a previous turn — the sources already gathered, the
    depth an expand click locked in, and the conversation so far. Everything else still blanks,
    so a follow-up is a fresh run that happens to know what came before.
    """
    state = GraphState(
        session_id=session_id or new_session_id(),
        original_query=query.strip(),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        query_depth=DEFAULT_DEPTH,
        depth_locked=False,
        history=[],
        resolved_query="",
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
    for key in SEEDABLE:
        if seed and key in seed:
            state[key] = seed[key]  # type: ignore[literal-required]
    return state


def subject(state: GraphState) -> str:
    """What to search and score sources against. Falls back to the query as the user typed it."""
    return state.get("resolved_query") or state["original_query"]


def source_count(state: GraphState) -> int:
    """Unique sources gathered so far. The supervisor routes on this number."""
    return len({item.get("url") for item in state["raw_research"] if item.get("url")})

"""Graph wiring. Forced hops are fixed edges; the supervisor owns the two real branch points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END

# GraphState must exist at runtime: langgraph resolves the route function's annotation
from amaris.graph.state import (
    ANALYST,
    CLARIFY,
    FINISH,
    PLANNER,
    RESEARCHER,
    TRIAGE,
    WRITER,
    GraphState,
)
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from langgraph.graph import StateGraph

EVALUATOR = "evaluator"
SUPERVISOR = "supervisor"
CRITIC_NODE = "critic"

# where the supervisor's two gates may send work; FINISH is the only key that leaves the loop
ROUTE_MAP = {
    PLANNER: PLANNER,
    RESEARCHER: RESEARCHER,
    ANALYST: ANALYST,
    WRITER: WRITER,
    FINISH: EVALUATOR,
}

TRIAGE_MAP = {PLANNER: PLANNER, CLARIFY: CLARIFY}


def route_from_triage(state: GraphState) -> str:
    """An unanswerable question is asked back instead of researched — it never reaches an agent."""
    if state.get("answerable", True):
        return PLANNER
    logger.bind(question=state.get("clarifying_question", "")[:80]).info("edges.clarify")
    return CLARIFY


def route_from_supervisor(state: GraphState) -> str:
    """Reads the value the supervisor wrote into next_agent. Nothing else decides a branch."""
    chosen = state.get("next_agent") or FINISH
    if chosen not in ROUTE_MAP:
        # the supervisor validates its own output, so this is a last line of defence
        logger.bind(next_agent=chosen).warning("edges.unknown_route")
        return FINISH
    return chosen


def register_edges(workflow: StateGraph) -> None:
    """Wire the graph. Only two edges are conditional, and both sit on a real decision."""
    workflow.add_conditional_edges(TRIAGE, route_from_triage, TRIAGE_MAP)
    workflow.add_edge(CLARIFY, END)

    # a plan always needs researching and research always needs writing up — asking a model
    # to confirm that cost a call per hop and never once chose differently
    workflow.add_edge(PLANNER, RESEARCHER)
    workflow.add_edge(ANALYST, WRITER)
    workflow.add_edge(WRITER, CRITIC_NODE)

    # gate 1 after research, gate 2 after review
    workflow.add_edge(RESEARCHER, SUPERVISOR)
    workflow.add_edge(CRITIC_NODE, SUPERVISOR)
    workflow.add_conditional_edges(SUPERVISOR, route_from_supervisor, ROUTE_MAP)

    workflow.add_edge(EVALUATOR, END)

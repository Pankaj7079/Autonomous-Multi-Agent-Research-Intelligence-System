"""The only routing in the system. One conditional edge, and every agent returns to supervisor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END

# GraphState must exist at runtime: langgraph resolves the route function's annotation
from amaris.graph.state import AGENTS, FINISH, GraphState
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from langgraph.graph import StateGraph

EVALUATOR = "evaluator"

# FINISH is the only key that leaves the agent loop
ROUTE_MAP = {**{agent: agent for agent in AGENTS}, FINISH: EVALUATOR}


def route_from_supervisor(state: GraphState) -> str:
    """Reads the value an LLM wrote into next_agent. Nothing else decides the path."""
    chosen = state.get("next_agent") or FINISH
    if chosen not in ROUTE_MAP:
        # the supervisor validates its own output, so this is a last line of defence
        logger.bind(next_agent=chosen).warning("edges.unknown_route")
        return FINISH
    return chosen


def register_edges(workflow: StateGraph) -> None:
    """Wire the graph: supervisor fans out, every agent comes back, evaluator ends the run."""
    workflow.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

    for agent in AGENTS:
        workflow.add_edge(agent, "supervisor")

    workflow.add_edge(EVALUATOR, END)

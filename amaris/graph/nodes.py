"""Node wrappers. These never raise — a partial report beats a stack trace."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from amaris.agents import (
    AnalystAgent,
    CriticAgent,
    PlannerAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from amaris.evaluation.report_eval import ReportEvaluator
from amaris.evaluation.retrieval_eval import RetrievalEvaluator

# GraphState must exist at runtime: langgraph resolves node annotations when compiling
from amaris.graph.state import FINISH, GraphState
from amaris.memory.mem0_memory import add_session_summary
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.agents.base_agent import BaseAgent


async def _run_node(agent_class: type[BaseAgent], state: GraphState) -> dict[str, Any]:
    """Run one agent and convert any failure into state["error"] for the supervisor to see."""
    name = agent_class.name
    logger.bind(node=name).debug("node.start")
    started = time.perf_counter()

    try:
        update = await agent_class().run(state)
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        logger.bind(node=name, ms=elapsed).exception("node.failed")
        # the supervisor reads error and routes to FINISH, so the run ends cleanly
        return {"error": f"{name} failed: {exc}"}

    logger.bind(node=name, ms=round((time.perf_counter() - started) * 1000, 1)).info(
        "node.complete"
    )
    return update


async def supervisor_node(state: GraphState) -> dict[str, Any]:
    """Decides next_agent. If this one fails the run cannot continue, so it finishes."""
    update = await _run_node(SupervisorAgent, state)
    if "error" in update:
        update["next_agent"] = FINISH
    return update


async def planner_node(state: GraphState) -> dict[str, Any]:
    return await _run_node(PlannerAgent, state)


async def researcher_node(state: GraphState) -> dict[str, Any]:
    return await _run_node(ResearcherAgent, state)


async def analyst_node(state: GraphState) -> dict[str, Any]:
    return await _run_node(AnalystAgent, state)


async def writer_node(state: GraphState) -> dict[str, Any]:
    return await _run_node(WriterAgent, state)


async def critic_node(state: GraphState) -> dict[str, Any]:
    return await _run_node(CriticAgent, state)


async def evaluator_node(state: GraphState) -> dict[str, Any]:
    """Terminal node: promotes the draft to final, scores Layers 1+2 inline, records the session.

    Layer 3 (trajectory) needs the full decision_log a finished run leaves behind and is meant
    for offline regression runs, not per-request cost — it runs from the harness only (ADR-017).
    """
    report = state["draft_report"]
    if not report and state["error"]:
        report = f"This run could not be completed: {state['error']}"
    state = {**state, "final_report": report}

    # scoring is advisory, so a failure here returns zeros rather than costing us a finished report
    retrieval_results = await RetrievalEvaluator().evaluate(state)
    report_results = await ReportEvaluator().evaluate(state)
    scores = {r.metric: r.score for r in (*retrieval_results, *report_results)}
    scores["overall"] = round(sum(scores.values()) / len(scores), 4) if scores else 0.0

    await add_session_summary(state["original_query"], report, scores)

    logger.bind(
        chars=len(report),
        sources=len(state["raw_research"]),
        score=round(state["quality_score"], 2),
        eval_overall=scores["overall"],
        path=" → ".join(state["agent_path"]),
    ).info("run.complete")

    return {"final_report": report, "evaluation_scores": scores}

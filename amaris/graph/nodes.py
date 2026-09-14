"""Node wrappers. These never raise — a partial report beats a stack trace."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from amaris.agents import (
    AnalystAgent,
    CriticAgent,
    PlannerAgent,
    ResearcherAgent,
    SupervisorAgent,
    TriageAgent,
    WriterAgent,
)
from amaris.evaluation.report_eval import ReportEvaluator
from amaris.evaluation.retrieval_eval import RetrievalEvaluator

# GraphState must exist at runtime: langgraph resolves node annotations when compiling
from amaris.graph.state import FINISH, GraphState
from amaris.memory.episodic import add_session_summary
from amaris.observability.logging import logger
from amaris.safety.guardrails import validate_output

if TYPE_CHECKING:
    from amaris.agents.base_agent import BaseAgent

CLARIFY_TEMPLATE = """## Answer

I need one more detail before I can research this.

**{question}**

{reason}"""


async def _run_node(
    agent_class: type[BaseAgent], state: GraphState, track_path: bool = True
) -> dict[str, Any]:
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

    # agent_path is "who actually ran": with forced hops as direct edges, the supervisor no
    # longer sees every step, so each node has to record its own
    if track_path:
        update["agent_path"] = [*state["agent_path"], name]

    logger.bind(node=name, ms=round((time.perf_counter() - started) * 1000, 1)).info(
        "node.complete"
    )
    return update


async def triage_node(state: GraphState) -> dict[str, Any]:
    """Runs first. Everything downstream reads the depth it sets."""
    return await _run_node(TriageAgent, state)


async def clarify_node(state: GraphState) -> dict[str, Any]:
    """Terminal, and deliberately free — asking for a missing detail must not cost a model call."""
    question = state["clarifying_question"] or "Could you add a bit more detail to that question?"
    report = CLARIFY_TEMPLATE.format(
        question=question,
        reason=state["triage_reason"] or "",
    ).strip()
    logger.bind(question=question[:120]).info("clarify.asked")
    return {
        "final_report": report,
        "draft_report": report,
        "next_agent": FINISH,
        "agent_path": [*state["agent_path"], "clarify"],
    }


async def supervisor_node(state: GraphState) -> dict[str, Any]:
    """Decides next_agent. If this one fails the run cannot continue, so it finishes."""
    # not tracked in agent_path: _gate reads the last entry to know which question it is answering
    update = await _run_node(SupervisorAgent, state, track_path=False)
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


async def _safe_scores(state: GraphState) -> dict[str, float]:
    """Layers 1+2, or {} if the judge is unavailable. Scoring must never cost us a finished report."""
    try:
        # gather, not sequential — two independent judges ran back to back and added ~25s
        # to every run, on top of a report that was already finished
        retrieval_results, report_results = await asyncio.gather(
            RetrievalEvaluator().evaluate(state), ReportEvaluator().evaluate(state)
        )
    except Exception as exc:
        logger.bind(error=str(exc)[:200]).warning("evaluator.scoring_failed")
        return {}

    # a metric the judge never scored is omitted, not stored as 0.0 — "not scored" and
    # "scored zero" mean opposite things and the ui cannot tell them apart downstream
    scores = {r.metric: r.score for r in (*retrieval_results, *report_results) if r.scored}
    scores["overall"] = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
    return scores


async def evaluator_node(state: GraphState) -> dict[str, Any]:
    """Terminal node: promotes the draft to final, scores Layers 1+2 inline, records the session.

    Layer 3 (trajectory) needs the full decision_log a finished run leaves behind and is meant
    for offline regression runs, not per-request cost — it runs from the harness only (ADR-017).
    """
    report = state["draft_report"]
    if not report and state["error"]:
        report = f"This run could not be completed: {state['error']}"
    # scraped pages can carry personal data into the draft, so it is masked before it ships
    report = validate_output(report).text
    # promoted before scoring: a judge failure used to take the finished report down with it
    state = {**state, "final_report": report}

    scores = await _safe_scores(state)

    try:
        await add_session_summary(state["original_query"], report, scores)
    except Exception as exc:
        logger.bind(error=str(exc)[:150]).warning("evaluator.memory_failed")

    logger.bind(
        chars=len(report),
        sources=len(state["raw_research"]),
        score=round(state["quality_score"], 2),
        eval_overall=scores.get("overall", 0.0),
        path=" → ".join(state["agent_path"]),
    ).info("run.complete")

    return {
        "final_report": report,
        "evaluation_scores": scores,
        "agent_path": [*state["agent_path"], "evaluator"],
    }

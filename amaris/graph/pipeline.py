"""Builds and compiles the agentic graph. Checkpointed so a crashed run can resume."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.graph import START, StateGraph

from amaris.config.settings import get_settings
from amaris.graph.edges import EVALUATOR, register_edges
from amaris.graph.nodes import (
    analyst_node,
    clarify_node,
    critic_node,
    evaluator_node,
    planner_node,
    researcher_node,
    supervisor_node,
    triage_node,
    writer_node,
)
from amaris.graph.state import GraphState, new_state
from amaris.observability.context import bind_session
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langgraph.graph.state import CompiledStateGraph

NODES = {
    "triage": triage_node,
    "clarify": clarify_node,
    "supervisor": supervisor_node,
    "planner": planner_node,
    "researcher": researcher_node,
    "analyst": analyst_node,
    "writer": writer_node,
    "critic": critic_node,
    EVALUATOR: evaluator_node,
}

_pipeline: CompiledStateGraph | None = None
_checkpointer: Any = None


def build_graph() -> StateGraph:
    """The graph shape. Kept separate from compilation so tests can inspect it."""
    workflow = StateGraph(GraphState)
    for name, node in NODES.items():
        workflow.add_node(name, node)

    # triage runs first: nothing else should be spent before we know what the question needs
    workflow.add_edge(START, "triage")
    register_edges(workflow)
    return workflow


async def _build_checkpointer() -> Any:
    """AsyncSqliteSaver on a long-lived connection. Sync SqliteSaver would block the loop."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = get_settings().sqlite_checkpoint_db
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(connection)
    await saver.setup()
    logger.bind(db=path).debug("pipeline.checkpointer_ready")
    return saver


async def get_pipeline() -> CompiledStateGraph:
    """Compiled graph singleton for this process."""
    global _pipeline, _checkpointer
    if _pipeline is not None:
        return _pipeline

    _checkpointer = await _build_checkpointer()
    _pipeline = build_graph().compile(checkpointer=_checkpointer)
    logger.bind(nodes=len(NODES)).info("pipeline.ready")
    return _pipeline


async def reset_pipeline() -> None:
    """Drop the singleton and close the checkpoint connection. Used by tests and shutdown."""
    global _pipeline, _checkpointer
    if _checkpointer is not None and hasattr(_checkpointer, "conn"):
        await _checkpointer.conn.close()
    _pipeline = None
    _checkpointer = None


def run_config(session_id: str) -> dict[str, Any]:
    """thread_id is the session, so a resumed run picks up its own checkpoints."""
    settings = get_settings()
    return {
        "configurable": {"thread_id": session_id},
        # each supervisor hop costs two graph steps, so the cap needs headroom
        "recursion_limit": settings.max_supervisor_steps * 2 + 5,
    }


async def run_research(query: str, session_id: str | None = None) -> GraphState:
    """Run the whole pipeline to completion and return the final state."""
    state = new_state(query, session_id=session_id)
    resolved = bind_session(state["session_id"])
    pipeline = await get_pipeline()

    logger.bind(query=query[:120], session_id=resolved).info("run.start")
    final: GraphState = await pipeline.ainvoke(state, config=run_config(resolved))
    return final


async def stream_research(
    query: str, session_id: str | None = None
) -> AsyncIterator[tuple[str, dict[str, Any], GraphState]]:
    """Yield (node, delta, state_so_far) per node transition until the run ends.

    One generator, two transports: the API publishes each event to pubsub, Streamlit in cloud
    mode consumes it directly. Raises whatever the graph raises — the caller marks the job failed.
    """
    state = new_state(query, session_id=session_id)
    resolved = bind_session(state["session_id"])
    pipeline = await get_pipeline()

    logger.bind(query=query[:120], session_id=resolved).info("run.stream_start")
    async for chunk in pipeline.astream(state, config=run_config(resolved), stream_mode="updates"):
        for node, delta in chunk.items():
            # GraphState declares no reducers, so a plain merge is exactly what langgraph did
            if delta:
                state = {**state, **delta}
            yield node, delta or {}, state


async def _main() -> None:
    """uv run python -m amaris.graph.pipeline --query "What is LangGraph?" """
    import argparse

    from amaris.observability.logging import configure_from_settings

    parser = argparse.ArgumentParser(description="run the AMARIS pipeline once")
    parser.add_argument("--query", required=True, help="the research question")
    parser.add_argument("--session-id", default=None, help="resume an earlier run")
    args = parser.parse_args()

    configure_from_settings()
    state = await run_research(args.query, session_id=args.session_id)

    logger.bind(
        path=" → ".join(state["agent_path"]),
        sources=len(state["raw_research"]),
        quality=round(state["research_quality"], 2),
        score=round(state["quality_score"], 2),
        revisions=state["revision_count"],
        error=state["error"] or "none",
    ).info("run.summary")

    # a 1200 word report does not belong in a log line, so it goes to a file
    report = state["final_report"] or state["draft_report"]
    out = Path(get_settings().log_dir) / f"report_{state['session_id']}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report or "no report was produced", encoding="utf-8")

    logger.bind(session_id=state["session_id"], chars=len(report), file=str(out)).info(
        "run.report_ready"
    )
    await reset_pipeline()


if __name__ == "__main__":
    asyncio.run(_main())

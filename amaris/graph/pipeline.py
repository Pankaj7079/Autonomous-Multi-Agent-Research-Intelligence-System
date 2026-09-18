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
    live_node,
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
    "live": live_node,
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
    await _prune_checkpoints(connection)
    logger.bind(db=path).debug("pipeline.checkpointer_ready")
    return saver


async def _prune_checkpoints(connection: Any) -> None:
    """Keep only the newest N threads. Checkpoints exist to resume a crashed run, so a
    finished run from last week is dead weight — and nothing ever deleted it: 79 runs had
    grown the file to 14MB, which on a container is a volume that fills up silently.
    """
    keep = get_settings().checkpoint_keep_threads
    try:
        cursor = await connection.execute(
            "SELECT thread_id FROM checkpoints GROUP BY thread_id "
            "ORDER BY MAX(rowid) DESC LIMIT -1 OFFSET ?",
            (keep,),
        )
        stale = [row[0] for row in await cursor.fetchall()]
        if not stale:
            return

        # pruning uses fixed SQL — rare operation, not worth dynamic queries
        for thread_id in stale:
            await connection.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            await connection.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        await connection.commit()
        logger.bind(pruned=len(stale), kept=keep).info("pipeline.checkpoints_pruned")
    except Exception as exc:
        # a run must never fail to start because housekeeping did
        logger.bind(error=str(exc)[:200]).warning("pipeline.prune_failed")


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
    from amaris.tools.scraper_tool import close_scraper

    # the scraper now keeps one browser alive for the process, so shutdown has to end it
    await close_scraper()
    if _checkpointer is not None and hasattr(_checkpointer, "conn"):
        await _checkpointer.conn.close()
    _pipeline = None
    _checkpointer = None


def run_config(session_id: str) -> dict[str, Any]:
    """thread_id is the session, so a resumed run picks up its own checkpoints."""

    settings = get_settings()
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        # each supervisor hop costs two graph steps, so the cap needs headroom
        "recursion_limit": settings.max_supervisor_steps * 2 + 5,
    }
    return config


async def run_research(
    query: str, session_id: str | None = None, *, seed: dict[str, Any] | None = None
) -> GraphState:
    """Run the whole pipeline to completion and return the final state."""
    state = new_state(query, session_id=session_id, seed=seed)
    resolved = bind_session(state["session_id"])
    pipeline = await get_pipeline()

    logger.bind(query=query[:120], session_id=resolved).info("run.start")
    final: GraphState = await pipeline.ainvoke(state, config=run_config(resolved))
    return final


async def stream_research(
    query: str, session_id: str | None = None, *, seed: dict[str, Any] | None = None
) -> AsyncIterator[tuple[str, dict[str, Any], GraphState]]:
    """Yield (node, delta, state_so_far) per node transition until the run ends.

    One generator, two transports: the API publishes each event to pubsub, Streamlit in cloud
    mode consumes it directly. Raises whatever the graph raises — the caller marks the job failed.
    """
    state = new_state(query, session_id=session_id, seed=seed)
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

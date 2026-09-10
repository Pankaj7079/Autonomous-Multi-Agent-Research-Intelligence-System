"""Smoke-check the logging stack: uv run python -m amaris.observability."""

from __future__ import annotations

from amaris.observability.context import agent_context, bind_session
from amaris.observability.logging import configure_logging, logger, timed


def main() -> None:
    configure_logging(level="DEBUG")
    session_id = bind_session()

    logger.bind(query="What is LangGraph?").info("run.start")

    with agent_context("supervisor"):
        logger.bind(next_agent="researcher", quality=0.0, revisions=0).info("supervisor.route")

    with agent_context("researcher"), timed("node.researcher", node="researcher") as fields:
        logger.bind(task_id="t1", iteration=1, action="web_search").debug("researcher.react_step")
        logger.bind(provider="groq", to="cerebras", reason="429").warning("llm.fallback")
        fields["sources"] = 7

    with agent_context("critic"):
        logger.bind(overall=0.81, routing_hint="approve").info("critic.score")

    try:
        with agent_context("writer"), timed("node.writer", node="writer"):
            raise RuntimeError("deliberate failure so errors.jsonl has a sample")
    except RuntimeError:
        pass  # we wanted the log record, not the exception

    logger.bind(total_ms=1234, agent_path="planner researcher writer critic FINISH").info(
        "run.complete"
    )
    logger.bind(session_id=session_id).info("selftest.done")


if __name__ == "__main__":
    main()

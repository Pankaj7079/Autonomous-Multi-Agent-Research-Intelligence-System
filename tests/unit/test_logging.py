"""Behaviour tests for the logging layer. Contract in docs/OBSERVABILITY.md."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from amaris.observability.context import agent_context, bind_session, get_agent
from amaris.observability.logging import configure_logging, logger, timed


@pytest.fixture
def json_log(tmp_path: Path) -> Path:
    """Fresh sinks in a temp dir, so each test reads only its own records."""
    configure_logging(level="DEBUG", log_dir=tmp_path, json_enabled=True, force=True)
    return tmp_path / "amaris.jsonl"


def _records(path: Path) -> list[dict]:
    logger.complete()  # enqueue=True writes on a background thread — flush it first
    return [json.loads(line)["record"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_session_and_agent_land_on_every_record(json_log: Path) -> None:
    bind_session("abc12345")
    with agent_context("researcher"):
        logger.info("researcher.react_step")

    record = _records(json_log)[-1]
    assert record["extra"]["session_id"] == "abc12345"
    assert record["extra"]["agent"] == "researcher"


def test_agent_context_restores_previous_agent() -> None:
    with agent_context("supervisor"):
        with agent_context("writer"):
            assert get_agent() == "writer"
        assert get_agent() == "supervisor"


async def test_parallel_tasks_keep_separate_agent_context() -> None:
    """gather() is how the researcher runs tasks — context must not leak across them."""

    async def run(name: str) -> str:
        with agent_context(name):
            await asyncio.sleep(0)
            return get_agent()

    assert await asyncio.gather(run("a"), run("b")) == ["a", "b"]


def test_timed_logs_duration_and_extra_fields(json_log: Path) -> None:
    with timed("node.researcher", node="researcher") as fields:
        fields["sources"] = 7

    record = _records(json_log)[-1]
    assert record["message"] == "node.researcher.complete"
    assert record["extra"]["sources"] == 7
    assert record["extra"]["ms"] >= 0


def test_timed_logs_failure_and_reraises(json_log: Path) -> None:
    with pytest.raises(RuntimeError), timed("node.writer", node="writer"):
        raise RuntimeError("boom")

    record = _records(json_log)[-1]
    assert record["message"] == "node.writer.failed"
    assert record["level"]["name"] == "ERROR"


def test_configure_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Streamlit reruns the script on every interaction — double config must not double sinks."""
    configure_logging(level="INFO", log_dir=tmp_path, json_enabled=False, force=True)
    configure_logging(level="INFO", log_dir=tmp_path, json_enabled=False)
    logger.info("once.only")

    assert capsys.readouterr().err.count("once.only") == 1

"""Graph shape, routing and the node error boundary. No LLM calls here."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.graph import nodes as nodes_module
from amaris.graph.edges import EVALUATOR, ROUTE_MAP, route_from_supervisor, route_from_triage
from amaris.graph.nodes import clarify_node, evaluator_node, researcher_node, supervisor_node
from amaris.graph.pipeline import build_graph, run_config
from amaris.graph.state import AGENTS, CLARIFY, FINISH, PLANNER, TRIAGE, new_state

# critic is not here on purpose: writer → critic is a fixed edge, never a routing choice
ROUTABLE = tuple(a for a in ROUTE_MAP if a != FINISH)

# ── edges ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("agent", ROUTABLE)
def test_every_gate_target_is_routable(agent: str) -> None:
    state = new_state("q")
    state["next_agent"] = agent
    assert route_from_supervisor(state) == agent


def test_finish_routes_to_the_evaluator_not_to_end() -> None:
    """The evaluator must run before the graph ends, or no final_report is ever set."""
    assert ROUTE_MAP[FINISH] == EVALUATOR


def test_unknown_next_agent_falls_back_to_finish() -> None:
    """A bad key here would raise inside langgraph instead of ending the run."""
    state = new_state("q")
    state["next_agent"] = "banana"
    assert route_from_supervisor(state) == FINISH


def test_empty_next_agent_falls_back_to_finish() -> None:
    assert route_from_supervisor(new_state("q")) == FINISH


# ── graph shape ───────────────────────────────────────────────────────────


def test_graph_has_every_node() -> None:
    compiled = build_graph().compile()
    names = {n for n in compiled.get_graph().nodes if not n.startswith("__")}
    assert names == {*AGENTS, TRIAGE, CLARIFY, "supervisor", EVALUATOR}


def test_triage_runs_before_anything_is_spent() -> None:
    """Nothing may run before the question has been sized."""
    compiled = build_graph().compile()
    edges = {(e.source, e.target) for e in compiled.get_graph().edges}
    assert ("__start__", TRIAGE) in edges


def test_forced_hops_are_edges_not_routing_calls() -> None:
    """A plan always needs researching; asking a model to confirm that cost a call per hop."""
    compiled = build_graph().compile()
    edges = {(e.source, e.target) for e in compiled.get_graph().edges}
    assert ("planner", "researcher") in edges
    assert ("analyst", "writer") in edges
    assert ("writer", "critic") in edges
    assert ("planner", "supervisor") not in edges


def test_the_supervisor_sits_on_exactly_the_two_real_gates() -> None:
    """Research sufficiency and post-review direction are the only undetermined branches."""
    compiled = build_graph().compile()
    into_supervisor = {e.source for e in compiled.get_graph().edges if e.target == "supervisor"}
    assert into_supervisor == {"researcher", "critic"}


def test_an_unanswerable_query_never_reaches_an_agent() -> None:
    state = new_state("tell about today weather")
    state["answerable"] = False
    assert route_from_triage(state) == CLARIFY

    state["answerable"] = True
    assert route_from_triage(state) == PLANNER


async def test_clarify_asks_a_question_and_spends_nothing() -> None:
    """The clarify node makes no LLM call at all — that is what makes it ~3s."""
    state = new_state("tell about today weather")
    state["clarifying_question"] = "Which city should I look up the weather for?"
    update = await clarify_node(state)

    assert "Which city" in update["final_report"]
    assert update["next_agent"] == FINISH
    assert update["agent_path"] == ["clarify"]


def test_recursion_limit_has_headroom_for_the_step_cap() -> None:
    """Each supervisor hop costs two graph steps, so the limit must exceed 2x the cap."""
    config = run_config("abc123")
    assert config["configurable"]["thread_id"] == "abc123"
    assert config["recursion_limit"] > 2 * 15


# ── node error boundary ───────────────────────────────────────────────────


async def test_node_converts_an_exception_into_error_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nodes never raise. A stack trace out of here would kill the whole graph."""

    class Exploding:
        name = "researcher"

        async def run(self, state: Any) -> dict[str, Any]:
            raise RuntimeError("search backend died")

    monkeypatch.setattr(nodes_module, "ResearcherAgent", Exploding)
    update = await researcher_node(new_state("q"))

    assert "search backend died" in update["error"]


async def test_supervisor_node_forces_finish_when_it_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If routing itself breaks, the graph has no next edge — it must end, not stall."""

    class Exploding:
        name = "supervisor"

        async def run(self, state: Any) -> dict[str, Any]:
            raise RuntimeError("groq is down")

    monkeypatch.setattr(nodes_module, "SupervisorAgent", Exploding)
    update = await supervisor_node(new_state("q"))

    assert update["next_agent"] == FINISH
    assert "groq is down" in update["error"]


async def test_evaluator_promotes_the_draft_to_final() -> None:
    state = new_state("q")
    state["draft_report"] = "## Executive Summary\nfindings [1]"
    assert (await evaluator_node(state))["final_report"] == state["draft_report"]


async def test_evaluator_explains_a_failed_run_instead_of_returning_nothing() -> None:
    """A user who asked a question deserves a reason, not an empty string."""
    state = new_state("q")
    state["error"] = "researcher failed: rate limited"
    report = (await evaluator_node(state))["final_report"]

    assert "could not be completed" in report
    assert "rate limited" in report


async def test_the_evaluator_spends_no_model_call_on_a_finished_report(monkeypatch) -> None:
    """RAGAS used to run here on every question — ~40s for four numbers the reader never asked
    for, after the answer was already written. It now benchmarks the golden set offline, and
    the per-request signal is a citation audit that cannot rate-limit because it is pure."""
    from amaris.graph import nodes

    async def no_memory(*args, **kwargs):
        return None

    monkeypatch.setattr(nodes, "add_session_summary", no_memory)

    state = new_state("what is mango?")
    state["draft_report"] = "India grew 24.7 million tonnes of mango in 2023 [1]."
    state["citations"] = [{"index": 1, "url": "https://a.com", "title": "Mango"}]
    state["raw_research"] = [
        {"url": "https://a.com", "content": "India grew 24.7 million tonnes of mango in 2023."}
    ]

    result = await nodes.evaluator_node(state)

    audit = result["citation_audit"]
    assert audit["checked"] == 1
    assert audit["grounded"] == 1
    assert audit["dead"] == 0
    # the evaluator no longer produces judge scores at all
    assert "evaluation_scores" not in result


async def test_a_claim_whose_page_does_not_back_it_is_named_not_averaged(monkeypatch) -> None:
    """The reason RAGAS went: one faithfulness number averages away the claim that is wrong.
    A report can score 0.94 as a whole and 0.61 per claim."""
    from amaris.graph import nodes

    async def no_memory(*args, **kwargs):
        return None

    monkeypatch.setattr(nodes, "add_session_summary", no_memory)

    state = new_state("what is mango?")
    state["draft_report"] = "The fruit can weigh 2.5 kg [1]. It cures diabetes in 92% of cases [2]."
    state["citations"] = [
        {"index": 1, "url": "https://a.com"},
        {"index": 2, "url": "https://b.com"},
    ]
    state["raw_research"] = [
        {"url": "https://a.com", "content": "A mango can weigh 2.5 kg when ripe."},
        {"url": "https://b.com", "content": "An article about container orchestration."},
    ]

    audit = (await nodes.evaluator_node(state))["citation_audit"]

    assert audit["grounded"] == 1 and audit["unsupported"] == 1
    unsupported = [c for c in audit["claims"] if not c["grounded"]]
    assert "diabetes" in unsupported[0]["text"]
    # and the answer names the figure it could not find, rather than looking as confident
    # as a verified one
    assert "92%" in audit["caveat"]


async def test_checkpoint_pruning_keeps_only_recent_threads(tmp_path, monkeypatch) -> None:
    """Nothing ever deleted checkpoints: 79 runs had grown the file to 14MB. They exist to
    resume a crashed run, so finished ones from last week are dead weight."""
    import aiosqlite

    from amaris.config.settings import get_settings
    from amaris.graph.pipeline import _prune_checkpoints

    get_settings.cache_clear()
    monkeypatch.setenv("CHECKPOINT_KEEP_THREADS", "2")

    db = tmp_path / "ckpt.db"
    async with aiosqlite.connect(db) as conn:
        await conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        await conn.execute("CREATE TABLE writes (thread_id TEXT)")
        for thread in ("old1", "old2", "keep1", "keep2"):
            await conn.execute("INSERT INTO checkpoints VALUES (?)", (thread,))
            await conn.execute("INSERT INTO writes VALUES (?)", (thread,))
        await conn.commit()

        await _prune_checkpoints(conn)

        cursor = await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
        remaining = {row[0] for row in await cursor.fetchall()}

    assert remaining == {"keep1", "keep2"}
    get_settings.cache_clear()


async def test_a_prune_failure_never_stops_a_run_from_starting(tmp_path) -> None:
    """Housekeeping is not worth failing a research run over."""
    import aiosqlite

    from amaris.graph.pipeline import _prune_checkpoints

    async with aiosqlite.connect(tmp_path / "empty.db") as conn:
        # no checkpoints table at all — must be swallowed, not raised
        await _prune_checkpoints(conn)

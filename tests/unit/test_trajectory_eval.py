"""Layer 3 is pure state math — every metric here must be checkable without an LLM call."""

from __future__ import annotations

from amaris.agents.supervisor import expected_route
from amaris.evaluation.trajectory_eval import TrajectoryEvaluator
from amaris.graph.state import ANALYST, CRITIC, FINISH, PLANNER, RESEARCHER, WRITER, new_state


def entry(
    step: int,
    from_agent: str,
    to_agent: str,
    *,
    llm_decided: bool = True,
    expected_agent: str | None = None,
    matched_rule: str = "rule_x",
    research_quality: float = 0.7,
    source_count: int = 8,
    quality_score: float = 0.0,
    analysis_done: bool = False,
    draft_exists: bool = False,
    revision_count: int = 0,
    plan_exists: bool = True,
    routing_hint: str = "none",
    error: str = "none",
) -> dict:
    return {
        "step": step,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "llm_decided": llm_decided,
        "expected_agent": expected_agent if expected_agent is not None else to_agent,
        "matched_rule": matched_rule,
        "reasoning": f"{matched_rule} → expected {expected_agent or to_agent}",
        "plan_exists": plan_exists,
        "research_quality": research_quality,
        "source_count": source_count,
        "analysis_done": analysis_done,
        "draft_exists": draft_exists,
        "quality_score": quality_score,
        "revision_count": revision_count,
        "routing_hint": routing_hint,
        "error": error,
    }


# ── expected_route (the rule table, re-derived) ─────────────────────────────


def test_expected_route_covers_the_documented_priority_order() -> None:
    assert expected_route(False, 0.0, 0, False, False, 0.0, 0, "none", 0.6, 0.72, 2) == (
        PLANNER,
        "rule_3_no_plan",
    )
    assert (
        expected_route(True, 0.0, 0, False, False, 0.0, 0, "need_more_research", 0.6, 0.72, 2)[0]
        == RESEARCHER
    )
    assert (
        expected_route(True, 0.0, 0, False, False, 0.0, 0, "fix_writing", 0.6, 0.72, 2)[0] == WRITER
    )
    assert expected_route(True, 0.3, 2, False, False, 0.0, 0, "none", 0.6, 0.72, 2)[0] == RESEARCHER
    assert expected_route(True, 0.8, 8, False, False, 0.0, 0, "none", 0.6, 0.72, 2)[0] == ANALYST
    assert expected_route(True, 0.8, 8, True, False, 0.0, 0, "none", 0.6, 0.72, 2)[0] == WRITER
    assert expected_route(True, 0.8, 8, True, True, 0.0, 0, "none", 0.6, 0.72, 2)[0] == CRITIC
    assert expected_route(True, 0.8, 8, True, True, 0.9, 0, "none", 0.6, 0.72, 2)[0] == FINISH
    assert expected_route(True, 0.8, 8, True, True, 0.4, 0, "none", 0.6, 0.72, 2)[0] == WRITER


def test_hint_rules_outrank_the_thin_research_rule() -> None:
    """Rule 4/5 sit above rule 6 — a hint must win even when research also looks thin."""
    agent, rule = expected_route(True, 0.1, 1, False, False, 0.0, 0, "fix_writing", 0.6, 0.72, 2)
    assert (agent, rule) == (WRITER, "rule_5_hint_fix_writing")


# ── routing_accuracy ─────────────────────────────────────────────────────────


async def test_routing_accuracy_is_perfect_when_every_choice_matches_the_rule() -> None:
    state = new_state("q")
    state["decision_log"] = [
        entry(1, "START", PLANNER, matched_rule="rule_3_no_plan"),
        entry(2, PLANNER, RESEARCHER, matched_rule="rule_6_thin_research"),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    routing = next(r for r in results if r.metric == "routing_accuracy")
    assert routing.score == 1.0
    assert routing.passed


async def test_routing_accuracy_catches_a_supervisor_that_ignores_the_rules() -> None:
    """This is the metric that would have caught a supervisor routing on vibes."""
    state = new_state("q")
    state["decision_log"] = [
        # LLM chose analyst when the rule said researcher — a real mismatch
        entry(1, "START", ANALYST, expected_agent=RESEARCHER, matched_rule="rule_6_thin_research"),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    routing = next(r for r in results if r.metric == "routing_accuracy")
    assert routing.score == 0.0
    assert "rule_6_thin_research" in routing.detail


async def test_routing_accuracy_ignores_code_decided_terminal_steps() -> None:
    """Rules 1-2 are never a routing decision the LLM made — scoring them would be meaningless."""
    state = new_state("q")
    state["decision_log"] = [
        entry(
            1,
            "writer",
            FINISH,
            llm_decided=False,
            expected_agent=FINISH,
            matched_rule="rule_1_error",
        ),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    routing = next(r for r in results if r.metric == "routing_accuracy")
    assert routing.detail == "no LLM-decided steps to check"


async def test_no_decision_log_returns_a_zero_result_not_a_crash() -> None:
    state = new_state("q")
    results = await TrajectoryEvaluator().evaluate(state)
    assert results[0].score == 0.0
    assert "no decision_log" in results[0].detail


# ── research_convergence ─────────────────────────────────────────────────────


async def test_research_convergence_rewards_steady_improvement() -> None:
    state = new_state("q")
    state["decision_log"] = [
        entry(1, RESEARCHER, ANALYST, research_quality=0.3),
        entry(2, RESEARCHER, ANALYST, research_quality=0.6),
        entry(3, RESEARCHER, ANALYST, research_quality=0.8),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    convergence = next(r for r in results if r.metric == "research_convergence")
    assert convergence.score == 1.0


async def test_research_convergence_penalises_a_flat_or_regressing_loop() -> None:
    """The ReAct loop ran twice and learned nothing — that's the unhealthy case this catches."""
    state = new_state("q")
    state["decision_log"] = [
        entry(1, RESEARCHER, ANALYST, research_quality=0.5),
        entry(2, RESEARCHER, ANALYST, research_quality=0.4),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    convergence = next(r for r in results if r.metric == "research_convergence")
    assert convergence.score == 0.0
    assert not convergence.passed


async def test_research_convergence_needs_at_least_two_calls_to_judge() -> None:
    state = new_state("q")
    state["decision_log"] = [entry(1, RESEARCHER, ANALYST, research_quality=0.5)]
    results = await TrajectoryEvaluator().evaluate(state)
    convergence = next(r for r in results if r.metric == "research_convergence")
    assert convergence.score == 1.0
    assert "only one researcher call" in convergence.detail


# ── loop_efficiency ───────────────────────────────────────────────────────────


async def test_loop_efficiency_penalises_a_step_that_changed_nothing() -> None:
    state = new_state("q")
    state["decision_log"] = [
        entry(1, "START", RESEARCHER, research_quality=0.3, source_count=2),
        # researcher ran again but the snapshot is identical — wasted
        entry(2, RESEARCHER, RESEARCHER, research_quality=0.3, source_count=2),
        entry(3, RESEARCHER, ANALYST, research_quality=0.7, source_count=8),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    efficiency = next(r for r in results if r.metric == "loop_efficiency")
    assert efficiency.score == 0.5  # 1 wasted out of 2 transitions


async def test_loop_efficiency_is_perfect_when_every_step_changes_something() -> None:
    state = new_state("q")
    state["decision_log"] = [
        entry(1, "START", RESEARCHER, research_quality=0.3, source_count=2),
        entry(2, RESEARCHER, ANALYST, research_quality=0.7, source_count=8),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    efficiency = next(r for r in results if r.metric == "loop_efficiency")
    assert efficiency.score == 1.0


# ── termination_quality ────────────────────────────────────────────────────


async def test_termination_quality_rewards_a_clean_approval() -> None:
    state = new_state("q")
    state["decision_log"] = [entry(1, "critic", FINISH)]
    state["quality_score"] = 0.9
    state["raw_research"] = [{"url": f"https://x.com/{n}"} for n in range(5)]
    results = await TrajectoryEvaluator().evaluate(state)
    termination = next(r for r in results if r.metric == "termination_quality")
    assert termination.score == 1.0


async def test_termination_quality_penalises_an_error() -> None:
    state = new_state("q")
    state["decision_log"] = [entry(1, "researcher", FINISH)]
    state["error"] = "researcher failed: rate limited"
    results = await TrajectoryEvaluator().evaluate(state)
    termination = next(r for r in results if r.metric == "termination_quality")
    assert termination.score == 0.0


async def test_termination_quality_flags_too_few_sources() -> None:
    state = new_state("q")
    state["decision_log"] = [entry(1, "critic", FINISH)]
    state["quality_score"] = 0.9
    state["raw_research"] = [{"url": "https://x.com/1"}]
    results = await TrajectoryEvaluator().evaluate(state)
    termination = next(r for r in results if r.metric == "termination_quality")
    assert termination.score == 0.3


async def test_termination_quality_flags_the_revision_cap() -> None:
    state = new_state("q")
    state["decision_log"] = [entry(1, "critic", FINISH)]
    state["quality_score"] = 0.5
    state["revision_count"] = 2
    state["raw_research"] = [{"url": f"https://x.com/{n}"} for n in range(5)]
    results = await TrajectoryEvaluator().evaluate(state)
    termination = next(r for r in results if r.metric == "termination_quality")
    assert termination.score == 0.5


# ── react_discipline ─────────────────────────────────────────────────────────


async def test_react_discipline_rewards_self_termination() -> None:
    state = new_state("q")
    state["decision_log"] = [entry(1, "planner", RESEARCHER)]
    state["react_stats"] = {
        "t1": {"iterations_used": 2, "self_terminated": True},
        "t2": {"iterations_used": 3, "self_terminated": True},
    }
    results = await TrajectoryEvaluator().evaluate(state)
    discipline = next(r for r in results if r.metric == "react_discipline")
    assert discipline.score == 1.0


async def test_react_discipline_flags_every_task_hitting_the_cap() -> None:
    """Hitting the cap every time means the stop condition isn't working."""
    state = new_state("q")
    state["decision_log"] = [entry(1, "planner", RESEARCHER)]
    state["react_stats"] = {
        "t1": {"iterations_used": 4, "self_terminated": False},
        "t2": {"iterations_used": 4, "self_terminated": False},
    }
    results = await TrajectoryEvaluator().evaluate(state)
    discipline = next(r for r in results if r.metric == "react_discipline")
    assert discipline.score == 0.0
    assert "never fired" in discipline.detail


async def test_react_discipline_with_no_researcher_tasks() -> None:
    state = new_state("q")
    state["decision_log"] = [entry(1, "planner", ANALYST)]
    results = await TrajectoryEvaluator().evaluate(state)
    discipline = next(r for r in results if r.metric == "react_discipline")
    assert discipline.score == 0.0
    assert "no researcher tasks" in discipline.detail

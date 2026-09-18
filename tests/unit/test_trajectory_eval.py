"""Layer 3 is pure state math — every metric here must be checkable without an LLM call."""

from __future__ import annotations

from amaris.agents.supervisor import expected_after_research, expected_after_review
from amaris.evaluation.trajectory_eval import TrajectoryEvaluator
from amaris.graph.state import (
    ANALYST,
    APPROVE,
    FINISH,
    NEED_MORE_RESEARCH,
    PLANNER,
    RESEARCHER,
    WRITER,
    WRONG_TOPIC,
    new_state,
)


def entry(
    step: int,
    from_agent: str,
    to_agent: str,
    *,
    llm_decided: bool = True,
    llm_called: bool | None = None,
    expected_agent: str | None = None,
    matched_rule: str = "some_rule",
    gate: str = "research_gate",
    research_quality: float = 0.7,
    source_count: int = 8,
    quality_score: float = 0.0,
    analysis_done: bool = False,
    draft_exists: bool = False,
    revision_count: int = 0,
    plan_exists: bool = True,
    routing_hint: str = "none",
    error: str = "none",
    shadow_choice: str | None = None,
    shadow_agreed: bool | None = None,
    shadow_error: str | None = None,
) -> dict:
    return {
        "step": step,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "llm_decided": llm_decided,
        # mirrors supervisor._decide's own default: undecided means the audit call still ran
        "llm_called": llm_decided if llm_called is None else llm_called,
        "expected_agent": expected_agent if expected_agent is not None else to_agent,
        "matched_rule": matched_rule,
        "gate": gate,
        "reasoning": f"{gate}: chose {to_agent}",
        "plan_exists": plan_exists,
        "research_quality": research_quality,
        "source_count": source_count,
        "analysis_done": analysis_done,
        "draft_exists": draft_exists,
        "quality_score": quality_score,
        "revision_count": revision_count,
        "routing_hint": routing_hint,
        "error": error,
        "shadow_choice": shadow_choice,
        "shadow_agreed": shadow_agreed,
        "shadow_error": shadow_error,
    }


# ── the routing invariants (weak expectations, not a rule table) ─────────────


def test_the_research_invariant_reads_the_obvious_cases() -> None:
    assert expected_after_research(0, 0.0, 0, 0.6) == (RESEARCHER, "no_sources")
    assert expected_after_research(5, 0.3, 1, 0.6)[0] == RESEARCHER
    assert expected_after_research(8, 0.8, 1, 0.6)[0] == ANALYST
    # a gate that can always ask for more research is an unbounded bill
    assert expected_after_research(2, 0.1, 3, 0.6)[0] == ANALYST


def test_the_review_invariant_reads_the_obvious_cases() -> None:
    assert expected_after_review(APPROVE, 0.9, 0, 0.72, 2)[0] == FINISH
    assert expected_after_review(NEED_MORE_RESEARCH, 0.4, 0, 0.72, 2)[0] == RESEARCHER
    assert expected_after_review(WRONG_TOPIC, 0.3, 0, 0.72, 2)[0] == PLANNER
    assert expected_after_review("none", 0.4, 0, 0.72, 2)[0] == WRITER


def test_the_revision_cap_outranks_every_other_reading() -> None:
    """Nothing may keep a run going once it has spent its revisions."""
    agent, rule = expected_after_review(NEED_MORE_RESEARCH, 0.1, 2, 0.72, 2)
    assert (agent, rule) == (FINISH, "revision_cap")


# ── routing_agreement and routing_economy ────────────────────────────────────


async def test_agreement_is_perfect_when_every_choice_matches_the_invariant() -> None:
    state = new_state("q")
    state["decision_log"] = [
        entry(1, RESEARCHER, ANALYST, matched_rule="research_sufficient"),
        entry(2, "critic", FINISH, matched_rule="approved", gate="review_gate"),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    routing = next(r for r in results if r.metric == "routing_agreement")
    assert routing.score == 1.0


async def test_agreement_reports_where_the_model_took_a_different_view() -> None:
    """Disagreement is informative — judging 20 off-topic sources thin SHOULD differ from a count."""
    state = new_state("q")
    state["decision_log"] = [
        entry(
            1, RESEARCHER, ANALYST, expected_agent=RESEARCHER, matched_rule="below_quality_floor"
        ),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    routing = next(r for r in results if r.metric == "routing_agreement")
    assert routing.score == 0.0
    assert "below_quality_floor" in routing.detail
    # not a failing grade: the gate is allowed to disagree with the naive reading
    assert routing.passed


async def test_agreement_ignores_decisions_no_model_made() -> None:
    state = new_state("q")
    state["decision_log"] = [
        entry(1, WRITER, FINISH, llm_decided=False, matched_rule="error_set", gate="review_gate"),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    routing = next(r for r in results if r.metric == "routing_agreement")
    assert "no model call was needed" in routing.detail


async def test_economy_measures_how_many_hops_made_no_model_call_at_all() -> None:
    """Keyed on llm_called, not llm_decided: a settled decision still pays for an audit call
    it cannot act on (ADR-050), so only a decision that skips the audit entirely — error_set —
    is genuinely free."""
    state = new_state("q")
    state["decision_log"] = [
        entry(1, WRITER, FINISH, llm_decided=False, llm_called=False, matched_rule="error_set"),
        entry(
            2,
            "critic",
            WRITER,
            llm_decided=False,
            llm_called=True,
            matched_rule="settled_quality_met",
            gate="review_gate",
        ),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    economy = next(r for r in results if r.metric == "routing_economy")
    assert economy.score == 0.5
    assert "1/2" in economy.detail


# ── shadow_agreement: the audit ADR-050 pays for ─────────────────────────────


async def test_shadow_agreement_scores_how_often_the_audit_call_agreed() -> None:
    state = new_state("q")
    state["decision_log"] = [
        entry(
            1,
            RESEARCHER,
            ANALYST,
            llm_decided=False,
            llm_called=True,
            matched_rule="settled_quality_met",
            shadow_choice=ANALYST,
            shadow_agreed=True,
        ),
        entry(
            2,
            "critic",
            FINISH,
            llm_decided=False,
            llm_called=True,
            matched_rule="revision_cap",
            gate="review_gate",
            shadow_choice=WRITER,
            shadow_agreed=False,
        ),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    shadow = next(r for r in results if r.metric == "shadow_agreement")
    assert shadow.score == 0.5
    assert "1/2" in shadow.detail


async def test_shadow_agreement_is_perfect_when_nothing_was_settled_to_audit() -> None:
    """Only genuine judgement calls ran — there is no settled decision to have a second
    opinion on, so this must read as trivially satisfied, not as a missing metric."""
    state = new_state("q")
    state["decision_log"] = [entry(1, RESEARCHER, ANALYST, llm_decided=True)]
    results = await TrajectoryEvaluator().evaluate(state)
    shadow = next(r for r in results if r.metric == "shadow_agreement")
    assert shadow.score == 1.0
    assert "no settled decision was audited" in shadow.detail


async def test_shadow_agreement_reports_when_every_audit_call_failed() -> None:
    """A rate limit on the audit call must show up as a fact, not silently score as agreement."""
    state = new_state("q")
    state["decision_log"] = [
        entry(
            1,
            RESEARCHER,
            ANALYST,
            llm_decided=False,
            llm_called=True,
            matched_rule="settled_quality_met",
            shadow_choice=None,
            shadow_error="rate limited",
        ),
    ]
    results = await TrajectoryEvaluator().evaluate(state)
    shadow = next(r for r in results if r.metric == "shadow_agreement")
    assert shadow.score == 0.0
    assert "every audit call failed" in shadow.detail


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

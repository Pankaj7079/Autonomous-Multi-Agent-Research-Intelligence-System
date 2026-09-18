"""Layer 3 — scores the PATH, not the answer. Custom because nothing off-the-shelf does this (ADR-017).

Runs from the harness only, on `decision_log` + `react_stats` — never inline on a real session,
since it needs a whole finished run to look back over. Zero LLM calls: every metric here is a
deterministic function of state the pipeline already recorded, which is what makes it free to run
on every golden-set query without touching quota.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from amaris.agents.supervisor import expected_after_research, expected_after_review
from amaris.config.settings import get_settings
from amaris.evaluation.base import EvalResult
from amaris.graph.state import source_count
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

LAYER = "trajectory"


def _routing_agreement(decision_log: list[dict]) -> EvalResult:
    """How often the gate agreed with the obvious reading of the state.

    Deliberately NOT called accuracy: the invariants in supervisor.py are a weak expectation,
    not a rule the prompt mirrors, and a gate judging that twenty off-topic sources are worse
    than five good ones SHOULD disagree with a source count. Low agreement is a prompt to go
    read the mismatches, not a failing grade.
    """
    llm_steps = [entry for entry in decision_log if entry.get("llm_decided")]
    if not llm_steps:
        detail = "every routing decision was settled by state — no model call was needed"
        return EvalResult(LAYER, "routing_agreement", 1.0, True, detail)

    matches = sum(1 for entry in llm_steps if entry["to_agent"] == entry["expected_agent"])
    score = round(matches / len(llm_steps), 4)
    mismatches = [
        f"step {e['step']}: chose {e['to_agent']}, invariant expected {e['expected_agent']} ({e['matched_rule']})"
        for e in llm_steps
        if e["to_agent"] != e["expected_agent"]
    ]
    detail = f"{matches}/{len(llm_steps)} model decisions matched the invariant"
    if mismatches:
        detail += "; " + "; ".join(mismatches[:3])
    return EvalResult(LAYER, "routing_agreement", score, True, detail)


def _routing_economy(decision_log: list[dict]) -> EvalResult:
    """What share of routing decisions made no model call at all.

    Keyed on llm_called, not llm_decided: a settled decision still asks the model for an audit
    opinion it is not allowed to act on (ADR-050), and that call has a real cost even though the
    model's answer never became the route. llm_decided alone would report this as still free.
    """
    if not decision_log:
        return EvalResult(LAYER, "routing_economy", 0.0, False, "no routing decisions were made")

    free = sum(1 for entry in decision_log if not entry.get("llm_called"))
    score = round(free / len(decision_log), 4)
    detail = f"{free}/{len(decision_log)} routing decisions made no model call at all"
    return EvalResult(LAYER, "routing_economy", score, score >= 0.5, detail)


def _shadow_agreement(decision_log: list[dict]) -> EvalResult:
    """When code already had the answer, did the model's own opinion agree with it?

    This is the audit ADR-050 pays for: every settled decision still asks the model, and this
    checks whether that second opinion would have chosen the same route. It never changed what
    happened — this metric is what tells you whether the extra spend was worth anything.
    """
    audited = [e for e in decision_log if e.get("llm_called") and not e.get("llm_decided")]
    if not audited:
        return EvalResult(LAYER, "shadow_agreement", 1.0, True, "no settled decision was audited")

    answered = [e for e in audited if e.get("shadow_choice")]
    if not answered:
        return EvalResult(
            LAYER, "shadow_agreement", 0.0, False, "every audit call failed to answer"
        )

    agreed = sum(1 for e in answered if e.get("shadow_agreed"))
    score = round(agreed / len(answered), 4)
    detail = f"{agreed}/{len(answered)} audited decisions had the model agree with the rule"
    failed = len(audited) - len(answered)
    if failed:
        detail += f"; {failed} audit call(s) failed and are excluded"
    return EvalResult(LAYER, "shadow_agreement", score, score >= 0.5, detail)


def _research_convergence(decision_log: list[dict]) -> EvalResult:
    """Quality across successive researcher calls should climb. Flat or oscillating means the
    ReAct loop isn't learning from its own results — it's just spending iterations."""
    from amaris.graph.state import RESEARCHER

    qualities = [e["research_quality"] for e in decision_log if e["from_agent"] == RESEARCHER]
    if len(qualities) < 2:
        detail = "only one researcher call — convergence needs at least two to compare"
        return EvalResult(LAYER, "research_convergence", 1.0, True, detail)

    diffs = [b - a for a, b in pairwise(qualities)]
    improved = sum(1 for d in diffs if d > 0)
    regressed = sum(1 for d in diffs if d < 0)
    score = round(max(0.0, improved / len(diffs) - 0.5 * (regressed / len(diffs))), 4)
    detail = f"quality across calls: {qualities} ({improved} up, {regressed} down)"
    return EvalResult(LAYER, "research_convergence", score, score >= 0.5, detail)


_SNAPSHOT_KEYS = (
    "research_quality",
    "source_count",
    "quality_score",
    "analysis_done",
    "draft_exists",
)


def _loop_efficiency(decision_log: list[dict]) -> EvalResult:
    """A step is wasted if the agent it ran changed nothing the supervisor could see."""
    if len(decision_log) < 2:
        return EvalResult(LAYER, "loop_efficiency", 1.0, True, "too few steps to have a loop")

    wasted = 0
    for prev, cur in pairwise(decision_log):
        if all(prev[key] == cur[key] for key in _SNAPSHOT_KEYS):
            wasted += 1

    total = len(decision_log) - 1
    score = round(1.0 - wasted / total, 4)
    return EvalResult(
        LAYER, "loop_efficiency", score, score >= 0.7, f"{wasted}/{total} steps changed nothing"
    )


def _termination_quality(state: GraphState) -> EvalResult:
    """Did it stop for the right reason, or just run out of road."""
    settings = get_settings()
    sources = source_count(state)

    if state.get("error"):
        return EvalResult(
            LAYER, "termination_quality", 0.0, False, f"errored out: {state['error'][:100]}"
        )
    if sources < 4:
        return EvalResult(
            LAYER, "termination_quality", 0.3, False, f"finished with only {sources} sources"
        )
    if state["quality_score"] >= settings.quality_approve_threshold:
        return EvalResult(
            LAYER, "termination_quality", 1.0, True, f"approved at {state['quality_score']:.2f}"
        )
    if state["revision_count"] >= settings.max_revisions:
        return EvalResult(
            LAYER, "termination_quality", 0.5, False, "hit the revision cap, not a clean finish"
        )
    return EvalResult(
        LAYER, "termination_quality", 0.4, False, "ended without matching a documented reason"
    )


def _react_discipline(react_stats: dict[str, dict]) -> EvalResult:
    """Self-terminating via sufficient=true means the stop condition is real. Always hitting the
    cap means it isn't — the agent never decides, it just runs until code cuts it off."""
    if not react_stats:
        return EvalResult(LAYER, "react_discipline", 0.0, False, "no researcher tasks ran")

    settings = get_settings()
    terminated = sum(1 for s in react_stats.values() if s.get("self_terminated"))
    at_cap = sum(
        1 for s in react_stats.values() if s.get("iterations_used") == settings.max_react_iterations
    )
    score = round(terminated / len(react_stats), 4)
    detail = f"{terminated}/{len(react_stats)} tasks self-stopped"
    if at_cap == len(react_stats):
        detail += " — every task hit the iteration cap, the stop condition never fired"
    return EvalResult(LAYER, "react_discipline", score, score >= 0.5, detail)


class TrajectoryEvaluator:
    """Harness-only: needs the full decision_log a finished run leaves behind."""

    async def evaluate(self, state: GraphState) -> list[EvalResult]:
        decision_log = state["decision_log"]
        if not decision_log:
            logger.bind(session=state["session_id"]).debug("trajectory_eval.no_decision_log")
            return [
                EvalResult(LAYER, "routing_agreement", 0.0, False, "no decision_log on this run")
            ]

        results = [
            _routing_agreement(decision_log),
            _routing_economy(decision_log),
            _shadow_agreement(decision_log),
            _research_convergence(decision_log),
            _loop_efficiency(decision_log),
            _termination_quality(state),
            _react_discipline(state["react_stats"]),
        ]
        logger.bind(**{r.metric: r.score for r in results}).info("trajectory_eval.scored")
        return results


# re-exported so a caller can check the supervisor's routing invariants independently
__all__ = ["TrajectoryEvaluator", "expected_after_research", "expected_after_review"]

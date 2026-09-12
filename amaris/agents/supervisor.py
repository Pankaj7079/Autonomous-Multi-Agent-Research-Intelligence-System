"""THE AGENTIC CORE. An LLM reads state and picks the next agent. No routing lives elsewhere."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from amaris.agents.base_agent import BaseAgent
from amaris.graph.state import (
    AGENTS,
    ANALYST,
    CRITIC,
    FINISH,
    FIX_WRITING,
    NEED_MORE_RESEARCH,
    PLANNER,
    RESEARCHER,
    WRITER,
    source_count,
)
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

ALLOWED = (*AGENTS, FINISH)

PROMPT = """You are the orchestrator of AMARIS, an agentic research system.
Your only job: decide which agent runs next. Output one word, nothing else.

Available agents:
  planner    — breaks the query into concrete research tasks
  researcher — searches the web and recalls memory
  analyst    — synthesizes research into structured insight
  writer     — writes the final cited report
  critic     — reviews report quality
  FINISH     — triggers evaluation and ends the run

Current state:
  query: {original_query}
  plan exists: {plan_exists}
  research quality (0-1, 0=none): {research_quality}
  sources found: {source_count}
  analysis complete: {analysis_done}
  draft exists: {draft_exists}
  quality score (0=unscored): {quality_score}
  revisions so far: {revision_count}
  critic routing hint: {routing_hint}
  error: {error}

Decision rules, highest priority first:
  1. error is set                                    → FINISH
  2. revision_count >= {max_revisions}               → FINISH
  3. no plan yet                                     → planner
  4. routing_hint == "need_more_research"            → researcher
  5. routing_hint == "fix_writing"                   → writer
  6. research_quality < {quality_floor} OR source_count < 4     → researcher
  7. research ok, analysis_done false                → analyst
  8. analysis done, no draft                         → writer
  9. draft exists, quality_score == 0                → critic
 10. quality_score >= {approve_floor}                → FINISH
 11. quality_score < {approve_floor} and revisions < {max_revisions} → writer

Output exactly one of: planner researcher analyst writer critic FINISH"""


def expected_route(
    plan_exists: bool,
    research_quality: float,
    src_count: int,
    analysis_done: bool,
    draft_exists: bool,
    quality_score: float,
    revision_count: int,
    routing_hint: str,
    quality_floor: float,
    approve_floor: float,
    max_revisions: int,
) -> tuple[str, str]:
    """Rules 3-11 from PROMPT, re-derived in plain code.

    Single source of truth for "what should the supervisor have picked here" — used to build
    the decision_log's reasoning field, and reused as-is by trajectory_eval.py's routing_accuracy
    so the two never drift apart. Rules 1-2 are terminal_reason's job: the LLM never sees that
    state, so they are not "routing decisions" this function needs to reproduce.
    """
    if not plan_exists:
        return PLANNER, "rule_3_no_plan"
    if routing_hint == NEED_MORE_RESEARCH:
        return RESEARCHER, "rule_4_hint_need_more_research"
    if routing_hint == FIX_WRITING:
        return WRITER, "rule_5_hint_fix_writing"
    if research_quality < quality_floor or src_count < 4:
        return RESEARCHER, "rule_6_thin_research"
    if not analysis_done:
        return ANALYST, "rule_7_no_analysis"
    if not draft_exists:
        return WRITER, "rule_8_no_draft"
    if quality_score == 0:
        return CRITIC, "rule_9_unscored"
    if quality_score >= approve_floor:
        return FINISH, "rule_10_approved"
    if quality_score < approve_floor and revision_count < max_revisions:
        return WRITER, "rule_11_needs_revision"
    # unreachable given rule 2's cap runs first in _terminal_reason, but a pure function returns
    return FINISH, "rule_fallthrough"


class SupervisorAgent(BaseAgent):
    """Writes next_agent. Every agent returns here, so this is the only place routing happens."""

    name = "supervisor"
    task_type = "supervisor"

    def _terminal_reason(self, state: GraphState) -> str | None:
        """Conditions where code decides, not the LLM — an LLM must not guard a billing loop."""
        if state.get("error"):
            return "error_set"
        if state["revision_count"] >= self.settings.max_revisions:
            return "revision_cap"
        if len(state["agent_path"]) >= self.settings.max_supervisor_steps:
            return "step_cap"
        return None

    def _snapshot(self, state: GraphState) -> dict[str, Any]:
        """Raw values behind every prompt field and every routing rule — built once, used twice."""
        return {
            "plan_exists": bool(state["research_plan"]),
            "research_quality": round(state["research_quality"], 2),
            "source_count": source_count(state),
            "analysis_done": bool(state["analyzed_data"]),
            "draft_exists": bool(state["draft_report"]),
            "quality_score": round(state["quality_score"], 2),
            "revision_count": state["revision_count"],
            "routing_hint": state["routing_hint"] or "none",
            "error": state.get("error") or "none",
        }

    def _build_prompt(self, state: GraphState, snapshot: dict[str, Any]) -> str:
        return PROMPT.format(
            original_query=state["original_query"],
            max_revisions=self.settings.max_revisions,
            quality_floor=self.settings.research_quality_threshold,
            approve_floor=self.settings.quality_approve_threshold,
            **snapshot,
        )

    def _log_decision(
        self,
        state: GraphState,
        *,
        to_agent: str,
        llm_decided: bool,
        snapshot: dict[str, Any],
        matched_rule: str,
        expected_agent: str,
    ) -> dict[str, Any]:
        """One decision_log entry — the only input trajectory_eval.py's Layer 3 needs."""
        return {
            "step": len(state["decision_log"]) + 1,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "from_agent": state["agent_path"][-1] if state["agent_path"] else "START",
            "to_agent": to_agent,
            "llm_decided": llm_decided,
            "expected_agent": expected_agent,
            "matched_rule": matched_rule,
            # the supervisor prompt outputs one word, nothing else — there is no real llm
            # reasoning to log, so this states which documented rule justifies the expected pick
            "reasoning": f"{matched_rule} → expected {expected_agent}",
            **snapshot,
        }

    async def _run(self, state: GraphState) -> dict[str, Any]:
        reason = self._terminal_reason(state)
        if reason:
            snapshot = self._snapshot(state)
            entry = self._log_decision(
                state,
                to_agent=FINISH,
                llm_decided=False,
                snapshot=snapshot,
                matched_rule=reason,
                expected_agent=FINISH,
            )
            logger.bind(next_agent=FINISH, reason=reason).info("supervisor.route")
            return {
                "next_agent": FINISH,
                "agent_path": [*state["agent_path"], FINISH],
                "decision_log": [*state["decision_log"], entry],
            }

        snapshot = self._snapshot(state)
        raw = await self._invoke(self._build_prompt(state, snapshot))
        chosen = self._match(raw)
        expected_agent, matched_rule = expected_route(
            snapshot["plan_exists"],
            snapshot["research_quality"],
            snapshot["source_count"],
            snapshot["analysis_done"],
            snapshot["draft_exists"],
            snapshot["quality_score"],
            snapshot["revision_count"],
            snapshot["routing_hint"],
            self.settings.research_quality_threshold,
            self.settings.quality_approve_threshold,
            self.settings.max_revisions,
        )
        entry = self._log_decision(
            state,
            to_agent=chosen,
            llm_decided=True,
            snapshot=snapshot,
            matched_rule=matched_rule,
            expected_agent=expected_agent,
        )

        logger.bind(
            next_agent=chosen,
            expected=expected_agent if expected_agent != chosen else None,
            quality=snapshot["research_quality"],
            score=snapshot["quality_score"],
            revisions=snapshot["revision_count"],
            hint=snapshot["routing_hint"],
            sources=snapshot["source_count"],
            raw=raw[:40] if chosen != raw else None,
        ).info("supervisor.route")

        return {
            "next_agent": chosen,
            "agent_path": [*state["agent_path"], chosen],
            "decision_log": [*state["decision_log"], entry],
        }

    def _match(self, raw: str) -> str:
        """Map a reply onto the allowed set. Anything unrecognised ends the run safely."""
        cleaned = raw.strip().strip(".\"'`*").lower()
        for candidate in ALLOWED:
            if cleaned == candidate.lower():
                return candidate

        # models sometimes answer in a sentence, so fall back to the first name mentioned
        for candidate in ALLOWED:
            if candidate.lower() in cleaned:
                return candidate

        logger.bind(raw=raw[:120]).warning("supervisor.unparsable")
        return FINISH

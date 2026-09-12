"""THE AGENTIC CORE. An LLM reads state and picks the next agent. No routing lives elsewhere."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amaris.agents.base_agent import BaseAgent
from amaris.graph.state import AGENTS, FINISH, source_count
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

    def _build_prompt(self, state: GraphState) -> str:
        return PROMPT.format(
            original_query=state["original_query"],
            plan_exists=bool(state["research_plan"]),
            research_quality=round(state["research_quality"], 2),
            source_count=source_count(state),
            analysis_done=bool(state["analyzed_data"]),
            draft_exists=bool(state["draft_report"]),
            quality_score=round(state["quality_score"], 2),
            revision_count=state["revision_count"],
            routing_hint=state["routing_hint"] or "none",
            error=state.get("error") or "none",
            max_revisions=self.settings.max_revisions,
            quality_floor=self.settings.research_quality_threshold,
            approve_floor=self.settings.quality_approve_threshold,
        )

    async def _run(self, state: GraphState) -> dict[str, Any]:
        reason = self._terminal_reason(state)
        if reason:
            logger.bind(next_agent=FINISH, reason=reason).info("supervisor.route")
            return {"next_agent": FINISH, "agent_path": [*state["agent_path"], FINISH]}

        raw = await self._invoke(self._build_prompt(state))
        chosen = self._match(raw)

        logger.bind(
            next_agent=chosen,
            quality=round(state["research_quality"], 2),
            score=round(state["quality_score"], 2),
            revisions=state["revision_count"],
            hint=state["routing_hint"] or "none",
            sources=source_count(state),
            raw=raw[:40] if chosen != raw else None,
        ).info("supervisor.route")

        return {"next_agent": chosen, "agent_path": [*state["agent_path"], chosen]}

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

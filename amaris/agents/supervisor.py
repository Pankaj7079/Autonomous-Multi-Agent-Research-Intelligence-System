"""THE AGENTIC CORE. Two gates where the next step is genuinely undecided — and nowhere else.

Forced hops are graph edges now. This agent is asked only the two questions state cannot
answer on its own: is the research good enough to write from, and what does a reviewed draft
need next. Code decides first whenever state already settles the answer — a hard cap, or both
signals agreeing there is nothing left to weigh — and a model is still asked for a second
opinion on every one of those settled calls, purely to audit agreement. It is never allowed to
override one: a cap that only holds when a rate-limited or malformed model call does not fire
is not a cap (ADR-050). The only place a model's own answer becomes the route is the genuinely
undetermined middle ground where no rule applies at all.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from amaris.agents.base_agent import BaseAgent
from amaris.agents.triage import budget_for
from amaris.graph.state import (
    ANALYST,
    APPROVE,
    FINISH,
    FIX_WRITING,
    NEED_MORE_RESEARCH,
    PLANNER,
    RESEARCHER,
    WRITER,
    WRONG_TOPIC,
    source_count,
)
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

RESEARCH_GATE = "research_gate"
REVIEW_GATE = "review_gate"

# what each gate is allowed to answer — the LLM never picks from the full agent list
GATE_CHOICES: dict[str, tuple[str, ...]] = {
    RESEARCH_GATE: (RESEARCHER, ANALYST),
    REVIEW_GATE: (WRITER, RESEARCHER, PLANNER, FINISH),
}

# hard cap on researcher revisits — stops paying for repeated identical pages
MAX_RESEARCH_VISITS = 3
MAX_PLAN_VISITS = 2
# below this a run has essentially nothing to write from, whatever the self-assessment said
MIN_USABLE_SOURCES = 3

RESEARCH_GATE_PROMPT = """You decide whether a research pass is good enough to write from.

The question: {original_query}
Depth this question was triaged as: {query_depth} (budget: {budget_sources} sources)

What the researcher came back with:
  sources kept after relevance filtering: {kept}
  sources discarded as off-topic: {discarded}
  researcher's own quality self-rating: {research_quality} (floor is {quality_floor})
  research passes so far: {research_visits} of {max_visits}

The best sources it found, by title:
{top_titles}

Judge whether these sources can actually answer the question that was asked.
Many sources about a neighbouring topic are worse than a few about this one.
Another pass is only worth its cost if there is a specific gap it would close.

Answer with one word:
  researcher — go back and search again
  analyst    — enough to work with, move on

Output exactly one of: researcher analyst"""

REVIEW_GATE_PROMPT = """A draft has been reviewed. Decide what it needs next.

The question: {original_query}
Reviewer scores: {scores}
Overall: {quality_score} (approval floor is {approve_floor})
Does it answer the question that was asked: {answer_fit}
The single biggest problem: {top_issue}
Revisions so far: {revision_count} of {max_revisions}

Pick the cheapest fix that addresses the real problem:
  writer     — the facts are there, the writing is the problem
  researcher — the writing is fine, the evidence is missing
  planner    — it answers a different question than the one asked, so the
               research tasks themselves were wrong
  FINISH     — good enough, or nothing further can realistically be fixed

Output exactly one of: writer researcher planner FINISH"""


def expected_after_research(
    src_count: int, research_quality: float, research_visits: int, quality_floor: float
) -> tuple[str, str]:
    """Weak invariant for the research gate — what a reasonable decision looks like here.

    Not a rule table the prompt mirrors. It exists so the decision_log can record whether the
    model agreed with the obvious reading, which trajectory_eval reports as routing_agreement.
    Disagreement is informative, not a failure: judging that 20 off-topic sources are worse
    than 5 good ones is exactly the call we want a model making.
    """
    if src_count == 0:
        return RESEARCHER, "no_sources"
    if research_visits >= MAX_RESEARCH_VISITS:
        return ANALYST, "research_visits_spent"
    if research_quality < quality_floor:
        return RESEARCHER, "below_quality_floor"
    return ANALYST, "research_sufficient"


def expected_after_review(
    routing_hint: str,
    quality_score: float,
    revision_count: int,
    approve_floor: float,
    max_revisions: int,
) -> tuple[str, str]:
    """Weak invariant for the review gate. Same contract as expected_after_research."""
    if revision_count >= max_revisions:
        return FINISH, "revision_cap"
    if routing_hint == WRONG_TOPIC:
        return PLANNER, "hint_wrong_topic"
    if quality_score >= approve_floor or routing_hint == APPROVE:
        return FINISH, "approved"
    if routing_hint == NEED_MORE_RESEARCH:
        return RESEARCHER, "hint_need_more_research"
    if routing_hint == FIX_WRITING:
        return WRITER, "hint_fix_writing"
    return WRITER, "below_approve_floor"


class SupervisorAgent(BaseAgent):
    """Writes next_agent at the two branch points. Everything else in the graph is a fixed edge."""

    name = "supervisor"
    task_type = "supervisor"

    def _gate(self, state: GraphState) -> str:
        """Which question is being asked — derived from who just ran, not from a flag."""
        last = state["agent_path"][-1] if state["agent_path"] else ""
        return REVIEW_GATE if last == "critic" else RESEARCH_GATE

    def _revision_cap(self, state: GraphState) -> int:
        """Depth-scaled. A brief answer does not earn the same rewrite budget as a deep one."""
        return min(self.settings.max_revisions, budget_for(state["query_depth"]).max_revisions)

    def _terminal_reason(self, state: GraphState) -> str | None:
        """Caps where code decides, not the LLM — an LLM must not guard a billing loop."""
        if state["revision_count"] >= self._revision_cap(state):
            return "revision_cap"
        if len(state["agent_path"]) >= self.settings.max_supervisor_steps:
            return "step_cap"
        return None

    def _proceed_target(self, state: GraphState) -> str:
        """Where 'enough research' leads. Shallow queries skip synthesis and go straight to writing."""
        return ANALYST if budget_for(state["query_depth"]).analysis else WRITER

    def _settled(
        self, state: GraphState, gate: str, snapshot: dict[str, Any]
    ) -> tuple[str, str] | None:
        """The answer when state already determines it, so no model call is made. None means decide.

        This is the whole point of the rewrite: a clean run reaches FINISH without the supervisor
        ever calling a model, and the calls that do happen are on genuinely contested state.
        """
        if gate == RESEARCH_GATE:
            if snapshot["source_count"] == 0 and snapshot["research_visits"] < MAX_RESEARCH_VISITS:
                return RESEARCHER, "settled_no_sources"
            if snapshot["research_visits"] >= MAX_RESEARCH_VISITS:
                return self._proceed_target(state), "settled_visits_spent"
            # both signals agree there is enough — there is nothing left to weigh
            if (
                snapshot["research_quality"] >= self.settings.research_quality_threshold
                and snapshot["source_count"] >= MIN_USABLE_SOURCES
            ):
                return self._proceed_target(state), "settled_quality_met"
            return None

        if (
            snapshot["routing_hint"] == APPROVE
            and snapshot["quality_score"] >= self.settings.quality_approve_threshold
        ):
            return FINISH, "settled_approved"
        if snapshot["routing_hint"] == WRONG_TOPIC and snapshot["plan_visits"] >= MAX_PLAN_VISITS:
            # re-planning twice and still answering the wrong question will not improve on a third
            return FINISH, "settled_replan_spent"
        return None

    def _snapshot(self, state: GraphState) -> dict[str, Any]:
        """Raw values behind every prompt field and every logged decision — built once, used twice."""
        return {
            "plan_exists": bool(state["research_plan"]),
            "research_quality": round(state["research_quality"], 2),
            "source_count": source_count(state),
            "analysis_done": bool(state["analyzed_data"]),
            "draft_exists": bool(state["draft_report"]),
            "quality_score": round(state["quality_score"], 2),
            "revision_count": state["revision_count"],
            "routing_hint": state["routing_hint"] or "none",
            "research_visits": state["agent_path"].count(RESEARCHER),
            "plan_visits": state["agent_path"].count(PLANNER),
            "query_depth": state["query_depth"],
            "error": state.get("error") or "none",
        }

    def _build_prompt(self, state: GraphState, gate: str, snapshot: dict[str, Any]) -> str:
        if gate == RESEARCH_GATE:
            budget = budget_for(state["query_depth"])
            kept = min(snapshot["source_count"], budget.max_sources)
            return RESEARCH_GATE_PROMPT.format(
                original_query=state["original_query"],
                query_depth=snapshot["query_depth"],
                budget_sources=budget.max_sources,
                kept=kept,
                discarded=max(0, len(state["raw_research"]) - kept),
                research_quality=snapshot["research_quality"],
                quality_floor=self.settings.research_quality_threshold,
                research_visits=snapshot["research_visits"],
                max_visits=MAX_RESEARCH_VISITS,
                top_titles=self._top_titles(state),
            )
        return REVIEW_GATE_PROMPT.format(
            original_query=state["original_query"],
            scores=", ".join(f"{k} {v:.2f}" for k, v in state["critic_scores"].items()) or "none",
            quality_score=snapshot["quality_score"],
            approve_floor=self.settings.quality_approve_threshold,
            answer_fit=state["critic_scores"].get("answer_fit", "not scored"),
            top_issue=state["top_issue"] or "none stated",
            revision_count=snapshot["revision_count"],
            max_revisions=self._revision_cap(state),
        )

    def _top_titles(self, state: GraphState) -> str:
        """Titles are scraped text, so they are shown as data the gate reads, never as instructions."""
        from amaris.safety.injection import wrap_untrusted

        titles = [str(item.get("title", ""))[:90] for item in state["raw_research"][:6]]
        if not titles:
            return "nothing was found"
        return wrap_untrusted("\n".join(f"- {t}" for t in titles if t))

    def _log_decision(
        self,
        state: GraphState,
        *,
        gate: str,
        to_agent: str,
        llm_decided: bool,
        llm_called: bool,
        snapshot: dict[str, Any],
        matched_rule: str,
        expected_agent: str,
        shadow_choice: str | None,
        shadow_agreed: bool | None,
        shadow_error: str | None,
    ) -> dict[str, Any]:
        """One decision_log entry — the only input trajectory_eval.py's Layer 3 needs.

        llm_decided is whether the model's answer chose the route. llm_called is whether a
        model was asked at all — the two differ now that a settled decision still asks for a
        second opinion it is not allowed to act on (ADR-050).
        """
        return {
            "step": len(state["decision_log"]) + 1,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "from_agent": state["agent_path"][-1] if state["agent_path"] else "START",
            "to_agent": to_agent,
            "gate": gate,
            "llm_decided": llm_decided,
            "llm_called": llm_called,
            "expected_agent": expected_agent,
            "matched_rule": matched_rule,
            "shadow_choice": shadow_choice,
            "shadow_agreed": shadow_agreed,
            "shadow_error": shadow_error,
            "reasoning": (
                f"{gate}: chose {to_agent}"
                + ("" if llm_decided else f" without letting the model decide ({matched_rule})")
            ),
            **snapshot,
        }

    def _expected(self, state: GraphState, gate: str, snapshot: dict[str, Any]) -> tuple[str, str]:
        if gate == RESEARCH_GATE:
            agent, rule = expected_after_research(
                snapshot["source_count"],
                snapshot["research_quality"],
                snapshot["research_visits"],
                self.settings.research_quality_threshold,
            )
            return (self._proceed_target(state) if agent == ANALYST else agent), rule
        return expected_after_review(
            snapshot["routing_hint"],
            snapshot["quality_score"],
            snapshot["revision_count"],
            self.settings.quality_approve_threshold,
            self._revision_cap(state),
        )

    async def _run(self, state: GraphState) -> dict[str, Any]:
        snapshot = self._snapshot(state)
        gate = self._gate(state)

        if state.get("error"):
            # skip audit call for caught exceptions — no sane alternative (ADR-050)
            return self._decide(state, gate, snapshot, FINISH, "error_set", llm_decided=False)

        settled = self._settled(state, gate, snapshot)
        # passing run + revision cap → approve, not give up
        if settled and settled[0] == FINISH:
            return await self._decide_settled(state, gate, snapshot, FINISH, settled[1])

        reason = self._terminal_reason(state)
        if reason:
            return await self._decide_settled(state, gate, snapshot, FINISH, reason)

        if settled:
            chosen, rule = settled
            return await self._decide_settled(state, gate, snapshot, chosen, rule)

        raw = await self._invoke(self._build_prompt(state, gate, snapshot))
        chosen = self._match(raw, gate)
        if chosen == ANALYST:
            chosen = self._proceed_target(state)
        expected_agent, matched_rule = self._expected(state, gate, snapshot)
        return self._decide(
            state,
            gate,
            snapshot,
            chosen,
            matched_rule,
            llm_decided=True,
            expected_agent=expected_agent,
            raw=raw,
        )

    async def _decide_settled(
        self, state: GraphState, gate: str, snapshot: dict[str, Any], chosen: str, rule: str
    ) -> dict[str, Any]:
        """Code already has the answer; ask the model anyway and log whether it agrees.

        The model's answer is recorded for audit, never routed on — a hard cap or a settled
        read of state must not depend on a rate-limited or malformed model call (ADR-050).
        """
        shadow_choice, shadow_error = await self._shadow(state, gate, snapshot)
        return self._decide(
            state,
            gate,
            snapshot,
            chosen,
            rule,
            llm_decided=False,
            llm_called=True,
            shadow_choice=shadow_choice,
            shadow_error=shadow_error,
        )

    async def _shadow(
        self, state: GraphState, gate: str, snapshot: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """What the model would have chosen, for comparison only. Never raises."""
        try:
            raw = await self._invoke(self._build_prompt(state, gate, snapshot))
        except Exception as exc:
            logger.bind(gate=gate, error=str(exc)[:150]).warning("supervisor.shadow_failed")
            return None, str(exc)[:200]
        chosen = self._match(raw, gate)
        if chosen == ANALYST:
            chosen = self._proceed_target(state)
        return chosen, None

    def _decide(
        self,
        state: GraphState,
        gate: str,
        snapshot: dict[str, Any],
        chosen: str,
        rule: str,
        *,
        llm_decided: bool,
        llm_called: bool | None = None,
        expected_agent: str | None = None,
        shadow_choice: str | None = None,
        shadow_error: str | None = None,
        raw: str = "",
    ) -> dict[str, Any]:
        """Build the state update for one routing decision, logged the same way either way."""
        called = llm_decided if llm_called is None else llm_called
        shadow_agreed = (shadow_choice == chosen) if shadow_choice else None
        entry = self._log_decision(
            state,
            gate=gate,
            to_agent=chosen,
            llm_decided=llm_decided,
            llm_called=called,
            snapshot=snapshot,
            matched_rule=rule,
            expected_agent=expected_agent or chosen,
            shadow_choice=shadow_choice,
            shadow_agreed=shadow_agreed,
            shadow_error=shadow_error,
        )
        logger.bind(
            gate=gate,
            next_agent=chosen,
            llm_decided=llm_decided,
            llm_called=called,
            rule=rule,
            expected=expected_agent if expected_agent and expected_agent != chosen else None,
            shadow=shadow_choice,
            shadow_agreed=shadow_agreed,
            quality=snapshot["research_quality"],
            score=snapshot["quality_score"],
            hint=snapshot["routing_hint"],
            sources=snapshot["source_count"],
            raw=raw[:40] if raw and chosen not in raw else None,
        ).info("supervisor.route")
        return {
            "next_agent": chosen,
            "decision_log": [*state["decision_log"], entry],
        }

    def _match(self, raw: str, gate: str) -> str:
        """Map a reply onto what this gate may answer. Anything unrecognised ends the run safely."""
        allowed = GATE_CHOICES[gate]
        cleaned = raw.strip().strip(".\"'`*").lower()
        for candidate in allowed:
            if cleaned == candidate.lower():
                return candidate

        # models sometimes answer in a sentence, so fall back to the first name mentioned
        for candidate in allowed:
            if candidate.lower() in cleaned:
                return candidate

        logger.bind(raw=raw[:120], gate=gate).warning("supervisor.unparsable")
        return FINISH if gate == REVIEW_GATE else allowed[-1]

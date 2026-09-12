"""Scores the draft and tells the supervisor where to route. routing_hint is the whole point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from amaris.agents.base_agent import BaseAgent
from amaris.graph.state import APPROVE, FIX_WRITING, NEED_MORE_RESEARCH
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

DIMENSIONS = ("faithfulness", "completeness", "coherence", "citation_quality")
VALID_HINTS = (NEED_MORE_RESEARCH, FIX_WRITING, APPROVE)
REPORT_CHARS = 6000

PROMPT = """You are a rigorous quality reviewer. Score 0.0-1.0 on each dimension:

faithfulness      1.0 every claim cited and supported · 0.0 many invented claims
completeness      1.0 fully answers the query · 0.0 barely addresses it
coherence         1.0 logical and clear · 0.0 disorganized
citation_quality  1.0 all cited, sources relevant · 0.0 missing or wrong

Then pick exactly one routing_hint:
  "need_more_research" — faithfulness < 0.65 because claims lack sources.
                          The problem is missing data, not bad writing.
  "fix_writing"        — completeness or coherence is the main issue.
  "approve"            — overall >= {approve_floor}, or nothing further can be fixed.

Be specific. Not "needs improvement" but:
"Section 2 states a 40% failure rate but no provided source contains that
figure — the researcher needs supporting data before writing can fix this."

Output JSON:
{{
  "scores": {{"faithfulness":0.0,"completeness":0.0,
             "coherence":0.0,"citation_quality":0.0}},
  "overall": 0.0,
  "feedback": "specific and actionable",
  "top_issue": "the single most important fix",
  "routing_hint": "need_more_research|fix_writing|approve"
}}

Query: {query}

Sources available to the writer: {source_count}

Report:
{report}"""


class CriticOutput(BaseModel):
    """Raw values only — _scores and _clamp still own the ranges."""

    model_config = ConfigDict(extra="ignore")

    scores: dict[str, Any] = Field(default_factory=dict)
    overall: Any = None
    routing_hint: str = ""
    feedback: str = ""
    top_issue: str = ""


class CriticAgent(BaseAgent):
    """Writes critic_scores, quality_score, critic_feedback, top_issue, routing_hint."""

    name = "critic"
    task_type = "critique"

    async def _run(self, state: GraphState) -> dict[str, Any]:
        prompt = PROMPT.format(
            approve_floor=self.settings.quality_approve_threshold,
            query=state["original_query"],
            source_count=len(state["raw_research"]),
            report=state["draft_report"][:REPORT_CHARS] or "no report was produced",
        )
        payload = await self._invoke_structured(prompt, CriticOutput)

        scores = self._scores(payload.scores)
        overall = self._overall(payload.overall, scores)
        hint = self._hint(payload.routing_hint, scores, overall)

        logger.bind(
            overall=overall,
            routing_hint=hint,
            revision=state["revision_count"] + 1,
            **scores,
        ).info("critic.score")

        return {
            "critic_scores": scores,
            "quality_score": overall,
            "critic_feedback": payload.feedback.strip(),
            "top_issue": payload.top_issue.strip(),
            "routing_hint": hint,
            # the critic counts the revision; the supervisor enforces the cap
            "revision_count": state["revision_count"] + 1,
        }

    def _scores(self, raw: Any) -> dict[str, float]:
        values = raw if isinstance(raw, dict) else {}
        return {name: self._clamp(values.get(name)) for name in DIMENSIONS}

    def _overall(self, raw: Any, scores: dict[str, float]) -> float:
        """Trust the model's own overall, but recompute if it gave nonsense."""
        value = self._clamp(raw)
        if value > 0.0:
            return value
        return round(sum(scores.values()) / len(DIMENSIONS), 3)

    def _hint(self, raw: Any, scores: dict[str, float], overall: float) -> str:
        """An unrecognised or self-contradictory hint would strand the supervisor."""
        cleaned = str(raw or "").strip().lower()

        # a passing score plus "fix_writing" sends the writer round forever, so trust the score
        if overall >= self.settings.quality_approve_threshold and cleaned != APPROVE:
            logger.bind(overall=overall, asked_for=cleaned[:30]).warning(
                "critic.contradictory_hint"
            )
            return APPROVE

        if cleaned in VALID_HINTS:
            return cleaned

        logger.bind(raw=cleaned[:60]).warning("critic.unparsable_hint")
        if scores["faithfulness"] < 0.65:
            return NEED_MORE_RESEARCH
        return FIX_WRITING

    def _clamp(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, round(float(value), 3)))
        except (TypeError, ValueError):
            return 0.0

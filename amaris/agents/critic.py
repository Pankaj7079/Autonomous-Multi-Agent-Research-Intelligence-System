"""Scores the draft and tells the supervisor where to route. routing_hint is the whole point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from amaris.agents.base_agent import BaseAgent
from amaris.agents.writer import SOURCE_CHARS as WRITER_SOURCE_CHARS
from amaris.graph.state import APPROVE, FIX_WRITING, NEED_MORE_RESEARCH, WRONG_TOPIC, subject
from amaris.observability.logging import logger
from amaris.tools.relevance import select_for_prompt

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

# answer_fit is the only dimension that can fail a report
DIMENSIONS = ("answer_fit", "faithfulness", "completeness", "coherence", "citation_quality")
VALID_HINTS = (NEED_MORE_RESEARCH, FIX_WRITING, WRONG_TOPIC, APPROVE)
REPORT_CHARS = 6000
# floor prevents scoring accurate reports 0.2 on faithfulness (live bug)
SOURCE_CHARS = WRITER_SOURCE_CHARS
# below this the report is not a weaker answer to the question, it is an answer to another one
ANSWER_FIT_FLOOR = 0.5

PROMPT = """You are a rigorous quality reviewer. Score 0.0-1.0 on each dimension:

answer_fit        1.0 answers the exact question asked · 0.0 answers a different
                  question. Judge this first and judge it strictly. A well-written
                  report about a related subject scores LOW here, however good it
                  looks. If the report itself admits the asked-for information is
                  unknown or unavailable, answer_fit cannot exceed 0.3.
faithfulness      1.0 every claim supported by the sources below · 0.0 many invented
completeness      1.0 covers what the question needs · 0.0 barely addresses it
coherence         1.0 logical and clear · 0.0 disorganized
citation_quality  1.0 all cited, sources relevant · 0.0 missing or wrong

Then pick exactly one routing_hint:
  "wrong_topic"        — answer_fit is low. The research tasks themselves were
                          aimed at the wrong thing; rewriting cannot fix it.
  "need_more_research" — right topic, but claims lack supporting sources.
  "fix_writing"        — the material is right, the writing is the problem.
  "approve"            — good enough, or nothing further can realistically be fixed.

Be specific. Not "needs improvement" but:
"Section 2 states a 40% failure rate but no provided source contains that
figure — the researcher needs supporting data before writing can fix this."

Output JSON:
{{
  "scores": {{"answer_fit":0.0,"faithfulness":0.0,"completeness":0.0,
             "coherence":0.0,"citation_quality":0.0}},
  "overall": 0.0,
  "feedback": "specific and actionable",
  "top_issue": "the single most important fix",
  "routing_hint": "wrong_topic|need_more_research|fix_writing|approve"
}}

Question: {query}

The sources the writer was given:
{sources}

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
            query=state["original_query"],
            sources=self._format_sources(state),
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

    def _format_sources(self, state: GraphState) -> str:
        """The critic could not previously see the evidence, so faithfulness was unverifiable."""
        from amaris.safety.injection import wrap_untrusted

        selected = select_for_prompt(
            subject(state),
            state["raw_research"],
            self.settings.max_sources_in_prompt,
            self.settings.relevance_floor,
        )
        if not selected:
            return "no sources were gathered — any specific claim in the report is unsupported"
        blocks = "\n\n".join(
            f"[{index}] {item.get('title', '')} — {item.get('url', '')}\n"
            f"{item.get('content', '')[:SOURCE_CHARS]}"
            for index, item in enumerate(selected, start=1)
        )
        return wrap_untrusted(blocks)

    def _scores(self, raw: Any) -> dict[str, float]:
        values = raw if isinstance(raw, dict) else {}
        return {name: self._clamp(values.get(name)) for name in DIMENSIONS}

    def _overall(self, raw: Any, scores: dict[str, float]) -> float:
        """Capped by answer_fit: a report that answers the wrong question cannot score well.

        The model's self-reported overall used to be taken verbatim, which is how a 2000-word
        essay about weather APIs scored 0.91 for a question about the weather.
        """
        value = self._clamp(raw)
        if value <= 0.0:
            value = round(sum(scores.values()) / len(DIMENSIONS), 3)
        return round(min(value, scores["answer_fit"]), 3)

    def _hint(self, raw: Any, scores: dict[str, float], overall: float) -> str:
        """An unrecognised or self-contradictory hint would strand the supervisor."""
        cleaned = str(raw or "").strip().lower()

        # approve + low answer_fit → override to wrong_topic, send to planner not writer
        if cleaned == APPROVE and scores["answer_fit"] < ANSWER_FIT_FLOOR:
            logger.bind(answer_fit=scores["answer_fit"]).warning("critic.approved_wrong_topic")
            return WRONG_TOPIC

        if cleaned in VALID_HINTS:
            return cleaned

        logger.bind(raw=cleaned[:60]).warning("critic.unparsable_hint")
        if scores["answer_fit"] < ANSWER_FIT_FLOOR:
            return WRONG_TOPIC
        if scores["faithfulness"] < 0.65:
            return NEED_MORE_RESEARCH
        if overall >= self.settings.quality_approve_threshold:
            return APPROVE
        return FIX_WRITING

    def _clamp(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, round(float(value), 3)))
        except (TypeError, ValueError):
            return 0.0

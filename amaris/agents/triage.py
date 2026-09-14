"""Reads the question before anything is spent and decides how much work it deserves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from amaris.agents.base_agent import BaseAgent
from amaris.graph.state import DEFAULT_DEPTH, DEPTHS
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState


@dataclass(frozen=True)
class Budget:
    """Everything downstream is allowed to spend on one query."""

    tasks: int
    react_iterations: int
    results_per_search: int
    max_sources: int
    # shallow depths go straight from researcher to writer — a synthesis pass over six
    # sources tells the writer nothing it cannot see itself, and costs a reasoning call
    analysis: bool
    word_target: int
    sections: tuple[str, ...]


# max_sources never exceeds what the writer actually reads, or we pay to gather and then discard
DEPTH_BUDGETS: dict[str, Budget] = {
    "direct": Budget(1, 1, 4, 6, False, 120, ("Answer",)),
    "brief": Budget(2, 2, 5, 8, False, 300, ("Answer", "Key Points", "References")),
    "standard": Budget(
        3, 3, 6, 12, True, 700, ("Answer", "Key Findings", "Analysis", "References")
    ),
    "deep": Budget(
        5,
        4,
        8,
        12,
        True,
        1100,
        (
            "Answer",
            "Background & Context",
            "Key Findings",
            "Analysis & Insights",
            "Conclusion & Recommendations",
            "References",
        ),
    ),
}

PROMPT = """You triage research questions before any work is spent on them.
Judge the question itself. Do not answer it and do not guess what the answer is.

Decide three things.

1. Is it answerable as written? It is NOT answerable when a parameter the answer
   depends on is missing — no location for a local question, no subject for a
   comparison, no timeframe where "now" cannot be looked up. If it is not
   answerable, write the single shortest question that would unblock it.

2. How many genuinely distinct angles does a good answer need?
     direct   — one settled fact or definition, one angle
     brief    — a couple of related points, no debate
     standard — several angles that must be weighed against each other
     deep     — contested, comparative, or needs evidence on multiple sides

3. What shape should the answer take? List the headings that actually earn their
   place. Short questions deserve short answers; do not pad one out to look thorough.

Output JSON:
{{
  "answerable": true,
  "clarifying_question": "",
  "depth": "direct|brief|standard|deep",
  "sections": ["Answer"],
  "word_target": 150,
  "reason": "one sentence on what the question needs"
}}

Question: {query}"""


class TriageOutput(BaseModel):
    """Raw triage reply. Every field has a safe default so a thin answer still runs."""

    model_config = ConfigDict(extra="ignore")

    depth: str = DEFAULT_DEPTH
    answerable: bool = True
    clarifying_question: str = ""
    sections: list[str] = Field(default_factory=list)
    word_target: int = 0
    reason: str = ""


def budget_for(depth: str) -> Budget:
    """The spend allowed at this depth. Unknown depths fall back to standard."""
    return DEPTH_BUDGETS.get(depth, DEPTH_BUDGETS[DEFAULT_DEPTH])


class TriageAgent(BaseAgent):
    """Writes query_depth, answerable, clarifying_question, report_sections, word_target."""

    name = "triage"
    task_type = "triage"

    async def _run(self, state: GraphState) -> dict[str, Any]:
        try:
            payload = await self._invoke_structured(
                PROMPT.format(query=state["original_query"]), TriageOutput
            )
        except Exception as exc:
            # broad on purpose: a provider 400 is not an AgentError, and letting it reach the
            # node wrapper sets state["error"], which poisons a run triage was only sizing
            logger.bind(error=str(exc)[:150]).warning("triage.failed")
            payload = TriageOutput()

        depth = payload.depth.strip().lower()
        if depth not in DEPTHS:
            depth = DEFAULT_DEPTH
        budget = budget_for(depth)

        answerable = payload.answerable or not payload.clarifying_question.strip()
        sections = [s.strip() for s in payload.sections if s.strip()] or list(budget.sections)
        # the answer always leads, whatever headings the model asked for
        if sections[0].lower() != "answer":
            sections = ["Answer", *[s for s in sections if s.lower() != "answer"]]

        word_target = (
            payload.word_target if 40 <= payload.word_target <= 2000 else budget.word_target
        )

        logger.bind(
            depth=depth,
            answerable=answerable,
            tasks=budget.tasks,
            words=word_target,
            sections=len(sections),
        ).info("triage.decided")

        return {
            "query_depth": depth,
            "answerable": answerable,
            "clarifying_question": payload.clarifying_question.strip(),
            "report_sections": sections,
            "word_target": word_target,
            "triage_reason": payload.reason.strip(),
        }

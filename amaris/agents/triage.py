"""Reads the question before anything is spent and decides how much work it deserves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from amaris.agents.base_agent import BaseAgent
from amaris.graph.state import DEFAULT_DEPTH, DEPTHS
from amaris.observability.logging import logger
from amaris.tools.weather_tool import place_for_weather_query

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
    # the critic pass this run is allowed to reach; 1 means review once and never rewrite,
    # which is what stops a 60-word answer paying for a 44s re-research round
    max_revisions: int


# max_sources never exceeds what the writer actually reads, or we pay to gather and then discard
DEPTH_BUDGETS: dict[str, Budget] = {
    "direct": Budget(1, 1, 4, 6, False, 120, ("Answer",), 1),
    "brief": Budget(2, 2, 5, 8, False, 300, ("Answer", "Key Points", "References"), 1),
    "standard": Budget(
        3, 3, 6, 12, True, 700, ("Answer", "Key Findings", "Analysis", "References"), 2
    ),
    "deep": Budget(
        5,
        4,
        8,
        # 20, not 12: deep used to read exactly as many sources as standard, so the only
        # thing it bought was a longer report written from the same evidence
        20,
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
        2,
    ),
}

# what "explain in detail" moves to. the caller applies this and sends the resolved depth,
# so triage never has to work out whether it was asked to bump or to obey
NEXT_DEPTH: dict[str, str] = {
    "direct": "brief",
    "brief": "standard",
    "standard": "deep",
    "deep": "deep",
}

PROMPT = """You triage research questions before any work is spent on them.
Judge the question itself. Do not answer it and do not guess what the answer is.

Decide three things.

1. Is it answerable as written? Assume yes. Asking the reader a question costs them
   a round trip, so only do it when no amount of searching could close the gap:
   a local question with no place, a comparison with only one side, "now" where
   there is no way to know when now is.

   A named thing is always answerable. If the question names something — a protocol,
   a library, a person, a product, an event — then searching for that name is the
   job, even when the name is ambiguous or you do not recognise it. Research it and
   say what you found; do not ask which one they meant.

   If it genuinely is not answerable, write the single shortest question that
   would unblock it.

2. How many genuinely distinct angles does a good answer need?
     direct   — one settled fact or definition, one angle
     brief    — a couple of related points, no debate
     standard — several angles that must be weighed against each other
     deep     — contested, comparative, or needs evidence on multiple sides

   When a question sits between two of these, choose the shallower one. The reader
   can ask for more depth in one click, and cannot un-read a report they did not want.

3. What shape should the answer take? List the headings that actually earn their
   place. Short questions deserve short answers; do not pad one out to look thorough.

4. Restate it as a question that stands on its own. If it leans on an earlier turn —
   a pronoun, or a bare noun phrase — put the subject back in, because this is the
   string the searches will actually run on. If it already stands alone, repeat it.

Output JSON:
{{
  "answerable": true,
  "resolved_query": "the question as a standalone sentence",
  "clarifying_question": "",
  "depth": "direct|brief|standard|deep",
  "sections": ["Answer"],
  "word_target": 150,
  "reason": "one sentence on what the question needs"
}}
{history}{attached}
Question: {query}"""

ATTACHED_BLOCK = """
The user has attached these files, and their text is already indexed and retrievable:
{files}

So a question about "this document", "the file", "this pdf" or "it" is fully answerable —
the attachment is the subject. Never ask which document they mean. Resolve the question to
name the file instead, and size it by what the question asks of the file, not by its length.
"""

HISTORY_BLOCK = """
Earlier in this conversation:
{turns}

Read the new question in that light — a pronoun or a bare noun phrase usually points
back at what was just answered, so resolve it rather than asking who or what is meant.
"""

# enough to resolve a pronoun, not enough to grow the prompt without bound
HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 600


def format_attachments(attachments: list[dict[str, Any]]) -> str:
    """The attached filenames as a prompt block, or "" when nothing is attached."""
    names = [str(item.get("name") or item.get("url", "")) for item in (attachments or [])]
    names = [name for name in names if name]
    if not names:
        return ""
    return ATTACHED_BLOCK.format(files="\n".join(f"  - {name}" for name in names))


def format_history(history: list[dict[str, str]]) -> str:
    """The last few turns as a prompt block, or "" on the first question of a conversation."""
    recent = [turn for turn in (history or []) if turn.get("query")][-HISTORY_TURNS:]
    if not recent:
        return ""
    turns = "\n\n".join(
        f"Q: {turn['query']}\nA: {(turn.get('answer') or '')[:HISTORY_ANSWER_CHARS]}"
        for turn in recent
    )
    return HISTORY_BLOCK.format(turns=turns)


class TriageOutput(BaseModel):
    """Raw triage reply. Every field has a safe default so a thin answer still runs."""

    model_config = ConfigDict(extra="ignore")

    depth: str = DEFAULT_DEPTH
    answerable: bool = True
    resolved_query: str = ""
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
        # checked before the depth lock: picking "deep" does not make scraping the right way to
        # find out what the temperature is right now (ADR-041)
        place = place_for_weather_query(state["original_query"])
        if place:
            return self._live(place)
        if state.get("depth_locked"):
            return self._locked(state)
        try:
            payload = await self._invoke_structured(
                PROMPT.format(
                    query=state["original_query"],
                    history=format_history(state.get("history", [])),
                    attached=format_attachments(state.get("attachments", [])),
                ),
                TriageOutput,
            )
        except Exception as exc:
            # broad on purpose: a provider 400 is not an AgentError, and letting it reach the
            # node wrapper sets state["error"], which poisons a run triage was only sizing
            logger.bind(error=str(exc)[:150]).warning("triage.failed")
            payload = TriageOutput()

        return self._settle(payload, state.get("attachments", []))

    def _live(self, place: str) -> dict[str, Any]:
        """A question about live state is handed to a data source, so no research is planned."""
        logger.bind(kind="weather", place=place[:60], llm_decided=False).info("triage.live")
        return {
            "query_depth": "direct",
            "depth_locked": False,
            "answerable": True,
            "clarifying_question": "",
            "live_data": {"kind": "weather", "place": place},
            "report_sections": ["Answer"],
            "word_target": budget_for("direct").word_target,
            "triage_reason": f"current conditions for {place}, read from a data source",
        }

    def _locked(self, state: GraphState) -> dict[str, Any]:
        """The user picked the depth, so it is already decided and costs no model call."""
        depth = state["query_depth"] if state["query_depth"] in DEPTHS else DEFAULT_DEPTH
        budget = budget_for(depth)
        logger.bind(depth=depth, llm_decided=False).info("triage.locked")
        return {
            "query_depth": depth,
            "depth_locked": False,
            "answerable": True,
            "clarifying_question": "",
            "report_sections": list(budget.sections),
            "word_target": budget.word_target,
            "triage_reason": f"{depth} requested by the user",
        }

    def _settle(
        self, payload: TriageOutput, attachments: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Clamp the model's answer to something every downstream budget can be derived from."""
        depth = payload.depth.strip().lower()
        if depth not in DEPTHS:
            depth = DEFAULT_DEPTH
        budget = budget_for(depth)

        answerable = payload.answerable or not payload.clarifying_question.strip()
        # the only thing triage asks back for is a missing subject, and an attached file is the
        # subject — asking "which document?" about the one file just uploaded is never right
        if attachments and not answerable:
            logger.bind(files=len(attachments)).info("triage.attachment_answers_it")
            answerable = True
            payload.clarifying_question = ""
        sections = [s.strip() for s in payload.sections if s.strip()] or list(budget.sections)
        # the answer always leads, whatever headings the model asked for
        if sections[0].lower() != "answer":
            sections = ["Answer", *[s for s in sections if s.lower() != "answer"]]

        word_target = (
            payload.word_target if 40 <= payload.word_target <= 2000 else budget.word_target
        )

        resolved = payload.resolved_query.strip()
        logger.bind(
            depth=depth,
            answerable=answerable,
            tasks=budget.tasks,
            words=word_target,
            sections=len(sections),
            rewritten=bool(resolved),
        ).info("triage.decided")

        return {
            "query_depth": depth,
            "resolved_query": resolved,
            "answerable": answerable,
            "clarifying_question": payload.clarifying_question.strip(),
            "report_sections": sections,
            "word_target": word_target,
            "triage_reason": payload.reason.strip(),
        }

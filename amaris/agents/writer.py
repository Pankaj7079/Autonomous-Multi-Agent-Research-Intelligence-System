"""Writes the cited report. Sources are numbered here so [n] markers line up with references."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from amaris.agents.base_agent import BaseAgent
from amaris.agents.triage import budget_for, format_history
from amaris.graph.state import subject
from amaris.observability.logging import logger
from amaris.safety.injection import wrap_untrusted
from amaris.tools.relevance import select_for_prompt

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

SOURCE_CHARS = 500
# source cap: 20 x 500 chars + 1100-word target was the timeout threshold
WRITER_SOURCE_CAP = 14

# fix: gpt-oss fullwidth brackets 【1†source】 broke downstream [n] matching
_CITATION_MARKER = re.compile(r"[【\[]\s*(\d+)\s*(?:†[^】\]]*)?[】\]]")
# the same model emits 【—】 and 【†source】 with no number at all; they cite nothing, so they go
_EMPTY_MARKER = re.compile(r"【[^】\d]*】")

PROMPT = """You are a senior research analyst writing up findings for a reader who is
technical, busy, and will check your sources. Use only the provided research and analysis.

Open with this heading and nothing before it:

## Answer

Under it, answer the question in 1-3 sentences. Plainly, in the first breath, the
way you would answer a colleague who asked. No throat-clearing, no restating the
question, no "this report examines". If the research does not actually answer the
question, say so in that first line, say what the sources DO establish, and say
exactly what is missing — do not fill the space with adjacent facts.

Then, and only then, use exactly these headings:
{sections}

{length_rule}

What makes this a research write-up rather than a summary:
  - Be specific. Named entities, figures, dates, versions, quantities. A sentence
    that would still read as true if you swapped the subject for something else is
    filler — delete it.
  - Say where sources disagree, and which is better supported. Agreement between
    two sources is worth stating; silence from all of them is worth stating too.
  - Every section must carry something the sections above it did not. If a heading
    has nothing behind it in the research, write one line saying the sources do not
    cover it rather than padding it out.
  - Separate what the sources establish from what you are inferring, and mark the
    inference as an inference.

Rules:
  - every factual claim gets an inline citation [1], [2]
  - no claim without a source; flag uncertainty explicitly
  - active voice: "researchers found" not "it was found that"
  - build on what an earlier turn already answered, never restate it
  - end with ## References listing only the sources you actually cited

{revision_block}
{history}
Question: {query}

Analysis:
{analysis}

Numbered sources — cite only the numbers that support a claim you make:
{sources}"""

# length instruction tied to depth — same "shorter is better" hurts "explain in detail"
SHORT_RULE = """Length: about {words} words. Shorter is better than padded — stop when
the question is answered rather than filling the space."""

LONG_RULE = """Length: about {words} words, and this one was asked in depth on purpose,
so develop it. A reader who asked for detail and got four lines was not served. Give
each heading real substance: the evidence behind the claim, the figures, the
disagreements, the caveats. Do not pad to reach the count — but do not stop at a
summary either, because a summary is what they already had."""

# above this target the reader asked for a report, not an answer
LONG_FROM_WORDS = 500

_MARKER = re.compile(r"\[(\d+)\]")
# rebuild reference list — model numbered correctly but listed out of order
_REFERENCES = re.compile(r"\n#{1,6}\s*references\b.*", re.IGNORECASE | re.DOTALL)


def _normalise_markers(report: str) -> str:
    """Every citation as [n], and numberless fullwidth markers removed."""
    return _EMPTY_MARKER.sub("", _CITATION_MARKER.sub(lambda m: f"[{m.group(1)}]", report))


def _renumber(report: str, citations: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Renumber citations to 1..N in order of first use and rebuild the reference list.

    Returns the rewritten report and only the citations actually cited. A report that cited
    nothing keeps its text and loses its citations, because none of them were used.
    """
    by_index = {item["index"]: item for item in citations}
    body = _REFERENCES.sub("", report).rstrip()

    # uncited entries don't earn a number
    order: list[int] = []
    for found in _MARKER.finditer(body):
        number = int(found.group(1))
        if number in by_index and number not in order:
            order.append(number)
    if not order:
        return body, []

    remap = {old: new for new, old in enumerate(order, start=1)}
    # unknown numbers are left alone: dropping them would silently delete a claim's only marker
    rewritten = _MARKER.sub(
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0),
        body,
    )
    kept = sorted(
        ({**by_index[old], "index": new} for old, new in remap.items()),
        key=lambda item: item["index"],
    )
    return f"{rewritten}\n\n{_reference_block(kept)}", kept


def _reference_block(citations: list[dict[str, Any]]) -> str:
    """Generated, not trusted — every line is a source the report actually cited, in order.

    Joined with a markdown hard break, or a renderer collapses the single newlines and every
    reference runs together into one paragraph.
    """
    lines = "  \n".join(
        f"[{item['index']}] {item['title']}" + (f" — {item['url']}" if item["url"] else "")
        for item in citations
    )
    return f"## References\n\n{lines}"


REVISION_BLOCK = """This is revision {n}. The critic said:
{critic_feedback}
Fix specifically: {top_issue}
Don't rewrite everything — address the stated issues."""


class WriterAgent(BaseAgent):
    """Writes draft_report and citations."""

    name = "writer"
    task_type = "writing"

    async def _run(self, state: GraphState) -> dict[str, Any]:
        selected = self._select(state)
        citations = self._build_citations(selected)
        words = state["word_target"] or budget_for(state["query_depth"]).word_target
        prompt = PROMPT.format(
            sections=self._sections(state),
            length_rule=self._length_rule(words),
            revision_block=self._revision_block(state),
            history=format_history(state.get("history", [])),
            query=state["original_query"],
            analysis=state["analyzed_data"] or "no separate analysis was produced",
            sources=self._format_sources(selected, citations),
        )

        report = _normalise_markers(await self._invoke(prompt))
        report, citations = _renumber(report, citations)
        logger.bind(
            chars=len(report),
            citations=len(citations),
            depth=state["query_depth"],
            revision=state["revision_count"],
        ).info("writer.done")

        return {
            "draft_report": report,
            "citations": citations,
            # consume the hint: leaving it set makes the supervisor route here forever
            "routing_hint": "",
        }

    def _select(self, state: GraphState) -> list[dict[str, Any]]:
        return select_for_prompt(
            subject(state),
            state["raw_research"],
            min(self.settings.max_sources_in_prompt, WRITER_SOURCE_CAP),
            self.settings.relevance_floor,
        )

    def _length_rule(self, words: int) -> str:
        """Brevity and depth need opposite instructions, so the target picks which one ships."""
        rule = LONG_RULE if words >= LONG_FROM_WORDS else SHORT_RULE
        return rule.format(words=words)

    def _sections(self, state: GraphState) -> str:
        """Headings after ## Answer. A short question does not get a six-part academic template."""
        sections = state["report_sections"] or list(budget_for(state["query_depth"]).sections)
        rest = [s for s in sections if s.strip().lower() != "answer"]
        if not rest:
            return "  (no further headings — the answer above is the whole report)"
        return "\n".join(f"  ## {s}" for s in rest)

    def _revision_block(self, state: GraphState) -> str:
        """Only present on a rewrite, so the first draft isn't told to fix nothing."""
        if state["revision_count"] == 0:
            return ""
        return REVISION_BLOCK.format(
            n=state["revision_count"],
            critic_feedback=state["critic_feedback"] or "no specific feedback given",
            top_issue=state["top_issue"] or "overall quality",
        )

    def _build_citations(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Number sources once, here — the writer and the UI must agree on what [3] means."""
        return [
            {
                "index": index,
                "title": item.get("title", "") or item.get("url", ""),
                "url": item.get("url", ""),
            }
            for index, item in enumerate(sources, start=1)
        ]

    def _format_sources(
        self, sources: list[dict[str, Any]], citations: list[dict[str, Any]]
    ) -> str:
        """Content included — a writer given only titles invents the rest."""
        if not citations:
            return "no sources available — say so explicitly instead of inventing citations"
        blocks = "\n\n".join(
            f"[{citation['index']}] {citation['title']} — {citation['url']}\n"
            f"{source.get('content', '')[:SOURCE_CHARS]}"
            for citation, source in zip(citations, sources, strict=False)
        )
        # security: scraped/MCP output treated as untrusted (README injection risk)
        return wrap_untrusted(blocks)

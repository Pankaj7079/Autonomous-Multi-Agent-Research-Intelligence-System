"""Writes the cited report. Sources are numbered here so [n] markers line up with references."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from amaris.agents.base_agent import BaseAgent
from amaris.agents.triage import budget_for
from amaris.observability.logging import logger
from amaris.tools.relevance import select_for_prompt

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

SOURCE_CHARS = 500

# gpt-oss reaches for fullwidth brackets when citing, which no [n] matcher downstream finds
_CITATION_MARKER = re.compile(r"[【\[]\s*(\d+)\s*[】\]]")

PROMPT = """You are a senior research writer. Use only the provided research and analysis.

Open with this heading and nothing before it:

## Answer

Under it, answer the question in 1-3 sentences. Plainly, in the first breath, the
way you would answer a colleague who asked. No throat-clearing, no restating the
question, no "this report examines". If the research does not actually answer the
question, say so in that first line and say what is missing — do not fill the space
with adjacent facts.

Then, and only then, use exactly these headings:
{sections}

Rules:
  - about {word_target} words in total, and shorter is better than padded
  - every factual claim gets an inline citation [1], [2]
  - no claim without a source; flag uncertainty explicitly
  - active voice: "researchers found" not "it was found that"
  - end with ## References listing only the sources you actually cited

{revision_block}

Question: {query}

Analysis:
{analysis}

Numbered sources — cite only the numbers that support a claim you make:
{sources}"""

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
        prompt = PROMPT.format(
            sections=self._sections(state),
            word_target=state["word_target"] or budget_for(state["query_depth"]).word_target,
            revision_block=self._revision_block(state),
            query=state["original_query"],
            analysis=state["analyzed_data"] or "no separate analysis was produced",
            sources=self._format_sources(selected, citations),
        )

        report = _CITATION_MARKER.sub(lambda m: f"[{m.group(1)}]", await self._invoke(prompt))
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
            state["original_query"],
            state["raw_research"],
            self.settings.max_sources_in_prompt,
            self.settings.relevance_floor,
        )

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
        return "\n\n".join(
            f"[{citation['index']}] {citation['title']} — {citation['url']}\n"
            f"{source.get('content', '')[:SOURCE_CHARS]}"
            for citation, source in zip(citations, sources, strict=False)
        )

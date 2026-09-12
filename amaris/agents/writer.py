"""Writes the cited report. Sources are numbered here so [n] markers line up with references."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from amaris.agents.base_agent import BaseAgent
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

SOURCE_CHARS = 500
MAX_SOURCES_IN_PROMPT = 12

PROMPT = """You are a senior research report writer. Use only the provided research
and analysis. Every factual claim gets an inline citation [1], [2].

Rules:
  - no claim without a source
  - flag uncertainty explicitly
  - active voice: "researchers found" not "it was found that"
  - 800-1200 words

Structure, exactly these headings:
  ## Executive Summary        (3-5 sentences: what was asked, found, implied)
  ## Background & Context
  ## Key Findings             (4-6 numbered: claim [n] + evidence + confidence)
  ## Analysis & Insights
  ## Conclusion & Recommendations
  ## References               ([1] Title — URL)

{revision_block}

Query: {query}

Analysis:
{analysis}

Numbered sources — cite these exact numbers:
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
        citations = self._build_citations(state["raw_research"])
        prompt = PROMPT.format(
            revision_block=self._revision_block(state),
            query=state["original_query"],
            analysis=state["analyzed_data"] or "no analysis was produced",
            sources=self._format_sources(state["raw_research"], citations),
        )

        report = await self._invoke(prompt)
        logger.bind(
            chars=len(report),
            citations=len(citations),
            revision=state["revision_count"],
        ).info("writer.done")

        return {
            "draft_report": report,
            "citations": citations,
            # consume the hint: leaving it set makes the supervisor route here forever
            "routing_hint": "",
        }

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
            for index, item in enumerate(sources[:MAX_SOURCES_IN_PROMPT], start=1)
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

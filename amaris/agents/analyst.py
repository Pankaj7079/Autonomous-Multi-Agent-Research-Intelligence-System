"""Synthesises research. Decides for itself whether computation is needed — that is the agentic bit."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from amaris.agents.base_agent import BaseAgent
from amaris.observability.logging import logger
from amaris.tools.code_executor import execute_python
from amaris.tools.vector_tool import search_knowledge_base

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

SOURCE_CHARS = 600
MAX_SOURCES_IN_PROMPT = 12
KB_LIMIT = 3

PROMPT = """You are a senior research analyst.

1. IDENTIFY the analysis type this needs:
   statistical (numbers present) | comparative | causal | conceptual
   Say which one and why.

2. SYNTHESIZE across sources — find what they say together, note where
   they conflict. Don't summarize each source separately.

3. QUANTIFY only if numbers are present. If so, state "I need to compute X"
   and the code will be run for you.

4. STRUCTURE output as:
   ## Analysis Type
   ## Key Findings (numbered)
   ## Evidence and Sources
   ## Confidence: HIGH | MEDIUM | LOW — with reason
   ## What remains uncertain

Be honest about LOW confidence. Never present a weak finding as settled fact.

If and only if step 3 applies, append one ```python block that prints the
numbers you need. No imports beyond statistics, math and json. Omit the block
entirely for conceptual or comparative work.

Query: {query}

Sources:
{sources}

Recalled from the knowledge base:
{knowledge}"""

CODE_BLOCK = re.compile(r"```python\s*(.+?)```", re.DOTALL)


class AnalystAgent(BaseAgent):
    """Writes analyzed_data and code_outputs."""

    name = "analyst"
    task_type = "analysis"

    async def _run(self, state: GraphState) -> dict[str, Any]:
        knowledge = await search_knowledge_base(state["original_query"], limit=KB_LIMIT)
        prompt = PROMPT.format(
            query=state["original_query"],
            sources=self._format_sources(state["raw_research"]),
            knowledge="; ".join(hit["text"][:200] for hit in knowledge) or "nothing stored yet",
        )

        analysis = await self._invoke(prompt)
        code_outputs = await self._maybe_compute(analysis)

        if code_outputs:
            analysis = f"{analysis}\n\n## Computed Results\n{code_outputs[0]['stdout']}"

        logger.bind(chars=len(analysis), computed=bool(code_outputs), kb_hits=len(knowledge)).info(
            "analyst.done"
        )
        return {"analyzed_data": analysis, "code_outputs": code_outputs}

    async def _maybe_compute(self, analysis: str) -> list[dict[str, Any]]:
        """Run code only if the analyst chose to emit some. Never forced."""
        match = CODE_BLOCK.search(analysis)
        if not match:
            return []

        code = match.group(1).strip()
        result = await execute_python(code)
        logger.bind(success=result["success"], chars=len(result["stdout"])).info("analyst.computed")

        if not result["success"]:
            return []
        return [{"code": code, "stdout": result["stdout"], "stderr": result["stderr"]}]

    def _format_sources(self, sources: list[dict[str, Any]]) -> str:
        if not sources:
            return "no sources were gathered"
        return "\n\n".join(
            f"[{index}] {item.get('title', '')} — {item.get('url', '')}\n"
            f"{item.get('content', '')[:SOURCE_CHARS]}"
            for index, item in enumerate(sources[:MAX_SOURCES_IN_PROMPT], start=1)
        )

"""ReAct research loop. The agent decides when it has enough — the graph does not decide for it."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from amaris.agents.base_agent import AgentError, BaseAgent
from amaris.memory.mem0_memory import add_research_finding, recall_related
from amaris.observability.logging import logger
from amaris.tools.scraper_tool import needs_scraping, scrape_url
from amaris.tools.search_tool import smart_search

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

SNIPPET_CHARS = 400
MEMORY_RECALL_LIMIT = 3

REACT_PROMPT = """You are a ReAct research agent. Think, act, observe, repeat.

Task: {task_description}
Sources found so far: {found_count}
Previous findings: {prev_summary}
Recalled from past sessions: {memory_context}
Iteration {iteration} of {max_iterations}

Output JSON exactly:
{{
  "thought": "what I have, what's missing, what angle to try next",
  "action": "web_search" | "scrape_webpage" | "stop",
  "action_input": "query or URL, null if stopping",
  "reasoning": "why this action now",
  "sufficient": false
}}

Set sufficient=true and action="stop" when you have 5+ relevant recent sources
covering the main angles. If results are thin, reformulate and search a
different angle instead of stopping early."""

ASSESS_PROMPT = """You gathered {n} sources across {t} tasks.
Rate overall research quality 0.0-1.0:
  0.0-0.3 very few or irrelevant sources
  0.3-0.6 some useful material, clear gaps remain
  0.6-0.8 good coverage of most angles
  0.8-1.0 excellent, diverse authoritative sources
Output only a float."""


class ResearcherAgent(BaseAgent):
    """Writes raw_research and research_quality."""

    name = "researcher"
    task_type = "react"

    async def _run(self, state: GraphState) -> dict[str, Any]:
        tasks = state["research_plan"] or [
            {"task_id": "t1", "description": state["original_query"]}
        ]
        recalled = await recall_related(state["original_query"], limit=MEMORY_RECALL_LIMIT)

        # gather, not a loop — tasks are independent and serial made a 4-task plan 4x slower
        outcomes = await asyncio.gather(
            *(self._research_task(task, recalled) for task in tasks),
            return_exceptions=True,
        )

        merged = self._merge(state["raw_research"], outcomes)
        quality = await self._self_assess(len(merged), len(tasks))
        await self._remember(state["original_query"], merged)

        logger.bind(
            sources=len(merged),
            new=len(merged) - len(state["raw_research"]),
            tasks=len(tasks),
            quality=quality,
            recalled=len(recalled),
        ).info("researcher.done")

        return {"raw_research": merged, "research_quality": quality}

    async def _research_task(
        self, task: dict[str, Any], recalled: list[str]
    ) -> list[dict[str, Any]]:
        """One ReAct loop. Returns whatever it gathered, even if a step failed."""
        task_id = str(task.get("task_id", "t?"))
        description = str(task.get("description", ""))
        found: list[dict[str, Any]] = []

        for iteration in range(1, self.settings.max_react_iterations + 1):
            try:
                decision = await self._decide(description, found, recalled, iteration)
            except AgentError as exc:
                logger.bind(task_id=task_id, iteration=iteration, error=str(exc)[:150]).warning(
                    "researcher.react_failed"
                )
                break

            action = str(decision.get("action", "stop")).strip()
            action_input = decision.get("action_input")
            sufficient = bool(decision.get("sufficient"))

            logger.bind(
                task_id=task_id,
                iteration=iteration,
                action=action,
                sufficient=sufficient,
                found=len(found),
            ).debug("researcher.react_step")

            if sufficient or action == "stop" or not action_input:
                break

            if action == "web_search":
                found.extend(await self._search(str(action_input), task_id, found))
            elif action == "scrape_webpage":
                found.extend(await self._scrape(str(action_input), task_id))
            else:
                logger.bind(task_id=task_id, action=action).warning("researcher.unknown_action")
                break

        return found

    async def _decide(
        self, description: str, found: list[dict[str, Any]], recalled: list[str], iteration: int
    ) -> dict[str, Any]:
        prompt = REACT_PROMPT.format(
            task_description=description,
            found_count=len(found),
            prev_summary=self._summarise(found),
            memory_context="; ".join(recalled)[:500] or "nothing recalled",
            iteration=iteration,
            max_iterations=self.settings.max_react_iterations,
        )
        payload = await self._invoke_json(prompt)
        return payload if isinstance(payload, dict) else {"action": "stop", "sufficient": False}

    async def _search(
        self, query: str, task_id: str, already: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results = await smart_search(query)
        collected = [self._as_source(item, task_id) for item in results]

        # only scrape when the top snippet is too thin to be worth anything on its own
        if collected and needs_scraping(collected[0]["content"]):
            body = await scrape_url(collected[0]["url"])
            if body:
                collected[0]["content"] = body[: SNIPPET_CHARS * 4]
                collected[0]["scraped"] = True

        seen = {item["url"] for item in already}
        return [item for item in collected if item["url"] not in seen]

    async def _scrape(self, url: str, task_id: str) -> list[dict[str, Any]]:
        body = await scrape_url(url)
        if not body:
            return []
        return [
            {
                "title": url,
                "url": url,
                "content": body[: SNIPPET_CHARS * 4],
                "task_id": task_id,
                "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "scraped": True,
            }
        ]

    def _as_source(self, item: dict[str, Any], task_id: str) -> dict[str, Any]:
        return {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "")[: SNIPPET_CHARS * 4],
            "task_id": task_id,
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def _merge(self, existing: list[dict[str, Any]], outcomes: list[Any]) -> list[dict[str, Any]]:
        """Dedupe by URL across tasks and across earlier researcher visits."""
        merged = list(existing)
        seen = {item.get("url") for item in existing}

        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                logger.bind(error=str(outcome)[:150]).warning("researcher.task_crashed")
                continue
            for source in outcome:
                url = source.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                merged.append(source)
        return merged

    async def _self_assess(self, sources: int, tasks: int) -> float:
        """The researcher rates its own work — the supervisor routes on this number."""
        if sources == 0:
            return 0.0
        try:
            raw = await self._invoke(ASSESS_PROMPT.format(n=sources, t=tasks))
            match = re.search(r"\d*\.?\d+", raw)
            if match:
                return max(0.0, min(1.0, float(match.group())))
            logger.bind(raw=raw[:80]).warning("researcher.unparsable_quality")
        except AgentError as exc:
            logger.bind(error=str(exc)[:150]).warning("researcher.assess_failed")

        # routing depends on this, so fall back to a count-based estimate rather than 0.0
        return min(1.0, round(sources / 8, 2))

    async def _remember(self, query: str, sources: list[dict[str, Any]]) -> None:
        """Store the top findings so a future run starts ahead. No-ops without the memory extra."""
        for source in sources[:3]:
            await add_research_finding(
                query,
                f"{source.get('title', '')}: {source.get('content', '')[:300]}",
                url=source.get("url", ""),
            )

    def _summarise(self, found: list[dict[str, Any]]) -> str:
        if not found:
            return "nothing yet"
        return "; ".join(item.get("title", "")[:80] for item in found[-5:])

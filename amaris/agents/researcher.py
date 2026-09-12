"""ReAct research loop. The agent decides when it has enough — the graph does not decide for it."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from amaris.agents.base_agent import AgentError, BaseAgent
from amaris.memory.mem0_memory import add_research_finding, recall_related
from amaris.observability.logging import logger
from amaris.safety.injection import UNTRUSTED_NOTICE, wrap_untrusted
from amaris.tools.scraper_tool import needs_scraping, scrape_url
from amaris.tools.search_tool import smart_search

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

SNIPPET_CHARS = 400
# above the default research floor so the run still progresses, but never 'confident'
UNASSESSED_QUALITY_CEILING = 0.7
MEMORY_RECALL_LIMIT = 3

REACT_PROMPT = """You are a ReAct research agent. Think, act, observe, repeat.

{untrusted_notice}

Task: {task_description}
Sources found so far: {found_count}
Previous findings: {prev_summary}
Recalled from past sessions: {memory_context}
Iteration {iteration} of {max_iterations}

Output JSON exactly:
{{
  "thought": "what I have, what's missing, what angle to try next",
  "action": "search" | "fetch" | "stop",
  "action_input": "query or URL, null if stopping",
  "reasoning": "why this action now",
  "sufficient": false
}}

Set sufficient=true and action="stop" when you have 5+ relevant recent sources
covering the main angles. If results are thin, reformulate and search a
different angle instead of stopping early.

CRITICAL: you have no tools and no browser access. Do not call any tool. Emit
only the JSON object above — a separate system executes the action for you."""

ASSESS_PROMPT = """You gathered {n} sources across {t} tasks.
Rate overall research quality 0.0-1.0:
  0.0-0.3 very few or irrelevant sources
  0.3-0.6 some useful material, clear gaps remain
  0.6-0.8 good coverage of most angles
  0.8-1.0 excellent, diverse authoritative sources
Output only a float."""


class ReActDecision(BaseModel):
    """One ReAct step. Defaults are the safe stop, so a thin reply ends the task quietly."""

    model_config = ConfigDict(extra="ignore")

    action: str = "stop"
    action_input: Any = None
    sufficient: bool = False


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

        sources_outcomes = [
            outcome if isinstance(outcome, BaseException) else outcome[0] for outcome in outcomes
        ]
        merged = self._merge(state["raw_research"], sources_outcomes)
        quality = await self._self_assess(len(merged), len(tasks))
        await self._remember(state["original_query"], merged)

        # react_discipline (Layer 3) needs this: did the loop decide it was done, or run out of road
        new_stats = dict(state["react_stats"])
        for task, outcome in zip(tasks, outcomes, strict=False):
            if not isinstance(outcome, BaseException):
                new_stats[str(task.get("task_id", "t?"))] = outcome[1]

        logger.bind(
            sources=len(merged),
            new=len(merged) - len(state["raw_research"]),
            tasks=len(tasks),
            quality=quality,
            recalled=len(recalled),
        ).info("researcher.done")

        # consume the hint: leaving it set makes the supervisor route here forever
        return {
            "raw_research": merged,
            "research_quality": quality,
            "routing_hint": "",
            "react_stats": new_stats,
        }

    async def _research_task(
        self, task: dict[str, Any], recalled: list[str]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """One ReAct loop. Returns what it gathered plus how it ended, even if a step failed."""
        task_id = str(task.get("task_id", "t?"))
        description = str(task.get("description", ""))
        found: list[dict[str, Any]] = []
        iterations_used = 0
        self_terminated = False

        for iteration in range(1, self.settings.max_react_iterations + 1):
            iterations_used = iteration
            try:
                decision = await self._decide(description, found, recalled, iteration)
            except Exception as exc:
                # provider errors surface here too, not just AgentError — one bad step must
                # not lose the sources this task already gathered
                logger.bind(task_id=task_id, iteration=iteration, error=str(exc)[:150]).warning(
                    "researcher.react_failed"
                )
                break

            action = decision.action.strip()
            action_input = decision.action_input
            sufficient = decision.sufficient

            logger.bind(
                task_id=task_id,
                iteration=iteration,
                action=action,
                sufficient=sufficient,
                found=len(found),
            ).debug("researcher.react_step")

            if sufficient or action == "stop" or not action_input:
                # only sufficient=true is a real decision — "stop" alone can mean a blank reply
                self_terminated = sufficient
                break

            if action == "search":
                found.extend(await self._search(str(action_input), task_id, found))
            elif action == "fetch":
                found.extend(await self._scrape(str(action_input), task_id))
            else:
                logger.bind(task_id=task_id, action=action).warning("researcher.unknown_action")
                break

        stats = {"iterations_used": iterations_used, "self_terminated": self_terminated}
        return found, stats

    async def _decide(
        self, description: str, found: list[dict[str, Any]], recalled: list[str], iteration: int
    ) -> dict[str, Any]:
        prompt = REACT_PROMPT.format(
            untrusted_notice=UNTRUSTED_NOTICE,
            task_description=description,
            found_count=len(found),
            prev_summary=self._summarise(found),
            memory_context="; ".join(recalled)[:500] or "nothing recalled",
            iteration=iteration,
            max_iterations=self.settings.max_react_iterations,
        )
        return await self._invoke_structured(prompt, ReActDecision)

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

        # routing depends on this, so fall back to a count-based estimate rather than 0.0.
        # capped well below 1.0: 62 sources scraped is not evidence of perfect research, and
        # letting an unassessed run claim 1.0 told the supervisor to stop looking at it
        return min(UNASSESSED_QUALITY_CEILING, round(sources / 8, 2))

    async def _remember(self, query: str, sources: list[dict[str, Any]]) -> None:
        """Store the top findings so a future run starts ahead. No-ops without the memory extra."""
        for source in sources[:3]:
            await add_research_finding(
                query,
                f"{source.get('title', '')}: {source.get('content', '')[:300]}",
                url=source.get("url", ""),
            )

    def _summarise(self, found: list[dict[str, Any]]) -> str:
        """Titles come off scraped pages, so they are untrusted text like the bodies are."""
        if not found:
            return "nothing yet"
        return wrap_untrusted("; ".join(item.get("title", "")[:80] for item in found[-5:]))

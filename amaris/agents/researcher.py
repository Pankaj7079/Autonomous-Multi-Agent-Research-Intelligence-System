"""ReAct research loop. The agent decides when it has enough — the graph does not decide for it."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from amaris.agents.base_agent import AgentError, BaseAgent
from amaris.agents.triage import budget_for
from amaris.graph.state import subject
from amaris.memory.mem0_memory import add_research_finding, recall_related
from amaris.observability.logging import logger
from amaris.safety.injection import UNTRUSTED_NOTICE, wrap_untrusted
from amaris.tools.relevance import mean_relevance, score_source
from amaris.tools.scraper_tool import needs_scraping, scrape_url
from amaris.tools.search_tool import smart_search

if TYPE_CHECKING:
    from amaris.agents.triage import Budget
    from amaris.graph.state import GraphState

SNIPPET_CHARS = 400
MEMORY_RECALL_LIMIT = 3

# outside this band the lexical signal is unambiguous and a judge call cannot change the answer
ASSESS_CONFIDENT_HIGH = 0.75
ASSESS_CONFIDENT_LOW = 0.20

REACT_PROMPT = """You are a ReAct research agent. Think, act, observe, repeat.

{untrusted_notice}

Task: {task_description}
Relevant sources found so far: {found_count} (you are budgeted {budget} for this task)
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

Stop as soon as the sources you have can answer the task — set sufficient=true
and action="stop". Reaching your budget is a reason to stop, not a target to
hit. Search again only when there is a specific gap you can name, and when you
do, change the angle rather than repeating a query that already ran.

CRITICAL: you have no tools and no browser access. Do not call any tool. Emit
only the JSON object above — a separate system executes the action for you."""

ASSESS_PROMPT = """Rate how well these sources answer the question, 0.0-1.0.

Question: {query}

Sources gathered ({n} across {t} tasks):
{titles}

Judge relevance to the question asked, not volume:
  0.0-0.3 the sources are about something else
  0.3-0.6 adjacent material, the actual question is not covered
  0.6-0.8 the question is covered, some angles thin
  0.8-1.0 the question is directly and authoritatively answered
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
        budget = budget_for(state["query_depth"])
        query = subject(state)
        tasks = state["research_plan"] or [{"task_id": "t1", "description": query}]
        recalled = await recall_related(query, limit=MEMORY_RECALL_LIMIT)

        # search apis and the headless browser both throttle, so fan out but not without bound
        gate = asyncio.Semaphore(self.settings.max_concurrent_research_tasks)

        async def run_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            async with gate:
                return await self._research_task(task, recalled, budget)

        # gather, not a loop — tasks are independent and serial made a 4-task plan 4x slower
        outcomes = await asyncio.gather(*(run_task(task) for task in tasks), return_exceptions=True)

        sources_outcomes = [
            outcome if isinstance(outcome, BaseException) else outcome[0] for outcome in outcomes
        ]
        attached = await self._attachment_sources(state, tasks)
        merged = self._merge(query, state["raw_research"], [*sources_outcomes, attached], budget)
        quality = await self._self_assess(query, merged, len(tasks))
        await self._remember(query, merged)

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
            depth=state["query_depth"],
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
        self, task: dict[str, Any], recalled: list[str], budget: Budget
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """One ReAct loop. Returns what it gathered plus how it ended, even if a step failed."""
        task_id = str(task.get("task_id", "t?"))
        description = str(task.get("description", ""))
        max_iterations = min(budget.react_iterations, self.settings.max_react_iterations)
        found: list[dict[str, Any]] = []
        iterations_used = 0
        self_terminated = False

        for iteration in range(1, max_iterations + 1):
            iterations_used = iteration
            try:
                decision = await self._decide(
                    description, found, recalled, iteration, max_iterations, budget
                )
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
                found.extend(
                    await self._search(str(action_input), task_id, found, budget.results_per_search)
                )
            elif action == "fetch":
                found.extend(await self._scrape(str(action_input), task_id))
            else:
                logger.bind(task_id=task_id, action=action).warning("researcher.unknown_action")
                break

            # the budget is already met, so another reasoning step would only spend a call
            if len(found) >= budget.max_sources:
                self_terminated = True
                break

        stats = {"iterations_used": iterations_used, "self_terminated": self_terminated}
        return found, stats

    async def _decide(
        self,
        description: str,
        found: list[dict[str, Any]],
        recalled: list[str],
        iteration: int,
        max_iterations: int,
        budget: Budget,
    ) -> ReActDecision:
        # with one iteration there is no loop to reason about: the only useful first move is
        # to search the task itself, so the reasoning call is skipped outright
        if max_iterations == 1 and not found:
            return ReActDecision(action="search", action_input=description, sufficient=False)

        prompt = REACT_PROMPT.format(
            untrusted_notice=UNTRUSTED_NOTICE,
            task_description=description,
            found_count=len(found),
            budget=budget.max_sources,
            prev_summary=self._summarise(found),
            memory_context="; ".join(recalled)[:500] or "nothing recalled",
            iteration=iteration,
            max_iterations=max_iterations,
        )
        return await self._invoke_structured(prompt, ReActDecision)

    async def _attachment_sources(
        self, state: GraphState, tasks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Retrieve the chunks of each attached file that match this plan, one source per file.

        Retrieval is per task, so a long document contributes only the parts that matter. The
        chunks are then folded back into a single source per file, or one PDF would occupy
        every citation slot in the report.
        """
        attachments = state.get("attachments") or []
        if not attachments:
            return []

        from amaris.tools.vector_tool import search_knowledge_base

        session_id = state["session_id"]
        top_k = self.settings.attachment_top_k
        queries = [str(task.get("description", "")) for task in tasks] or [subject(state)]
        results = await asyncio.gather(
            *(search_knowledge_base(q, limit=top_k, session_id=session_id) for q in queries),
            return_exceptions=True,
        )

        # url is the citation target, so chunks are grouped by the file they came from
        excerpts: dict[str, list[str]] = {}
        for result in results:
            if isinstance(result, BaseException):
                logger.bind(error=str(result)[:150]).warning("researcher.attachment_lookup_failed")
                continue
            for hit in result:
                seen = excerpts.setdefault(hit["url"], [])
                if hit["text"] not in seen:
                    seen.append(hit["text"])

        sources = []
        for attachment in attachments:
            url = str(attachment.get("url", ""))
            chunks = excerpts.get(url)
            if not chunks:
                continue
            sources.append(
                {
                    "title": str(attachment.get("name", url)),
                    "url": url,
                    "content": "\n\n".join(chunks)[: SNIPPET_CHARS * 6],
                    "task_id": "attachment",
                    "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    # exempts it from the relevance floor in _merge — the user chose this file
                    "from_attachment": True,
                }
            )

        logger.bind(files=len(attachments), matched=len(sources)).info(
            "researcher.attachments_used"
        )
        return sources

    async def _search(
        self, query: str, task_id: str, already: list[dict[str, Any]], max_results: int
    ) -> list[dict[str, Any]]:
        results = await smart_search(query, max_results=max_results)
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

    def _merge(
        self,
        query: str,
        existing: list[dict[str, Any]],
        outcomes: list[Any],
        budget: Budget,
    ) -> list[dict[str, Any]]:
        """Dedupe, score against the query, drop the off-topic, and keep only what fits the budget.

        Gathering 69 sources and handing 12 downstream was pure waste — the cut happens here now,
        once, so every agent reads the same evidence and nothing is paid for twice.
        """
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

        scored = [{**item, "relevance": score_source(query, item)} for item in merged]
        # an attached file is evidence the user chose deliberately, so it is never cut for
        # scoring below the floor — that floor exists to drop junk search results
        kept = [
            item
            for item in scored
            if item["relevance"] >= self.settings.relevance_floor or item.get("from_attachment")
        ]
        dropped = len(scored) - len(kept)
        # everything scoring zero would leave the writer with nothing, so keep the best available
        ranked = sorted(kept or scored, key=lambda item: item["relevance"], reverse=True)

        if dropped:
            logger.bind(dropped=dropped, kept=len(kept), floor=self.settings.relevance_floor).info(
                "researcher.filtered"
            )
        return ranked[: min(budget.max_sources, self.settings.max_sources_in_prompt)]

    async def _self_assess(self, query: str, sources: list[dict[str, Any]], tasks: int) -> float:
        """The researcher rates its own work against the question — the supervisor routes on it."""
        if not sources:
            return 0.0

        lexical = mean_relevance(query, sources)
        # clearly on-topic or clearly off-topic needs no judge: the call could not move the routing
        if lexical >= ASSESS_CONFIDENT_HIGH or lexical <= ASSESS_CONFIDENT_LOW:
            logger.bind(quality=lexical, llm=False).debug("researcher.assessed")
            return lexical

        try:
            prompt = ASSESS_PROMPT.format(
                query=query,
                n=len(sources),
                t=tasks,
                titles=self._summarise(sources),
            )
            raw = await self._invoke(prompt)
            match = re.search(r"\d*\.?\d+", raw)
            if match:
                return max(0.0, min(1.0, float(match.group())))
            logger.bind(raw=raw[:80]).warning("researcher.unparsable_quality")
        except AgentError as exc:
            logger.bind(error=str(exc)[:150]).warning("researcher.assess_failed")

        # counting sources said 69 junk pages were good research, so fall back to how well
        # the ones we kept actually match the question
        return lexical

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

"""Turns one query into as many concrete research tasks as its triaged depth justifies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from amaris.agents.base_agent import BaseAgent
from amaris.agents.triage import budget_for
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

PROMPT = """You are a senior research strategist.
Turn the query into at most {max_tasks} concrete, non-overlapping research tasks.

Use fewer if fewer will do. A task only earns its place if it would find
something the others would miss — padding the plan costs a web search each.
Order by dependency: foundational tasks first.

Bad task:  "Research AI agents"
Good task: "Find production case studies of multi-agent systems deployed in
            2025-2026 and note the specific failure modes reported"

Output JSON:
{{
  "tasks": [
    {{"task_id":"t1","description":"...","assigned_to":"researcher",
     "status":"pending","result":null}}
  ],
  "research_strategy": "one sentence on the overall angle"
}}

Query: {query}"""


class PlannerOutput(BaseModel):
    """The planner's contract. Items stay raw — _normalise_tasks owns their shape."""

    model_config = ConfigDict(extra="ignore")

    tasks: list[Any] = Field(default_factory=list)
    research_strategy: str = ""


class PlannerAgent(BaseAgent):
    """Writes research_plan and research_strategy."""

    name = "planner"
    task_type = "planning"

    async def _run(self, state: GraphState) -> dict[str, Any]:
        query = state["original_query"]
        max_tasks = budget_for(state["query_depth"]).tasks

        # a one-task plan for a one-angle question is the query itself, so asking a model to
        # restate it is a call that cannot change the outcome
        if max_tasks <= 1:
            logger.bind(tasks=1, depth=state["query_depth"], llm=False).info("planner.planned")
            return {
                "research_plan": [self._task(1, query)],
                "research_strategy": "answer the question directly from the best sources",
            }

        payload = await self._invoke_structured(
            PROMPT.format(query=query, max_tasks=max_tasks), PlannerOutput
        )
        tasks = self._normalise_tasks(payload.tasks, query, max_tasks)

        logger.bind(tasks=len(tasks), depth=state["query_depth"], llm=True).info("planner.planned")
        return {
            "research_plan": tasks,
            "research_strategy": payload.research_strategy.strip(),
        }

    def _task(self, index: int, description: str) -> dict[str, Any]:
        return {
            "task_id": f"t{index}",
            "description": description,
            "assigned_to": "researcher",
            "status": "pending",
            "result": None,
        }

    def _normalise_tasks(self, raw: Any, query: str, max_tasks: int) -> list[dict[str, Any]]:
        """Force the shape the researcher expects, whatever the model actually returned."""
        tasks: list[dict[str, Any]] = []
        if isinstance(raw, list):
            for index, item in enumerate(raw[:max_tasks], start=1):
                description = (
                    str(item.get("description", "")).strip()
                    if isinstance(item, dict)
                    else str(item)
                )
                if not description:
                    continue
                tasks.append(self._task(index, description))

        # the pipeline cannot move without at least one task, so fall back to the raw query
        if not tasks:
            logger.bind(query=query[:80]).warning("planner.empty_plan")
            tasks = [self._task(1, query)]
        return tasks

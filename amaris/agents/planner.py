"""Turns one query into 3-5 concrete research tasks the researcher can work through."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from amaris.agents.base_agent import BaseAgent
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

MIN_TASKS = 3
MAX_TASKS = 5

PROMPT = """You are a senior research strategist.
Turn the query into 3-5 concrete, non-overlapping research tasks.

Each task must be specific enough that one agent can complete it alone.
Order by dependency — foundational tasks first.
Cover different angles: current state, evidence, counterpoints, implications.

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
        payload = await self._invoke_structured(
            PROMPT.format(query=state["original_query"]), PlannerOutput
        )
        tasks = self._normalise_tasks(payload.tasks, state["original_query"])

        logger.bind(tasks=len(tasks)).info("planner.planned")
        return {
            "research_plan": tasks,
            "research_strategy": payload.research_strategy.strip(),
        }

    def _normalise_tasks(self, raw: Any, query: str) -> list[dict[str, Any]]:
        """Force the shape the researcher expects, whatever the model actually returned."""
        tasks: list[dict[str, Any]] = []
        if isinstance(raw, list):
            for index, item in enumerate(raw[:MAX_TASKS], start=1):
                description = (
                    str(item.get("description", "")).strip()
                    if isinstance(item, dict)
                    else str(item)
                )
                if not description:
                    continue
                tasks.append(
                    {
                        "task_id": f"t{index}",
                        "description": description,
                        "assigned_to": "researcher",
                        "status": "pending",
                        "result": None,
                    }
                )

        # the pipeline cannot move without at least one task, so fall back to the raw query
        if not tasks:
            logger.bind(query=query[:80]).warning("planner.empty_plan")
            tasks = [
                {
                    "task_id": "t1",
                    "description": query,
                    "assigned_to": "researcher",
                    "status": "pending",
                    "result": None,
                }
            ]
        return tasks

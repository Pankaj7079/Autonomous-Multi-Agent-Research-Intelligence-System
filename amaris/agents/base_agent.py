"""Shared agent plumbing: retries, JSON parsing, log context. Agents hold no state."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from amaris.config.settings import get_settings
from amaris.llm.router import invoke_with_fallback, is_retryable
from amaris.observability.context import agent_context
from amaris.observability.logging import logger, timed

if TYPE_CHECKING:
    from amaris.graph.state import GraphState

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5


class AgentError(RuntimeError):
    """An agent could not produce usable output. Node wrappers turn this into state["error"]."""


def strip_code_fences(text: str) -> str:
    """Remove a ```json / ``` wrapper if the model added one."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class BaseAgent(ABC):
    """One job per agent. No agent knows what runs before or after it."""

    name: ClassVar[str]
    task_type: ClassVar[str] = "reasoning"

    def __init__(self) -> None:
        self.settings = get_settings()

    @abstractmethod
    async def _run(self, state: GraphState) -> dict[str, Any]:
        """Do the work and return only the state fields this agent owns."""

    async def run(self, state: GraphState) -> dict[str, Any]:
        """Entry point. Binds the agent name onto every log line and times the work."""
        with agent_context(self.name), timed(f"agent.{self.name}") as fields:
            update = await self._run(state)
            fields["wrote"] = ",".join(update)
            return update

    async def _invoke(self, prompt: str, task_type: str | None = None) -> str:
        """LLM text with retries. Raises AgentError only after every attempt failed."""
        last_error: Exception | None = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await invoke_with_fallback(prompt, task_type=task_type or self.task_type)
                text = response.text.strip()
                if text:
                    return text
                # groq returns empty content on a first call sometimes, for no real reason
                logger.bind(agent=self.name, attempt=attempt + 1).warning("agent.empty_response")
            except Exception as exc:
                # a bad request or bad key fails identically on a retry, so don't waste one
                if not is_retryable(exc):
                    raise
                last_error = exc
                logger.bind(agent=self.name, attempt=attempt + 1, error=str(exc)[:150]).warning(
                    "agent.retry"
                )

            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(BACKOFF_BASE_SECONDS * 2**attempt)

        raise AgentError(
            f"{self.name}: no usable response after {MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _parse_json(self, text: str) -> Any:
        """Parse a JSON reply. Tier 1 patch 2 replaces this with validate-and-repair."""
        cleaned = strip_code_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # models like to wrap json in prose, so retry on the outermost braces
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise AgentError(f"{self.name}: no JSON object in response")
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AgentError(f"{self.name}: malformed JSON: {exc}") from exc

    async def _invoke_json(self, prompt: str, task_type: str | None = None) -> Any:
        """Convenience for the agents whose contract is JSON."""
        return self._parse_json(await self._invoke(prompt, task_type=task_type))

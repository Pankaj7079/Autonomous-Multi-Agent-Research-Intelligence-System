"""Shared agent plumbing: retries, JSON parsing, log context. Agents hold no state."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from amaris.config.settings import get_settings
from amaris.llm.router import (
    ProvidersCoolingDown,
    chain_wait_seconds,
    invoke_with_fallback,
    is_retryable,
)
from amaris.llm.structured import StructuredOutputError, extract_json, invoke_structured
from amaris.observability.context import agent_context
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from pydantic import BaseModel

    from amaris.graph.state import GraphState

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
# floor so a chain a hair from expiring cannot spin the retry loop on near-zero waits
MIN_RETRY_SLEEP_SECONDS = 0.25


def _retry_delay(attempt: int) -> float:
    """How long before the next attempt — only a fully rate-limited chain is worth a long wait."""
    blocked = chain_wait_seconds()
    if blocked > 0.0:
        return max(blocked, MIN_RETRY_SLEEP_SECONDS)
    # a free provider means the failure was transient, so retry soon and don't read its hint
    return BACKOFF_BASE_SECONDS * 2**attempt


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
        """Entry point. Binds the agent name onto every log line, including from inside tools."""
        # timing and error handling belong to the node wrapper, not here
        with agent_context(self.name):
            return await self._run(state)

    async def _invoke(
        self, prompt: str, task_type: str | None = None, json_mode: bool = False
    ) -> str:
        """LLM text with retries that respect the provider's stated wait. Raises AgentError if spent."""
        last_error: Exception | None = None
        attempts = 0
        waited = 0.0
        budget = self.settings.llm_retry_budget_seconds

        while attempts < MAX_ATTEMPTS:
            try:
                response = await invoke_with_fallback(
                    prompt, task_type=task_type or self.task_type, json_mode=json_mode
                )
                attempts += 1
                text = response.text.strip()
                if text:
                    return text
                # groq returns empty content on a first call sometimes, for no real reason
                logger.bind(agent=self.name, attempt=attempts).warning("agent.empty_response")
            except ProvidersCoolingDown as exc:
                # nothing was called, so this spends waiting budget but not one of the attempts
                last_error = exc
                logger.bind(agent=self.name, seconds=round(exc.seconds, 2)).warning(
                    "agent.chain_parked"
                )
            except Exception as exc:
                attempts += 1
                # a bad request or bad key fails identically on a retry, so don't waste one
                if not is_retryable(exc):
                    raise
                last_error = exc
                logger.bind(agent=self.name, attempt=attempts, error=str(exc)[:150]).warning(
                    "agent.retry"
                )

            if attempts >= MAX_ATTEMPTS:
                break
            delay = _retry_delay(max(attempts - 1, 0))
            if waited + delay > budget:
                # a window longer than the budget can't be waited out, so let the node write a partial report
                logger.bind(agent=self.name, delay=round(delay, 1), budget=budget).warning(
                    "agent.retry_budget_spent"
                )
                break
            logger.bind(agent=self.name, seconds=round(delay, 2)).info("agent.retry_wait")
            await asyncio.sleep(delay)
            waited += delay

        raise AgentError(
            f"{self.name}: no usable response after {attempts} attempts "
            f"({waited:.0f}s of {budget:.0f}s retry budget used): {last_error}"
        )

    def _parse_json(self, text: str) -> Any:
        """Parse a JSON reply with no schema attached. Prefer _invoke_structured where a model exists."""
        try:
            return json.loads(extract_json(text))
        except (StructuredOutputError, json.JSONDecodeError) as exc:
            raise AgentError(f"{self.name}: {exc}") from exc

    async def _invoke_json(self, prompt: str, task_type: str | None = None) -> Any:
        """For agents whose contract is JSON. Requests provider json mode where supported."""
        return self._parse_json(await self._invoke(prompt, task_type=task_type, json_mode=True))

    async def _invoke_structured[T: BaseModel](
        self, prompt: str, model: type[T], task_type: str | None = None
    ) -> T:
        """JSON validated against a schema, re-asking the model to repair its own bad output."""

        async def call(text: str) -> str:
            return await self._invoke(text, task_type=task_type, json_mode=True)

        try:
            return await invoke_structured(call, prompt, model)
        except StructuredOutputError as exc:
            raise AgentError(f"{self.name}: {exc}") from exc

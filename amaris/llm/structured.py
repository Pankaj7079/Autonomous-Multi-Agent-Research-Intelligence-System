"""Validate an LLM's JSON against a pydantic model, and ask it to repair its own mistakes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from amaris.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

MAX_PARSE_RETRIES = 2
# the repair prompt echoes the bad output back, so cap it or a long ramble doubles the call
_ECHO_LIMIT = 1500
_ERROR_LIMIT = 600

_REPAIR_TEMPLATE = """Your previous reply could not be parsed as the required JSON.

What you returned:
{echo}

What went wrong:
{error}

Reply with corrected JSON only. No prose, no code fences, no explanation."""


class StructuredOutputError(RuntimeError):
    """The model could not produce valid JSON for this schema, even after repair attempts."""


def extract_json(text: str) -> str:
    """Pull the JSON out of a reply, whether it is raw, fenced, or buried in prose."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if cleaned.startswith(("{", "[")):
        return cleaned

    # models like to wrap json in prose, so fall back to the outermost brackets
    candidates = [
        cleaned[start : end + 1]
        for start, end in ((cleaned.find(o), cleaned.rfind(c)) for o, c in ("{}", "[]"))
        if start != -1 and end > start
    ]
    if not candidates:
        raise StructuredOutputError("no JSON object or array found in the reply")
    # the outermost structure wins when a reply contains both
    return max(candidates, key=len)


def parse_structured[T: BaseModel](text: str, model: type[T]) -> T:
    """Extract, decode and validate in one step. Raises StructuredOutputError on any failure."""
    payload = extract_json(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"malformed JSON: {exc}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(f"{model.__name__} validation failed: {exc}") from exc


async def invoke_structured[T: BaseModel](
    invoke: Callable[[str], Awaitable[str]],
    prompt: str,
    model: type[T],
    max_parse_retries: int = MAX_PARSE_RETRIES,
) -> T:
    """Invoke, validate, and re-ask with the error quoted back. Raises after the retries."""
    attempt = 0
    text = ""
    last_error: StructuredOutputError | None = None

    while attempt <= max_parse_retries:
        current = prompt if attempt == 0 else _repair_prompt(prompt, text, last_error)
        text = await invoke(current)
        try:
            return parse_structured(text, model)
        except StructuredOutputError as exc:
            last_error = exc
            attempt += 1
            logger.bind(schema=model.__name__, attempt=attempt, error=str(exc)[:200]).warning(
                "structured.repair"
            )

    raise StructuredOutputError(
        f"{model.__name__}: no valid output after {max_parse_retries} repair attempts: {last_error}"
    )


def _repair_prompt(original: str, bad_output: str, error: StructuredOutputError | None) -> str:
    """Original task plus what went wrong — the model needs both to fix its own reply."""
    repair = _REPAIR_TEMPLATE.format(
        echo=bad_output[:_ECHO_LIMIT] or "(empty reply)",
        error=str(error)[:_ERROR_LIMIT],
    )
    return f"{original}\n\n---\n\n{repair}"

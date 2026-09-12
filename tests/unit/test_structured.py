"""Patch 2 — the parser and the repair loop. No real LLM is ever called here."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from amaris.llm.structured import (
    StructuredOutputError,
    extract_json,
    invoke_structured,
    parse_structured,
)


class Sample(BaseModel):
    name: str
    count: int = 0
    tags: list[str] = Field(default_factory=list)


class Recorder:
    """Returns queued replies and remembers the prompts it was asked with."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "still broken"


def test_raw_json_parses() -> None:
    assert parse_structured('{"name": "x", "count": 2}', Sample).count == 2


def test_fenced_json_parses() -> None:
    text = '```json\n{"name": "x"}\n```'
    assert parse_structured(text, Sample).name == "x"


def test_an_unlabelled_fence_parses() -> None:
    assert parse_structured('```\n{"name": "y"}\n```', Sample).name == "y"


def test_prose_wrapped_json_parses() -> None:
    text = 'Sure! Here is the plan:\n{"name": "z", "count": 5}\nHope that helps.'
    assert parse_structured(text, Sample).count == 5


def test_a_json_array_is_found_too() -> None:
    """The planner's tasks can come back as a bare list, not only wrapped in an object."""
    assert extract_json("here you go: [1, 2, 3] done") == "[1, 2, 3]"


def test_the_outermost_structure_wins() -> None:
    assert extract_json('{"a": {"b": 1}}') == '{"a": {"b": 1}}'


def test_no_json_at_all_raises() -> None:
    with pytest.raises(StructuredOutputError, match="no JSON"):
        parse_structured("I would rather not", Sample)


def test_malformed_json_raises() -> None:
    with pytest.raises(StructuredOutputError, match="malformed"):
        parse_structured('{"name": "x",}', Sample)


def test_valid_json_for_the_wrong_schema_raises() -> None:
    """Parsing is not enough — a missing required field is exactly what validation is for."""
    with pytest.raises(StructuredOutputError, match="validation failed"):
        parse_structured('{"count": 3}', Sample)


async def test_a_good_first_reply_costs_one_call() -> None:
    llm = Recorder('{"name": "ok"}')
    result = await invoke_structured(llm, "do the thing", Sample)
    assert result.name == "ok"
    assert len(llm.prompts) == 1


async def test_a_malformed_reply_is_repaired_on_the_second_call() -> None:
    llm = Recorder("not json at all", '{"name": "fixed"}')
    result = await invoke_structured(llm, "do the thing", Sample)
    assert result.name == "fixed"
    assert len(llm.prompts) == 2


def _repair_prompt_of(llm: Recorder) -> str:
    return llm.prompts[1]


async def test_the_repair_prompt_quotes_the_error_and_the_bad_output() -> None:
    """The model cannot fix what it is not shown, so both must survive into the retry."""
    llm = Recorder('{"count": 1}', '{"name": "fixed"}')
    await invoke_structured(llm, "original task", Sample)
    repair = _repair_prompt_of(llm)
    assert "original task" in repair
    assert '{"count": 1}' in repair
    assert "validation failed" in repair


async def test_it_gives_up_after_the_retry_budget() -> None:
    llm = Recorder("nope", "still nope", "nope again")
    with pytest.raises(StructuredOutputError, match="no valid output"):
        await invoke_structured(llm, "do the thing", Sample, max_parse_retries=2)
    assert len(llm.prompts) == 3


async def test_zero_retries_means_one_attempt() -> None:
    llm = Recorder("nope")
    with pytest.raises(StructuredOutputError):
        await invoke_structured(llm, "do the thing", Sample, max_parse_retries=0)
    assert len(llm.prompts) == 1


async def test_an_empty_reply_is_described_as_empty_in_the_repair() -> None:
    llm = Recorder("", '{"name": "fixed"}')
    await invoke_structured(llm, "task", Sample)
    assert "(empty reply)" in _repair_prompt_of(llm)

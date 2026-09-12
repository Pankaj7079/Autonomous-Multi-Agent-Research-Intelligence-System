"""Test doubles for agent tests. Fixtures live in conftest.py; these are imported directly."""

from __future__ import annotations

from typing import Any

import pytest


class ScriptedLLM:
    """Returns queued replies in order, so a ReAct loop can be driven step by step."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, task_type: str | None = None) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            return "stop"
        # hold the last reply so a loop running longer than the script doesn't break
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def patch_invoke(monkeypatch: pytest.MonkeyPatch, agent: Any, llm: ScriptedLLM) -> ScriptedLLM:
    """Swap the agent's LLM call for a scripted one. Signature must match _invoke exactly."""
    monkeypatch.setattr(
        type(agent),
        "_invoke",
        lambda self, prompt, task_type=None, json_mode=False: llm(prompt),
    )
    return llm

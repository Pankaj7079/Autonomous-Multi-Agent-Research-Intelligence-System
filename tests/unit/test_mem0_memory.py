"""mem0 is optional — absence must be a quieter run, never a failed one (ADR-009)."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.memory import mem0_memory
from amaris.memory.mem0_memory import (
    USER_ID,
    _build_config,
    _extract_memories,
    add_research_finding,
    is_available,
    recall_related,
)


@pytest.fixture(autouse=True)
def _clear_client_cache() -> None:
    mem0_memory._memory.cache_clear()


@pytest.fixture
def no_mem0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mem0_memory, "_memory", lambda: None)


class FakeMemory:
    """Stands in for the mem0 client so no model or Qdrant is needed."""

    def __init__(self, results: Any = None, fail: bool = False) -> None:
        self.results = results if results is not None else {"results": []}
        self.fail = fail
        self.added: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> Any:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        return self.results

    def add(self, messages: Any, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.added.append({"messages": messages, **kwargs})


async def test_recall_is_empty_without_the_extra(no_mem0: None) -> None:
    assert await recall_related("anything") == []


async def test_store_reports_false_without_the_extra(no_mem0: None) -> None:
    assert await add_research_finding("q", "a finding") is False


def test_is_available_reflects_the_client(no_mem0: None) -> None:
    assert is_available() is False


async def test_recall_returns_memory_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMemory({"results": [{"memory": "supervisors route agentically"}, {"memory": "b"}]})
    monkeypatch.setattr(mem0_memory, "_memory", lambda: fake)
    assert await recall_related("routing") == ["supervisors route agentically", "b"]


@pytest.mark.parametrize(
    "raw",
    [
        {"results": [{"memory": "m1"}]},  # mem0 2.x shape
        [{"memory": "m1"}],  # mem0 0.1.x shape
        ["m1"],  # bare strings
    ],
)
def test_both_mem0_response_shapes_are_accepted(raw: Any) -> None:
    """mem0 changed its return shape between majors — we must survive either."""
    assert _extract_memories(raw) == ["m1"]


def test_garbage_response_yields_nothing_rather_than_raising() -> None:
    assert _extract_memories(None) == []
    assert _extract_memories({"unexpected": "shape"}) == []


async def test_a_broken_backend_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mem0_memory, "_memory", lambda: FakeMemory(fail=True))
    assert await recall_related("q") == []
    assert await add_research_finding("q", "f") is False


async def test_store_sends_a_user_assistant_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMemory()
    monkeypatch.setattr(mem0_memory, "_memory", lambda: fake)

    assert await add_research_finding("the query", "the finding", url="https://a.com") is True
    call = fake.added[0]
    assert call["user_id"] == USER_ID
    assert [m["role"] for m in call["messages"]] == ["user", "assistant"]
    assert call["metadata"] == {"url": "https://a.com"}


async def test_empty_findings_are_not_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMemory()
    monkeypatch.setattr(mem0_memory, "_memory", lambda: fake)
    assert await add_research_finding("q", "   ") is False
    assert fake.added == []


def test_config_uses_free_providers_only() -> None:
    """An OpenAI default would silently need a paid key — the whole project runs on free tiers."""
    config = _build_config()
    assert config["llm"]["provider"] == "groq"
    assert config["embedder"]["provider"] == "huggingface"
    assert config["vector_store"]["provider"] == "qdrant"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 384


def test_config_switches_to_cloud_qdrant_when_a_url_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from amaris.config.settings import Settings

    monkeypatch.setattr(
        mem0_memory,
        "get_settings",
        lambda: Settings(
            _env_file=None, qdrant_url="https://x.cloud.qdrant.io", qdrant_api_key="k"
        ),
    )
    vector_config = _build_config()["vector_store"]["config"]
    assert vector_config["url"] == "https://x.cloud.qdrant.io"
    assert "host" not in vector_config

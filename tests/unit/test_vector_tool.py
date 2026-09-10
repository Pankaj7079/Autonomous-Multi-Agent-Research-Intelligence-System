"""Qdrant and the embedder are both optional — absence must degrade, never crash."""

from __future__ import annotations

import pytest

from amaris.tools import vector_tool
from amaris.tools.vector_tool import search_knowledge_base, upsert_documents


@pytest.fixture(autouse=True)
def _clear_embedder_cache() -> None:
    vector_tool._embedder.cache_clear()


async def test_search_returns_empty_without_the_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """sentence-transformers lives behind the memory extra (ADR-009)."""
    monkeypatch.setattr(vector_tool, "_embedder", lambda: None)
    assert await search_knowledge_base("anything") == []


async def test_search_returns_empty_when_qdrant_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_embed(text: str) -> list[float]:
        return [0.0] * vector_tool.VECTOR_SIZE

    monkeypatch.setattr(vector_tool, "_embed", fake_embed)
    monkeypatch.setattr(vector_tool, "_client", lambda: None)
    assert await search_knowledge_base("anything") == []


async def test_search_maps_qdrant_points_to_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePoint:
        def __init__(self) -> None:
            self.payload = {"text": "langgraph routes with a supervisor", "url": "https://a.com"}
            self.score = 0.87

    class FakeResponse:
        def __init__(self) -> None:
            self.points = [FakePoint()]

    class FakeClient:
        async def query_points(self, **kwargs: object) -> FakeResponse:
            return FakeResponse()

        async def close(self) -> None:
            return None

    async def fake_embed(text: str) -> list[float]:
        return [0.1] * vector_tool.VECTOR_SIZE

    monkeypatch.setattr(vector_tool, "_embed", fake_embed)
    monkeypatch.setattr(vector_tool, "_client", FakeClient)

    hits = await search_knowledge_base("supervisor routing")
    assert hits == [
        {"text": "langgraph routes with a supervisor", "url": "https://a.com", "score": 0.87}
    ]


async def test_upsert_of_nothing_is_a_no_op() -> None:
    assert await upsert_documents([]) == 0


async def test_upsert_returns_zero_when_collection_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_collection() -> bool:
        return False

    monkeypatch.setattr(vector_tool, "ensure_collection", no_collection)
    assert await upsert_documents([{"text": "t", "url": "u"}]) == 0

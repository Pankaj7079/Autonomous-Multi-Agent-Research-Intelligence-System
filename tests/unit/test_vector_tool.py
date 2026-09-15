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
            self.payload = {
                "text": "langgraph routes with a supervisor",
                "url": "https://a.com",
                "title": "routing docs",
            }
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
        {
            "text": "langgraph routes with a supervisor",
            "url": "https://a.com",
            "title": "routing docs",
            "score": 0.87,
        }
    ]


async def test_search_filters_to_one_session_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collection is shared, so an uploaded file must not surface in someone else's run."""
    seen: dict[str, object] = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.points: list[object] = []

    class FakeClient:
        async def query_points(self, **kwargs: object) -> FakeResponse:
            seen.update(kwargs)
            return FakeResponse()

        async def close(self) -> None:
            return None

    async def fake_embed(text: str) -> list[float]:
        return [0.1] * vector_tool.VECTOR_SIZE

    monkeypatch.setattr(vector_tool, "_embed", fake_embed)
    monkeypatch.setattr(vector_tool, "_client", FakeClient)

    await search_knowledge_base("anything", session_id="abc123")
    assert seen["query_filter"] is not None

    seen.clear()
    await search_knowledge_base("anything")
    # without a session id the search spans the whole collection, which is the old behaviour
    assert seen["query_filter"] is None


async def test_upsert_of_nothing_is_a_no_op() -> None:
    assert await upsert_documents([]) == 0


async def test_upsert_returns_zero_when_collection_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_collection(collection: str = "") -> bool:
        return False

    monkeypatch.setattr(vector_tool, "ensure_collection", no_collection)
    assert await upsert_documents([{"text": "t", "url": "u"}]) == 0


class _FakeQdrant:
    """Counts a filter, records the delete it was asked for. Set `points` per test."""

    points = 0
    last_delete: object = None

    async def count(self, **kwargs: object) -> object:
        return type("Counted", (), {"count": type(self).points})()

    async def delete(self, **kwargs: object) -> None:
        type(self).last_delete = kwargs.get("points_selector")

    async def close(self) -> None:
        return None


async def test_deleting_nothing_touches_nothing() -> None:
    """Without a scope this would build an unfiltered delete, which is the whole collection."""
    assert await vector_tool.delete_documents() == 0


async def test_delete_removes_one_conversations_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """An upload is the user's own file: when the conversation ends it should not stay in a
    shared collection where nothing can retrieve it anyway (ADR-045)."""

    class Client(_FakeQdrant):
        points = 2
        last_delete = None

    monkeypatch.setattr(vector_tool, "_client", Client)

    removed = await vector_tool.delete_documents(session_id="abc", urls=["file://cv.pdf"])

    assert removed == 2
    assert Client.last_delete is not None


async def test_a_purge_window_of_zero_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 must mean "keep everything", not "delete everything"."""

    class Client(_FakeQdrant):
        points = 9
        last_delete = None

    monkeypatch.setattr(vector_tool, "_client", Client)

    assert await vector_tool.purge_stale_attachments(0) == 0
    assert Client.last_delete is None


async def test_a_dead_qdrant_is_not_a_failed_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cleanup runs on "start over" and at boot; neither may raise in the user's face."""
    monkeypatch.setattr(vector_tool, "_client", lambda: None)
    assert await vector_tool.delete_documents(session_id="abc") == 0
    assert await vector_tool.purge_stale_attachments(24) == 0


async def test_a_collection_that_does_not_exist_yet_is_empty_not_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal state of a fresh qdrant. The 404 body reads like a real error in the console,
    which is alarming for something that is simply "nothing stored yet"."""

    class Client:
        async def query_points(self, **kwargs: object) -> object:
            raise RuntimeError(
                "Unexpected Response: 404 Collection `amaris_research` doesn't exist!"
            )

        async def close(self) -> None:
            return None

    async def fake_embed(text: str) -> list[float]:
        return [0.1] * vector_tool.VECTOR_SIZE

    monkeypatch.setattr(vector_tool, "_embed", fake_embed)
    monkeypatch.setattr(vector_tool, "_client", Client)

    assert await search_knowledge_base("anything") == []

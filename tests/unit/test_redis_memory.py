"""The two backends must be interchangeable — that is the whole point of the interface."""

from __future__ import annotations

import asyncio

import pytest

from amaris.config.settings import Settings
from amaris.memory import redis_memory
from amaris.memory.redis_memory import (
    InMemoryJobStore,
    JobStore,
    RedisJobStore,
    _decode,
    _encode,
    get_job_store,
    reset_job_store,
)


@pytest.fixture(autouse=True)
async def _fresh_singleton() -> None:
    await reset_job_store()
    yield
    await reset_job_store()


@pytest.fixture
def store() -> InMemoryJobStore:
    return InMemoryJobStore()


async def test_create_then_read_back(store: InMemoryJobStore) -> None:
    await store.create_job("j1", "what is langgraph?")
    job = await store.get_job("j1")
    assert job["status"] == "queued"
    assert job["progress_pct"] == 0
    assert job["query"] == "what is langgraph?"


async def test_update_merges_without_wiping_other_fields(store: InMemoryJobStore) -> None:
    await store.create_job("j1", "q")
    await store.update_job("j1", status="running", current_agent="researcher")
    await store.update_job("j1", progress_pct=45)

    job = await store.get_job("j1")
    assert (job["status"], job["current_agent"], job["progress_pct"]) == (
        "running",
        "researcher",
        45,
    )


async def test_unknown_job_is_none_not_an_error(store: InMemoryJobStore) -> None:
    assert await store.get_job("never-existed") is None


async def test_update_of_unknown_job_is_ignored(store: InMemoryJobStore) -> None:
    await store.update_job("never-existed", status="done")
    assert await store.get_job("never-existed") is None


async def test_get_job_returns_a_copy(store: InMemoryJobStore) -> None:
    """A caller mutating the returned dict must not corrupt the store."""
    await store.create_job("j1", "q")
    job = await store.get_job("j1")
    job["status"] = "tampered"
    assert (await store.get_job("j1"))["status"] == "queued"


async def test_subscriber_receives_published_events(store: InMemoryJobStore) -> None:
    received = []

    async def listen() -> None:
        async for event in store.subscribe_progress("j1"):
            received.append(event)
            if len(received) == 2:
                return

    task = asyncio.create_task(listen())
    await asyncio.sleep(0)
    await store.publish_progress("j1", {"agent": "planner", "progress_pct": 15})
    await store.publish_progress("j1", {"agent": "writer", "progress_pct": 85})
    await asyncio.wait_for(task, timeout=2)

    assert [e["agent"] for e in received] == ["planner", "writer"]


async def test_publishing_with_no_subscribers_is_harmless(store: InMemoryJobStore) -> None:
    await store.publish_progress("nobody-listening", {"agent": "writer"})


async def test_two_subscribers_both_get_the_event(store: InMemoryJobStore) -> None:
    """The API and a second browser tab can both watch one run."""
    got: list[str] = []

    async def listen(name: str) -> None:
        async for event in store.subscribe_progress("j1"):
            got.append(f"{name}:{event['agent']}")
            return

    tasks = [asyncio.create_task(listen("a")), asyncio.create_task(listen("b"))]
    await asyncio.sleep(0)
    await store.publish_progress("j1", {"agent": "critic"})
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

    assert sorted(got) == ["a:critic", "b:critic"]


async def test_subscriber_cleanup_survives_a_closed_store(store: InMemoryJobStore) -> None:
    """Closing the store before the stream is finalised must not raise. Caught by the selftest."""
    stream = store.subscribe_progress("j1")
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)  # the queue only registers once the generator first runs
    await store.publish_progress("j1", {"agent": "planner"})
    assert (await asyncio.wait_for(pending, timeout=2))["agent"] == "planner"

    await store.close()
    await stream.aclose()  # used to raise ValueError: x not in list


def test_encode_decode_roundtrips_nested_values() -> None:
    """Redis hashes only hold strings, so result must survive the trip."""
    original = {"status": "done", "progress_pct": 100, "result": {"score": 0.81, "cites": [1, 2]}}
    restored = _decode(_encode(original))
    assert restored["result"] == {"score": 0.81, "cites": [1, 2]}
    assert restored["progress_pct"] == 100


def test_decode_survives_a_corrupt_field() -> None:
    assert _decode({"result": "{not json"})["result"] is None


def test_encode_drops_none_so_it_never_overwrites_with_null() -> None:
    assert "current_agent" not in _encode({"current_agent": None, "status": "running"})


async def test_cloud_mode_never_touches_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        redis_memory, "get_settings", lambda: Settings(_env_file=None, deployment_mode="cloud")
    )
    store = await get_job_store()
    assert store.backend == "memory"


async def test_local_mode_falls_back_when_redis_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """A demo must not die because docker is down — this is the fallback that saves it."""
    monkeypatch.setattr(
        redis_memory, "get_settings", lambda: Settings(_env_file=None, deployment_mode="local")
    )

    def refuse(url: str) -> RedisJobStore:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(redis_memory, "RedisJobStore", refuse)
    store = await get_job_store()
    assert store.backend == "memory"


async def test_store_is_a_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        redis_memory, "get_settings", lambda: Settings(_env_file=None, deployment_mode="cloud")
    )
    assert await get_job_store() is await get_job_store()


def test_both_backends_implement_the_full_interface() -> None:
    """If these ever drift, a caller written against one breaks on the other."""
    contract = {name for name in vars(JobStore) if not name.startswith("_")}
    for backend in (InMemoryJobStore, RedisJobStore):
        assert contract <= set(dir(backend))

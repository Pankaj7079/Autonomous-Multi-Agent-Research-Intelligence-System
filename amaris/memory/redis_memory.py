"""Job status and progress events. Redis in local mode, an in-memory dict everywhere else."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

JobState = Literal["queued", "running", "done", "failed"]

JOB_TTL_SECONDS = 86_400  # a job nobody polled for in a day is not coming back

# fields that are not plain strings, so they need encoding in the redis hash
_JSON_FIELDS = ("result", "scores")
_INT_FIELDS = ("progress_pct",)


def _job_key(job_id: str) -> str:
    return f"amaris:job:{job_id}"


def _progress_channel(job_id: str) -> str:
    return f"amaris:progress:{job_id}"


def _encode(fields: dict[str, Any]) -> dict[str, str]:
    """Redis hashes only hold strings, so dicts go in as JSON."""
    encoded = {}
    for key, value in fields.items():
        if value is None:
            continue
        encoded[key] = json.dumps(value) if key in _JSON_FIELDS else str(value)
    return encoded


def _decode(raw: dict[str, str]) -> dict[str, Any]:
    """Inverse of _encode. A corrupt field degrades to None rather than killing the read."""
    job: dict[str, Any] = dict(raw)
    for key in _JSON_FIELDS:
        if key in job:
            try:
                job[key] = json.loads(job[key])
            except (json.JSONDecodeError, TypeError):
                job[key] = None
    for key in _INT_FIELDS:
        if key in job:
            try:
                job[key] = int(job[key])
            except (TypeError, ValueError):
                job[key] = 0
    return job


class JobStore(ABC):
    """The interface both backends implement. Callers never know which one they got."""

    @abstractmethod
    async def create_job(self, job_id: str, query: str) -> None:
        """Register a queued job."""

    @abstractmethod
    async def update_job(self, job_id: str, **fields: Any) -> None:
        """Merge fields into an existing job. Unknown job ids are ignored."""

    @abstractmethod
    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Current job record, or None when it never existed or expired."""

    @abstractmethod
    async def publish_progress(self, job_id: str, event: dict[str, Any]) -> None:
        """Fan one progress event out to every live subscriber."""

    @abstractmethod
    def subscribe_progress(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield progress events until the consumer stops iterating."""

    @abstractmethod
    async def ping(self) -> bool:
        """True when the backend is actually usable."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections. Safe to call twice."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Name for logs and the /ready probe."""


class InMemoryJobStore(JobStore):
    """Cloud mode and the local fallback. State dies with the process, which is fine."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    @property
    def backend(self) -> str:
        return "memory"

    async def create_job(self, job_id: str, query: str) -> None:
        self._jobs[job_id] = {
            "job_id": job_id,
            "query": query,
            "status": "queued",
            "current_agent": "",
            "progress_pct": 0,
            "created_at": time.time(),
        }

    async def update_job(self, job_id: str, **fields: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update({k: v for k, v in fields.items() if v is not None})

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return dict(job) if job else None

    async def publish_progress(self, job_id: str, event: dict[str, Any]) -> None:
        for queue in self._subscribers.get(job_id, []):
            queue.put_nowait(event)

    async def subscribe_progress(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            # the generator can be finalised after close() already cleared the list
            subscribers = self._subscribers.get(job_id)
            if subscribers and queue in subscribers:
                subscribers.remove(queue)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self._jobs.clear()
        self._subscribers.clear()


class RedisJobStore(JobStore):
    """Local mode. Survives an API restart, which is the only reason it exists."""

    def __init__(self, url: str) -> None:
        # redis.asyncio, never the sync client — a blocking call here stalls the whole loop
        from redis.asyncio import from_url

        self._client = from_url(url, decode_responses=True)

    @property
    def backend(self) -> str:
        return "redis"

    async def create_job(self, job_id: str, query: str) -> None:
        key = _job_key(job_id)
        await self._client.hset(
            key,
            mapping=_encode(
                {
                    "job_id": job_id,
                    "query": query,
                    "status": "queued",
                    "current_agent": "",
                    "progress_pct": 0,
                    "created_at": time.time(),
                }
            ),
        )
        await self._client.expire(key, JOB_TTL_SECONDS)

    async def update_job(self, job_id: str, **fields: Any) -> None:
        encoded = _encode(fields)
        if not encoded:
            return
        key = _job_key(job_id)
        await self._client.hset(key, mapping=encoded)
        # refresh the ttl on every write so an active job never expires mid-run
        await self._client.expire(key, JOB_TTL_SECONDS)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        raw = await self._client.hgetall(_job_key(job_id))
        return _decode(raw) if raw else None

    async def publish_progress(self, job_id: str, event: dict[str, Any]) -> None:
        await self._client.publish(_progress_channel(job_id), json.dumps(event))

    async def subscribe_progress(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        async with self._client.pubsub() as pubsub:
            await pubsub.subscribe(_progress_channel(job_id))
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    yield json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    logger.bind(job_id=job_id).warning("job_store.bad_event")

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


_store: JobStore | None = None


async def get_job_store() -> JobStore:
    """Singleton store for this process. Falls back to memory when Redis is unreachable."""
    global _store
    if _store is not None:
        return _store

    settings = get_settings()
    if settings.is_cloud:
        _store = InMemoryJobStore()
        logger.bind(backend="memory", reason="cloud_mode").info("job_store.ready")
        return _store

    try:
        candidate = RedisJobStore(settings.redis_url)
        if await candidate.ping():
            _store = candidate
            logger.bind(backend="redis", url=settings.redis_url).info("job_store.ready")
            return _store
        await candidate.close()
    except Exception as exc:
        logger.bind(error=str(exc)[:200]).debug("job_store.redis_unavailable")

    # a demo must not die because docker is down
    _store = InMemoryJobStore()
    logger.bind(backend="memory", reason="redis_unreachable").warning("job_store.degraded")
    return _store


async def reset_job_store() -> None:
    """Drop the singleton. Used by tests and the API lifespan shutdown."""
    global _store
    if _store is not None:
        await _store.close()
    _store = None


async def _selftest() -> None:
    """Exercise whichever backend is live: uv run python -m amaris.memory.redis_memory --selftest."""
    from amaris.observability.logging import configure_logging

    configure_logging(level="INFO", json_enabled=False)
    store = await get_job_store()
    logger.bind(backend=store.backend).info("selftest.backend")

    await store.create_job("selftest1", "does the store work?")
    await store.update_job(
        "selftest1", status="running", current_agent="researcher", progress_pct=45
    )
    job = await store.get_job("selftest1")
    logger.bind(status=job["status"], agent=job["current_agent"], pct=job["progress_pct"]).info(
        "selftest.read_back"
    )
    assert job["progress_pct"] == 45, "progress_pct must decode back to an int"

    received: list[dict[str, Any]] = []

    async def listen() -> None:
        async for event in store.subscribe_progress("selftest1"):
            received.append(event)
            return

    task = asyncio.create_task(listen())
    await asyncio.sleep(0.2)  # let the subscriber attach before publishing
    await store.publish_progress("selftest1", {"agent": "writer", "progress_pct": 85})
    try:
        await asyncio.wait_for(task, timeout=3)
    except TimeoutError:
        task.cancel()

    logger.bind(events=len(received), first=received[0] if received else None).info(
        "selftest.pubsub"
    )
    assert received, "no progress event arrived"

    logger.bind(missing=await store.get_job("does-not-exist")).info("selftest.unknown_job")
    await reset_job_store()
    logger.info("selftest.passed")


if __name__ == "__main__":
    asyncio.run(_selftest())

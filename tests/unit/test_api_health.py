"""Probes. /health must never touch a dependency, /ready must not 503 on an optional one."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from amaris.api.main import create_app
from amaris.api.routes import health
from amaris.memory import redis_memory
from amaris.tools import vector_tool


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # pre-seed the singleton so the lifespan never tries to reach a real redis
    redis_memory._store = redis_memory.InMemoryJobStore()

    async def no_qdrant() -> bool:
        return False

    monkeypatch.setattr(vector_tool, "ping", no_qdrant)
    with TestClient(create_app()) as test_client:
        yield test_client
    redis_memory._store = None


def _with_providers(monkeypatch: pytest.MonkeyPatch, providers: list[str]) -> None:
    settings = health.get_settings()
    monkeypatch.setattr(
        type(settings), "configured_llm_providers", property(lambda self: providers)
    )


def test_health_is_up_without_any_dependency(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["checks"] == {}


def test_ready_passes_with_an_llm_key_even_when_qdrant_is_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qdrant is advisory — vector_tool degrades to search, so it must not fail readiness."""
    _with_providers(monkeypatch, ["groq"])
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["checks"] == {"llm": True, "job_store": True, "qdrant": False}


def test_ready_is_503_with_no_provider_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_providers(monkeypatch, [])
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


async def test_a_hanging_check_is_treated_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe that blocks behind a dead dependency is worse than one that reports it down."""
    import asyncio

    monkeypatch.setattr(health, "PROBE_TIMEOUT_SECONDS", 0.05)

    async def hangs() -> bool:
        await asyncio.sleep(5)
        return True

    assert await health._guarded(hangs()) is False

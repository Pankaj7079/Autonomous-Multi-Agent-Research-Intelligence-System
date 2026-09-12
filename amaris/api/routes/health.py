"""Liveness and readiness probes. /health never touches a dependency."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response, status

from amaris.api.schemas import HealthStatus
from amaris.config.settings import get_settings
from amaris.config.validate import validate_config
from amaris.memory.redis_memory import get_job_store
from amaris.observability.logging import logger

router = APIRouter(tags=["health"])

# a probe must not hang behind a dead dependency, so every check is capped
PROBE_TIMEOUT_SECONDS = 3.0


async def _guarded(coro) -> bool:
    """Run a check, treating a timeout or any error as 'not available'."""
    try:
        return bool(await asyncio.wait_for(coro, timeout=PROBE_TIMEOUT_SECONDS))
    except Exception:
        return False


@router.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    """200 whenever the process is up. Deliberately checks nothing external."""
    return HealthStatus(status="ok", mode=get_settings().deployment_mode)


@router.get("/ready", response_model=HealthStatus)
async def ready(response: Response) -> HealthStatus:
    """503 only when the run could not work at all — an LLM key and a job store."""
    from amaris.tools import vector_tool

    settings = get_settings()
    store = await get_job_store()
    report = validate_config()

    checks = {
        # one source of truth for "is this configured" — the lifespan uses the same call
        "config": report.ok,
        "llm": bool(report.providers),
        "job_store": await _guarded(store.ping()),
        # qdrant is advisory: vector_tool degrades to search when it is missing (ADR-009)
        "qdrant": await _guarded(vector_tool.ping()),
    }
    required = checks["config"] and checks["job_store"]

    if not required:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    logger.bind(**checks, backend=store.backend, warnings=len(report.warnings)).debug("api.ready")
    return HealthStatus(
        status="ready" if required else "not_ready", mode=settings.deployment_mode, checks=checks
    )

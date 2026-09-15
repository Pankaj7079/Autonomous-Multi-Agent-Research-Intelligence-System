"""POST / GET / WS for a research run. Job bookkeeping only — no agent logic lives here."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from amaris.api.schemas import (
    DONE_PROGRESS,
    JobStatus,
    ProgressEvent,
    ResearchAccepted,
    ResearchRequest,
    build_progress_event,
    result_from_state,
)
from amaris.config.settings import use_session_keys
from amaris.graph.pipeline import stream_research
from amaris.memory.redis_memory import get_job_store
from amaris.observability.context import new_session_id
from amaris.observability.logging import logger
from amaris.safety.guardrails import validate_input

if TYPE_CHECKING:
    from amaris.graph.state import GraphState
    from amaris.memory.redis_memory import JobStore

router = APIRouter(prefix="/research", tags=["research"])

TERMINAL = ("done", "failed")
# the websocket falls back to polling the job so a terminal event it missed still closes it
WS_POLL_SECONDS = 5.0

# create_task alone is not enough — without a strong reference the loop can collect a
# running job mid-flight, and a 90 second pipeline is a very collectable task
_running: set[asyncio.Task[None]] = set()


async def _publish(store: JobStore, job_id: str, event: ProgressEvent) -> None:
    """Store the snapshot and fan the event out. A pubsub failure must not kill the run."""
    await store.update_job(
        job_id,
        status=event.status,
        current_agent=event.agent,
        progress_pct=event.progress_pct,
    )
    with contextlib.suppress(Exception):
        await store.publish_progress(job_id, event.model_dump())


async def _run_job(
    job_id: str,
    query: str,
    session_id: str,
    seed: dict[str, Any] | None = None,
    api_keys: dict[str, str] | None = None,
) -> None:
    """Drive the pipeline and mirror every node transition into the job store."""
    # bound inside the task, not in the request handler: a task gets its own copy of the
    # context, so one caller's key cannot reach another caller's job (ADR-037)
    if api_keys:
        use_session_keys(api_keys)
    store = await get_job_store()
    pct = 0
    started = time.perf_counter()
    final: GraphState | None = None
    try:
        async for node, delta, state in stream_research(query, session_id=session_id, seed=seed):
            final = state
            event = build_progress_event(node, delta, pct, elapsed_s=time.perf_counter() - started)
            pct = event.progress_pct
            await _publish(store, job_id, event)

        result = result_from_state(final) if final else None
        # an errored run that still produced a report is a degraded success, not a failure
        failed = bool(final and final["error"] and not (result and result.report))
        await store.update_job(
            job_id,
            status="failed" if failed else "done",
            current_agent="",
            progress_pct=DONE_PROGRESS,
            result=result.model_dump() if result else None,
            error=(final["error"] if final else "the pipeline produced no state") or None,
        )
        await _publish(
            store,
            job_id,
            ProgressEvent(
                agent="",
                status="failed" if failed else "done",
                message="run finished",
                progress_pct=DONE_PROGRESS,
                elapsed_s=round(time.perf_counter() - started, 2),
            ),
        )
        logger.bind(job_id=job_id, session_id=session_id, pct=DONE_PROGRESS).info("api.job_done")
    except Exception as exc:
        await store.update_job(job_id, status="failed", error=str(exc)[:500])
        await _publish(
            store,
            job_id,
            ProgressEvent(agent="", status="failed", message=str(exc)[:200], progress_pct=pct),
        )
        logger.bind(job_id=job_id, error=str(exc)[:200]).exception("api.job_failed")


def _spawn(
    job_id: str,
    query: str,
    session_id: str,
    seed: dict[str, Any] | None = None,
    api_keys: dict[str, str] | None = None,
) -> None:
    """Fire the run off the request thread and keep it referenced until it ends."""
    task = asyncio.create_task(_run_job(job_id, query, session_id, seed, api_keys))
    _running.add(task)
    task.add_done_callback(_running.discard)


@router.post("", response_model=ResearchAccepted, status_code=status.HTTP_202_ACCEPTED)
async def start_research(request: ResearchRequest) -> ResearchAccepted:
    """Queue a run and return immediately. The pipeline takes 60-90s, so nothing blocks here."""
    guard = validate_input(request.query)
    if not guard.ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=guard.reason)

    job_id = uuid.uuid4().hex
    session_id = request.session_id or new_session_id()

    store = await get_job_store()
    await store.create_job(job_id, guard.text)
    await store.update_job(job_id, session_id=session_id)
    _spawn(job_id, guard.text, session_id, request.seed(), request.api_keys)

    logger.bind(job_id=job_id, session_id=session_id, query=guard.text[:120]).info("api.job_queued")
    return ResearchAccepted(job_id=job_id, session_id=session_id)


@router.get("/{job_id}", response_model=JobStatus)
async def get_research(job_id: str) -> JobStatus:
    """Current job record. 404 once the job has expired out of the store."""
    store = await get_job_store()
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown job_id")
    return JobStatus(**{k: v for k, v in job.items() if k in JobStatus.model_fields})


def _snapshot_event(job: dict[str, Any]) -> dict[str, Any]:
    """The job's current position, so a client joining mid-run is not left blank."""
    return ProgressEvent(
        agent=job.get("current_agent") or "",
        status=job.get("status") or "queued",
        message=f"joined an existing job at {job.get('progress_pct', 0)}%",
        progress_pct=int(job.get("progress_pct") or 0),
    ).model_dump()


async def _forward(store: JobStore, job_id: str, out: asyncio.Queue[dict[str, Any]]) -> None:
    """Pump the pubsub subscription into a queue the socket can wait on with a timeout."""
    async for event in store.subscribe_progress(job_id):
        await out.put(event)


@router.websocket("/{job_id}/stream")
async def stream_progress(websocket: WebSocket, job_id: str) -> None:
    """Forward progress events until the run ends, then close."""
    await websocket.accept()
    store = await get_job_store()

    job = await store.get_job(job_id)
    if job is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="unknown job_id")
        return

    await websocket.send_json(_snapshot_event(job))
    if job.get("status") in TERMINAL:
        await websocket.close()
        return

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    pump = asyncio.create_task(_forward(store, job_id, queue))
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=WS_POLL_SECONDS)
            except TimeoutError:
                # covers the window between get_job above and the subscription attaching
                current = await store.get_job(job_id)
                if current and current.get("status") in TERMINAL:
                    await websocket.send_json(_snapshot_event(current))
                    break
                continue
            await websocket.send_json(event)
            if event.get("status") in TERMINAL:
                break
    except WebSocketDisconnect:
        logger.bind(job_id=job_id).debug("api.ws_disconnected")
    finally:
        pump.cancel()
        # CancelledError derives from BaseException, so suppress(Exception) let it escape
        # into the ASGI app and print a traceback on every clean socket close
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pump
        with contextlib.suppress(Exception):
            await websocket.close()

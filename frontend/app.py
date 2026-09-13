"""Streamlit entrypoint. Local mode talks to the API, cloud mode runs the pipeline in-process."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import httpx
import streamlit as st
from websockets.asyncio.client import connect

from amaris.api.schemas import (
    DONE_PROGRESS,
    ProgressEvent,
    ResearchResult,
    build_progress_event,
    result_from_state,
)
from amaris.config.settings import get_settings
from amaris.config.validate import ConfigReport, log_report, validate_config
from amaris.observability.logging import configure_from_settings, logger
from amaris.safety.guardrails import validate_input
from frontend.components import (
    example_queries,
    hero,
    history_sidebar,
    label,
    provider_strip,
    record_run,
    topbar,
)
from frontend.styles import inject_css
from frontend.views import landing, live_run, results

TERMINAL = ("done", "failed")
HTTP_TIMEOUT_SECONDS = 30.0

EventSink = Callable[[ProgressEvent], None]
RunOutcome = tuple[ResearchResult | None, str, str | None]

# one place to clear, so a new run never leaves half of the previous one on screen
RUN_KEYS = ("events", "result", "session_id", "error", "elapsed", "last_query")


@st.cache_resource
def _boot() -> ConfigReport:
    """Streamlit reruns the script constantly, so this must happen exactly once."""
    configure_from_settings()
    report = validate_config()
    log_report(report)
    return report


async def _run_local(query: str, on_event: EventSink) -> RunOutcome:
    """POST the job, follow it over the WebSocket, then read the finished record back."""
    base = get_settings().api_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{base}/research", json={"query": query})
        response.raise_for_status()
        accepted = response.json()

    job_id, session_id = accepted["job_id"], accepted["session_id"]
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://")
    async with connect(f"{ws_url}/research/{job_id}/stream") as socket:
        while True:
            event = ProgressEvent(**json.loads(await socket.recv()))
            on_event(event)
            if event.status in TERMINAL:
                break

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        job = (await client.get(f"{base}/research/{job_id}")).json()
    result = ResearchResult(**job["result"]) if job.get("result") else None
    return result, session_id, job.get("error")


async def _run_cloud(query: str, on_event: EventSink) -> RunOutcome:
    """No API and no Redis here — the same generator the API drives, consumed in-process."""
    from amaris.graph.pipeline import stream_research

    pct = 0
    final = None
    started = time.perf_counter()
    async for node, delta, state in stream_research(query):
        final = state
        event = build_progress_event(node, delta, pct, elapsed_s=time.perf_counter() - started)
        pct = event.progress_pct
        on_event(event)

    on_event(
        ProgressEvent(
            agent="",
            status="done",
            message="run finished",
            progress_pct=DONE_PROGRESS,
            elapsed_s=round(time.perf_counter() - started, 2),
        )
    )
    if final is None:
        return None, "", "the pipeline produced no state"
    return result_from_state(final), final["session_id"], final["error"]


def _execute(query: str, is_cloud: bool) -> None:
    """Drive one run, repainting the live view on every event rather than polling for state."""
    for key in RUN_KEYS:
        st.session_state.pop(key, None)

    events: list[ProgressEvent] = []
    bar = st.progress(0, text="starting")
    track = st.empty()

    def on_event(event: ProgressEvent) -> None:
        events.append(event)
        bar.progress(event.progress_pct / 100, text=event.message or event.agent)
        with track.container():
            live_run(events)

    runner = _run_cloud(query, on_event) if is_cloud else _run_local(query, on_event)
    started = time.perf_counter()
    try:
        result, session_id, error = asyncio.run(runner)
    finally:
        # the live widgets are replaced by the persistent render below, not stacked with it
        bar.empty()
        track.empty()

    elapsed = time.perf_counter() - started
    st.session_state.update(
        events=events,
        result=result,
        session_id=session_id,
        error=error,
        last_query=query,
        elapsed=elapsed,
    )
    record_run(query, result, events, session_id, error, elapsed)
    logger.bind(session_id=session_id, cloud=is_cloud).info("frontend.run_finished")


def _sidebar() -> None:
    """Deliberately lean — status and history only. Configuration lives in the System tab."""
    st.markdown(
        '<div class="sb-mark"><i>◆</i> AMARIS</div><div class="sb-sub">research console</div>',
        unsafe_allow_html=True,
    )
    label("providers")
    provider_strip()
    st.caption("first is primary · a seconds value is a cooldown")

    label("history")
    history_sidebar()

    if st.session_state.get("result") is not None:
        label("session")
        if st.button("new question", use_container_width=True):
            for key in RUN_KEYS:
                st.session_state.pop(key, None)
            st.rerun()


def main() -> None:
    # set_page_config must be the first streamlit call on the page; the sidebar carries
    # status and history, so it must not open collapsed
    st.set_page_config(
        page_title="AMARIS",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    report = _boot()
    settings = get_settings()
    inject_css()

    from amaris.llm.router import configured_chain, provider_status

    with st.sidebar:
        _sidebar()

    chain = configured_chain()
    topbar(settings.deployment_mode, chain, provider_status())
    for item in report.errors:
        st.error(item)

    # the hero only makes sense before a run; afterwards the result is the headline
    showing_result = st.session_state.get("result") is not None or st.session_state.get("error")
    if not showing_result:
        hero(chain)

    # a key separate from the widget's own — writing into the widget's key after it is
    # instantiated is what crashed this page before
    prefill = st.session_state.pop("prefill_query", "")
    with st.form("research_form"):
        field, action = st.columns([9, 1], gap="small")
        query = field.text_input(
            "Research question",
            value=prefill,
            placeholder="ask a research question…",
            label_visibility="collapsed",
        )
        submitted = action.form_submit_button("run", type="primary", use_container_width=True)
    example_queries()

    if submitted and query.strip():
        guard = validate_input(query)
        if not guard.ok:
            st.error(guard.reason)
            return
        try:
            _execute(guard.text.strip(), settings.is_cloud)
        except (httpx.HTTPError, OSError) as exc:
            # the single most likely local-mode mistake is the api simply not being up
            st.error(
                f"could not reach the API at {settings.api_base_url} — "
                f"start it with `uv run uvicorn amaris.api.main:app` ({exc})"
            )
            return
        except Exception as exc:
            logger.bind(error=str(exc)[:200]).exception("frontend.run_failed")
            st.error(f"the run failed: {exc}")
            return

    if st.session_state.get("result") is not None or st.session_state.get("error"):
        results(
            st.session_state.get("result"),
            st.session_state.get("session_id", ""),
            st.session_state.get("error"),
            st.session_state.get("elapsed"),
            st.session_state.get("events", []),
            report,
        )
    else:
        landing()


if __name__ == "__main__":
    main()

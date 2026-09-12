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
    agent_progress_tracker,
    citation_list,
    critic_verdict,
    decision_trace,
    evaluation_layers,
    event_log,
    raw_inspector,
    react_discipline,
    run_header,
    run_stats,
    score_dashboard,
    system_panel,
)
from frontend.styles import inject_css

TERMINAL = ("done", "failed")
HTTP_TIMEOUT_SECONDS = 30.0

EventSink = Callable[[ProgressEvent], None]
RunOutcome = tuple[ResearchResult | None, str, str | None]


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


def _header(mode: str) -> None:
    from amaris.llm.router import configured_chain

    chain = " → ".join(configured_chain()) or "no provider"
    st.markdown(
        '<div class="amaris-mast">'
        '<span class="amaris-title">AMARIS</span>'
        '<span class="amaris-sub">autonomous multi-agent research</span>'
        f'<span class="amaris-pill">{mode}</span>'
        f'<span class="amaris-pill">{chain}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def _config_notice(report: ConfigReport) -> None:
    """Errors are loud, warnings stay folded away — a demo should not open on a yellow wall."""
    for item in report.errors:
        st.error(item)
    if report.warnings:
        with st.expander(f"{len(report.warnings)} configuration warning(s)"):
            for item in report.warnings:
                st.warning(item)


def _render_result(
    result: ResearchResult | None,
    session_id: str,
    error: str | None,
    elapsed: float | None = None,
    events: list[ProgressEvent] | None = None,
) -> None:
    events = events or []
    if error:
        st.warning(f"the run reported an error: {error}")
    if result is None or not result.report:
        st.error("no report was produced")
        if events:
            event_log(events)
        return

    run_stats(result.trace, result.agent_path, elapsed)
    score_dashboard(result.scores)

    # tabs, not a single scroll — each one answers a different question about the run
    report_tab, exec_tab, sources_tab, eval_tab, raw_tab = st.tabs(
        ["Report", "Execution", "Sources", "Evaluation", "Raw"]
    )

    with report_tab, st.container(border=True):
        st.markdown(result.report)

    with exec_tab:
        st.caption(f"routing path · {' → '.join(result.agent_path)}")
        agent_progress_tracker(events)
        decision_trace(result.trace)
        react_discipline(result.trace)
        critic_verdict(result.trace)
        st.markdown('<div class="amaris-section">event log</div>', unsafe_allow_html=True)
        event_log(events)

    with sources_tab:
        citation_list(result.citations)

    with eval_tab:
        evaluation_layers(result.scores)

    with raw_tab:
        raw_inspector(result, events)

    st.caption("session id — replays this run from logs/amaris.jsonl")
    st.code(session_id, language=None)


def _execute(query: str, is_cloud: bool) -> None:
    """Drive one run, repainting the tracker on every event rather than polling for state."""
    # drop the previous run first, or a failed new query leaves the old report on screen
    for key in ("events", "result", "session_id", "error", "elapsed", "last_query"):
        st.session_state.pop(key, None)

    events: list[ProgressEvent] = []
    bar = st.progress(0, text="starting")
    track = st.empty()

    def on_event(event: ProgressEvent) -> None:
        events.append(event)
        bar.progress(event.progress_pct / 100, text=event.message or event.agent)
        with track.container():
            run_header(events)
            agent_progress_tracker(events)
            event_log(events)

    runner = _run_cloud(query, on_event) if is_cloud else _run_local(query, on_event)
    started = time.perf_counter()
    try:
        result, session_id, error = asyncio.run(runner)
    finally:
        # the live widgets are replaced by the persistent render below, not stacked with it
        bar.empty()
        track.empty()

    st.session_state.update(
        events=events,
        result=result,
        session_id=session_id,
        error=error,
        last_query=query,
        elapsed=time.perf_counter() - started,
    )
    logger.bind(session_id=session_id, cloud=is_cloud).info("frontend.run_finished")


def main() -> None:
    # set_page_config must be the first streamlit call on the page
    st.set_page_config(page_title="AMARIS", page_icon="◆", layout="wide")
    report = _boot()
    settings = get_settings()
    inject_css()
    _header(settings.deployment_mode)
    with st.sidebar:
        system_panel()
    _config_notice(report)

    with st.form("research_form"):
        query = st.text_input(
            "Research question",
            placeholder="What is the Model Context Protocol?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Research", type="primary")

    if submitted and query.strip():
        guard = validate_input(query)
        if not guard.ok:
            st.error(guard.reason)
            return
        query = guard.text
        try:
            _execute(query.strip(), settings.is_cloud)
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
        run_header(st.session_state.get("events", []), st.session_state.get("session_id", ""))
        _render_result(
            st.session_state.get("result"),
            st.session_state.get("session_id", ""),
            st.session_state.get("error"),
            st.session_state.get("elapsed"),
            st.session_state.get("events", []),
        )


if __name__ == "__main__":
    main()

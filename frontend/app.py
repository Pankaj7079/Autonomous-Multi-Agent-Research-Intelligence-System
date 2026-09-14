"""Streamlit entrypoint. Local mode talks to the API, cloud mode runs the pipeline in-process."""

from __future__ import annotations

import asyncio
import html
import json
import time
from collections.abc import Callable
from typing import Any

import httpx
import streamlit as st
from websockets.asyncio.client import connect

from amaris.agents.triage import budget_for
from amaris.api.schemas import (
    DONE_PROGRESS,
    ProgressEvent,
    ResearchRequest,
    ResearchResult,
    build_progress_event,
    result_from_state,
)
from amaris.config.settings import get_settings
from amaris.config.validate import ConfigReport, log_report, validate_config
from amaris.observability.logging import configure_from_settings, logger
from amaris.safety.guardrails import validate_input
from frontend import thread
from frontend.components import asking, hero, label, provider_strip
from frontend.styles import inject_css, wordmark
from frontend.views import inspection, landing, live_run

TERMINAL = ("done", "failed")
HTTP_TIMEOUT_SECONDS = 30.0
# mirrors ResearchRequest.query's min_length so both deployment modes reject the same input
MIN_QUERY_CHARS = 3

# what the composer's paperclip accepts; images are read by the vision model, not by OCR
ATTACHMENT_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]

# "auto" is not a depth, it is the absence of one — triage sizes the question as it always has
DEPTH_CHOICES = ("auto", "brief", "standard", "deep")
DEPTH_KEY = "depth_choice"
# streamlit drops the state of a widget that a rerun did not draw, and a run returns early
# before the picker — so the choice is mirrored here, where nothing collects it
DEPTH_SAVED = "depth_choice_saved"
# from real runs, not from the budget numbers: brief measured 94s and deep 206s. scraping and
# the evaluator's ~45s dominate, so even the shallow depths are a minute rather than seconds
DEPTH_ETA = {"brief": "~1-2 min", "standard": "~2-3 min", "deep": "~3-6 min"}

# each one triages to a different depth, so clicking any of them shows the budget logic working
EXAMPLES = (
    "what is the MCP protocol?",
    "how does LangGraph differ from CrewAI?",
    "why evaluate agent trajectories, not just answers?",
)

EventSink = Callable[[ProgressEvent], None]
RunOutcome = tuple[ResearchResult | None, str, str | None]


@st.cache_resource
def _boot() -> ConfigReport:
    """Streamlit reruns the script constantly, so this must happen exactly once."""
    configure_from_settings()
    report = validate_config()
    log_report(report)
    return report


def _payload(action: dict[str, Any]) -> dict[str, Any]:
    """The POST body for one pending action — a new question or an expand of an old one."""
    body: dict[str, Any] = {
        "query": action["query"],
        "history": action.get("history", []),
        "attachments": action.get("attachments", []),
    }
    if action.get("depth"):
        body["depth"] = action["depth"]
    if action.get("prior_sources"):
        body["prior_sources"] = action["prior_sources"]
    return body


async def _run_local(action: dict[str, Any], on_event: EventSink) -> RunOutcome:
    """POST the job, follow it over the WebSocket, then read the finished record back."""
    base = get_settings().api_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{base}/research", json=_payload(action))
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


async def _run_cloud(action: dict[str, Any], on_event: EventSink) -> RunOutcome:
    """No API and no Redis here — the same generator the API drives, consumed in-process."""
    from amaris.graph.pipeline import stream_research

    # the request model owns how a payload becomes a seed, so both modes build it the same way
    seed = ResearchRequest(**_payload(action)).seed()

    pct = 0
    final = None
    started = time.perf_counter()
    async for node, delta, state in stream_research(action["query"], seed=seed):
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


def _execute(action: dict[str, Any], is_cloud: bool) -> None:
    """Drive one run, repainting the live view on every event rather than polling for state."""
    events: list[ProgressEvent] = []
    # the question stays on screen for the whole run — for a spoken one this is the only
    # confirmation of what was actually heard
    head = st.empty()
    with head.container():
        asking(action["query"], spoken=bool(action.get("spoken")))
    bar = st.progress(0, text=action.get("label", "starting"))
    track = st.empty()

    def on_event(event: ProgressEvent) -> None:
        events.append(event)
        bar.progress(event.progress_pct / 100, text=event.message or event.agent)
        with track.container():
            live_run(events)

    runner = _run_cloud(action, on_event) if is_cloud else _run_local(action, on_event)
    started = time.perf_counter()
    try:
        result, session_id, error = asyncio.run(runner)
    finally:
        # the live widgets are replaced by the finished turn card, not stacked with it
        head.empty()
        bar.empty()
        track.empty()

    elapsed = time.perf_counter() - started
    thread.append(action["query"], result, events, session_id, error, elapsed)
    logger.bind(
        session_id=session_id,
        cloud=is_cloud,
        depth=action.get("depth") or "auto",
        reused_sources=len(action.get("prior_sources") or []),
    ).info("frontend.run_finished")


def _sidebar(mode: str) -> None:
    """Deliberately lean — status and the thread only. Configuration lives in the System tab."""
    st.markdown(
        f'{wordmark("sb-mark")}<div class="sb-sub">research console · {html.escape(mode)}</div>',
        unsafe_allow_html=True,
    )
    label("providers")
    provider_strip()
    st.caption("first is primary · a number is its cooldown")

    files = thread.attachments()
    if files:
        label("attached", f"{len(files)}")
        for item in files:
            st.markdown(
                f'<span class="chip accent"><span class="dot"></span>'
                f"{html.escape(str(item['name']))} · {item['chunks']} chunks</span>",
                unsafe_allow_html=True,
            )
        st.caption("retrieved per task, cited like any other source")

    label("this conversation")
    thread.thread_sidebar()

    if thread.turns():
        label("session")
        if st.button("start over", use_container_width=True):
            thread.clear()
            st.rerun()


def _ingest_files(files: list[Any]) -> bool:
    """Extract and index each attached file. False when one failed and the run must not start.

    Ingestion happens here rather than inside an agent: it is I/O at the edge, like input
    validation, and the same session id ties the stored chunks to this conversation.
    """
    from amaris.tools.attachments import AttachmentError, ingest

    session_id = thread.attachment_session()
    stored = thread.attachments()
    known = {item["name"] for item in stored}

    for upload in files:
        if upload.name in known:
            continue
        try:
            record = asyncio.run(
                ingest(upload.name, upload.getvalue(), upload.type or "", session_id)
            )
        except AttachmentError as exc:
            st.error(f"{upload.name}: {exc}")
            return False
        except Exception as exc:
            logger.bind(file=upload.name, error=str(exc)[:200]).exception("frontend.ingest_failed")
            st.error(f"{upload.name} could not be read: {exc}")
            return False
        stored.append(record)
        how = " (scanned, read by OCR)" if record.get("scanned") else ""
        st.toast(f"indexed {record['name']} — {record['chunks']} chunks{how}")
    return True


def _transcribe(audio: Any) -> str | None:
    """Spoken question to text. None when it could not be heard, with the reason shown.

    The transcript is surfaced rather than run silently: whisper can mishear a technical term,
    and a wrong question costs a full pipeline run to find out about.
    """
    from amaris.tools.transcribe import TranscriptionError, transcribe

    try:
        spoken = asyncio.run(transcribe(audio.getvalue(), audio.name or "question.wav"))
    except TranscriptionError as exc:
        st.warning(str(exc))
        return None
    except Exception as exc:
        logger.bind(error=str(exc)[:200]).exception("frontend.transcribe_failed")
        st.error(f"speech input failed: {exc}")
        return None

    # no toast here: the rerun that starts the run discards it before it can be read. The
    # transcript is shown by the run header instead, which lasts the whole run.
    return spoken


def _examples() -> None:
    """One click into a real run — an empty page with only a text box offers nothing to try."""
    label("try one", "runs a full pipeline")
    for column, question in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
        with column:
            if st.button(question, use_container_width=True, key=f"eg{hash(question)}"):
                st.session_state[thread.PENDING] = thread.ask_request(question)
                st.rerun()


def _depth_picker() -> str:
    """The depth the next question runs at, or "" to let triage size it."""
    picked = st.segmented_control(
        "depth",
        DEPTH_CHOICES,
        default=str(st.session_state.get(DEPTH_SAVED, "auto")),
        key=DEPTH_KEY,
        label_visibility="collapsed",
    )
    # clicking the selected chip deselects it, which reads as "stop forcing a depth"
    choice = str(picked) if picked else "auto"
    st.session_state[DEPTH_SAVED] = choice

    if choice == "auto":
        st.caption("auto · triage sizes the question before anything is spent on it")
    else:
        budget = budget_for(choice)
        st.caption(
            f"{choice} · {budget.tasks} tasks · {budget.react_iterations} react loops · "
            f"up to {budget.max_sources} sources · ~{budget.word_target} words · "
            f"{DEPTH_ETA.get(choice, '')} · triage spends no model call"
        )
    return "" if choice == "auto" else str(choice)


def _dispatch(action: dict[str, Any], settings: Any) -> None:
    """Run one pending action and convert the two likely failures into readable messages."""
    try:
        _execute(action, settings.is_cloud)
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
    st.rerun()


def main() -> None:
    # set_page_config must be the first streamlit call on the page; the sidebar carries
    # status and the thread, so it must not open collapsed
    st.set_page_config(
        page_title="AMARIS",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    report = _boot()
    settings = get_settings()
    inject_css()

    from amaris.llm.router import configured_chain

    with st.sidebar:
        _sidebar(settings.deployment_mode)

    chain = configured_chain()
    for item in report.errors:
        st.error(item)

    turns = thread.turns()
    if not turns:
        hero(chain)
        landing()
        _examples()

    for index, turn in enumerate(turns):
        thread.turn_card(turn, index, is_last=index == len(turns) - 1)

    # a pending action runs here so its live view lands under the thread, where the new
    # turn card will appear once it finishes
    pending = st.session_state.pop(thread.PENDING, None)
    if pending:
        _dispatch(pending, settings)
        return

    if turns:
        focused = thread.selected()
        if focused is not None:
            inspection(
                focused.get("result"),
                focused.get("session_id", ""),
                focused.get("error"),
                focused.get("elapsed"),
                focused.get("events", []),
                report,
            )

    chosen_depth = _depth_picker()
    submitted = st.chat_input(
        "ask a research question, or use the mic…",
        key="composer",
        accept_file="multiple",
        file_type=ATTACHMENT_TYPES,
        accept_audio=True,
    )
    if not submitted:
        return

    # accept_file/accept_audio make this a ChatInputValue rather than a str
    asked = submitted.text or ""
    if submitted.files and not _ingest_files(submitted.files):
        return

    by_voice = bool(submitted.audio) and not asked.strip()
    if by_voice:
        spoken = _transcribe(submitted.audio)
        if spoken is None:
            return
        asked = spoken

    if asked.strip():
        # whatever rejects the question below must still say what was heard, or a misheard
        # word looks like the microphone failing
        heard = f' — heard "{asked.strip()}"' if by_voice else ""
        # the same floor ResearchRequest enforces, checked here so cloud mode rejects a
        # two-character question as readably as the API does instead of raising mid-run
        if len(asked.strip()) < MIN_QUERY_CHARS:
            st.error(f"a question needs at least {MIN_QUERY_CHARS} characters{heard}")
            return
        guard = validate_input(asked)
        if not guard.ok:
            st.error(f"{guard.reason}{heard}")
            return
        request = thread.ask_request(guard.text.strip(), chosen_depth)
        request["spoken"] = by_voice
        st.session_state[thread.PENDING] = request
        st.rerun()
    elif submitted.files:
        # a file with no question is a valid thing to do — it is ready for the next question
        st.rerun()


if __name__ == "__main__":
    main()

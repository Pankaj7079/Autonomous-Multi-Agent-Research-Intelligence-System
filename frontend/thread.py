"""The conversation thread — one turn per question, oldest first.

Turns are the unit the UI is built around: a question, the answer it produced, and everything
the run left behind so the inspection tabs can rebind to any of them. State lives in
st.session_state and dies with the browser session, which is the whole scope of the feature.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import streamlit as st

from amaris.agents.triage import HISTORY_TURNS, NEXT_DEPTH
from amaris.config.settings import get_settings
from frontend.styles import badge_class

if TYPE_CHECKING:
    from amaris.api.schemas import ProgressEvent, ResearchResult

TURNS = "turns"
SELECTED = "selected_turn"
PENDING = "pending"
ATTACHMENTS = "attachments"
ATTACHMENT_SESSION = "attachment_session"

# the thread is a demo surface, not a transcript archive — old turns fall off the end
MAX_TURNS = 12
# a follow-up only needs enough of the previous answer to resolve what it refers to
ANSWER_CONTEXT_CHARS = 600


def turns() -> list[dict[str, Any]]:
    return st.session_state.setdefault(TURNS, [])


def append(
    query: str,
    result: ResearchResult | None,
    events: list[ProgressEvent],
    session_id: str,
    error: str | None,
    elapsed: float,
) -> None:
    """Record a finished run and focus the inspection tabs on it."""
    thread = turns()
    thread.append(
        {
            "query": query,
            "result": result,
            "events": events,
            "session_id": session_id,
            "error": error,
            "elapsed": elapsed,
        }
    )
    del thread[:-MAX_TURNS]
    st.session_state[SELECTED] = len(thread) - 1


def clear() -> None:
    st.session_state.pop(TURNS, None)
    st.session_state.pop(SELECTED, None)
    # a new conversation gets a new attachment scope; the old chunks stay in Qdrant but
    # nothing can retrieve them again, which is what "start over" should mean
    st.session_state.pop(ATTACHMENTS, None)
    st.session_state.pop(ATTACHMENT_SESSION, None)


def selected() -> dict[str, Any] | None:
    """The turn the inspection tabs are showing — the newest unless one was clicked."""
    thread = turns()
    if not thread:
        return None
    index = st.session_state.get(SELECTED, len(thread) - 1)
    return thread[min(max(index, 0), len(thread) - 1)]


def depth_of(turn: dict[str, Any]) -> str:
    result = turn.get("result")
    if result is None or result.trace is None:
        return ""
    return str(result.trace.triage.get("depth", ""))


def answer_of(turn: dict[str, Any]) -> str:
    result = turn.get("result")
    return result.report if result is not None else ""


def history_payload() -> list[dict[str, str]]:
    """Prior turns for the next question, trimmed to what a pronoun actually needs."""
    return [
        {"query": str(turn["query"]), "answer": answer_of(turn)[:ANSWER_CONTEXT_CHARS]}
        for turn in turns()[-HISTORY_TURNS:]
        if turn.get("result") is not None
    ]


def can_expand(turn: dict[str, Any]) -> bool:
    """False at the deepest depth, on a failed run, and on a question that was asked back."""
    result = turn.get("result")
    if result is None or result.awaiting_clarification or not result.report:
        return False
    depth = depth_of(turn)
    return bool(depth) and NEXT_DEPTH.get(depth, depth) != depth


def expand_request(turn: dict[str, Any]) -> dict[str, Any]:
    """A pending action that re-runs the same question one depth deeper, sources kept."""
    result = turn["result"]
    return {
        "query": str(turn["query"]),
        "expand_from_depth": depth_of(turn),
        "prior_sources": list(result.trace.sources) if result.trace else [],
        "history": history_payload()[:-1],
        # the deeper pass must still see the uploaded file, or expanding loses it
        "attachments": list(attachments()),
        "label": f"expanding to {NEXT_DEPTH.get(depth_of(turn), '')}",
    }


def attachments() -> list[dict[str, Any]]:
    """Files indexed in this browser session. They outlive a turn, so a document can be asked
    several questions without re-uploading it."""
    return st.session_state.setdefault(ATTACHMENTS, [])


def attachment_session() -> str:
    """One id for everything uploaded in this conversation — it scopes the Qdrant lookup.

    Deliberately not a run's session_id: those change per turn, and a file uploaded for the
    first question must still be retrievable by the third.
    """
    if ATTACHMENT_SESSION not in st.session_state:
        st.session_state[ATTACHMENT_SESSION] = uuid4().hex
    return str(st.session_state[ATTACHMENT_SESSION])


def ask_request(query: str) -> dict[str, Any]:
    """A pending action for a new question, carrying the conversation so far."""
    return {
        "query": query,
        "history": history_payload(),
        "attachments": list(attachments()),
        "label": "researching",
    }


def _meta_html(turn: dict[str, Any]) -> str:
    """The instrument strip under an answer: did it pass, and what did it cost."""
    result = turn.get("result")
    if result is None or result.trace is None:
        return ""
    trace = result.trace
    floor = get_settings().quality_approve_threshold
    score = trace.quality_score

    if result.awaiting_clarification:
        chip = '<span class="vchip warn"><span class="dot"></span>needs a detail</span>'
    else:
        state = badge_class(score, floor)
        verdict = "approved" if score >= floor else "below floor"
        chip = f'<span class="vchip {state}"><span class="dot"></span>{score:.2f} {verdict}</span>'

    cells = [
        (str(trace.source_count), "sources"),
        (str(len(result.agent_path)), "hops"),
        (str(len(trace.decisions)), "routing calls"),
    ]
    if trace.revision_count:
        cells.append((str(trace.revision_count), "revisions"))
    if turn.get("elapsed"):
        cells.append((f"{turn['elapsed']:.1f}", "seconds"))

    bars = '<span class="bar"></span>'.join(
        f'<span class="m"><b>{value}</b>{unit}</span>' for value, unit in cells
    )
    return f'<div class="turn-meta">{chip}<span class="bar"></span>{bars}</div>'


def _render_markdown(text: str) -> str:
    """Report markdown to HTML for the card body.

    html=False so anything HTML-shaped in a report — which is written from scraped pages —
    is escaped rather than injected into the page.
    """
    from markdown_it import MarkdownIt

    # no linkify: it needs linkify-it-py, which is not a dependency, and the Sources tab
    # already gives every url as a real link
    return MarkdownIt("commonmark", {"html": False}).enable("table").render(text)


def turn_card(turn: dict[str, Any], index: int, *, is_last: bool) -> None:
    """One question and its answer, as a single piece of markup this file fully controls.

    Rendered in one block rather than several Streamlit elements: a bordered container with
    widgets stacked inside it cannot be made to look like one designed card.
    """
    depth = depth_of(turn)
    depth_tag = f'<span class="depth-chip">{html.escape(depth)}</span>' if depth else ""
    answer = answer_of(turn)
    body = (
        f'<div class="turn-a">{_render_markdown(answer)}</div>'
        if answer
        else '<div class="turn-a"><p><em>no report was produced</em></p></div>'
    )

    st.markdown(
        f'<div class="turn {"current" if is_last else "past"}">'
        f'<div class="turn-head"><span class="who">question</span>'
        f'<span class="rule"></span>{depth_tag}</div>'
        f'<div class="turn-q">{html.escape(str(turn["query"]))}</div>'
        f"{body}{_meta_html(turn)}</div>",
        unsafe_allow_html=True,
    )

    if turn.get("error") and turn.get("result") is None:
        st.error(f"the run failed: {turn['error']}")
        return

    _actions(turn, index, is_last=is_last)


def _actions(turn: dict[str, Any], index: int, *, is_last: bool) -> None:
    """Expand, inspect, download. Rendered as real buttons so keyboard and screen readers work."""
    slots = st.columns([3, 1.5, 1.7, 4], gap="small")

    if can_expand(turn):
        target = NEXT_DEPTH.get(depth_of(turn), "")
        if slots[0].button(
            f"explain in detail  →  {target}",
            key=f"exp_{index}",
            use_container_width=True,
            help="re-runs one depth deeper and keeps the sources already gathered",
        ):
            st.session_state[PENDING] = expand_request(turn)
            st.rerun()

    if not is_last and slots[1].button(
        "inspect",
        key=f"ins_{index}",
        use_container_width=True,
        help="point the execution tabs below at this turn",
    ):
        st.session_state[SELECTED] = index
        st.rerun()

    if answer_of(turn):
        slots[2].download_button(
            "download .md",
            data=answer_of(turn),
            file_name=f"amaris_{str(turn['session_id'])[:8] or 'report'}.md",
            mime="text/markdown",
            key=f"dl_{index}",
            use_container_width=True,
        )


def thread_sidebar() -> None:
    """Every turn in this conversation, one click to rebind the inspection tabs to it."""
    thread = turns()
    if not thread:
        st.caption("no questions yet")
        return

    focused = st.session_state.get(SELECTED, len(thread) - 1)
    for index, turn in reversed(list(enumerate(thread))):
        result = turn.get("result")
        score = result.trace.quality_score if result and result.trace else None
        tag = f"{score:.2f}" if score else ("err" if turn.get("error") else "—")
        query = str(turn["query"])
        text = query if len(query) <= 20 else f"{query[:20]}…"
        mark = "▸ " if index == focused else ""
        if st.button(f"{mark}{tag}  {text}", key=f"thr_{index}", use_container_width=True):
            st.session_state[SELECTED] = index
            st.rerun()
        st.markdown(
            f'<div class="hist-m">{depth_of(turn) or "—"} · {turn.get("elapsed") or 0:.0f}s</div>',
            unsafe_allow_html=True,
        )

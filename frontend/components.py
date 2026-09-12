"""Pure render functions. They never fetch — all I/O stays in app.py so both modes share them."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import streamlit as st

from amaris.agents.critic import DIMENSIONS
from amaris.config.settings import get_settings
from frontend.styles import badge_class

if TYPE_CHECKING:
    from amaris.api.schemas import ProgressEvent

# the supervisor routes rather than produces, so it gets no card of its own
WORKERS = ("planner", "researcher", "analyst", "writer", "critic", "evaluator")
TERMINAL = ("done", "failed")


def _card(name: str, note: str, state: str) -> str:
    return (
        f'<div class="amaris-card {state}">'
        f'<div class="name">{html.escape(name)}</div>'
        f'<div class="note">{html.escape(note)}</div>'
        f"</div>"
    )


def agent_progress_tracker(events: list[ProgressEvent]) -> None:
    """One card per agent visit, in execution order. A repeat visit gets its own card."""
    visits = [e for e in events if e.agent and e.agent != "supervisor"]
    finished = any(e.status in TERMINAL for e in events)
    failed = any(e.status == "failed" for e in events)

    cards = []
    for position, event in enumerate(visits):
        last = position == len(visits) - 1
        if last and failed:
            state = "failed"
        elif last and not finished:
            state = "active"
        else:
            state = "done"
        cards.append(_card(event.agent, event.message, state))

    # a repeat visit is the point, so dedupe only for working out what never ran
    seen = {e.agent for e in visits}
    cards += [_card(name, "not visited", "pending") for name in WORKERS if name not in seen]

    st.markdown(f'<div class="amaris-track">{"".join(cards)}</div>', unsafe_allow_html=True)


def score_dashboard(scores: dict[str, float]) -> None:
    """The critic's four dimensions as badges, with the evaluator's layers underneath."""
    good_floor = get_settings().quality_approve_threshold
    present = [(name, scores[name]) for name in DIMENSIONS if name in scores]
    if not present:
        st.caption("no scores — the critic did not run")
        return

    badges = [
        f'<div class="amaris-badge {badge_class(value, good_floor)}">'
        f'<div class="label">{name.replace("_", " ")}</div>'
        f'<div class="value">{value:.2f}</div>'
        f"</div>"
        for name, value in present
    ]
    st.markdown(f'<div class="amaris-scores">{"".join(badges)}</div>', unsafe_allow_html=True)

    layered = {k[5:]: v for k, v in scores.items() if k.startswith("eval_")}
    if layered:
        line = " · ".join(f"{k} {v:.2f}" for k, v in layered.items())
        st.caption(f"evaluation layers — {line}")
        # the API cannot tell a real 0.0 from a judge that was rate limited mid-scoring
        if any(v == 0.0 for v in layered.values()):
            st.caption("a 0.00 here may mean the judge was unavailable, not that the run was bad")


def citation_list(citations: list[dict[str, Any]]) -> None:
    """Numbered exactly as the writer numbered them — [3] in the report is [3] here."""
    if not citations:
        st.caption("no citations")
        return

    with st.expander(f"Sources ({len(citations)})"):
        rows = []
        for item in citations:
            title = html.escape(str(item.get("title") or item.get("url") or "untitled"))
            url = html.escape(str(item.get("url") or ""), quote=True)
            label = f'<a href="{url}" target="_blank">{title}</a>' if url else title
            rows.append(
                f'<div class="amaris-cite">'
                f'<span class="num">[{item.get("index", "?")}]</span>{label}</div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

"""Pure render functions. They never fetch — all I/O stays in app.py so both modes share them."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import streamlit as st

from amaris.agents.critic import DIMENSIONS
from amaris.config.settings import get_settings
from frontend.styles import badge_class

if TYPE_CHECKING:
    from amaris.api.schemas import ProgressEvent, RunTrace

# the supervisor routes rather than produces, so it gets no card of its own
WORKERS = ("planner", "researcher", "analyst", "writer", "critic", "evaluator")
TERMINAL = ("done", "failed")


def _step(name: str, note: str, state: str, at: str) -> str:
    return (
        f'<div class="amaris-step {state}">'
        f'<span class="dot"></span>'
        f'<span class="name">{html.escape(name)}</span>'
        f'<span class="at">{html.escape(at)}</span>'
        f'<span class="note">{html.escape(note)}</span>'
        f"</div>"
    )


def agent_progress_tracker(events: list[ProgressEvent]) -> None:
    """One row per agent visit with the time it landed. A repeat visit gets its own row."""
    finished = any(e.status in TERMINAL for e in events)
    failed = any(e.status == "failed" for e in events)

    # how long a step took is the useful number; cumulative elapsed lives in the event log
    visits: list[tuple[ProgressEvent, float]] = []
    previous = 0.0
    for event in events:
        if event.agent and event.agent != "supervisor":
            visits.append((event, max(event.elapsed_s - previous, 0.0)))
        previous = event.elapsed_s or previous

    rows = []
    for position, (event, took) in enumerate(visits):
        last = position == len(visits) - 1
        if last and failed:
            state = "failed"
        elif last and not finished:
            state = "active"
        else:
            state = "done"
        at = f"{took:5.1f}s" if took else ""
        rows.append(_step(event.agent, event.message, state, at))

    # a repeat visit is the point, so dedupe only for working out what never ran
    seen = {event.agent for event, _ in visits}
    rows += [_step(name, "not visited", "pending", "") for name in WORKERS if name not in seen]

    st.markdown(f'<div class="amaris-timeline">{"".join(rows)}</div>', unsafe_allow_html=True)


def run_header(events: list[ProgressEvent], session_id: str = "") -> None:
    """Session, current stage, elapsed and progress — the strip a build tool puts up top."""
    if not events:
        return
    latest = events[-1]
    stage = latest.agent or ("finished" if latest.status in TERMINAL else "starting")
    bits = [
        f"<span>run <b>{html.escape(session_id or '—')}</b></span>",
        '<span class="sep">|</span>',
        f"<span>stage <b>{html.escape(stage)}</b></span>",
        '<span class="sep">|</span>',
        f"<span>{latest.progress_pct}%</span>",
        '<span class="sep">|</span>',
        f"<span>{latest.elapsed_s:.1f}s</span>",
        '<span class="sep">|</span>',
        f"<span>{len(events)} events</span>",
    ]
    st.markdown(f'<div class="amaris-runbar">{"".join(bits)}</div>', unsafe_allow_html=True)


def event_log(events: list[ProgressEvent]) -> None:
    """Every transition as it happened. The most literal answer to "what is it doing"."""
    if not events:
        return
    lines = []
    for event in events:
        cls = "err" if event.status == "failed" else ("ok" if event.status == "done" else "m")
        lines.append(
            f'<div><span class="t">{event.elapsed_s:7.2f}s</span> '
            f'<span class="a">{html.escape((event.agent or "run").ljust(11))}</span> '
            f'<span class="{cls}">{html.escape(event.message)}</span></div>'
        )
    st.markdown(f'<div class="amaris-log">{"".join(lines)}</div>', unsafe_allow_html=True)


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


def citation_list(citations: list[dict[str, Any]]) -> None:
    """Numbered exactly as the writer numbered them — [3] in the report is [3] here."""
    if not citations:
        st.caption("no citations")
        return

    with st.expander(f"Sources ({len(citations)})", expanded=True):
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


# the layers the evaluator can report, so a missing one reads as "not scored" not as zero
EVAL_METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")


def run_stats(trace: RunTrace | None, agent_path: list[str], elapsed: float | None) -> None:
    """The numbers that explain the run, above the report rather than buried under it."""
    if trace is None:
        return
    cells = [
        ("sources", str(trace.source_count)),
        ("research quality", f"{trace.research_quality:.2f}"),
        ("revisions", str(trace.revision_count)),
        ("agent hops", str(len(agent_path))),
        ("supervisor calls", str(len(trace.decisions))),
    ]
    if elapsed is not None:
        cells.append(("elapsed", f"{elapsed:.0f}s"))

    html_cells = "".join(
        f'<div class="amaris-stat"><div class="label">{label}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in cells
    )
    st.markdown(f'<div class="amaris-stats">{html_cells}</div>', unsafe_allow_html=True)


def _decision_row(entry: dict[str, Any]) -> str:
    chosen = str(entry.get("to_agent", "?"))
    expected = str(entry.get("expected_agent", "?"))
    diverged = chosen != expected
    source = "LLM" if entry.get("llm_decided") else "rule"
    flag = ' <span class="amaris-diverged">diverged</span>' if diverged else ""
    return (
        f"<tr>"
        f"<td>{entry.get('step', '?')}</td>"
        f"<td>{html.escape(str(entry.get('from_agent', '?')))} → "
        f"<b>{html.escape(chosen)}</b>{flag}</td>"
        f"<td>{source}</td>"
        f"<td><code>{html.escape(str(entry.get('matched_rule', '?')))}</code></td>"
        f"<td>{float(entry.get('research_quality', 0.0)):.2f}</td>"
        f"<td>{float(entry.get('quality_score', 0.0)):.2f}</td>"
        f"</tr>"
    )


def decision_trace(trace: RunTrace | None) -> None:
    """Every supervisor call: what it chose, what the documented rule expected, and why.

    This is the one view that proves the system is agentic rather than a chain — and the
    one thing the UI used to hide completely.
    """
    if trace is None or not trace.decisions:
        return

    diverged = sum(1 for d in trace.decisions if d.get("to_agent") != d.get("expected_agent"))
    label = f"Routing decisions ({len(trace.decisions)})"
    if diverged:
        label += f" · {diverged} diverged from the rule"

    with st.expander(label, expanded=True):
        st.caption(
            "The supervisor is an LLM choosing the next agent. 'rule' rows are the "
            "deterministic guards (error, revision cap, step cap); 'LLM' rows are its own "
            "choice, compared against the rule table in docs/AGENTS.md."
        )
        rows = "".join(_decision_row(entry) for entry in trace.decisions)
        st.markdown(
            '<table class="amaris-table"><thead><tr>'
            "<th>#</th><th>route</th><th>by</th><th>rule</th>"
            "<th>research</th><th>quality</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )


def react_discipline(trace: RunTrace | None) -> None:
    """Whether the researcher stopped because it was satisfied or because it ran out of budget."""
    if trace is None or not trace.react_stats:
        return

    rows = "".join(
        f"<tr><td><code>{html.escape(str(task))}</code></td>"
        f"<td>{stats.get('iterations_used', '?')}</td>"
        f"<td>{'self-stopped' if stats.get('self_terminated') else 'hit the cap'}</td></tr>"
        for task, stats in trace.react_stats.items()
    )
    stopped = sum(1 for s in trace.react_stats.values() if s.get("self_terminated"))
    with st.expander(f"ReAct loop ({stopped}/{len(trace.react_stats)} tasks self-stopped)"):
        st.caption(
            "The researcher decides for itself when it has enough. Hitting the cap is not a "
            "failure, but a run where nothing self-stops means the cap is doing the deciding."
        )
        st.markdown(
            '<table class="amaris-table"><thead><tr><th>task</th><th>iterations</th>'
            f"<th>ended because</th></tr></thead><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )


def critic_verdict(trace: RunTrace | None) -> None:
    """What the critic actually said, rather than only the number it produced."""
    if trace is None or not (trace.critic_feedback or trace.top_issue):
        return
    with st.expander("Critic feedback"):
        if trace.top_issue:
            st.markdown(f"**Top issue** — {trace.top_issue}")
        if trace.critic_feedback:
            st.markdown(trace.critic_feedback)


def evaluation_layers(scores: dict[str, float]) -> None:
    """Layers 1 and 2, with unscored metrics named as unscored instead of shown as 0.00."""
    good_floor = get_settings().quality_approve_threshold
    rows = []
    for metric in EVAL_METRICS:
        value = scores.get(f"eval_{metric}")
        if value is None:
            rows.append(
                f"<tr><td>{metric.replace('_', ' ')}</td>"
                '<td class="amaris-unscored">not scored — judge unavailable</td></tr>'
            )
        else:
            state = badge_class(value, good_floor)
            rows.append(
                f"<tr><td>{metric.replace('_', ' ')}</td>"
                f'<td class="amaris-{state}">{value:.2f}</td></tr>'
            )

    with st.expander("Evaluation layers", expanded=True):
        st.caption(
            "Layer 1 scores the sources against the query; Layer 2 scores the report against "
            "the sources. Both need an LLM judge, so a rate-limited run reports 'not scored' "
            "rather than pretending the answer was bad."
        )
        st.markdown(
            '<table class="amaris-table"><thead><tr><th>metric</th><th>score</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>",
            unsafe_allow_html=True,
        )


def system_panel() -> None:
    """Live configuration in the sidebar. A developer tool should state its own wiring."""
    from amaris.config.settings import get_settings
    from amaris.config.validate import validate_config
    from amaris.llm.router import configured_chain, provider_status

    settings = get_settings()
    report = validate_config()
    chain = configured_chain()
    cooling = provider_status()

    st.markdown('<div class="amaris-section">providers</div>', unsafe_allow_html=True)
    pills = (
        "".join(
            f'<span class="amaris-pill {"off" if cooling.get(name) else "on"}">'
            f"{name}{f' · {cooling[name]:.0f}s' if cooling.get(name) else ''}</span> "
            for name in chain
        )
        or '<span class="amaris-pill off">none configured</span>'
    )
    st.markdown(pills, unsafe_allow_html=True)
    st.caption("first is primary · a seconds value means that provider is rate limited")

    # a run started against a fully parked chain crawls through retries and then fails;
    # saying so up front is cheaper than watching it for nineteen minutes
    ready = [name for name in chain if not cooling.get(name)]
    if chain and not ready:
        st.error(
            "every provider is rate limited — a run started now will crawl through retries "
            "and is likely to fail. Wait for the cooldown or add another key."
        )
    elif chain and len(ready) < len(chain):
        st.warning(f"only {', '.join(ready)} usable right now — expect slower runs.")

    rows = {
        "mode": settings.deployment_mode,
        "primary": settings.primary_provider,
        "judge": settings.eval_provider,
        "reasoning model": settings.groq_model_reasoning,
        "fast model": settings.groq_model_fast,
        "research floor": f"{settings.research_quality_threshold:.2f}",
        "approve floor": f"{settings.quality_approve_threshold:.2f}",
        "max revisions": str(settings.max_revisions),
        "react cap": str(settings.max_react_iterations),
        "supervisor cap": str(settings.max_supervisor_steps),
        "retry budget": f"{settings.llm_retry_budget_seconds:.0f}s",
    }
    st.markdown('<div class="amaris-section">configuration</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(
            f'<div class="amaris-kv"><span class="k">{k}</span>'
            f'<span class="v">{html.escape(str(v))}</span></div>'
            for k, v in rows.items()
        ),
        unsafe_allow_html=True,
    )

    if report.warnings or report.errors:
        st.markdown('<div class="amaris-section">health</div>', unsafe_allow_html=True)
        for item in report.errors:
            st.error(item)
        for item in report.warnings:
            st.warning(item)


def raw_inspector(result: Any, events: list[ProgressEvent]) -> None:
    """The unfiltered payload. Nothing the system produced should be unreachable from the UI."""
    st.caption(
        "Exactly what the API returned, plus every progress event. If something is visible "
        "anywhere above, it came from here."
    )
    st.json(
        {
            "result": result.model_dump() if result is not None else None,
            "events": [e.model_dump() for e in events],
        },
        expanded=False,
    )

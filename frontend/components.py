"""Pure render functions. They never fetch — all I/O stays in app.py so both modes share them."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from amaris.agents.critic import DIMENSIONS
from amaris.config.settings import get_settings
from frontend.styles import TOKENS, badge_class

if TYPE_CHECKING:
    from amaris.api.schemas import ProgressEvent, ResearchResult, RunTrace

# the supervisor routes rather than produces, so it gets no lane in the worker flow
WORKERS = ("triage", "planner", "researcher", "analyst", "writer", "critic", "evaluator")
# an unanswerable question takes this lane instead and never touches an agent
CLARIFY_LANE = ("triage", "clarify")
TERMINAL = ("done", "failed")

FLOW_TAGS = {
    "triage": "TRI",
    "clarify": "ASK",
    "planner": "PLN",
    "researcher": "RSH",
    "analyst": "ANL",
    "writer": "WRT",
    "critic": "CRT",
    "evaluator": "EVL",
}

# what each agent DECIDES is the interesting column — a list of duties would not show autonomy
AGENT_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("TRI", "triage", "budget setter", "how much work the question is worth, before any is spent"),
    ("SUP", "supervisor", "agentic core", "the two calls state cannot settle on its own"),
    ("PLN", "planner", "decomposition", "how few tasks the question can be answered with"),
    ("RSH", "researcher", "ReAct loop", "when it has gathered enough — the graph never stops it"),
    ("ANL", "analyst", "synthesis", "whether the question needs real computation"),
    ("WRT", "writer", "drafting", "what to cite and how the report is structured"),
    ("CRT", "critic", "review + routing", "approve, rewrite, re-research, or re-plan"),
    ("EVL", "evaluator", "scoring", "nothing — it scores the finished run after the fact"),
)

# a real transcript. the point is no longer that everything is routed, but that routing is
# paid for only where state leaves the answer open
ROUTE_DEMO: tuple[tuple[str, str, str, bool], ...] = (
    ("triage", "no location given", "clarify", False),
    ("triage", "one settled fact, depth=direct", "planner", False),
    ("edge", "a plan always needs researching", "researcher", False),
    ("gate 1", "quality 0.41 with 20 off-topic hits", "researcher", True),
    ("gate 1", "quality 0.78, sources on topic", "analyst", False),
    ("edge", "analysis always needs writing up", "writer", False),
    ("edge", "a draft always needs reviewing", "critic", False),
    ("gate 2", "answer_fit 0.2 — wrong question", "planner", True),
    ("gate 2", "quality 0.81, critic approves", "FINISH", False),
)

EXAMPLE_QUERIES = (
    ("mcp", "What is the Model Context Protocol (MCP)?"),
    ("langgraph vs crewai", "Compare LangGraph and CrewAI for building multi-agent systems"),
    ("langchain stack", "What is the relationship between LangChain, LangGraph, and LangSmith?"),
    ("anthropic models", "What AI models has Anthropic released most recently?"),
)

HISTORY_LIMIT = 8
EVAL_METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")


def label(text: str, note: str = "") -> None:
    """Section heading. The count or qualifier goes in `note` and renders as a pill."""
    extra = f'<span class="n">{html.escape(note)}</span>' if note else ""
    st.markdown(f'<div class="lbl">{html.escape(text)}{extra}</div>', unsafe_allow_html=True)


def describe(text: str) -> None:
    """One line under a label saying what the panel is. No panel ships unexplained."""
    st.markdown(f'<div class="desc">{html.escape(text)}</div>', unsafe_allow_html=True)


def topbar(mode: str, chain: list[str], cooling: dict[str, float]) -> None:
    """Identity, mode and live provider health in one 44px strip instead of a hero block."""
    chips = [f'<span class="chip accent"><span class="dot"></span>{html.escape(mode)}</span>']
    for name in chain:
        seconds = cooling.get(name) or 0.0
        state = "hot" if seconds else "on"
        suffix = f" {seconds:.0f}s" if seconds else ""
        chips.append(f'<span class="chip {state}"><span class="dot"></span>{name}{suffix}</span>')
    if not chain:
        chips.append('<span class="chip off">no provider configured</span>')
    st.markdown(
        '<div class="topbar">'
        '<span class="mark">AMARIS</span>'
        '<span class="what">multi-agent research</span>'
        '<span class="grow"></span>'
        f"{''.join(chips)}</div>",
        unsafe_allow_html=True,
    )


def agent_grid() -> None:
    """The agents as cards. What each one DECIDES is the line that shows autonomy."""
    cards = []
    for index, (tag, name, role, decides) in enumerate(AGENT_ROWS):
        core = " core" if name == "supervisor" else ""
        # stagger the entrance so the grid assembles instead of snapping in
        delay = f"animation-delay:{index * 45}ms;"
        cards.append(
            f'<div class="agent{core}" style="{delay}"><div class="badge">{tag}</div>'
            f'<div class="nm">{html.escape(name)}</div>'
            f'<div class="role">{html.escape(role)}</div>'
            f'<div class="dec">{html.escape(decides)}</div></div>'
        )
    st.markdown(f'<div class="agents">{"".join(cards)}</div>', unsafe_allow_html=True)


def hero(chain: list[str]) -> None:
    """Gradient headline and a one-line promise. Compact — the command bar follows it."""
    live = f"{len(chain)} providers wired" if chain else "no provider configured"
    st.markdown(
        '<div class="hero">'
        f'<div class="eyebrow"><span class="pip"></span>{html.escape(live)} · $0 / month</div>'
        # a div, not an h1 — streamlit styles headings itself and wins the specificity fight
        '<div class="h1">Research that shows<br/>its own reasoning.</div>'
        '<p class="lede">Ask a question and it is triaged first, then planned, searched, '
        "analysed, written up with citations and reviewed. A <b>supervisor LLM is spent only "
        "where state leaves the next step genuinely open</b> — and every one of those calls, "
        "and every one it skipped, is on screen.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def stat_strip() -> None:
    """Four numbers that frame the system before a run exists to describe it."""
    cells = [
        ("8", "autonomous agents"),
        ("4", "depth budgets"),
        ("3", "evaluation layers"),
        ("$0", "monthly cost"),
    ]
    st.markdown(
        '<div class="stats">'
        + "".join(
            f'<div class="stat" style="animation-delay:{i * 60}ms">'
            f'<div class="n">{n}</div><div class="l">{label_text}</div></div>'
            for i, (n, label_text) in enumerate(cells)
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def route_demo() -> None:
    """A real routing transcript. The two marked lines are decisions a fixed chain cannot make."""
    lines = []
    for who, why, to, looped in ROUTE_DEMO:
        mark = ' <span class="loop">&#8635; re-route</span>' if looped else ""
        lines.append(
            f'<div><span class="who">{who.ljust(11)}</span>'
            f'<span class="why">{html.escape(why).ljust(34)}</span>'
            f'<span class="arr">&rarr;</span> <span class="to">{to}</span>{mark}</div>'
        )
    st.markdown(f'<div class="trace">{"".join(lines)}</div>', unsafe_allow_html=True)


def _visits(events: list[ProgressEvent]) -> list[tuple[ProgressEvent, float]]:
    """One (event, seconds-spent) pair per agent visit — shared by the timeline and chart."""
    visits: list[tuple[ProgressEvent, float]] = []
    previous = 0.0
    for event in events:
        if event.agent and event.agent != "supervisor":
            visits.append((event, max(event.elapsed_s - previous, 0.0)))
        previous = event.elapsed_s or previous
    return visits


def pipeline_flow(events: list[ProgressEvent]) -> None:
    """The worker lane as boxes, with a multiplier on any agent the supervisor ran twice."""
    finished = any(e.status in TERMINAL for e in events)
    failed = any(e.status == "failed" for e in events)
    counts: dict[str, int] = {}
    for event, _ in _visits(events):
        counts[event.agent] = counts.get(event.agent, 0) + 1
    current = next((e.agent for e in reversed(events) if e.agent and e.agent != "supervisor"), "")

    # a clarification run never enters the worker lane, so showing six pending boxes would
    # read as a failed run rather than a question asked
    lane = CLARIFY_LANE if "clarify" in counts else WORKERS

    nodes = []
    for index, name in enumerate(lane):
        if index:
            nodes.append('<span class="sep">&rsaquo;</span>')
        if name == current and not finished:
            state = "failed" if failed else "active"
        elif name in counts:
            state = "done"
        else:
            state = "pending"
        # always emit the line, blank when visited once, or the nodes sit on a ragged baseline
        seen_count = counts.get(name, 0)
        hits = f"&times;{seen_count}" if seen_count > 1 else "&nbsp;"
        nodes.append(
            f'<div class="node {state}"><div class="box">{FLOW_TAGS[name]}</div>'
            f'<div class="nm">{name}</div><div class="hits">{hits}</div></div>'
        )
    st.markdown(f'<div class="flow">{"".join(nodes)}</div>', unsafe_allow_html=True)


def _row(name: str, note: str, state: str, at: str) -> str:
    return (
        f'<div class="tl-row {state}"><span class="dot"></span>'
        f'<span class="nm">{html.escape(name)}</span>'
        f'<span class="at">{html.escape(at)}</span>'
        f'<span class="msg">{html.escape(note)}</span></div>'
    )


def agent_timeline(events: list[ProgressEvent]) -> None:
    """One row per agent visit with how long it took. A repeat visit gets its own row."""
    finished = any(e.status in TERMINAL for e in events)
    failed = any(e.status == "failed" for e in events)
    visits = _visits(events)

    rows = []
    for position, (event, took) in enumerate(visits):
        last = position == len(visits) - 1
        if last and failed:
            state = "failed"
        elif last and not finished:
            state = "active"
        else:
            state = "done"
        rows.append(_row(event.agent, event.message, state, f"{took:.1f}s" if took else ""))

    seen = {event.agent for event, _ in visits}
    rows += [_row(name, "not visited", "pending", "") for name in WORKERS if name not in seen]
    st.markdown(f'<div class="timeline">{"".join(rows)}</div>', unsafe_allow_html=True)


def duration_chart(events: list[ProgressEvent]) -> None:
    """Per-agent wall time — the same numbers as the timeline, easier to compare."""
    visits = _visits(events)
    if not visits:
        return
    totals: dict[str, float] = {}
    for event, took in visits:
        totals[event.agent] = totals.get(event.agent, 0.0) + took
    # horizontal, or altair rotates the agent names vertically and they become unreadable
    st.bar_chart(
        pd.Series(totals, name="seconds"),
        color=TOKENS["accent"],
        height=230,
        horizontal=True,
    )


def run_header(events: list[ProgressEvent], session_id: str = "") -> None:
    """Session, stage, elapsed and event count — the strip a build tool puts up top."""
    if not events:
        return
    latest = events[-1]
    done = latest.status in TERMINAL
    stage = latest.agent or ("finished" if done else "starting")
    live = "" if done else '<span class="live">live</span><span class="sep">|</span>'
    st.markdown(
        f'<div class="runbar">{live}'
        f"<span>run <b>{html.escape(session_id or '—')}</b></span>"
        '<span class="sep">|</span>'
        f"<span>stage <b>{html.escape(stage)}</b></span>"
        '<span class="sep">|</span>'
        f"<span><b>{latest.progress_pct}%</b></span>"
        '<span class="sep">|</span>'
        f"<span><b>{latest.elapsed_s:.1f}s</b></span>"
        '<span class="sep">|</span>'
        f"<span>{len(events)} events</span></div>",
        unsafe_allow_html=True,
    )


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
    st.markdown(f'<div class="log">{"".join(lines)}</div>', unsafe_allow_html=True)


def verdict_banner(result: ResearchResult | None, error: str | None) -> None:
    """One glanceable line: did the critic approve this, and on which pass."""
    if error and result is None:
        st.markdown(
            '<div class="verdict bad"><span class="ic">&#10005;</span>'
            f"<div><b>run failed</b><span class='sub'>{html.escape(error)}</span></div></div>",
            unsafe_allow_html=True,
        )
        return
    if result is None or result.trace is None:
        return

    if result.awaiting_clarification:
        st.markdown(
            '<div class="verdict warn"><span class="ic">&#63;</span>'
            "<div><b>the question needs one more detail</b>"
            "<span class='sub'>triage stopped the run before anything was spent — "
            "answer above and ask again</span></div></div>",
            unsafe_allow_html=True,
        )
        return

    floor = get_settings().quality_approve_threshold
    overall = result.scores.get("overall", 0.0)
    revisions = result.trace.revision_count
    if overall >= floor:
        css, icon = "good", "&#10003;"
        headline = (
            "approved on the first pass"
            if not revisions
            else f"approved after {revisions} revision(s)"
        )
    else:
        css, icon = "warn", "&#9888;"
        headline = (
            f"below the approval floor — best available draft shown ({revisions} revision(s))"
        )

    st.markdown(
        f'<div class="verdict {css}"><span class="ic">{icon}</span>'
        f"<div><b>{html.escape(headline)}</b>"
        f"<span class='sub'>overall {overall:.2f} · floor {floor:.2f} · "
        f"hint {html.escape(result.trace.routing_hint or 'n/a')}</span></div></div>",
        unsafe_allow_html=True,
    )


def run_metrics(trace: RunTrace | None, agent_path: list[str], elapsed: float | None) -> None:
    """The numbers that explain the run, above the report rather than buried under it."""
    if trace is None:
        return
    floor = get_settings().research_quality_threshold
    cells = [
        ("sources", str(trace.source_count), ""),
        ("research q", f"{trace.research_quality:.2f}", badge_class(trace.research_quality, floor)),
        ("revisions", str(trace.revision_count), ""),
        ("agent hops", str(len(agent_path)), ""),
        ("routing calls", str(len(trace.decisions)), ""),
    ]
    if elapsed is not None:
        cells.append(("elapsed", f"{elapsed:.0f}s", ""))
    st.markdown(
        '<div class="metrics">'
        + "".join(
            f'<div class="metric {state}"><div class="k">{k}</div>'
            f'<div class="v">{html.escape(v)}</div></div>'
            for k, v, state in cells
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def score_dashboard(scores: dict[str, float]) -> None:
    """The critic's four dimensions, coloured against the approval floor."""
    floor = get_settings().quality_approve_threshold
    present = [(name, scores[name]) for name in DIMENSIONS if name in scores]
    if not present:
        st.caption("no scores — the critic did not run")
        return
    st.markdown(
        '<div class="metrics">'
        + "".join(
            f'<div class="metric {badge_class(value, floor)}">'
            f'<div class="k">{name.replace("_", " ")}</div>'
            f'<div class="v">{value:.2f}</div></div>'
            for name, value in present
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def research_plan(trace: RunTrace | None) -> None:
    """The planner's actual task list — it drove the whole run and used to be invisible."""
    if trace is None or not trace.research_plan:
        return
    label("research plan", f"{len(trace.research_plan)} tasks")
    describe(
        "What the planner broke the question into. Every source is tagged with the task it "
        "came from, so the research is traceable back to a task."
    )
    if trace.research_strategy:
        st.markdown(
            f'<div class="desc"><b>strategy</b> — {html.escape(trace.research_strategy)}</div>',
            unsafe_allow_html=True,
        )
    rows = []
    for task in trace.research_plan:
        rows.append(
            f'<div class="task"><div class="id">{html.escape(str(task.get("task_id", "t?")))}</div>'
            f'<div><div class="w">{html.escape(str(task.get("description", "")))}</div>'
            f'<div class="m">assigned to {html.escape(str(task.get("assigned_to", "researcher")))}'
            "</div></div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def analysis_view(trace: RunTrace | None) -> None:
    """The analyst's synthesis. It sits between the sources and the report and was hidden."""
    if trace is None or not trace.analysis:
        return
    label("analyst synthesis")
    describe(
        "What the analyst concluded from the raw sources. The writer drafts from this, not "
        "directly from the search results."
    )
    with st.container(border=True):
        st.markdown(trace.analysis)
    for output in trace.code_outputs:
        with st.expander(f"code execution · {output.get('purpose', 'computation')}"):
            st.code(str(output.get("code", "")), language="python")
            st.code(str(output.get("output", "")), language=None)


def _decision_row(entry: dict[str, Any]) -> str:
    chosen = str(entry.get("to_agent", "?"))
    expected = str(entry.get("expected_agent", "?"))
    source = "LLM" if entry.get("llm_decided") else "rule"
    flag = ' <span class="diverged">diverged</span>' if chosen != expected else ""
    return (
        f"<tr><td>{entry.get('step', '?')}</td>"
        f"<td>{html.escape(str(entry.get('from_agent', '?')))} &rarr; "
        f"<b>{html.escape(chosen)}</b>{flag}</td>"
        f"<td>{source}</td>"
        f"<td><code>{html.escape(str(entry.get('matched_rule', '?')))}</code></td>"
        f"<td>{float(entry.get('research_quality', 0.0)):.2f}</td>"
        f"<td>{float(entry.get('quality_score', 0.0)):.2f}</td></tr>"
    )


def decision_trace(trace: RunTrace | None) -> None:
    """Every supervisor call: what it chose, what the documented rule expected, and why."""
    if trace is None or not trace.decisions:
        return
    diverged = sum(1 for d in trace.decisions if d.get("to_agent") != d.get("expected_agent"))
    settled = sum(1 for d in trace.decisions if not d.get("llm_decided"))
    label("routing decisions", f"{len(trace.decisions)}")
    describe(
        f"{settled} of {len(trace.decisions)} were settled by state and cost no model call; "
        "the rest were genuine judgement calls at one of the two gates. "
        + (
            f"{diverged} diverged from the naive reading of the state — that is the system "
            "reasoning, not a bug."
            if diverged
            else "None diverged from the naive reading of the state on this run."
        )
    )
    st.markdown(
        '<table class="tbl"><thead><tr>'
        "<th>#</th><th>route</th><th>by</th><th>rule</th><th>research</th><th>quality</th>"
        f"</tr></thead><tbody>{''.join(_decision_row(e) for e in trace.decisions)}</tbody></table>",
        unsafe_allow_html=True,
    )


def react_discipline(trace: RunTrace | None) -> None:
    """Whether the researcher stopped because it was satisfied or ran out of budget."""
    if trace is None or not trace.react_stats:
        return
    stopped = sum(1 for s in trace.react_stats.values() if s.get("self_terminated"))
    label("react discipline", f"{stopped}/{len(trace.react_stats)} tasks self-terminated")
    describe(
        "The researcher decides for itself when it has enough evidence. Hitting the iteration "
        "cap is not a failure, but a run where nothing self-terminates means the cap is doing "
        "the deciding instead of the agent."
    )
    rows = "".join(
        f"<tr><td><code>{html.escape(str(task))}</code></td>"
        f"<td>{stats.get('iterations_used', '?')}</td>"
        f"<td>{'self-stopped' if stats.get('self_terminated') else 'hit the cap'}</td></tr>"
        for task, stats in trace.react_stats.items()
    )
    st.markdown(
        '<table class="tbl"><thead><tr><th>task</th><th>iterations</th>'
        f"<th>ended because</th></tr></thead><tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )


def critic_verdict(trace: RunTrace | None) -> None:
    """What the critic actually said, rather than only the number it produced."""
    if trace is None or not (trace.critic_feedback or trace.top_issue):
        return
    label("critic feedback")
    describe(
        "The critic's own words. Its routing hint is what sent the draft back, or approved it."
    )
    if trace.top_issue:
        st.markdown(f"**Top issue** — {trace.top_issue}")
    if trace.critic_feedback:
        st.markdown(trace.critic_feedback)


def evaluation_layers(scores: dict[str, float]) -> None:
    """Layers 1 and 2, with unscored metrics named as unscored instead of shown as 0.00."""
    floor = get_settings().quality_approve_threshold
    rows = []
    for metric in EVAL_METRICS:
        value = scores.get(f"eval_{metric}")
        if value is None:
            rows.append(
                f"<tr><td>{metric.replace('_', ' ')}</td>"
                '<td class="unscored">not scored — judge unavailable</td></tr>'
            )
        else:
            rows.append(
                f"<tr><td>{metric.replace('_', ' ')}</td>"
                f'<td class="v-{badge_class(value, floor)}">{value:.2f}</td></tr>'
            )
    st.markdown(
        '<table class="tbl"><thead><tr><th>metric</th><th>score</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>",
        unsafe_allow_html=True,
    )


def source_list(citations: list[dict[str, Any]], sources: list[dict[str, Any]]) -> None:
    """Cited references first, then everything the researcher read but the writer did not use."""
    cited_urls = {str(item.get("url", "")) for item in citations}
    if citations:
        rows = []
        for item in citations:
            title = html.escape(str(item.get("title") or item.get("url") or "untitled"))
            url = html.escape(str(item.get("url") or ""), quote=True)
            link = f'<a href="{url}" target="_blank">{title}</a>' if url else title
            rows.append(
                f'<div class="src"><div class="hd">'
                f'<span class="n">[{item.get("index", "?")}]</span>{link}</div>'
                f'<div class="u">{html.escape(str(item.get("url") or ""))}</div></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
    else:
        st.caption("no citations")

    uncited = [s for s in sources if str(s.get("url", "")) not in cited_urls]
    if not uncited:
        return
    with st.expander(f"gathered but not cited ({len(uncited)})"):
        st.caption(
            "The researcher read these and the writer chose not to cite them. A large gap here "
            "means the research was broad but the report was selective."
        )
        rows = []
        for item in uncited:
            title = html.escape(str(item.get("title") or item.get("url") or "untitled"))
            url = html.escape(str(item.get("url") or ""), quote=True)
            link = f'<a href="{url}" target="_blank">{title}</a>' if url else title
            rows.append(
                f'<div class="src"><div class="hd">'
                f'<span class="n">{html.escape(str(item.get("task_id", "")))}</span>{link}</div>'
                f'<div class="u">{html.escape(str(item.get("url") or ""))}</div>'
                f'<div class="s">{html.escape(str(item.get("snippet", ""))[:240])}</div></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)


def example_queries() -> None:
    """Short labels, not truncated sentences — four grey boxes ending in an ellipsis read broken."""
    cols = st.columns([1, 1, 1, 1, 2])
    for col, (tag, query) in zip(cols, EXAMPLE_QUERIES, strict=False):
        if col.button(tag, key=f"ex_{tag}", use_container_width=True):
            # these chips render below the form, which streamlit has already drawn by now,
            # so the value only lands if we re-run the script from the top
            st.session_state["prefill_query"] = query
            st.rerun()


def provider_strip() -> None:
    """Which providers are usable right now, and a loud warning when none are."""
    from amaris.llm.router import configured_chain, provider_status

    chain = configured_chain()
    cooling = provider_status()
    chips = (
        "".join(
            f'<span class="chip {"hot" if cooling.get(name) else "on"}"><span class="dot"></span>'
            f"{name}{f' {cooling[name]:.0f}s' if cooling.get(name) else ''}</span> "
            for name in chain
        )
        or '<span class="chip off">none configured</span>'
    )
    st.markdown(chips, unsafe_allow_html=True)

    # a run started against a fully parked chain crawls through retries and then fails;
    # saying so up front is cheaper than watching it for nineteen minutes
    ready = [name for name in chain if not cooling.get(name)]
    if chain and not ready:
        st.error("every provider is rate limited — a run started now will crawl and likely fail.")
    elif chain and len(ready) < len(chain):
        st.warning(f"only {', '.join(ready)} usable — expect slower runs.")


def kv_rows(rows: dict[str, str], boxed: bool = False) -> None:
    """Mono key/value list, for configuration that is read rather than interacted with."""
    body = "".join(
        f'<div class="kv"><span class="k">{html.escape(k)}</span>'
        f'<span class="v">{html.escape(str(v))}</span></div>'
        for k, v in rows.items()
    )
    # boxed when it sits beside a bordered panel, or the column reads as unfinished
    st.markdown(f'<div class="boxed">{body}</div>' if boxed else body, unsafe_allow_html=True)


def record_run(
    query: str,
    result: ResearchResult | None,
    events: list[ProgressEvent],
    session_id: str,
    error: str | None,
    elapsed: float,
) -> None:
    """Keep the last few finished runs so starting a new query doesn't erase the last one."""
    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    history.insert(
        0,
        {
            "query": query,
            "result": result,
            "events": events,
            "session_id": session_id,
            "error": error,
            "elapsed": elapsed,
        },
    )
    del history[HISTORY_LIMIT:]


def history_sidebar() -> None:
    """Past runs, one click to reopen a finished one without paying for it again."""
    history: list[dict[str, Any]] = st.session_state.get("history", [])
    if not history:
        st.caption("no runs yet")
        return
    for index, entry in enumerate(history):
        result: ResearchResult | None = entry["result"]
        overall = result.scores.get("overall") if result else None
        tag = f"{overall:.2f}" if overall is not None else ("err" if entry["error"] else "—")
        query = str(entry["query"])
        label_text = query if len(query) <= 22 else f"{query[:22]}…"
        if st.button(f"{tag}  {label_text}", key=f"hist_{index}", use_container_width=True):
            st.session_state.update(
                result=entry["result"],
                events=entry["events"],
                session_id=entry["session_id"],
                error=entry["error"],
                elapsed=entry["elapsed"],
                last_query=entry["query"],
            )
        st.markdown(
            f'<div class="hist-m">{entry["elapsed"] or 0:.0f}s · {len(entry["events"])} ev</div>',
            unsafe_allow_html=True,
        )


def raw_inspector(result: Any, events: list[ProgressEvent]) -> None:
    """The unfiltered payload. Nothing the system produced should be unreachable from the UI."""
    st.json(
        {
            "result": result.model_dump() if result is not None else None,
            "events": [e.model_dump() for e in events],
        },
        expanded=False,
    )

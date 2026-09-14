"""Composed screens. Everything here renders — app.py owns the I/O and hands results in."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import streamlit as st

from amaris.config.settings import get_settings
from frontend.components import (
    agent_grid,
    agent_timeline,
    analysis_view,
    critic_verdict,
    decision_trace,
    describe,
    duration_chart,
    evaluation_layers,
    event_log,
    kv_rows,
    label,
    pipeline_flow,
    raw_inspector,
    react_discipline,
    research_plan,
    route_demo,
    run_header,
    run_metrics,
    score_dashboard,
    source_list,
    stat_strip,
    verdict_banner,
)

if TYPE_CHECKING:
    from amaris.api.schemas import ProgressEvent, ResearchResult
    from amaris.config.validate import ConfigReport


def landing() -> None:
    """The empty state. It sells the system and explains it, rather than sitting blank."""
    stat_strip()

    label("the agents")
    describe(
        "Each agent owns a decision rather than a step. Triage decides how much the question "
        "is worth spending before anything is spent, and the supervisor decides the rest — but "
        "only where the state of the run leaves the next step genuinely open."
    )
    agent_grid()

    left, right = st.columns([3, 2], gap="medium")
    with left:
        label("routing", "example run")
        describe(
            "Forced hops are graph edges and cost nothing. Only the gate lines are model calls, "
            "and a fixed chain cannot produce the two marked ones."
        )
        route_demo()
    with right:
        label("state that carries the autonomy")
        describe(
            "Three fields in GraphState are what make the routing dynamic rather than declared."
        )
        kv_rows(
            {
                "query_depth": "triage sets every budget from it",
                "next_agent": "supervisor writes, graph routes",
                "research_quality": "researcher self-scores 0-1",
                "routing_hint": "approve / fix / research / re-plan",
                "decision_log": "every hop, and what it cost",
                "react_stats": "did it self-stop or hit the cap",
            },
            boxed=True,
        )
        label("stack")
        kv_rows(
            {
                "orchestration": "langgraph StateGraph",
                "providers": "groq → gemini → glm",
                "evaluation": "ragas + trajectory",
                "transport": "fastapi + websocket",
                "memory": "redis / qdrant / mem0",
            },
            boxed=True,
        )


def live_run(events: list[ProgressEvent]) -> None:
    """Repainted on every progress event while a run is in flight."""
    run_header(events)
    pipeline_flow(events)
    agent_timeline(events)
    event_log(events)


def _report_tab(result: ResearchResult, session_id: str) -> None:
    label("report")
    describe(
        "Written by the writer agent from the analyst's synthesis. Every [n] marker points at "
        "a real source in the Sources tab."
    )
    with st.container(border=True):
        st.markdown(result.report)
    st.download_button(
        "download .md",
        data=result.report,
        file_name=f"amaris_{session_id[:8] or 'report'}.md",
        mime="text/markdown",
    )


def _execution_tab(result: ResearchResult, events: list[ProgressEvent]) -> None:
    label("route taken", f"{len(result.agent_path)} hops")
    describe(
        f"{' → '.join(result.agent_path)}. An agent appearing twice means the supervisor "
        "sent work back to it."
    )
    pipeline_flow(events)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        label("timeline")
        describe("Per-visit duration, in the order the agents actually ran.")
        agent_timeline(events)
    with right:
        label("time per agent")
        describe("A researcher or analyst spike usually means the provider chain fell back.")
        duration_chart(events)

    decision_trace(result.trace)
    react_discipline(result.trace)
    critic_verdict(result.trace)

    label("event stream", f"{len(events)}")
    describe("Every node transition the run emitted, in order, with timings.")
    event_log(events)


def _evidence_tab(result: ResearchResult) -> None:
    research_plan(result.trace)
    analysis_view(result.trace)


def _sources_tab(result: ResearchResult) -> None:
    sources = result.trace.sources if result.trace else []
    label("sources", f"{len(result.citations)} cited of {len(sources)} gathered")
    describe(
        "Cited references are numbered exactly as the report numbers them. Everything the "
        "researcher read but the writer left out is listed underneath."
    )
    source_list(result.citations, sources)


def _evaluation_tab(result: ResearchResult) -> None:
    label("critic scores", "layer 3")
    describe(
        "The critic reads the draft against the sources and scores five dimensions. answer_fit "
        "caps the overall score, so a polished report about the wrong subject cannot pass."
    )
    score_dashboard(result.scores)
    label("ragas evaluation", "layers 1-2")
    describe(
        "Layer 1 scores the gathered sources against the question. Layer 2 scores the report "
        "against those sources. Both need a separate LLM judge, so a rate-limited run says "
        "'not scored' rather than pretending the answer was bad."
    )
    evaluation_layers(result.scores)


def system_tab(report: ConfigReport | None = None) -> None:
    """Full configuration. This lives in the main pane because the sidebar has no room for it."""
    settings = get_settings()
    left, right = st.columns(2, gap="medium")
    with left:
        label("model routing")
        describe(
            "Calls try each provider in order and fall through on a rate limit or an error. The "
            "evaluator's judge uses a different provider from the pipeline on purpose, so it is "
            "not starved by the rate window the run just spent."
        )
        kv_rows(
            {
                "primary": settings.primary_provider,
                "eval judge": settings.eval_provider,
                "groq reasoning": settings.groq_model_reasoning,
                "groq fast": settings.groq_model_fast,
                "gemini": settings.gemini_model_fallback,
                "glm": settings.glm_model,
                "anthropic": settings.anthropic_model,
            },
            boxed=True,
        )
        label("deployment")
        describe(
            "Local mode runs a FastAPI backend with Redis and Qdrant. Cloud mode runs the same "
            "graph in-process, so the demo works without Docker."
        )
        kv_rows(
            {
                "mode": settings.deployment_mode,
                "api base": settings.api_base_url,
                "qdrant": settings.qdrant_url or f"{settings.qdrant_host}:{settings.qdrant_port}",
                "checkpoints": settings.sqlite_checkpoint_db,
                "logs": settings.log_dir,
            },
            boxed=True,
        )
    with right:
        label("agent limits")
        describe(
            "The caps that stop a run looping forever. Each is a deliberate ceiling, and the "
            "Execution tab reports whenever an agent stopped because of a cap rather than by "
            "its own judgement."
        )
        kv_rows(
            {
                "research floor": f"{settings.research_quality_threshold:.2f}",
                "approve floor": f"{settings.quality_approve_threshold:.2f}",
                "max revisions": str(settings.max_revisions),
                "react cap": str(settings.max_react_iterations),
                "supervisor cap": str(settings.max_supervisor_steps),
                "retry budget": f"{settings.llm_retry_budget_seconds:.0f}s",
                "ragas timeout": f"{settings.ragas_timeout_seconds:.0f}s",
            },
            boxed=True,
        )
        if report and (report.errors or report.warnings):
            label("config health")
            for item in report.errors:
                st.error(item)
            for item in report.warnings:
                st.warning(item)


def results(
    result: ResearchResult | None,
    session_id: str,
    error: str | None,
    elapsed: float | None,
    events: list[ProgressEvent],
    config_report: ConfigReport | None = None,
) -> None:
    """Everything a finished run produced, split so each tab answers one question."""
    run_header(events, session_id)
    verdict_banner(result, error)
    if error and result is not None:
        st.warning(f"the run reported an error: {error}")
    if result is None or not result.report:
        st.error("no report was produced")
        if events:
            label("what happened before it stopped")
            pipeline_flow(events)
            event_log(events)
        return

    run_metrics(result.trace, result.agent_path, elapsed)

    report_tab, exec_tab, evidence_tab, sources_tab, eval_tab, sys_tab, raw_tab = st.tabs(
        ["report", "execution", "evidence", "sources", "evaluation", "system", "raw"]
    )
    with report_tab:
        _report_tab(result, session_id)
    with exec_tab:
        _execution_tab(result, events)
    with evidence_tab:
        _evidence_tab(result)
    with sources_tab:
        _sources_tab(result)
    with eval_tab:
        _evaluation_tab(result)
    with sys_tab:
        system_tab(config_report)
    with raw_tab:
        label("raw payload")
        describe(
            "Exactly what the API returned plus every progress event. If a number appears "
            "anywhere above, it came from here."
        )
        raw_inspector(result, events)
        st.download_button(
            "download .json",
            data=json.dumps(
                {"result": result.model_dump(), "events": [e.model_dump() for e in events]},
                indent=2,
            ),
            file_name=f"amaris_{session_id[:8] or 'run'}_raw.json",
            mime="application/json",
        )

    st.caption(f"session {session_id} — replays this run from logs/amaris.jsonl")

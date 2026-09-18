"""Composed screens. Everything here renders — app.py owns the I/O and hands results in."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import streamlit as st

from amaris.config.settings import get_settings
from amaris.observability.tracing import active_backend
from frontend.components import (
    agent_timeline,
    analysis_view,
    citation_audit,
    critic_verdict,
    decision_trace,
    describe,
    duration_chart,
    event_log,
    kv_rows,
    label,
    pipeline_flow,
    raw_inspector,
    react_discipline,
    research_plan,
    run_header,
    run_metrics,
    score_dashboard,
    source_list,
    stat_strip,
    tool_calls,
    verdict_banner,
)

if TYPE_CHECKING:
    from amaris.api.schemas import ProgressEvent, ResearchResult
    from amaris.config.validate import ConfigReport


def landing() -> None:
    """The framing numbers. Not called by app.py any more — the landing page is the wordmark,
    the examples and the composer, and four numbers above them read as a brochure."""
    stat_strip()


def live_run(events: list[ProgressEvent]) -> None:
    """Repainted on every progress event while a run is in flight."""
    run_header(events)
    pipeline_flow(events)
    agent_timeline(events)
    event_log(events)


def _execution_tab(result: ResearchResult, events: list[ProgressEvent]) -> None:
    label("route taken", f"{len(result.agent_path)} hops")
    describe(f"{' → '.join(result.agent_path)} · a repeat means the supervisor sent work back")
    pipeline_flow(events)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        label("timeline")
        describe("Per visit, in execution order.")
        agent_timeline(events)
    with right:
        label("time per agent")
        describe("A spike usually means the provider chain fell back.")
        duration_chart(events)

    decision_trace(result.trace)
    react_discipline(result.trace)
    tool_calls(result.trace)
    critic_verdict(result.trace)

    label("event stream", f"{len(events)}")
    describe("Every node transition, with timings.")
    event_log(events)


def _evidence_tab(result: ResearchResult) -> None:
    if result.trace and result.trace.triage:
        triage = result.trace.triage
        label("triage", str(triage.get("depth", "")))
        describe("Read before anything was spent. Every budget below derives from it.")
        kv_rows(
            {
                "depth": str(triage.get("depth", "")),
                "word target": str(triage.get("word_target", "")),
                "sections": ", ".join(triage.get("sections", [])) or "—",
                "reason": str(triage.get("reason", "")) or "—",
            },
            boxed=True,
        )
    research_plan(result.trace)
    analysis_view(result.trace)


def _sources_tab(result: ResearchResult) -> None:
    sources = result.trace.sources if result.trace else []
    label("sources", f"{len(result.citations)} cited of {len(sources)} gathered")
    describe("Numbered exactly as the report numbers them. Uncited sources follow.")
    source_list(result.citations, sources)


def _evaluation_tab(result: ResearchResult) -> None:
    label("critic scores", "in-run review")
    describe("answer_fit caps the overall, so a polished answer to the wrong question cannot pass.")
    score_dashboard(result.scores)
    label("citation audit", "no model call")
    describe("Every figure in the answer, looked up in the page that sentence cites.")
    citation_audit(result.trace.audit if result.trace else {})


def system_tab(report: ConfigReport | None = None) -> None:
    """Full configuration. This lives in the main pane because the sidebar has no room for it."""
    settings = get_settings()
    left, right = st.columns(2, gap="medium")
    with left:
        label("model routing")
        describe(
            "Tried in order, falling through on a rate limit. The offline judge uses the tail."
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
        describe("Cloud mode runs the same graph in-process, with no Docker.")
        kv_rows(
            {
                "mode": settings.deployment_mode,
                "api base": settings.api_base_url,
                "qdrant": settings.qdrant_url or f"{settings.qdrant_host}:{settings.qdrant_port}",
                "checkpoints": settings.sqlite_checkpoint_db,
                "logs": settings.log_dir,
                "tracing": active_backend() or "off",
            },
            boxed=True,
        )
    with right:
        label("agent limits")
        describe("Ceilings, not targets. A shallow question gets a lower revision cap.")
        kv_rows(
            {
                "research floor": f"{settings.research_quality_threshold:.2f}",
                "approve floor": f"{settings.quality_approve_threshold:.2f}",
                "max revisions": str(settings.max_revisions),
                "react cap": str(settings.max_react_iterations),
                "supervisor cap": str(settings.max_supervisor_steps),
                "retry budget": f"{settings.llm_retry_budget_seconds:.0f}s",
                "ragas timeout": f"{settings.ragas_timeout_seconds:.0f}s (offline only)",
            },
            boxed=True,
        )
        if report and (report.errors or report.warnings):
            label("config health")
            for item in report.errors:
                st.error(item)
            for item in report.warnings:
                st.warning(item)


def inspection(
    result: ResearchResult | None,
    session_id: str,
    error: str | None,
    elapsed: float | None,
    events: list[ProgressEvent],
    config_report: ConfigReport | None = None,
) -> None:
    """Everything one turn left behind, split so each tab answers a single question."""
    if result is None or not result.report:
        verdict_banner(result, error)
        if events:
            label("what happened before it stopped")
            pipeline_flow(events)
            event_log(events)
        return

    if error:
        st.warning(f"the run reported an error: {error}")

    label("how this run went", session_id[:8])
    describe("The evidence behind the answer above, and what the run chose to spend.")
    run_metrics(result.trace, result.agent_path, elapsed)

    exec_tab, evidence_tab, sources_tab, eval_tab, sys_tab, raw_tab = st.tabs(
        ["execution", "evidence", "sources", "evaluation", "system", "raw"]
    )
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
        describe("Every number shown above came from here.")
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

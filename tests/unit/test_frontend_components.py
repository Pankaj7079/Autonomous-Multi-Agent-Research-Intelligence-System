"""Components render and never fetch, so they are testable with a fake streamlit."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from amaris.api.schemas import ProgressEvent
from frontend import components
from frontend.styles import WARN_FLOOR, badge_class


class FakeStreamlit:
    """Records what a component drew instead of drawing it."""

    def __init__(self) -> None:
        self.html: list[str] = []
        self.captions: list[str] = []
        self.codes: list[str] = []

    def markdown(self, body: str, **_: Any) -> None:
        self.html.append(body)

    def caption(self, body: str, **_: Any) -> None:
        self.captions.append(body)

    def code(self, body: str, **_: Any) -> None:
        self.codes.append(str(body))

    @contextlib.contextmanager
    def expander(self, label: str, **_: Any):
        self.captions.append(label)
        yield

    @contextlib.contextmanager
    def container(self, **_: Any):
        yield

    @property
    def drawn(self) -> str:
        return "".join(self.html)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeStreamlit:
    recorder = FakeStreamlit()
    monkeypatch.setattr(components, "st", recorder)
    return recorder


def _event(agent: str, status: str = "running", pct: int = 45) -> ProgressEvent:
    return ProgressEvent(agent=agent, status=status, message=f"{agent} working", progress_pct=pct)


def test_badge_class_uses_the_critics_threshold_not_a_ui_constant() -> None:
    """0.72 is the approval threshold — a UI that greens 0.70 calls a rejected run good."""
    assert badge_class(0.72, 0.72) == "good"
    assert badge_class(0.71, 0.72) == "warn"
    assert badge_class(WARN_FLOOR - 0.01, 0.72) == "bad"


# ── the execution timeline ────────────────────────────────────────────────


def test_a_repeat_visit_gets_its_own_row(fake: FakeStreamlit) -> None:
    """Two researcher rows are the visible proof the supervisor re-routed."""
    events = [_event("planner"), _event("researcher"), _event("researcher")]
    components.agent_timeline(events)
    assert fake.drawn.count(">researcher<") == 2


def test_the_supervisor_never_gets_a_row(fake: FakeStreamlit) -> None:
    components.agent_timeline([_event("supervisor"), _event("planner")])
    assert ">supervisor<" not in fake.drawn


def test_the_current_agent_is_active_and_the_rest_are_done(fake: FakeStreamlit) -> None:
    components.agent_timeline([_event("planner"), _event("researcher")])
    drawn = fake.drawn
    assert drawn.index("tl-row done") < drawn.index("tl-row active")


def test_a_finished_run_has_no_active_row(fake: FakeStreamlit) -> None:
    events = [_event("planner"), _event("writer"), _event("", status="done", pct=100)]
    components.agent_timeline(events)
    assert "tl-row active" not in fake.drawn


def test_a_failed_run_marks_the_last_row_failed(fake: FakeStreamlit) -> None:
    events = [_event("planner"), _event("researcher"), _event("", status="failed")]
    components.agent_timeline(events)
    assert "tl-row failed" in fake.drawn


def test_unvisited_agents_show_as_pending(fake: FakeStreamlit) -> None:
    components.agent_timeline([_event("planner")])
    assert fake.drawn.count("tl-row pending") == len(components.WORKERS) - 1


def test_the_timeline_shows_how_long_each_step_took(fake: FakeStreamlit) -> None:
    """Duration is the number worth showing; cumulative elapsed lives in the event log."""
    events = [_event("supervisor"), _event("planner"), _event("researcher")]
    events[0].elapsed_s = 0.4
    events[1].elapsed_s = 4.1
    events[2].elapsed_s = 61.8
    components.agent_timeline(events)
    assert "3.7s" in fake.drawn, "planner took 4.1 - 0.4"
    assert "57.7s" in fake.drawn, "researcher took 61.8 - 4.1"


# ── the pipeline flow strip ───────────────────────────────────────────────


def test_the_flow_marks_the_current_agent_active(fake: FakeStreamlit) -> None:
    components.pipeline_flow([_event("planner"), _event("researcher")])
    drawn = fake.drawn
    assert "node active" in drawn
    assert "node pending" in drawn


def test_the_flow_counts_a_repeat_visit(fake: FakeStreamlit) -> None:
    """One node cannot show two rows, so a re-route has to show as a multiplier."""
    events = [_event("researcher"), _event("critic"), _event("researcher")]
    components.pipeline_flow(events)
    assert "&times;2" in fake.drawn


def test_the_flow_has_no_active_node_once_the_run_is_done(fake: FakeStreamlit) -> None:
    events = [_event("writer"), _event("", status="done", pct=100)]
    components.pipeline_flow(events)
    assert "node active" not in fake.drawn


# ── scores ────────────────────────────────────────────────────────────────


def test_score_dashboard_draws_the_four_critic_dimensions(fake: FakeStreamlit) -> None:
    scores = {
        "faithfulness": 0.95,
        "completeness": 0.75,
        "coherence": 0.60,
        "citation_quality": 0.40,
    }
    components.score_dashboard(scores)
    drawn = fake.drawn
    assert drawn.count("metric ") == 4
    assert "metric good" in drawn
    assert "metric warn" in drawn
    assert "metric bad" in drawn


def test_an_answer_with_no_citations_says_so_rather_than_showing_zeroes(
    fake: FakeStreamlit,
) -> None:
    """Empty metrics read as a bad run; "cited nothing" reads as what actually happened."""
    components.citation_audit({})
    assert fake.captions
    assert "cited nothing" in " ".join(fake.captions)


def test_an_unsupported_claim_is_named_rather_than_averaged_into_a_score(
    fake: FakeStreamlit,
) -> None:
    """The whole reason RAGAS left the request path: a single faithfulness number hides the
    one claim that is wrong. The claim itself has to be reachable from the UI."""
    components.citation_audit(
        {
            "checked": 2,
            "grounded": 1,
            "unsupported": 1,
            "grounded_ratio": 0.5,
            "dead": 0,
            "cited_sources": 2,
            "domains": 2,
            "dated": 0,
            "recent": 0,
            "claims": [
                {
                    "text": "Mango is a stone fruit [1].",
                    "citation": 1,
                    "grounded": True,
                    "missing": [],
                },
                {
                    "text": "Mango cures diabetes in 92% of cases [2].",
                    "citation": 2,
                    "grounded": False,
                    "missing": ["92%"],
                },
            ],
        }
    )
    drawn = fake.drawn
    assert "1/2" in drawn
    assert "cures diabetes" in drawn
    # the reader is told which fact to go and look at, not given a score to interpret
    assert "92%" in drawn


def test_a_dead_citation_is_flagged_as_bad_not_merely_counted(fake: FakeStreamlit) -> None:
    """A citation pointing at a page that was never stored is unverifiable, which is worse
    than a low score — the reader cannot check it at all."""
    components.citation_audit(
        {
            "checked": 1,
            "grounded": 0,
            "unsupported": 1,
            "grounded_ratio": 0.0,
            "dead": 1,
            "cited_sources": 1,
            "domains": 1,
            "dated": 0,
            "recent": 0,
            "claims": [],
        }
    )
    assert "metric bad" in fake.drawn


def test_no_critic_scores_says_so_instead_of_drawing_empty_badges(fake: FakeStreamlit) -> None:
    components.score_dashboard({})
    assert "metric" not in fake.drawn
    assert fake.captions


# ── sources ───────────────────────────────────────────────────────────────


def test_citations_keep_the_writers_numbering(fake: FakeStreamlit) -> None:
    """[3] in the report must be [3] in the list, so the index is never recomputed here."""
    components.source_list([{"index": 7, "title": "Spec", "url": "https://example.com/spec"}], [])
    assert "[7]" in fake.drawn
    assert 'href="https://example.com/spec"' in fake.drawn


def test_citation_titles_are_escaped(fake: FakeStreamlit) -> None:
    components.source_list([{"index": 1, "title": "<script>x</script>", "url": ""}], [])
    assert "<script>" not in fake.drawn


def test_no_citations_renders_a_note_not_an_empty_list(fake: FakeStreamlit) -> None:
    components.source_list([], [])
    assert fake.drawn == ""
    assert fake.captions == ["no citations"]


def test_gathered_but_uncited_sources_are_still_reachable(fake: FakeStreamlit) -> None:
    """The researcher read 2 pages and the writer cited 1 — hiding the other is a black box."""
    citations = [{"index": 1, "title": "Used", "url": "https://a.test"}]
    sources = [
        {"title": "Used", "url": "https://a.test", "task_id": "t1", "snippet": "kept"},
        {"title": "Dropped", "url": "https://b.test", "task_id": "t2", "snippet": "unused"},
    ]
    components.source_list(citations, sources)
    assert "Dropped" in fake.drawn
    assert "b.test" in fake.drawn
    assert any("not cited (1)" in c for c in fake.captions)


# ── the views that stop the run being a black box ─────────────────────────


def _trace(**overrides):
    from amaris.api.schemas import RunTrace

    base = {
        "decisions": [
            {
                "step": 1,
                "from_agent": "START",
                "to_agent": "planner",
                "expected_agent": "planner",
                "matched_rule": "no_plan",
                "llm_decided": True,
                "research_quality": 0.0,
                "quality_score": 0.0,
            }
        ],
        "react_stats": {"t1": {"iterations_used": 2, "self_terminated": True}},
        "research_quality": 0.8,
        "revision_count": 1,
        "source_count": 12,
        "critic_feedback": "the draft is well sourced",
        "top_issue": "thin on recent developments",
    }
    return RunTrace(**{**base, **overrides})


def test_run_stats_surfaces_the_numbers_that_explain_the_run(fake: FakeStreamlit) -> None:
    components.run_metrics(_trace(), ["planner", "researcher"], 84.0)
    drawn = fake.drawn
    assert "12" in drawn and "0.80" in drawn and "84s" in drawn


def test_run_stats_without_a_trace_draws_nothing(fake: FakeStreamlit) -> None:
    components.run_metrics(None, [], None)
    assert fake.drawn == ""


def test_the_decision_trace_shows_the_rule_behind_each_hop(fake: FakeStreamlit) -> None:
    components.decision_trace(_trace())
    assert "no_plan" in fake.drawn
    assert "planner" in fake.drawn


def test_a_divergence_from_the_documented_rule_is_flagged(fake: FakeStreamlit) -> None:
    """The LLM overruling the rule table is the most interesting thing a run can do."""
    trace = _trace(
        decisions=[
            {
                "step": 1,
                "from_agent": "critic",
                "to_agent": "researcher",
                "expected_agent": "writer",
                "matched_rule": "fix_writing",
                "llm_decided": True,
                "llm_called": True,
                "research_quality": 0.5,
                "quality_score": 0.6,
            }
        ]
    )
    components.decision_trace(trace)
    drawn = fake.drawn
    assert "diverged" in drawn
    assert "1 genuine judgement call" in drawn, "the heading must say so, not only the row"
    assert "0 of 1 needed no model call at all" in drawn


def test_a_settled_decision_shows_whether_the_audit_call_agreed(fake: FakeStreamlit) -> None:
    """The whole point of ADR-050: a settled decision still shows the model's own opinion,
    even though that opinion never had the power to change what happened."""
    trace = _trace(
        decisions=[
            {
                "step": 1,
                "from_agent": "researcher",
                "to_agent": "analyst",
                "expected_agent": "analyst",
                "matched_rule": "settled_quality_met",
                "llm_decided": False,
                "llm_called": True,
                "shadow_choice": "researcher",
                "shadow_agreed": False,
                "research_quality": 0.8,
                "quality_score": 0.0,
            }
        ]
    )
    components.decision_trace(trace)
    drawn = fake.drawn
    assert "would have chosen researcher" in drawn
    assert "0 of 1 needed no model call at all" in drawn
    assert "agreed on 0 of 1" in drawn


def test_a_failed_audit_call_is_shown_not_hidden(fake: FakeStreamlit) -> None:
    trace = _trace(
        decisions=[
            {
                "step": 1,
                "from_agent": "researcher",
                "to_agent": "analyst",
                "expected_agent": "analyst",
                "matched_rule": "settled_quality_met",
                "llm_decided": False,
                "llm_called": True,
                "shadow_error": "rate limited",
                "research_quality": 0.8,
                "quality_score": 0.0,
            }
        ]
    )
    components.decision_trace(trace)
    assert "audit call failed" in fake.drawn


def test_react_discipline_reports_how_each_task_ended(fake: FakeStreamlit) -> None:
    components.react_discipline(_trace())
    assert "self-stopped" in fake.drawn
    assert "1/1 tasks self-terminated" in fake.drawn


def test_hitting_the_cap_is_reported_as_such(fake: FakeStreamlit) -> None:
    components.react_discipline(
        _trace(react_stats={"t1": {"iterations_used": 4, "self_terminated": False}})
    )
    assert "hit the cap" in fake.drawn


def test_critic_feedback_is_shown_not_just_its_number(fake: FakeStreamlit) -> None:
    components.critic_verdict(_trace())
    assert "well sourced" in fake.drawn
    assert "thin on recent developments" in fake.drawn


def test_critic_verdict_is_silent_when_the_critic_said_nothing(fake: FakeStreamlit) -> None:
    components.critic_verdict(_trace(critic_feedback="", top_issue=""))
    assert fake.drawn == ""


def test_the_research_plan_is_visible(fake: FakeStreamlit) -> None:
    """The planner drove the whole run and used to leave no trace on screen at all."""
    trace = _trace(
        research_plan=[
            {"task_id": "t1", "description": "find the spec", "assigned_to": "researcher"}
        ],
        research_strategy="breadth first",
    )
    components.research_plan(trace)
    drawn = fake.drawn
    assert "find the spec" in drawn
    assert "t1" in drawn
    assert "breadth first" in drawn


def test_the_research_plan_is_silent_when_the_planner_produced_nothing(
    fake: FakeStreamlit,
) -> None:
    components.research_plan(_trace(research_plan=[]))
    assert fake.drawn == ""


def test_the_analyst_synthesis_is_visible(fake: FakeStreamlit) -> None:
    """The analyst sits between the sources and the report and was never shown."""
    components.analysis_view(_trace(analysis="the three findings are ..."))
    assert "the three findings are ..." in fake.drawn


def test_plan_and_analysis_escape_injected_markup(fake: FakeStreamlit) -> None:
    trace = _trace(research_plan=[{"task_id": "t1", "description": "<script>x</script>"}])
    components.research_plan(trace)
    assert "<script>" not in fake.drawn


# ── the verdict banner ────────────────────────────────────────────────────


def _result(**overrides):
    from amaris.api.schemas import ResearchResult

    base = {
        "report": "# body",
        "citations": [],
        "scores": {},
        "agent_path": ["planner"],
        "trace": _trace(quality_score=0.81),
    }
    return ResearchResult(**{**base, **overrides})


def test_an_approved_run_reads_as_approved(fake: FakeStreamlit) -> None:
    components.verdict_banner(_result(), None)
    drawn = fake.drawn
    assert "verdict good" in drawn
    assert "approved" in drawn.lower()


def test_a_run_below_the_floor_does_not_claim_approval(fake: FakeStreamlit) -> None:
    """0.55 is under the 0.72 approve floor — calling that approved would be a lie."""
    components.verdict_banner(_result(trace=_trace(quality_score=0.55)), None)
    assert "verdict warn" in fake.drawn
    assert "below the approval floor" in fake.drawn.lower()


def test_the_banner_reads_the_shape_result_from_state_actually_produces(
    fake: FakeStreamlit, state
) -> None:
    """The old fixture hand-built scores={"overall": ...}, a key result_from_state never emits."""
    from amaris.api.schemas import result_from_state

    state["quality_score"] = 0.78
    state["critic_scores"] = {"answer_fit": 0.8}
    state["draft_report"] = "## Answer\nyes."
    components.verdict_banner(result_from_state(state), None)

    assert "verdict good" in fake.drawn
    assert "0.78" in fake.drawn


def test_a_failed_run_shows_the_error(fake: FakeStreamlit) -> None:
    components.verdict_banner(None, "provider chain exhausted")
    drawn = fake.drawn
    assert "verdict bad" in drawn
    assert "provider chain exhausted" in drawn


# ── run header, log and raw ───────────────────────────────────────────────


def test_the_run_header_reports_stage_progress_and_elapsed(fake: FakeStreamlit) -> None:
    event = _event("researcher", pct=45)
    event.elapsed_s = 31.2
    components.run_header([event], "5e700386")
    drawn = fake.drawn
    assert "5e700386" in drawn
    assert "researcher" in drawn
    assert "45%" in drawn
    assert "31.2s" in drawn


def test_the_run_header_marks_an_unfinished_run_live(fake: FakeStreamlit) -> None:
    components.run_header([_event("researcher")], "abc")
    assert "live" in fake.drawn


def test_the_run_header_is_silent_before_the_first_event(fake: FakeStreamlit) -> None:
    components.run_header([], "abc")
    assert fake.drawn == ""


def test_the_event_log_keeps_every_transition_in_order(fake: FakeStreamlit) -> None:
    events = [_event("planner"), _event("supervisor"), _event("researcher")]
    components.event_log(events)
    drawn = fake.drawn
    assert drawn.index("planner") < drawn.index("supervisor") < drawn.index("researcher")


def test_the_event_log_marks_a_failure_differently(fake: FakeStreamlit) -> None:
    components.event_log([_event("writer", status="failed")])
    assert 'class="log"' in fake.drawn
    assert 'class="err"' in fake.drawn


def test_the_raw_inspector_exposes_result_and_events(fake: FakeStreamlit) -> None:
    """Nothing the system produced should be unreachable from the UI."""
    payloads: list[dict] = []
    fake.json = lambda body, **k: payloads.append(body)
    components.raw_inspector(None, [_event("planner")])
    assert payloads and set(payloads[0]) == {"result", "events"}
    assert payloads[0]["events"][0]["agent"] == "planner"


# ── provider status ───────────────────────────────────────────────────────


def test_the_sidebar_flags_a_parked_provider(fake: FakeStreamlit, monkeypatch) -> None:
    """Found live: groq and gemini were parked for an hour and the sidebar still read healthy,
    so a run was started that crawled for nineteen minutes before failing."""
    from amaris.llm import router

    monkeypatch.setattr(router, "provider_status", lambda: {"groq": 3600.0, "glm": 0.0})
    monkeypatch.setattr(router, "configured_chain", lambda: ["groq", "glm"])
    warnings: list[str] = []
    fake.warning = warnings.append
    fake.error = lambda body: warnings.append(f"ERROR {body}")

    components.system_panel({})
    components.provider_alert()

    assert "3600s" in fake.drawn
    assert any("only glm usable" in w for w in warnings)


def test_a_parked_chain_is_reported_outside_the_panel_so_collapsing_cannot_hide_it(
    fake: FakeStreamlit, monkeypatch
) -> None:
    from amaris.llm import router

    monkeypatch.setattr(router, "provider_status", lambda: {"groq": 120.0, "glm": 30.0})
    monkeypatch.setattr(router, "configured_chain", lambda: ["groq", "glm"])
    errors: list[str] = []
    fake.error = errors.append
    fake.warning = lambda body: None

    components.provider_alert()
    assert any("every provider is rate limited" in e for e in errors)


def test_the_question_is_shown_while_its_run_is_in_flight(fake: FakeStreamlit) -> None:
    """A spoken question had no visible confirmation: the toast was discarded by the rerun
    that starts the run, so for 60-90s the user could not see what was heard."""
    components.asking("what is the MCP protocol?", spoken=True)
    drawn = fake.drawn

    assert 'class="asking"' in drawn
    assert "what is the MCP protocol?" in drawn
    # a spoken question is labelled as heard, so a mishearing is obvious rather than puzzling
    assert ">heard<" in drawn


def test_a_typed_question_is_not_labelled_as_heard(fake: FakeStreamlit) -> None:
    components.asking("what is langgraph?")
    drawn = fake.drawn

    assert ">asking<" in drawn
    assert ">heard<" not in drawn


def test_the_question_is_escaped_before_it_is_drawn(fake: FakeStreamlit) -> None:
    """It is user input rendered into unsafe_allow_html markup."""
    components.asking("<img src=x onerror=alert(1)>")
    drawn = fake.drawn

    assert "<img" not in drawn
    assert "&lt;img" in drawn


def test_a_live_capability_is_visually_distinct_from_a_missing_one(
    fake: FakeStreamlit, monkeypatch
) -> None:
    """ADR-009 degrades silently on purpose, so the only signal a user gets is this card."""
    from amaris.llm import router

    monkeypatch.setattr(router, "provider_status", lambda: {})
    monkeypatch.setattr(router, "configured_chain", lambda: ["groq"])
    fake.warning = lambda body: None

    components.system_panel({"memory": True, "docx": False})

    assert 'class="sys-item on"' in fake.drawn
    assert 'class="sys-item off"' in fake.drawn
    assert "memory" in fake.drawn and "docx" in fake.drawn
    # the counts are what make the card glanceable — a 1/2 says something is off without reading
    assert '<span class="pill warn">1/2</span>' in fake.drawn


def test_the_system_card_is_one_element_so_nothing_can_be_laid_over_it(
    fake: FakeStreamlit, monkeypatch
) -> None:
    """Two labelled sections of chips is what this replaced, and each wrapper was a collision."""
    from amaris.llm import router

    monkeypatch.setattr(router, "provider_status", lambda: {})
    monkeypatch.setattr(router, "configured_chain", lambda: ["groq", "gemini"])
    fake.warning = lambda body: None

    components.system_panel({"memory": True})

    assert len(fake.html) == 1
    assert '<span class="pill ok">2/2</span>' in fake.drawn


def test_a_healthy_system_summarises_itself_without_making_the_reader_do_arithmetic(
    fake: FakeStreamlit, monkeypatch
) -> None:
    """The panel is collapsed by default, so its summary row is the only thing most sessions
    ever read. Counts earn their width only when one of them is short."""
    from amaris.llm import router

    monkeypatch.setattr(router, "provider_status", lambda: {})
    monkeypatch.setattr(router, "configured_chain", lambda: ["groq", "gemini"])

    text, opens = components.system_summary({"memory": True, "docx": True})
    assert text == "System · all live"
    assert opens is False


def test_a_degraded_system_names_the_shortfall_and_opens_its_own_panel(
    fake: FakeStreamlit, monkeypatch
) -> None:
    from amaris.llm import router

    monkeypatch.setattr(router, "provider_status", lambda: {"groq": 60.0})
    monkeypatch.setattr(router, "configured_chain", lambda: ["groq", "gemini"])

    text, opens = components.system_summary({"memory": False, "docx": True})
    assert "1/2 providers" in text and "1/2 live" in text
    # a problem the reader has to click to discover is a problem the reader will not discover
    assert opens is True


def test_the_agent_list_marks_the_supervisor_because_deciding_is_its_whole_job(
    fake: FakeStreamlit,
) -> None:
    """The decision column is the point — a list of duties would read as a fixed pipeline."""
    components.agent_list()
    drawn = fake.drawn

    assert 'class="ag-row core"' in drawn
    for _tag, name, _role, _decides in components.AGENT_ROWS:
        assert name in drawn
    # the researcher's line is the clearest statement that the graph does not drive it
    assert "the graph never stops it" in drawn


def test_mini_metrics_draws_nothing_when_there_is_nothing_to_report(
    fake: FakeStreamlit,
) -> None:
    """An empty strip of zeroes reads as a broken session rather than a new one."""
    components.mini_metrics({})
    assert fake.drawn == ""


def test_mini_metrics_keeps_the_metric_markup_so_it_inherits_the_strip_styling(
    fake: FakeStreamlit,
) -> None:
    components.mini_metrics({"avg score": "0.88", "sources": "31"})

    assert 'class="metrics mini"' in fake.drawn
    assert "avg score" in fake.drawn and "0.88" in fake.drawn


def test_a_label_hint_ships_inside_the_same_element_as_the_heading(fake: FakeStreamlit) -> None:
    """As its own st.caption, Streamlit laid the hint 11px inside the heading's rule."""
    components.label("providers", hint="fallback order")

    assert len(fake.html) == 1
    assert 'class="lbl"' in fake.drawn and 'class="lbl-hint"' in fake.drawn


def test_a_label_without_a_hint_emits_no_empty_hint_div(fake: FakeStreamlit) -> None:
    components.label("providers")
    assert "lbl-hint" not in fake.drawn


def test_the_depth_table_is_read_from_the_budgets_so_it_cannot_drift(
    fake: FakeStreamlit,
) -> None:
    """A hand-written table here would quietly disagree with what a run actually spends."""
    from amaris.agents.triage import budget_for

    components.depth_table()

    deep = budget_for("deep")
    assert f"<span>{deep.max_sources}</span>" in fake.drawn
    assert f"<span>{deep.word_target}</span>" in fake.drawn
    for depth in ("direct", "brief", "standard", "deep"):
        assert depth in fake.drawn


def test_no_text_in_the_depth_table_is_below_a_readable_size() -> None:
    """The header shipped at 0.52rem — about 8px — which no contrast fix could rescue."""
    import re

    from frontend.styles import _CSS

    tiny = [m for m in re.findall(r"font-size:\s*(0\.\d+)rem", _CSS) if float(m) < 0.62]
    assert not tiny, f"font sizes below 0.62rem are unreadable: {tiny}"


def _call(**overrides):
    base = {
        "tool": "web_search",
        "kind": "tool",
        "agent": "researcher",
        "target": "what is MCP",
        "ok": True,
        "ms": 412.4,
        "detail": "8 results",
        "at": "2026-09-17T12:00:00+00:00",
    }
    return {**base, **overrides}


def test_the_tool_panel_names_every_call_and_who_made_it(fake: FakeStreamlit) -> None:
    """The whole point: an answer used to arrive with no way to see what was fetched."""
    components.tool_calls(_trace(tool_calls=[_call(), _call(tool="scrape_page", agent="analyst")]))

    assert "web_search" in fake.drawn
    assert "scrape_page" in fake.drawn
    assert "researcher" in fake.drawn
    assert "analyst" in fake.drawn


def test_an_mcp_call_is_marked_so_it_is_not_mistaken_for_a_local_tool(
    fake: FakeStreamlit,
) -> None:
    components.tool_calls(
        _trace(tool_calls=[_call(tool="arxiv/search_papers", kind="mcp", target="stdio")])
    )

    assert "arxiv/search_papers" in fake.drawn
    assert ">mcp<" in fake.drawn
    assert "1 of them to an MCP server" in fake.drawn


def test_a_run_that_reached_no_mcp_server_says_so_rather_than_staying_silent(
    fake: FakeStreamlit,
) -> None:
    """Silence read as "MCP works"; it actually meant the client was switched off."""
    components.tool_calls(_trace(tool_calls=[_call()]))

    assert "no MCP server was reached" in fake.drawn


def test_a_failed_call_is_shown_as_failed_not_dropped(fake: FakeStreamlit) -> None:
    """A dead tool is the most interesting row on the table — hiding it hides the why."""
    components.tool_calls(_trace(tool_calls=[_call(ok=False, detail="timed out after 25s")]))

    assert "failed" in fake.drawn
    assert "timed out after 25s" in fake.drawn


def test_the_target_is_escaped_because_it_is_the_users_own_query(fake: FakeStreamlit) -> None:
    """target carries the query verbatim into a table rendered with unsafe_allow_html."""
    components.tool_calls(_trace(tool_calls=[_call(target="<img src=x onerror=alert(1)>")]))

    assert "<img src=x" not in fake.drawn
    assert "&lt;img" in fake.drawn


def test_a_run_with_no_tool_calls_draws_nothing(fake: FakeStreamlit) -> None:
    components.tool_calls(_trace(tool_calls=[]))

    assert fake.drawn == ""

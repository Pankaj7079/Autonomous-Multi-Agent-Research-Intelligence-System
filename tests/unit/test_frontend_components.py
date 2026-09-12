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

    def markdown(self, body: str, **_: Any) -> None:
        self.html.append(body)

    def caption(self, body: str, **_: Any) -> None:
        self.captions.append(body)

    @contextlib.contextmanager
    def expander(self, label: str, **_: Any):
        self.captions.append(label)
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


def test_a_repeat_visit_gets_its_own_card(fake: FakeStreamlit) -> None:
    """Three researcher cards are the visible proof the supervisor re-routed."""
    events = [_event("planner"), _event("researcher"), _event("researcher")]
    components.agent_progress_tracker(events)
    assert fake.drawn.count(">researcher<") == 2


def test_the_supervisor_never_gets_a_card(fake: FakeStreamlit) -> None:
    components.agent_progress_tracker([_event("supervisor"), _event("planner")])
    assert ">supervisor<" not in fake.drawn


def test_the_current_agent_is_active_and_the_rest_are_done(fake: FakeStreamlit) -> None:
    components.agent_progress_tracker([_event("planner"), _event("researcher")])
    drawn = fake.drawn
    assert drawn.index("amaris-step done") < drawn.index("amaris-step active")


def test_a_finished_run_has_no_active_card(fake: FakeStreamlit) -> None:
    events = [_event("planner"), _event("writer"), _event("", status="done", pct=100)]
    components.agent_progress_tracker(events)
    assert "amaris-step active" not in fake.drawn


def test_a_failed_run_marks_the_last_card_failed(fake: FakeStreamlit) -> None:
    events = [_event("planner"), _event("researcher"), _event("", status="failed")]
    components.agent_progress_tracker(events)
    assert "amaris-step failed" in fake.drawn


def test_unvisited_agents_show_as_pending(fake: FakeStreamlit) -> None:
    components.agent_progress_tracker([_event("planner")])
    assert fake.drawn.count("amaris-step pending") == len(components.WORKERS) - 1


def test_score_dashboard_draws_the_four_critic_dimensions(fake: FakeStreamlit) -> None:
    scores = {
        "faithfulness": 0.95,
        "completeness": 0.75,
        "coherence": 0.60,
        "citation_quality": 0.40,
    }
    components.score_dashboard(scores)
    drawn = fake.drawn
    assert drawn.count("amaris-badge") == 4
    assert "amaris-badge good" in drawn
    assert "amaris-badge warn" in drawn
    assert "amaris-badge bad" in drawn


def test_an_absent_eval_metric_reads_as_not_scored(fake: FakeStreamlit) -> None:
    """A rate-limited judge omits the metric; showing 0.00 would claim the run was bad."""
    components.evaluation_layers({"faithfulness": 0.9})
    assert "not scored" in fake.drawn


def test_a_real_eval_zero_is_still_shown_as_zero(fake: FakeStreamlit) -> None:
    components.evaluation_layers({"eval_faithfulness": 0.0})
    assert "0.00" in fake.drawn
    assert "not scored" not in fake.drawn.split("faithfulness")[1][:60]


def test_no_critic_scores_says_so_instead_of_drawing_empty_badges(fake: FakeStreamlit) -> None:
    components.score_dashboard({})
    assert "amaris-badge" not in fake.drawn
    assert fake.captions


def test_citations_keep_the_writers_numbering(fake: FakeStreamlit) -> None:
    """[3] in the report must be [3] in the list, so the index is never recomputed here."""
    components.citation_list([{"index": 7, "title": "Spec", "url": "https://example.com/spec"}])
    assert "[7]" in fake.drawn
    assert 'href="https://example.com/spec"' in fake.drawn


def test_citation_titles_are_escaped(fake: FakeStreamlit) -> None:
    components.citation_list([{"index": 1, "title": "<script>x</script>", "url": ""}])
    assert "<script>" not in fake.drawn


def test_no_citations_renders_a_note_not_an_empty_expander(fake: FakeStreamlit) -> None:
    components.citation_list([])
    assert fake.drawn == ""
    assert fake.captions == ["no citations"]


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
    components.run_stats(_trace(), ["planner", "researcher"], 84.0)
    drawn = fake.drawn
    assert "12" in drawn and "0.80" in drawn and "84s" in drawn


def test_run_stats_without_a_trace_draws_nothing(fake: FakeStreamlit) -> None:
    components.run_stats(None, [], None)
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
                "research_quality": 0.5,
                "quality_score": 0.6,
            }
        ]
    )
    components.decision_trace(trace)
    assert "diverged" in fake.drawn
    assert any("diverged" in c for c in fake.captions)


def test_react_discipline_reports_how_each_task_ended(fake: FakeStreamlit) -> None:
    components.react_discipline(_trace())
    assert "self-stopped" in fake.drawn
    assert any("1/1 tasks self-stopped" in c for c in fake.captions)


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


# ── the dense execution views ─────────────────────────────────────────────


def test_the_timeline_shows_how_long_each_step_took(fake: FakeStreamlit) -> None:
    """Duration is the number worth showing; cumulative elapsed lives in the event log."""
    events = [_event("supervisor"), _event("planner"), _event("researcher")]
    events[0].elapsed_s = 0.4
    events[1].elapsed_s = 4.1
    events[2].elapsed_s = 61.8
    components.agent_progress_tracker(events)
    assert "3.7s" in fake.drawn, "planner took 4.1 - 0.4"
    assert "57.7s" in fake.drawn, "researcher took 61.8 - 4.1"


def test_the_run_header_reports_stage_progress_and_elapsed(fake: FakeStreamlit) -> None:
    event = _event("researcher", pct=45)
    event.elapsed_s = 31.2
    components.run_header([event], "5e700386")
    drawn = fake.drawn
    assert "5e700386" in drawn
    assert "researcher" in drawn
    assert "45%" in drawn
    assert "31.2s" in drawn


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
    assert "amaris-log" in fake.drawn
    assert 'class="err"' in fake.drawn


def test_the_raw_inspector_exposes_result_and_events(fake: FakeStreamlit) -> None:
    """Nothing the system produced should be unreachable from the UI."""
    payloads: list[dict] = []
    fake.json = lambda body, **k: payloads.append(body)
    components.raw_inspector(None, [_event("planner")])
    assert payloads and set(payloads[0]) == {"result", "events"}
    assert payloads[0]["events"][0]["agent"] == "planner"


def test_the_sidebar_flags_a_parked_provider(fake: FakeStreamlit, monkeypatch) -> None:
    """Found live: groq and gemini were parked for an hour and the sidebar still read healthy,
    so a run was started that crawled for nineteen minutes before failing."""
    from amaris.llm import router

    monkeypatch.setattr(router, "provider_status", lambda: {"groq": 3600.0, "glm": 0.0})
    monkeypatch.setattr(router, "configured_chain", lambda: ["groq", "glm"])
    warnings: list[str] = []
    fake.warning = warnings.append
    fake.error = lambda body: warnings.append(f"ERROR {body}")

    components.system_panel()

    assert "3600s" in fake.drawn
    assert any("only glm usable" in w for w in warnings)


def test_the_sidebar_refuses_a_fully_parked_chain(fake: FakeStreamlit, monkeypatch) -> None:
    from amaris.llm import router

    monkeypatch.setattr(router, "provider_status", lambda: {"groq": 120.0, "glm": 30.0})
    monkeypatch.setattr(router, "configured_chain", lambda: ["groq", "glm"])
    errors: list[str] = []
    fake.error = errors.append
    fake.warning = lambda body: None

    components.system_panel()
    assert any("every provider is rate limited" in e for e in errors)

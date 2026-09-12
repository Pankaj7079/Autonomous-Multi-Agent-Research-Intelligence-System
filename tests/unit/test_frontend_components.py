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
    assert drawn.index("amaris-card done") < drawn.index("amaris-card active")


def test_a_finished_run_has_no_active_card(fake: FakeStreamlit) -> None:
    events = [_event("planner"), _event("writer"), _event("", status="done", pct=100)]
    components.agent_progress_tracker(events)
    assert "amaris-card active" not in fake.drawn


def test_a_failed_run_marks_the_last_card_failed(fake: FakeStreamlit) -> None:
    events = [_event("planner"), _event("researcher"), _event("", status="failed")]
    components.agent_progress_tracker(events)
    assert "amaris-card failed" in fake.drawn


def test_unvisited_agents_show_as_pending(fake: FakeStreamlit) -> None:
    components.agent_progress_tracker([_event("planner")])
    assert fake.drawn.count("amaris-card pending") == len(components.WORKERS) - 1


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


def test_a_zero_eval_score_is_flagged_as_possibly_unavailable(fake: FakeStreamlit) -> None:
    """A rate-limited judge and a genuinely bad run both arrive as 0.0 over the API."""
    components.score_dashboard({"faithfulness": 0.9, "eval_faithfulness": 0.0})
    assert any("judge was unavailable" in c for c in fake.captions)


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

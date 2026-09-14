"""Boots the real script through streamlit's own runner — catches import and ordering errors."""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

# AppTest resolves a relative path against the caller, not the repo root
APP = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")


def test_the_app_renders_without_raising() -> None:
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, [e.value for e in app.exception]
    # the wordmark is split so the second half can take the accent colour
    assert any("AMA" in block.value and "RIS" in block.value for block in app.markdown)
    # the composer is a chat input pinned to the bottom, not a form with a run button
    assert app.chat_input


def test_the_empty_state_is_the_mark_the_numbers_and_the_agents() -> None:
    """The routing transcript, depth table and GraphState list were removed — they explained
    the system in prose. The stat strip and agent grid stayed, because they show it."""
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    drawn = "".join(b.value for b in app.markdown if "<style>" not in b.value)

    assert 'class="hero"' in drawn
    assert 'class="tagline"' in drawn
    assert 'class="stats"' in drawn
    assert 'class="agents"' in drawn
    # the supervisor card is marked because deciding is its whole job
    assert 'class="agent core"' in drawn
    assert 'class="trace"' not in drawn


def test_no_run_output_is_rendered_before_a_query_is_submitted() -> None:
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    assert not app.error
    # the stylesheet names every class, so match the rendered markup rather than the css
    drawn = "".join(b.value for b in app.markdown if "<style>" not in b.value)
    # the verdict banner and the run bar only exist once a run has produced something
    assert 'class="verdict' not in drawn
    assert 'class="runbar' not in drawn


def test_the_empty_state_offers_real_questions_but_no_explainer_paragraph() -> None:
    """A page with only a text box gives a first-time reader nothing to try. The examples are
    whole questions that dispatch a real run — the marketing explainer stays gone."""
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    drawn = "".join(b.value for b in app.markdown if "<style>" not in b.value)

    labels = [b.label for b in app.button]
    assert any(label.endswith("?") for label in labels), labels
    # bare keyword chips were the filler that got removed; full questions are the replacement
    assert not any(label in ("mcp", "langgraph vs crewai") for label in labels)
    assert 'class="lede"' not in drawn
    # the wordmark and its one line are what remain
    assert "AMA" in drawn and "RIS" in drawn
    assert "show every decision" in drawn


def test_the_wordmark_has_no_flex_gap_between_its_halves() -> None:
    """ "AMA" and the RIS span are separate flex items, so any gap on .wordmark renders the
    product name as two words. This shipped at 96px in the hero before it was caught."""
    from frontend.styles import _CSS

    rule = _CSS.split(".wordmark {")[1].split("}")[0]
    assert "gap: 0;" in rule, rule

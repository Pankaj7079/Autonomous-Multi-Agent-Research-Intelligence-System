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
    assert any("AMARIS" in block.value for block in app.markdown)
    # example-query chips sit after the submit button, so find it by label not position
    assert any(button.label == "run" for button in app.button)


def test_the_landing_page_explains_the_system_instead_of_sitting_blank() -> None:
    """An empty state that teaches is the point — a reviewer arrives knowing nothing."""
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    drawn = "".join(block.value for block in app.markdown)
    assert "supervisor" in drawn
    assert "agentic" in drawn.lower()


def test_no_run_output_is_rendered_before_a_query_is_submitted() -> None:
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    assert not app.error
    # the stylesheet names every class, so match the rendered markup rather than the css
    drawn = "".join(b.value for b in app.markdown if "<style>" not in b.value)
    # the verdict banner and the run bar only exist once a run has produced something
    assert 'class="verdict' not in drawn
    assert 'class="runbar' not in drawn


def test_clicking_an_example_fills_the_query_box() -> None:
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    example = next(b for b in app.button if b.label == "mcp")
    example.click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert "Model Context Protocol" in app.text_input[0].value

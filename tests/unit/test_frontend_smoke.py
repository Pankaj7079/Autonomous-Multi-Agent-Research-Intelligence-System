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
    assert app.button[0].label == "Research"


def test_nothing_is_rendered_before_a_query_is_submitted() -> None:
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=30).run()
    assert not app.error
    # the session id block only exists once a run has produced a result
    assert not app.code

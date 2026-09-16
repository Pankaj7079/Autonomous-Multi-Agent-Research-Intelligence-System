"""Tracing setup. The keys existed long before anything read them (ADR-043, ADR-046)."""

from __future__ import annotations

import os

import pytest

from amaris.config.settings import get_settings
from amaris.observability import tracing


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Every test starts with no tracer configured and leaves no environment behind."""
    for name in ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT"):
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_no_key_means_no_tracer() -> None:
    """A run without tracing configured must behave exactly as before."""
    assert tracing.configure_tracing() == ""
    assert tracing.active_backend() == ""


def test_a_langsmith_key_turns_langchain_tracing_on(monkeypatch) -> None:
    """Langsmith needs no callback — langchain reads these variables itself, which is why the
    key sat in .env doing nothing until something set them."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls_example")
    monkeypatch.setenv("LANGSMITH_PROJECT", "amaris-test")
    get_settings.cache_clear()

    assert tracing.configure_tracing() == "langsmith"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "amaris-test"
    assert tracing.active_backend() == "langsmith"


def test_the_project_falls_back_to_a_default(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls_example")
    get_settings.cache_clear()

    assert tracing.configure_tracing() == "langsmith"
    assert os.environ["LANGSMITH_PROJECT"] == tracing.DEFAULT_PROJECT


def test_reconfiguring_without_a_key_turns_the_backend_back_off(monkeypatch) -> None:
    """_backend is module state, so a later call must clear it rather than leave it stale."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls_example")
    get_settings.cache_clear()
    assert tracing.configure_tracing() == "langsmith"

    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    get_settings.cache_clear()
    assert tracing.configure_tracing() == ""
    assert tracing.active_backend() == ""


def test_the_graph_run_config_attaches_no_callbacks() -> None:
    """Langsmith traces through environment variables, so nothing is attached to the run —
    and langchain treats an empty callbacks list as a real override, so the key stays absent."""
    from amaris.graph.pipeline import run_config

    assert "callbacks" not in run_config("session-1")

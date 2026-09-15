"""Tracing setup. The keys existed long before anything read them (ADR-043)."""

from __future__ import annotations

import pytest

from amaris.config.settings import get_settings
from amaris.observability import tracing


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Every test starts with no tracer configured and leaves no environment behind."""
    for name in (
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_no_keys_means_no_tracer_and_no_callbacks() -> None:
    """A run without tracing configured must behave exactly as before."""
    assert tracing.configure_tracing() == ""
    assert tracing.callbacks() == []
    assert tracing.active_backend() == ""


def test_a_langsmith_key_turns_langchain_tracing_on(monkeypatch) -> None:
    """Langsmith needs no callback — langchain reads these variables itself, which is why the
    key sat in .env doing nothing until something set them."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls_example")
    monkeypatch.setenv("LANGSMITH_PROJECT", "amaris-test")
    get_settings.cache_clear()

    assert tracing.configure_tracing() == "langsmith"

    import os

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "amaris-test"
    # nothing to attach to the graph: the integration is entirely in those variables
    assert tracing.callbacks() == []


def test_langfuse_needs_both_halves_of_its_key_pair(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_example")
    get_settings.cache_clear()
    assert tracing.configure_tracing() == ""


def test_a_tracer_that_cannot_start_never_stops_a_run(monkeypatch) -> None:
    """Same rule as every other optional extra: a missing tracer is a reduced run, not a crash."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_example")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_example")
    get_settings.cache_clear()

    def explode(*args, **kwargs):
        raise RuntimeError("langfuse server unreachable")

    monkeypatch.setattr("langfuse.langchain.CallbackHandler", explode)

    assert tracing.configure_tracing() == ""
    assert tracing.callbacks() == []


def test_the_graph_run_config_carries_the_handler(monkeypatch) -> None:
    """LangGraph passes callbacks down, so one attachment at the top traces every nested call."""
    from amaris.graph.pipeline import run_config

    sentinel = object()
    monkeypatch.setattr(tracing, "callbacks", lambda: [sentinel])
    assert run_config("session-1")["callbacks"] == [sentinel]

    monkeypatch.setattr(tracing, "callbacks", list)
    # absent, not empty: langchain treats an empty callbacks list as a real override
    assert "callbacks" not in run_config("session-1")

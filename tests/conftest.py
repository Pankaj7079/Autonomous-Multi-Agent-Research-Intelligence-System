"""Shared fixtures. Phase 9 adds sample_state and mock_llm here."""

from __future__ import annotations

from pathlib import Path

import pytest

from amaris.llm.router import reset_cooldowns
from amaris.observability.logging import configure_logging


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point sinks at a temp dir so tests never write into the real logs/."""
    log_dir: Path = tmp_path_factory.mktemp("logs")
    configure_logging(level="DEBUG", log_dir=log_dir, json_enabled=False, force=True)


@pytest.fixture(autouse=True)
def _no_parked_providers() -> None:
    """Provider cooldowns are module state, so one rate-limit test would skew every later one."""
    reset_cooldowns()

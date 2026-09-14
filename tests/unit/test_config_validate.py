"""Patch 3 — the checks Settings itself cannot express."""

from __future__ import annotations

import pytest

from amaris.config import validate as cv
from amaris.config.settings import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch):
    """Build settings from explicit values, ignoring any real .env on disk."""

    def _apply(**values) -> Settings:
        built = Settings(_env_file=None, **values)
        monkeypatch.setattr(cv, "get_settings", lambda: built)
        return built

    return _apply


def test_no_provider_key_is_an_error_with_instructions(settings) -> None:
    settings()
    report = cv.validate_config()
    assert not report.ok
    assert "console.groq.com" in report.errors[0]


def test_one_provider_is_valid_but_warns_about_no_fallback(settings) -> None:
    settings(groq_api_key="gsk_test")
    report = cv.validate_config()
    assert report.ok
    assert any("fall back" in w for w in report.warnings)


def test_two_providers_produce_no_warning(settings) -> None:
    settings(groq_api_key="gsk_test", gemini_api_key="g_test")
    report = cv.validate_config()
    assert report.ok
    assert report.warnings == []
    assert report.providers == ["groq", "gemini"]


def test_approve_below_research_is_an_error(settings) -> None:
    """Settings bounds each threshold 0-1; only this catches the relationship between them."""
    settings(
        groq_api_key="gsk_test",
        gemini_api_key="g_test",
        research_quality_threshold=0.9,
        quality_approve_threshold=0.2,
    )
    report = cv.validate_config()
    assert not report.ok
    assert "RESEARCH_QUALITY_THRESHOLD" in report.errors[0]


def test_equal_thresholds_are_also_rejected(settings) -> None:
    settings(
        groq_api_key="gsk_test",
        gemini_api_key="g_test",
        research_quality_threshold=0.7,
        quality_approve_threshold=0.7,
    )
    assert not cv.validate_config().ok


def test_cloud_without_qdrant_warns_rather_than_fails(settings) -> None:
    """Long-term memory no-ops without Qdrant — degraded, not broken, so it must not block startup."""
    settings(groq_api_key="gsk_test", gemini_api_key="g_test", deployment_mode="cloud")
    report = cv.validate_config()
    assert report.ok
    assert any("long-term memory" in w for w in report.warnings)


def test_cloud_with_qdrant_configured_is_quiet(settings) -> None:
    settings(
        groq_api_key="gsk_test",
        gemini_api_key="g_test",
        deployment_mode="cloud",
        qdrant_url="https://example.cloud.qdrant.io",
        qdrant_api_key="qk_test",
    )
    assert cv.validate_config().warnings == []


def test_the_report_prints_readably(settings) -> None:
    settings()
    text = str(cv.validate_config())
    assert "INVALID" in text
    assert "error:" in text

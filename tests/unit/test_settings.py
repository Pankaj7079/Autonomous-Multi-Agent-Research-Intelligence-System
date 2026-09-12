"""Settings must import cleanly with no .env and expose the right deployment flags."""

from __future__ import annotations

from amaris.config.settings import Settings, get_settings


def _settings(**overrides: object) -> Settings:
    # _env_file=None so a developer's real .env can never change a test result
    return Settings(_env_file=None, **overrides)


def test_imports_with_no_env_file() -> None:
    s = _settings()
    assert s.deployment_mode == "local"
    assert s.max_revisions == 2
    assert s.research_quality_threshold == 0.60


def test_thresholds_match_the_supervisor_prompt() -> None:
    """These numbers are duplicated in docs/AGENTS.md — drift breaks routing silently."""
    s = _settings()
    assert s.research_quality_threshold == 0.60
    assert s.quality_approve_threshold == 0.72
    assert s.max_react_iterations == 4


def test_deployment_flags_are_mutually_exclusive() -> None:
    assert _settings(deployment_mode="local").is_local is True
    assert _settings(deployment_mode="cloud").is_local is False
    assert _settings(deployment_mode="cloud").is_cloud is True


def test_key_returns_none_when_unset_and_plain_text_when_set() -> None:
    assert _settings().key("groq_api_key") is None
    assert _settings(groq_api_key="gsk_abc").key("groq_api_key") == "gsk_abc"


def test_unedited_env_example_placeholders_read_as_unset() -> None:
    """Copying .env.example without editing must give the setup error, not a 401 from groq."""
    assert _settings(groq_api_key="gsk_your_key_here").key("groq_api_key") is None
    assert _settings(gemini_api_key="AIza_your_key_here").key("gemini_api_key") is None
    assert _settings(groq_api_key="   ").key("groq_api_key") is None
    assert _settings(groq_api_key="gsk_realLooKingKey123").key("groq_api_key") is not None


def test_secret_keys_are_masked_in_repr() -> None:
    """A settings object gets logged eventually — the key must not come with it."""
    assert "gsk_abc" not in repr(_settings(groq_api_key="gsk_abc"))


def test_configured_providers_follow_fallback_order() -> None:
    s = _settings(gemini_api_key="g", groq_api_key="k")
    assert s.configured_llm_providers == ["groq", "gemini"]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()

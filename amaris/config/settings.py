"""Typed settings loaded from .env. One source of truth for every knob."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DeploymentMode = Literal["local", "cloud"]

# substrings that only appear in .env.example's dummy values
_PLACEHOLDER_MARKERS = ("your_key_here", "your-key-here", "your_key", "changeme")


class Settings(BaseSettings):
    """Every field mirrors a var in .env.example. Defaults must work with no .env at all."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # local = docker redis + qdrant + fastapi, cloud = streamlit in-process
    deployment_mode: DeploymentMode = "local"

    # which provider leads the fallback chain; the rest keep their documented order
    primary_provider: str = "groq"
    # the judge needs to be fast AND hold quota the pipeline did not just spend. gemini
    # bills per day and evaluation is only a few calls, so it suits this and groq does not.
    eval_provider: str = "gemini"

    # SecretStr so a stray repr() or log of the settings object can't leak a key
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    glm_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_json_enabled: bool = True

    langsmith_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3000"

    # the streamlit frontend calls this in local mode; cloud mode ignores it entirely
    api_base_url: str = "http://localhost:8000"

    redis_url: str = "redis://localhost:6379"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None

    sqlite_checkpoint_db: str = "./amaris_checkpoints.db"

    # bounds only — the cross-field check (research < approve) belongs to Tier 1 patch 3
    max_revisions: int = Field(default=2, ge=1)
    max_react_iterations: int = Field(default=4, ge=1)
    # hard stop on the supervisor loop: an llm picks the path, so code guarantees it ends
    max_supervisor_steps: int = Field(default=15, ge=3)
    research_quality_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    quality_approve_threshold: float = Field(default=0.72, ge=0.0, le=1.0)

    # how many sources any agent prompt may carry; the researcher's cap never exceeds it
    max_sources_in_prompt: int = Field(default=12, ge=1)
    # browsers and search apis both throttle, so tasks fan out but not without bound
    max_concurrent_research_tasks: int = Field(default=3, ge=1)
    # 0.35 drops a source that matched only half the query and only in its body text —
    # exactly the "weather API pricing" pages a "today's weather" query used to keep
    relevance_floor: float = Field(default=0.35, ge=0.0, le=1.0)

    # attachments: extraction is capped so one upload cannot blow up every downstream prompt
    attachment_max_bytes: int = Field(default=10_000_000, ge=1)
    attachment_max_pages: int = Field(default=40, ge=1)
    attachment_max_chars: int = Field(default=120_000, ge=1_000)
    # ~1200 chars keeps a chunk inside bge-small's 512-token window with room to spare
    attachment_chunk_chars: int = Field(default=1_200, ge=200)
    attachment_chunk_overlap: int = Field(default=150, ge=0)
    # how many document chunks the researcher may pull back per task
    attachment_top_k: int = Field(default=4, ge=1)

    # groq's tokens-per-minute window is ~60s, so a shorter retry budget can never recover from it
    llm_retry_budget_seconds: float = Field(default=75.0, ge=0.0)
    # ragas makes one judge call per metric per context, so it needs its own ceiling
    ragas_timeout_seconds: float = Field(default=120.0, ge=1.0)

    groq_model_reasoning: str = "openai/gpt-oss-120b"
    groq_model_fast: str = "openai/gpt-oss-20b"
    gemini_model_fallback: str = "gemini-3.6-flash"
    # z.ai speaks the openai protocol, so langchain-openai drives it with a base url override
    glm_model: str = "glm-4.5-flash"
    glm_base_url: str = "https://api.z.ai/api/paas/v4"
    # last-resort fallback only — not a free-tier provider, skipped silently if no key
    anthropic_model: str = "claude-sonnet-5"

    @property
    def is_local(self) -> bool:
        """True when Redis, Qdrant and FastAPI are expected to be reachable."""
        return self.deployment_mode == "local"

    @property
    def is_cloud(self) -> bool:
        """True on Streamlit Community Cloud, where nothing is dockerised."""
        return self.deployment_mode == "cloud"

    def key(self, name: str) -> str | None:
        """Plain value of a SecretStr field, or None when it isn't really configured."""
        secret: SecretStr | None = getattr(self, name, None)
        if not secret:
            return None
        value = secret.get_secret_value().strip()
        # an unedited copy of .env.example must read as unset, not as a bad key
        if not value or any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS):
            return None
        return value

    @property
    def configured_llm_providers(self) -> list[str]:
        """Providers with a key present, in fallback order."""
        pairs = [
            ("groq", "groq_api_key"),
            ("gemini", "gemini_api_key"),
            ("glm", "glm_api_key"),
            ("anthropic", "anthropic_api_key"),
        ]
        return [provider for provider, field in pairs if self.key(field)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call get_settings.cache_clear() in tests."""
    return Settings()

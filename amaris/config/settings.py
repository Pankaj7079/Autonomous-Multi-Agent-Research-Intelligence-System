"""Typed settings loaded from .env. One source of truth for every knob."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DeploymentMode = Literal["local", "cloud"]


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

    # SecretStr so a stray repr() or log of the settings object can't leak a key
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_json_enabled: bool = True

    langsmith_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3000"

    redis_url: str = "redis://localhost:6379"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None

    sqlite_checkpoint_db: str = "./amaris_checkpoints.db"

    # bounds only — the cross-field check (research < approve) belongs to Tier 1 patch 3
    max_revisions: int = Field(default=2, ge=1)
    max_react_iterations: int = Field(default=4, ge=1)
    research_quality_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    quality_approve_threshold: float = Field(default=0.72, ge=0.0, le=1.0)

    groq_model_reasoning: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"
    cerebras_model: str = "gpt-oss-120b"
    gemini_model_fallback: str = "gemini-2.0-flash"

    @property
    def is_local(self) -> bool:
        """True when Redis, Qdrant and FastAPI are expected to be reachable."""
        return self.deployment_mode == "local"

    @property
    def is_cloud(self) -> bool:
        """True on Streamlit Community Cloud, where nothing is dockerised."""
        return self.deployment_mode == "cloud"

    def key(self, name: str) -> str | None:
        """Plain value of a SecretStr field, or None when it isn't configured."""
        secret: SecretStr | None = getattr(self, name, None)
        return secret.get_secret_value() if secret else None

    @property
    def configured_llm_providers(self) -> list[str]:
        """Providers with a key present, in fallback order."""
        pairs = [
            ("groq", "groq_api_key"),
            ("cerebras", "cerebras_api_key"),
            ("gemini", "gemini_api_key"),
        ]
        return [provider for provider, field in pairs if self.key(field)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call get_settings.cache_clear() in tests."""
    return Settings()

"""Typed settings loaded from .env. One source of truth for every knob."""

from __future__ import annotations

from contextvars import ContextVar
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
    # a classic PAT is enough for the remote GitHub MCP endpoint; no OAuth app needed
    github_token: SecretStr | None = None
    # off by default: a research run would otherwise spawn npx/uvx subprocesses, which is a
    # surprise in tests and a cold start in cloud mode where no node runtime exists
    # uploaded chunks older than this are swept at startup; 0 disables the sweep. "start over"
    # already deletes them, but a closed browser tab never presses a button
    attachment_retention_hours: float = Field(default=24.0, ge=0.0)
    mcp_client_enabled: bool = False
    # empty disables the filesystem MCP server: pointing an agent at a whole disk is not a default
    mcp_filesystem_root: str = ""

    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_json_enabled: bool = True

    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "amaris"

    # the streamlit frontend calls this in local mode; cloud mode ignores it entirely
    api_base_url: str = "http://localhost:8000"

    redis_url: str = "redis://localhost:6379"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None

    sqlite_checkpoint_db: str = "./checkpoints.db"
    # checkpoints resume a crashed run, so only recent threads are worth keeping — nothing
    # pruned them before and the file grew ~176KB per run forever
    checkpoint_keep_threads: int = Field(default=50, ge=1)

    # bounds only — the cross-field check (research < approve) belongs to Tier 1 patch 3
    max_revisions: int = Field(default=2, ge=1)
    max_react_iterations: int = Field(default=4, ge=1)
    # hard stop on the supervisor loop: an llm picks the path, so code guarantees it ends
    max_supervisor_steps: int = Field(default=15, ge=3)
    research_quality_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    quality_approve_threshold: float = Field(default=0.72, ge=0.0, le=1.0)

    # how many sources any agent prompt may carry; the researcher's cap never exceeds it.
    # 20 so the deep budget can actually use its headroom — at 12 this silently clamped it
    # back down to the standard budget and deep read no more than standard did
    max_sources_in_prompt: int = Field(default=20, ge=1)
    # browsers and search apis both throttle, so tasks fan out but not without bound
    max_concurrent_research_tasks: int = Field(default=3, ge=1)
    # 0.35 drops a source that matched only half the query and only in its body text —
    # exactly the "weather API pricing" pages a "today's weather" query used to keep
    relevance_floor: float = Field(default=0.35, ge=0.0, le=1.0)

    # attachments: extraction is capped so one upload cannot blow up every downstream prompt
    attachment_max_bytes: int = Field(default=10_000_000, ge=1)
    attachment_max_pages: int = Field(default=40, ge=1)
    # a scan costs a render plus an OCR pass per page, so it gets a tighter ceiling
    attachment_max_scan_pages: int = Field(default=10, ge=1)
    # 2x render is the point where OCR stops mangling 10pt body text
    attachment_scan_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    attachment_max_chars: int = Field(default=120_000, ge=1_000)
    # ~1200 chars keeps a chunk inside bge-small's 512-token window with room to spare
    attachment_chunk_chars: int = Field(default=1_200, ge=200)
    attachment_chunk_overlap: int = Field(default=150, ge=0)
    # how many document chunks the researcher may pull back per task
    attachment_top_k: int = Field(default=4, ge=1)

    # speech input — groq serves whisper on the same free key, so there is no local model
    stt_model: str = "whisper-large-v3-turbo"
    stt_max_bytes: int = Field(default=25_000_000, ge=1)
    # below this there is no speech in the file, only a mis-tap on the mic button
    stt_min_bytes: int = Field(default=2_000, ge=0)

    # emailing a report — resend speaks plain http, so httpx covers it and there is no new dep
    resend_api_key: SecretStr | None = None
    email_from: str = "AMARIS <onboarding@resend.dev>"
    # empty means any recipient. a public demo that mails anywhere is an open relay, so cloud
    # mode refuses to send until this names the domains it is allowed to reach
    email_allowed_domains: list[str] = Field(default_factory=list)
    email_max_per_session: int = Field(default=5, ge=1)

    # must exceed the slowest single request (GLM, 90s) or the budget is spent before the first
    # attempt returns and no retry is ever possible — which is how a deep writer call died once
    llm_retry_budget_seconds: float = Field(default=150.0, ge=0.0)
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
        """Plain value of a SecretStr field, or None when it isn't really configured.

        A key bound to this context by `use_session_keys` wins, so a cloud visitor can run on
        their own quota without their key reaching anyone else's run.
        """
        supplied = (_SESSION_KEYS.get() or {}).get(name)
        if supplied:
            return supplied
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


# a visitor's own key on a shared cloud process. a contextvar and not os.environ or the settings
# singleton: streamlit runs each session's script in its own thread, so a value set here is
# visible to that run's coroutines and to nobody else's. the process-global alternatives are a
# cross-visitor key leak on any deployment serving more than one person at a time.
_SESSION_KEYS: ContextVar[dict[str, str] | None] = ContextVar("amaris_session_keys", default=None)


def use_session_keys(keys: dict[str, str]) -> None:
    """Bind caller-supplied API keys to this context. Pass {} to go back to the configured ones."""
    bound = {name: value.strip() for name, value in keys.items() if value and value.strip()}
    # deliberately silent: observability.logging imports this module, so a logger here is a
    # circular import — and a key value is the one thing that must never reach a log line anyway
    _SESSION_KEYS.set(bound)


def session_keys() -> dict[str, str]:
    """What is currently bound to this context. Values included — callers must not log them."""
    return dict(_SESSION_KEYS.get() or {})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call get_settings.cache_clear() in tests."""
    return Settings()

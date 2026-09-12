"""Fail fast on a misconfigured .env instead of crashing four agents into a run."""

from __future__ import annotations

from dataclasses import dataclass, field

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

NO_PROVIDER_ERROR = (
    "No LLM provider key found. Set GROQ_API_KEY in .env — it is free at "
    "console.groq.com and needs no card. GEMINI_API_KEY, GLM_API_KEY and "
    "ANTHROPIC_API_KEY are optional fallbacks."
)


@dataclass(frozen=True)
class ConfigReport:
    """Errors block startup; warnings are things that will quietly degrade."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = [
            f"config: {'ok' if self.ok else 'INVALID'} · providers: {', '.join(self.providers) or 'none'}"
        ]
        lines += [f"  error:   {item}" for item in self.errors]
        lines += [f"  warning: {item}" for item in self.warnings]
        return "\n".join(lines)


def validate_config() -> ConfigReport:
    """Check the settings a run depends on. Pure — dependency reachability is /ready's job."""
    settings = get_settings()
    errors: list[str] = []
    warnings: list[str] = []
    providers = settings.configured_llm_providers

    if not providers:
        errors.append(NO_PROVIDER_ERROR)
    elif len(providers) == 1:
        warnings.append(
            f"only {providers[0]} is configured — a rate limit will stall the run "
            f"with no provider to fall back to"
        )

    research = settings.research_quality_threshold
    approve = settings.quality_approve_threshold
    # Settings already bounds each one 0-1; what it cannot express is the relationship
    if not 0 < research < approve < 1:
        errors.append(
            f"thresholds must satisfy 0 < RESEARCH_QUALITY_THRESHOLD ({research}) < "
            f"QUALITY_APPROVE_THRESHOLD ({approve}) < 1 — otherwise the supervisor either "
            f"never leaves research or approves work the critic would reject"
        )

    if settings.is_cloud and not (settings.qdrant_url and settings.key("qdrant_api_key")):
        warnings.append(
            "cloud mode without QDRANT_URL and QDRANT_API_KEY — mem0 and vector search no-op"
        )

    return ConfigReport(errors=errors, warnings=warnings, providers=providers)


def log_report(report: ConfigReport) -> None:
    """Emit the report on the logging contract so a bad boot is greppable like anything else."""
    for item in report.errors:
        logger.bind(detail=item).error("config.invalid")
    for item in report.warnings:
        logger.bind(detail=item).warning("config.degraded")
    if report.ok and not report.warnings:
        logger.bind(providers=report.providers).info("config.valid")

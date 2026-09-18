"""Loguru setup: console + JSONL sinks. Contract in docs/OBSERVABILITY.md."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from loguru import logger

from amaris.observability.context import context_patcher

_configured = False

# session:agent first — easiest thing to scan when two runs interleave
_CONSOLE_BASE = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <7}</level> "
    "<cyan>{extra[session_id]}</cyan>:<magenta>{extra[agent]}</magenta> "
    "<level>{message}</level>"
)


def _console_format(record: dict) -> str:
    """Render bound fields as k=v after the event name."""
    extras = {k: v for k, v in record["extra"].items() if k not in ("session_id", "agent")}
    if not extras:
        return _CONSOLE_BASE + "\n{exception}"
    # escape braces or loguru re-reads a dict value as format placeholders
    rendered = " ".join(f"{k}={v}" for k, v in extras.items()).replace("{", "{{").replace("}", "}}")
    return _CONSOLE_BASE + f" <dim>{rendered}</dim>" + "\n{exception}"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(
    *,
    level: str | None = None,
    log_dir: str | Path | None = None,
    json_enabled: bool | None = None,
    force: bool = False,
) -> None:
    """Install the sinks. Idempotent — Streamlit reruns the script on every click."""
    global _configured
    if _configured and not force:
        return

    # read from env, not settings — logging must work before settings exist
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    json_enabled = _env_bool("LOG_JSON_ENABLED", True) if json_enabled is None else json_enabled
    log_path = Path(log_dir or os.getenv("LOG_DIR", "./logs"))

    logger.remove()
    logger.configure(patcher=context_patcher)

    logger.add(
        sys.stderr,
        level=level,
        format=_console_format,
        colorize=True,
        backtrace=False,  # the JSONL sink keeps the full trace; the console stays readable
        diagnose=False,  # diagnose=True prints local variables, which can include api keys
    )

    if json_enabled:
        log_path.mkdir(parents=True, exist_ok=True)

        # always DEBUG on disk — the noisy records are the ones you need after a failure
        logger.add(
            log_path / "amaris.jsonl",
            level="DEBUG",
            serialize=True,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,  # background writer — the pipeline never blocks on disk
            backtrace=True,
            diagnose=False,
        )
        logger.add(
            log_path / "errors.jsonl",
            level="ERROR",
            serialize=True,
            rotation="10 MB",
            retention="30 days",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    _configured = True
    logger.bind(level=level, log_dir=str(log_path), json=json_enabled).debug("logging.configured")


@contextmanager
def timed(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log {event}.complete with ms, or {event}.failed. Yields a dict for extra fields."""
    # re-raises on purpose — this measures, the node wrapper does the catching
    extra: dict[str, Any] = dict(fields)
    started = time.perf_counter()
    try:
        yield extra
    except Exception:
        elapsed = (time.perf_counter() - started) * 1000
        logger.bind(ms=round(elapsed, 1), **extra).exception(f"{event}.failed")
        raise
    else:
        elapsed = (time.perf_counter() - started) * 1000
        logger.bind(ms=round(elapsed, 1), **extra).info(f"{event}.complete")


def configure_from_settings(force: bool = False) -> None:
    """Same as configure_logging but reads the values from Settings."""
    # imported here, not at module level, so logging stays usable before settings load
    from amaris.config.settings import get_settings

    s = get_settings()
    configure_logging(
        level=s.log_level, log_dir=s.log_dir, json_enabled=s.log_json_enabled, force=force
    )
    # centralized tracer init: all entrypoints call this (ADR-043)
    from amaris.observability.tracing import configure_tracing

    configure_tracing()


__all__ = ["configure_from_settings", "configure_logging", "logger", "timed"]

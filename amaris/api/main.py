"""FastAPI app. Local mode only — cloud mode runs the pipeline in-process from Streamlit."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from amaris.api.routes import health, research
from amaris.config.settings import get_settings
from amaris.graph.pipeline import reset_pipeline
from amaris.memory.redis_memory import get_job_store, reset_job_store
from amaris.observability.logging import configure_from_settings, logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# the Streamlit frontend is the only browser client, and it is not same-origin in local mode
ALLOWED_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Logging first, then warm the job store so the first request does not pay for it."""
    configure_from_settings()
    settings = get_settings()
    store = await get_job_store()
    logger.bind(
        mode=settings.deployment_mode,
        job_store=store.backend,
        providers=settings.configured_llm_providers,
    ).info("api.startup")

    yield

    await reset_pipeline()
    await reset_job_store()
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    """Build the app. A factory so tests get a fresh instance per module."""
    app = FastAPI(
        title="AMARIS",
        description="Autonomous Multi-Agent Research Intelligence System",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(research.router)
    return app


app = create_app()


def _main() -> None:
    """uv run uvicorn amaris.api.main:app --reload, or python -m amaris.api.main."""
    import uvicorn

    configure_from_settings()
    uvicorn.run("amaris.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    _main()

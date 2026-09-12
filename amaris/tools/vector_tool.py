"""Semantic search over Qdrant. Works against a local container or Qdrant Cloud."""

from __future__ import annotations

import asyncio
import time
import uuid
from functools import lru_cache
from typing import Any, TypedDict

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

COLLECTION = "amaris_research"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384


class VectorHit(TypedDict):
    """One match from the knowledge base."""

    text: str
    url: str
    score: float


@lru_cache(maxsize=1)
def _embedder() -> Any | None:
    """MiniLM from the `memory` extra. None when it isn't installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.bind(tool="vector", model=EMBED_MODEL).warning("tool.extra_missing")
        return None
    return SentenceTransformer(EMBED_MODEL)


async def _embed(text: str) -> list[float] | None:
    model = _embedder()
    if model is None:
        return None
    loop = asyncio.get_running_loop()
    # encoding is CPU-bound, so it must not run on the event loop
    vector = await loop.run_in_executor(None, lambda: model.encode(text).tolist())
    return list(vector)


def _client() -> Any | None:
    """Qdrant client for whichever mode is configured, or None if unreachable."""
    settings = get_settings()
    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError:
        return None

    try:
        if settings.qdrant_url:
            return AsyncQdrantClient(
                url=settings.qdrant_url, api_key=settings.key("qdrant_api_key"), timeout=10
            )
        return AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=10)
    except Exception as exc:
        logger.bind(tool="vector", error=str(exc)[:200]).warning("tool.failed")
        return None


async def ping() -> bool:
    """True when Qdrant answers. Used by the /ready probe, never by the agents."""
    client = _client()
    if client is None:
        return False
    try:
        await client.get_collections()
        return True
    except Exception:
        return False
    finally:
        await client.close()


async def ensure_collection() -> bool:
    """Create the collection if missing. False when Qdrant isn't reachable."""
    client = _client()
    if client is None:
        return False
    try:
        from qdrant_client.models import Distance, VectorParams

        existing = await client.get_collections()
        if COLLECTION not in {c.name for c in existing.collections}:
            await client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.bind(collection=COLLECTION).info("vector.collection_created")
        return True
    except Exception as exc:
        logger.bind(tool="vector", error=str(exc)[:200]).warning("tool.failed")
        return False
    finally:
        await client.close()


async def upsert_documents(documents: list[dict[str, Any]]) -> int:
    """Store {text, url} documents. Returns how many landed, 0 on any failure."""
    if not documents or not await ensure_collection():
        return 0

    client = _client()
    if client is None:
        return 0

    try:
        from qdrant_client.models import PointStruct

        points = []
        for doc in documents:
            vector = await _embed(doc.get("text", ""))
            if vector is None:
                return 0
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": doc.get("text", ""), "url": doc.get("url", "")},
                )
            )
        await client.upsert(collection_name=COLLECTION, points=points)
        logger.bind(tool="vector", stored=len(points)).debug("tool.call")
        return len(points)
    except Exception as exc:
        logger.bind(tool="vector", error=str(exc)[:200]).warning("tool.failed")
        return 0
    finally:
        await client.close()


async def search_knowledge_base(query: str, limit: int = 5) -> list[VectorHit]:
    """Semantic search over stored research. Returns [] when unavailable — never raises."""
    started = time.perf_counter()
    vector = await _embed(query)
    if vector is None:
        return []

    client = _client()
    if client is None:
        return []

    try:
        response = await client.query_points(
            collection_name=COLLECTION, query=vector, limit=limit, with_payload=True
        )
        hits = [
            VectorHit(
                text=str((point.payload or {}).get("text", "")),
                url=str((point.payload or {}).get("url", "")),
                score=float(point.score),
            )
            for point in response.points
        ]
    except Exception as exc:
        logger.bind(tool="search_knowledge_base", error=str(exc)[:200]).warning("tool.failed")
        return []
    finally:
        await client.close()

    logger.bind(
        tool="search_knowledge_base",
        ms=round((time.perf_counter() - started) * 1000, 1),
        results=len(hits),
    ).debug("tool.call")
    return hits

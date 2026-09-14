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
# bge-small is 384-dim, the size this collection is already created with, so moving off
# sentence-transformers needed no migration. fastembed runs it on ONNX and pulls no torch.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384


class VectorHit(TypedDict):
    """One match from the knowledge base."""

    text: str
    url: str
    title: str
    score: float


@lru_cache(maxsize=1)
def _embedder() -> Any | None:
    """bge-small from the `files` extra. None when it isn't installed."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        logger.bind(tool="vector", model=EMBED_MODEL).warning("tool.extra_missing")
        return None
    return TextEmbedding(model_name=EMBED_MODEL)


async def _embed_many(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch in one pass. None when the extra is missing — never raises."""
    model = _embedder()
    if model is None or not texts:
        return None
    loop = asyncio.get_running_loop()
    # embedding is CPU-bound, so it must not run on the event loop
    vectors = await loop.run_in_executor(None, lambda: list(model.embed(texts)))
    return [[float(value) for value in vector] for vector in vectors]


async def _embed(text: str) -> list[float] | None:
    vectors = await _embed_many([text])
    return vectors[0] if vectors else None


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


async def upsert_documents(documents: list[dict[str, Any]], session_id: str = "") -> int:
    """Store {text, url, title} documents. Returns how many landed, 0 on any failure.

    `session_id` scopes the points to one run. The collection is shared, so an attachment
    written without it would surface in someone else's search.
    """
    if not documents or not await ensure_collection():
        return 0

    client = _client()
    if client is None:
        return 0

    try:
        from qdrant_client.models import PointStruct

        vectors = await _embed_many([str(doc.get("text", "")) for doc in documents])
        if vectors is None:
            return 0

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": doc.get("text", ""),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "session_id": session_id,
                },
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        await client.upsert(collection_name=COLLECTION, points=points)
        logger.bind(tool="vector", stored=len(points), session_id=session_id).debug("tool.call")
        return len(points)
    except Exception as exc:
        logger.bind(tool="vector", error=str(exc)[:200]).warning("tool.failed")
        return 0
    finally:
        await client.close()


async def search_knowledge_base(
    query: str, limit: int = 5, session_id: str = ""
) -> list[VectorHit]:
    """Semantic search over stored research. Returns [] when unavailable — never raises.

    Pass `session_id` to see only what this run uploaded; without it the search spans
    everything in the collection.
    """
    started = time.perf_counter()
    vector = await _embed(query)
    if vector is None:
        return []

    client = _client()
    if client is None:
        return []

    try:
        query_filter = None
        if session_id:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )

        response = await client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        hits = [
            VectorHit(
                text=str((point.payload or {}).get("text", "")),
                url=str((point.payload or {}).get("url", "")),
                title=str((point.payload or {}).get("title", "")),
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

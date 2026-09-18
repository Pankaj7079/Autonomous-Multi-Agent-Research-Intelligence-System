"""Semantic search over Qdrant. Works against a local container or Qdrant Cloud."""

from __future__ import annotations

import asyncio
import time
import uuid
from functools import lru_cache
from typing import Any, TypedDict

from amaris.config.settings import get_settings
from amaris.observability.logging import logger
from amaris.observability.tool_trace import record

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


async def ensure_collection(collection: str = COLLECTION) -> bool:
    """Create the collection if missing. False when Qdrant isn't reachable."""
    client = _client()
    if client is None:
        return False
    try:
        from qdrant_client.models import Distance, VectorParams

        existing = await client.get_collections()
        if collection not in {c.name for c in existing.collections}:
            await client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.bind(collection=collection).info("vector.collection_created")
        return True
    except Exception as exc:
        logger.bind(tool="vector", error=str(exc)[:200]).warning("tool.failed")
        return False
    finally:
        await client.close()


async def upsert_documents(
    documents: list[dict[str, Any]], session_id: str = "", collection: str = COLLECTION
) -> int:
    """Store {text, url, title} documents. Returns how many landed, 0 on any failure.

    `session_id` scopes the points to one run. The collection is shared, so an attachment
    written without it would surface in someone else's search.
    """
    if not documents or not await ensure_collection(collection):
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
                # extra keys ride along: episodic memory tags points with kind and created_at,
                # and dropping them here would lose that silently
                payload={
                    **doc,
                    "text": doc.get("text", ""),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "session_id": session_id,
                },
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        await client.upsert(collection_name=collection, points=points)
        logger.bind(tool="vector", stored=len(points), session_id=session_id).debug("tool.call")
        return len(points)
    except Exception as exc:
        logger.bind(tool="vector", error=str(exc)[:200]).warning("tool.failed")
        return 0
    finally:
        await client.close()


async def search_knowledge_base(
    query: str,
    limit: int = 5,
    session_id: str = "",
    urls: list[str] | None = None,
    collection: str = COLLECTION,
) -> list[VectorHit]:
    """Semantic search over stored research. Returns [] when unavailable — never raises.

    `urls` restricts the search to specific documents and is the filter attachments use:
    GraphState carries exactly which files belong to the conversation, so it is authoritative
    where a session id is not — ingestion and the run that reads it have different ids.
    `session_id` narrows to one uploader. Passing neither searches the whole collection,
    which for a shared collection means other people's documents.
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
        if session_id or urls:
            from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

            conditions = []
            if session_id:
                conditions.append(
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                )
            if urls:
                conditions.append(FieldCondition(key="url", match=MatchAny(any=list(urls))))
            query_filter = Filter(must=conditions)

        response = await client.query_points(
            collection_name=collection,
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
        # a collection that does not exist yet holds nothing, which is an answer, not a failure —
        # it is the normal state on a fresh qdrant and the 404 body reads like a real error
        if "doesn't exist" in str(exc) or "Not found" in str(exc):
            logger.bind(collection=collection).debug("vector.collection_missing")
            return []
        logger.bind(tool="search_knowledge_base", error=str(exc)[:200]).warning("tool.failed")
        return []
    finally:
        await client.close()

    elapsed = (time.perf_counter() - started) * 1000
    logger.bind(
        tool="search_knowledge_base",
        ms=round(elapsed, 1),
        results=len(hits),
    ).debug("tool.call")
    record(
        "vector_search",
        target=query,
        ok=bool(hits),
        ms=elapsed,
        detail=f"{len(hits)} chunks from {collection}",
    )
    return hits


async def delete_documents(
    session_id: str = "", urls: list[str] | None = None, collection: str = COLLECTION
) -> int:
    """Delete points for one attachment session, or for specific urls. Returns points removed.

    An upload is the user's own file; leaving it in a shared collection after the conversation
    ends is storage nobody asked for and evidence nobody can retrieve (ADR-045).
    """
    if not session_id and not urls:
        return 0

    client = _client()
    if client is None:
        return 0

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        must = []
        if session_id:
            must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
        if urls:
            must.append(FieldCondition(key="url", match=MatchAny(any=list(urls))))
        selector = Filter(must=must)

        # counted before deleting: qdrant does not report how many points a filter removed
        before = await client.count(collection_name=collection, count_filter=selector, exact=True)
        if not before.count:
            return 0
        await client.delete(collection_name=collection, points_selector=selector)
    except Exception as exc:
        logger.bind(tool="vector", error=str(exc)[:200]).warning("tool.failed")
        return 0
    else:
        logger.bind(
            tool="vector", removed=before.count, session_id=session_id, urls=len(urls or [])
        ).info("vector.deleted")
        return int(before.count)
    finally:
        await client.close()


async def purge_stale_attachments(max_age_hours: float, collection: str = COLLECTION) -> int:
    """Delete attachment chunks older than max_age_hours. Returns how many went.

    "Start over" cleans up the conversations that end properly. This is for the ones that do
    not: a closed browser tab never presses a button (ADR-045).
    """
    if max_age_hours <= 0:
        return 0

    client = _client()
    if client is None:
        return 0

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        cutoff = time.time() - max_age_hours * 3600
        selector = Filter(
            must=[
                FieldCondition(key="kind", match=MatchValue(value="attachment")),
                FieldCondition(key="created_ts", range=Range(lt=cutoff)),
            ]
        )
        before = await client.count(collection_name=collection, count_filter=selector, exact=True)
        if not before.count:
            return 0
        await client.delete(collection_name=collection, points_selector=selector)
    except Exception as exc:
        logger.bind(tool="vector", error=str(exc)[:200]).warning("tool.failed")
        return 0
    else:
        logger.bind(removed=before.count, older_than_h=max_age_hours).info("vector.purged")
        return int(before.count)
    finally:
        await client.close()

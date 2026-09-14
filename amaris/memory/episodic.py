"""Long-term memory across sessions: what past runs found, and what they concluded.

Replaces mem0 (ADR-034). mem0 needed sentence-transformers, which pulls torch (~2GB) and is
therefore disallowed in cloud mode — so in practice it was never installed and every call was
a silent no-op. This backs the same three functions with the Qdrant and fastembed that the
rest of the system already runs on, which makes long-term memory actually work.

Storing costs no LLM call. mem0's value was model-driven fact extraction; that is exactly the
decorative LLM use ADR-030 removed, and near-duplicate suppression by vector distance gets
most of the benefit for none of the latency.
"""

from __future__ import annotations

from datetime import UTC, datetime

from amaris.observability.logging import logger
from amaris.tools.vector_tool import search_knowledge_base, upsert_documents

COLLECTION = "amaris_memories"

# above this cosine score a new finding says what a stored one already says
DUPLICATE_SCORE = 0.95
# a finding only has to be specific enough to be worth recalling
MIN_FINDING_CHARS = 40
# a full report is mostly headings and citations, which add nothing to a recall
SUMMARY_CHAR_LIMIT = 2000

FINDING = "finding"
SUMMARY = "session_summary"


async def is_available() -> bool:
    """True when recall and store will actually do something."""
    from amaris.tools.vector_tool import _embedder, ping

    return _embedder() is not None and await ping()


async def _is_duplicate(text: str) -> bool:
    """True when something close enough is already stored — cheaper than an LLM dedup."""
    hits = await search_knowledge_base(text, limit=1, collection=COLLECTION)
    return bool(hits and hits[0]["score"] >= DUPLICATE_SCORE)


async def recall_related(query: str, limit: int = 3) -> list[str]:
    """What past sessions found about this. [] when Qdrant or the embedder is unavailable."""
    hits = await search_knowledge_base(query, limit=limit, collection=COLLECTION)
    memories = [hit["text"] for hit in hits if hit["text"]]
    logger.bind(query=query[:80], recalled=len(memories)).debug("episodic.recall")
    return memories


async def _store(text: str, kind: str, query: str, url: str = "") -> bool:
    if await _is_duplicate(text):
        logger.bind(kind=kind).debug("episodic.duplicate_skipped")
        return False

    stored = await upsert_documents(
        [
            {
                "text": text,
                "url": url,
                "title": f"{kind} · {query[:80]}",
                "kind": kind,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        ],
        collection=COLLECTION,
    )
    return bool(stored)


async def add_research_finding(query: str, finding: str, url: str = "") -> bool:
    """Store one finding for future runs. False when it did not land or was already known."""
    text = finding.strip()
    if len(text) < MIN_FINDING_CHARS:
        return False

    ok = await _store(text, FINDING, query, url)
    if ok:
        logger.bind(chars=len(text), url=url[:120]).debug("episodic.stored")
    return ok


async def add_session_summary(query: str, report: str, scores: dict[str, float]) -> bool:
    """Store a finished run so a later session recalls the conclusion, not just the findings."""
    conclusion = report.strip()[:SUMMARY_CHAR_LIMIT]
    if not conclusion:
        return False

    ok = await _store(f"{query}\n\n{conclusion}", SUMMARY, query)
    if ok:
        logger.bind(chars=len(conclusion), scores=len(scores)).debug("episodic.session_stored")
    return ok

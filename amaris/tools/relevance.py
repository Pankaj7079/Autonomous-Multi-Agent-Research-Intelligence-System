"""Lexical relevance scoring. Deliberately no LLM — a reranker would put judge latency on every run."""

from __future__ import annotations

import re
from typing import Any

# words that appear in almost any question and so cannot discriminate between sources
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "of",
        "on",
        "or",
        "tell",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "txt",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "about",
        "give",
        "show",
        "explain",
        "describe",
        "please",
        "can",
        "could",
        "would",
        "should",
        "do",
        "does",
        "did",
    ]
)

_TOKEN = re.compile(r"[a-z0-9]+")

# a query term in the title is a much stronger signal than the same term buried in body text
_TITLE_WEIGHT = 1.0
_CONTENT_WEIGHT = 0.5
# only the snippet is scored: search results pad the tail with navigation and boilerplate
_CONTENT_SCAN_CHARS = 1200


def _tokens(text: str) -> set[str]:
    """Content words only, crudely singularised so 'agents' and 'agent' match."""
    words = _TOKEN.findall(text.lower())
    return {w.rstrip("s") if len(w) > 3 and w.endswith("s") else w for w in words} - _STOPWORDS


def score_source(query: str, source: dict[str, Any]) -> float:
    """0-1 relevance of one source to the query. Returns 0.0 when nothing in the query matches."""
    wanted = _tokens(query)
    if not wanted:
        return 0.0

    title = _tokens(str(source.get("title", "")))
    content = _tokens(str(source.get("content", ""))[:_CONTENT_SCAN_CHARS])

    # each query term scores where it was found, so half a query matched in body text
    # lands near 0.25 while a real hit titled with the whole query lands near 1.0
    earned = sum(
        _TITLE_WEIGHT if term in title else _CONTENT_WEIGHT if term in content else 0.0
        for term in wanted
    )
    return round(earned / len(wanted), 3)


def select_for_prompt(
    query: str,
    sources: list[dict[str, Any]],
    limit: int,
    floor: float,
) -> list[dict[str, Any]]:
    """The sources worth spending prompt tokens on: scored, filtered, ranked, capped.

    The single selector for analyst, writer and critic, so all three read the same evidence.
    Each returned dict carries its score under "relevance".
    """
    if not sources:
        return []

    scored = [{**item, "relevance": score_source(query, item)} for item in sources]
    scored.sort(key=lambda item: item["relevance"], reverse=True)

    kept = [item for item in scored if item["relevance"] >= floor]
    # a query whose wording shares nothing with any source would otherwise hand the writer
    # nothing at all, which is worse than handing it the closest matches available
    return (kept or scored)[:limit]


def mean_relevance(query: str, sources: list[dict[str, Any]], top: int = 8) -> float:
    """Average relevance of the best sources — the researcher's fallback when self-assessment fails."""
    if not sources:
        return 0.0
    ranked = sorted((score_source(query, item) for item in sources), reverse=True)[:top]
    return round(sum(ranked) / len(ranked), 3)

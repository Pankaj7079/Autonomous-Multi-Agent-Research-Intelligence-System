"""Loads tests/golden/queries.yaml — the fixed query set the harness runs every layer against."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path("tests/golden/queries.yaml")


@dataclass(frozen=True)
class GoldenQuery:
    """One entry. `reference_answer` is optional — only queries that set it get context_recall."""

    id: str
    query: str
    category: str
    expected_min_sources: int
    expected_agents_involved: list[str]
    should_require_revision: bool
    notes: str
    reference_answer: str | None = None


def load_golden_set(path: Path | str = DEFAULT_PATH) -> list[GoldenQuery]:
    """Raises FileNotFoundError/ValueError early — a bad golden set should stop the harness, not
    silently run zero queries and report a clean pass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"golden set not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must contain a non-empty list of queries")

    queries = []
    seen_ids: set[str] = set()
    for entry in raw:
        query = _parse_entry(entry, path)
        if query.id in seen_ids:
            raise ValueError(f"duplicate golden query id: {query.id}")
        seen_ids.add(query.id)
        queries.append(query)
    return queries


def _parse_entry(entry: dict[str, Any], path: Path) -> GoldenQuery:
    missing = {"id", "query", "category", "expected_min_sources"} - entry.keys()
    if missing:
        raise ValueError(f"{path}: entry missing required fields {missing}: {entry}")

    return GoldenQuery(
        id=str(entry["id"]),
        query=str(entry["query"]),
        category=str(entry["category"]),
        expected_min_sources=int(entry["expected_min_sources"]),
        expected_agents_involved=list(entry.get("expected_agents_involved", [])),
        should_require_revision=bool(entry.get("should_require_revision", False)),
        notes=str(entry.get("notes", "")),
        reference_answer=entry.get("reference_answer"),
    )

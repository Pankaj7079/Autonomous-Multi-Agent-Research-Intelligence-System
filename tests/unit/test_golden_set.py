"""Golden set loading — a bad file must fail loudly, not silently run zero queries."""

from __future__ import annotations

from pathlib import Path

import pytest

from amaris.evaluation.golden_set import load_golden_set


def test_the_real_golden_set_loads_and_covers_every_category() -> None:
    queries = load_golden_set()
    categories = {q.category for q in queries}
    assert categories == {"factual", "comparative", "multi_hop", "recent_events", "ambiguous"}
    assert len(queries) >= 10


def test_the_real_golden_set_has_deliberately_hard_queries() -> None:
    """An eval set where everything passes measures nothing."""
    hard = [q for q in load_golden_set() if "deliberately hard" in q.notes]
    assert len(hard) >= 3


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden_set(tmp_path / "nope.yaml")


def test_empty_list_raises(tmp_path: Path) -> None:
    path = tmp_path / "queries.yaml"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_golden_set(path)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "queries.yaml"
    path.write_text("- {id: x1, query: 'q', category: factual}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_golden_set(path)


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    path = tmp_path / "queries.yaml"
    path.write_text(
        "- {id: x1, query: 'a', category: factual, expected_min_sources: 1}\n"
        "- {id: x1, query: 'b', category: factual, expected_min_sources: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_golden_set(path)


def test_optional_fields_default_sensibly(tmp_path: Path) -> None:
    path = tmp_path / "queries.yaml"
    path.write_text(
        "- {id: x1, query: 'a', category: factual, expected_min_sources: 1}\n", encoding="utf-8"
    )
    query = load_golden_set(path)[0]
    assert query.expected_agents_involved == []
    assert query.should_require_revision is False
    assert query.reference_answer is None


def test_reference_answer_is_carried_through(tmp_path: Path) -> None:
    path = tmp_path / "queries.yaml"
    path.write_text(
        "- {id: x1, query: 'a', category: factual, expected_min_sources: 1, "
        "reference_answer: 'the answer'}\n",
        encoding="utf-8",
    )
    assert load_golden_set(path)[0].reference_answer == "the answer"

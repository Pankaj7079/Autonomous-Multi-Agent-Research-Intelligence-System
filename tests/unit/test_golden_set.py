"""Golden set loading — a bad file must fail loudly, not silently run zero queries."""

from __future__ import annotations

from pathlib import Path

import pytest

from amaris.evaluation.golden_set import load_golden_set


def test_the_real_golden_set_loads_and_covers_every_category() -> None:
    queries = load_golden_set()
    categories = {q.category for q in queries}
    assert categories == {
        "factual",
        "comparative",
        "multi_hop",
        "recent_events",
        "ambiguous",
        "clarification",
        "direct",
        "live",
    }
    assert len(queries) >= 10


def test_the_cheap_categories_expect_fewer_agents_not_more() -> None:
    """A question that needs no research must not be scored as though it did."""
    cheap = [q for q in load_golden_set() if q.category in ("clarification", "direct", "live")]
    assert cheap, "the golden set must cover questions that do not deserve a full run"
    for query in cheap:
        assert "analyst" not in query.expected_agents_involved, query.id
        # presence alone cannot say "the analyst never ran" — that is what forbidden_agents is for
        assert "analyst" in query.forbidden_agents, query.id
        if query.category == "clarification":
            assert query.expected_min_sources == 0
            assert "researcher" not in query.expected_agents_involved, query.id
            assert "researcher" in query.forbidden_agents, query.id


def test_live_and_clarification_never_reach_the_researcher() -> None:
    """Both categories exist to prove a whole research run was avoided, not just shortened."""
    shortcut = [q for q in load_golden_set() if q.category in ("clarification", "live")]
    assert len(shortcut) >= 3
    for query in shortcut:
        assert "researcher" in query.forbidden_agents, query.id
        assert "planner" in query.forbidden_agents, query.id


def test_context_recall_has_something_to_score() -> None:
    """context_recall only runs on queries with a reference, so at least one must supply it."""
    referenced = [q for q in load_golden_set() if q.reference_answer]
    assert referenced, "no reference_answer anywhere means context_recall is never scored"
    for query in referenced:
        # a reference on a fast-moving fact rots — recent_events answers must stay unreferenced
        assert query.category != "recent_events", query.id


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

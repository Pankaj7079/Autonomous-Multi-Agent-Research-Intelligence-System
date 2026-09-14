"""The report as a .docx — what survives the flattening, and what happens without the extra."""

from __future__ import annotations

from importlib.util import find_spec
from io import BytesIO
from typing import ClassVar

import pytest

from amaris.export.document import Block, build_docx, docx_filename, to_blocks

REPORT = """## Answer

LangGraph routes with a supervisor [1].

## Key Findings

- it self-terminates
- the critic can route backwards

See the [docs](https://langchain.com/langgraph) for more.

## References

[1] routing docs — https://a.com"""

needs_docx = pytest.mark.skipif(find_spec("docx") is None, reason="export extra not installed")


class _Trace:
    triage: ClassVar[dict[str, str]] = {"depth": "deep"}
    quality_score = 0.81
    source_count = 7


class _Result:
    report = REPORT
    trace = _Trace()


def test_headings_keep_their_level() -> None:
    blocks = to_blocks(REPORT)
    headings = [b for b in blocks if b.kind == "heading"]
    assert [b.text for b in headings] == ["Answer", "Key Findings", "References"]
    assert {b.level for b in headings} == {2}


def test_bullets_are_not_flattened_into_prose() -> None:
    """A list rendered as paragraphs reads as a wall of text in Word."""
    assert Block("bullet", "it self-terminates", 1) in to_blocks(REPORT)


def test_a_link_keeps_its_url_because_paper_cannot_be_hovered() -> None:
    body = " ".join(b.text for b in to_blocks(REPORT) if b.kind == "body")
    assert "docs (https://langchain.com/langgraph)" in body


def test_a_bare_url_is_not_repeated_after_itself() -> None:
    """The reference list is "[1] title — url"; echoing the url would double every line."""
    blocks = to_blocks("[1] routing docs — https://a.com")
    assert blocks[0].text.count("https://a.com") == 1


def test_an_empty_report_produces_no_blocks() -> None:
    assert to_blocks("") == []


def test_the_filename_is_recognisable_and_safe() -> None:
    name = docx_filename("Tell me about God Rama!")
    assert name.startswith("amaris_tell-me-about-god-rama_")
    assert name.endswith(".docx")
    assert " " not in name and "!" not in name


def test_a_filename_survives_a_query_with_nothing_sluggable() -> None:
    assert docx_filename("???").startswith("amaris_report_")


@needs_docx
def test_the_document_opens_and_carries_the_question_and_the_score() -> None:
    from docx import Document

    data = build_docx(_Result(), "what is langgraph", session_id="abc12345", elapsed=41.2)

    text = "\n".join(p.text for p in Document(BytesIO(data)).paragraphs)
    assert "what is langgraph" in text
    assert "deep research" in text
    assert "quality 0.81" in text
    assert "it self-terminates" in text
    # the run is identified, so a document can be traced back to the log line that made it
    assert "run abc1234" in text


@needs_docx
def test_a_failed_run_still_exports_rather_than_raising() -> None:
    """A turn with no result is offered no button, but the renderer must not be the thing
    that decides that — an empty document is a better failure than a stack trace."""
    assert build_docx(None, "a question that failed")

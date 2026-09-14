"""Attachment ingestion: extraction, chunking, and the caps that keep prompts bounded."""

from __future__ import annotations

import pytest

from amaris.tools import attachments
from amaris.tools.attachments import AttachmentError, chunk_text, ingest


def test_chunks_never_exceed_the_budget_and_carry_overlap() -> None:
    text = "\n\n".join(f"Paragraph {i} about supervisor routing." for i in range(20))
    chunks = chunk_text(text, size=200, overlap=40)

    assert len(chunks) > 1
    # every chunk after the first is prefixed with the tail of the one before it
    assert all(chunks[i].startswith(chunks[i - 1][-40:]) for i in range(1, len(chunks)))


def test_a_paragraph_longer_than_the_budget_is_split_rather_than_dropped() -> None:
    """A single wall-of-text page must not silently vanish for having no paragraph breaks."""
    chunks = chunk_text("x" * 900, size=200, overlap=0)
    assert chunks
    assert "".join(chunks).count("x") >= 900


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("", size=200, overlap=20) == []


async def test_an_unsupported_type_is_refused_by_name() -> None:
    with pytest.raises(AttachmentError, match="not supported"):
        await ingest("notes.docx", b"x", "application/msword", "s1")


async def test_a_file_over_the_size_cap_is_refused_before_any_parsing() -> None:
    settings = attachments.get_settings()
    oversized = b"x" * (settings.attachment_max_bytes + 1)
    with pytest.raises(AttachmentError, match="larger than"):
        await ingest("big.pdf", oversized, "application/pdf", "s1")


async def test_a_scanned_pdf_says_so_instead_of_indexing_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan parses fine and yields no text — the user needs to be told to upload it as an image."""
    monkeypatch.setattr(attachments, "extract_pdf", lambda data, pages: "")

    with pytest.raises(AttachmentError):
        await ingest("scan.pdf", b"%PDF-1.4", "application/pdf", "s1")


async def test_ingest_stores_chunks_scoped_to_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    stored: dict[str, object] = {}

    async def fake_upsert(documents: list[dict[str, object]], session_id: str = "") -> int:
        stored["documents"] = documents
        stored["session_id"] = session_id
        return len(documents)

    monkeypatch.setattr(attachments, "extract_pdf", lambda data, pages: "Routing is decided.")
    monkeypatch.setattr("amaris.tools.vector_tool.upsert_documents", fake_upsert)

    result = await ingest("spec.pdf", b"%PDF-1.4", "application/pdf", "sess-9")

    assert result["chunks"] == 1
    assert stored["session_id"] == "sess-9"
    # the url is the citation target, so it has to name the file rather than a web page
    assert result["url"] == "file://spec.pdf"
    documents = stored["documents"]
    assert isinstance(documents, list)
    assert documents[0]["title"] == "spec.pdf"


async def test_a_file_that_cannot_be_indexed_is_reported_not_silently_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning success with nothing stored would leave the user waiting on a source that
    the researcher can never retrieve."""

    async def no_storage(documents: list[dict[str, object]], session_id: str = "") -> int:
        return 0

    monkeypatch.setattr(attachments, "extract_pdf", lambda data, pages: "Some text.")
    monkeypatch.setattr("amaris.tools.vector_tool.upsert_documents", no_storage)

    with pytest.raises(AttachmentError, match="could not be indexed"):
        await ingest("spec.pdf", b"%PDF-1.4", "application/pdf", "s1")

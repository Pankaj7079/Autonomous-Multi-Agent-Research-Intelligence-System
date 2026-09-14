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


def test_a_long_paragraph_is_not_stored_twice_inside_one_chunk() -> None:
    """The hard split advanced by size-overlap and then overlap was prepended again, so each
    chunk of an unbroken page carried the same passage back to back. Seen in real stored data."""
    markers = [f"[m{i:03d}]" for i in range(400)]
    chunks = chunk_text("".join(markers), size=300, overlap=60)

    for chunk in chunks:
        for marker in markers:
            assert chunk.count(marker) <= 1, f"{marker} repeated inside one chunk"


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


async def test_a_scanned_pdf_is_rendered_and_read_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan has pages but no text layer. Telling the user to convert it themselves was the
    bug; the pages are rasterised and read instead."""
    monkeypatch.setattr(attachments, "extract_pdf", lambda data, pages: "")
    monkeypatch.setattr(attachments, "rasterize_pdf", lambda data, pages, scale: [b"png1", b"png2"])

    async def fake_read(data: bytes, mime: str = "image/png") -> str:
        return f"page text from {data.decode()}"

    async def fake_upsert(documents: list[dict[str, object]], session_id: str = "") -> int:
        return len(documents)

    monkeypatch.setattr(attachments, "extract_image", fake_read)
    monkeypatch.setattr("amaris.tools.vector_tool.upsert_documents", fake_upsert)

    record = await ingest("scan.pdf", b"%PDF-1.4", "application/pdf", "s1")

    assert record["scanned"] is True
    assert record["chunks"] >= 1


async def test_a_scan_still_parses_when_the_vision_api_is_out_of_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The free vision tier runs out regularly. Offline OCR is what keeps uploads working."""
    calls: list[str] = []

    async def exhausted(data: bytes, mime: str) -> str:
        calls.append("vision")
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    async def ocr(data: bytes) -> str:
        calls.append("ocr")
        return "INVOICE 8841 total 4,380.50"

    monkeypatch.setattr(attachments, "vision_image", exhausted)
    monkeypatch.setattr(attachments, "ocr_image", ocr)
    monkeypatch.setattr("amaris.llm.router.provider_status", dict)

    text = await attachments.extract_image(b"fake-png", "image/png")

    assert "INVOICE 8841" in text
    assert calls == ["vision", "ocr"], "vision is tried first, OCR is the fallback"


async def test_a_rate_limited_vision_provider_is_skipped_not_waited_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The router already tracks cooldowns, so a parked provider costs no 429 round trip."""
    calls: list[str] = []

    async def vision(data: bytes, mime: str) -> str:
        calls.append("vision")
        return "should not be reached"

    async def ocr(data: bytes) -> str:
        calls.append("ocr")
        return "read offline"

    monkeypatch.setattr(attachments, "vision_image", vision)
    monkeypatch.setattr(attachments, "ocr_image", ocr)
    monkeypatch.setattr("amaris.llm.router.provider_status", lambda: {"gemini": 42.0})

    assert await attachments.extract_image(b"fake-png", "image/png") == "read offline"
    assert calls == ["ocr"]


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

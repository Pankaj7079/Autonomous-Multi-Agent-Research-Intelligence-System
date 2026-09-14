"""Turn an uploaded PDF or image into chunks the researcher can retrieve and cite.

Extraction happens once at upload, not inside an agent: it is I/O at the edge, like input
validation. Chunks go to Qdrant scoped to the session, so a document is retrieved on relevance
rather than pasted whole into every downstream prompt.
"""

from __future__ import annotations

import base64
import io
import re
from itertools import pairwise
from typing import Any

from amaris.config.settings import get_settings
from amaris.observability.logging import logger

# the only provider in the chain that reads images — groq and glm are text only
VISION_PROVIDER = "gemini"

PDF_TYPES = ("application/pdf",)
IMAGE_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/webp")
SUPPORTED_TYPES = PDF_TYPES + IMAGE_TYPES

# what the vision model is asked for — a description would be useless as a citable source
VISION_PROMPT = (
    "Transcribe every piece of text in this image exactly, preserving reading order. "
    "Then, in one short paragraph, describe any chart, diagram or table it contains, "
    "including the actual numbers. Do not comment on the image quality or add preamble."
)

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class AttachmentError(Exception):
    """Raised when a file cannot be read at all. The caller shows this to the user."""


def _clean(text: str) -> str:
    """PDF extraction leaves ragged spacing that wastes tokens and hurts chunk boundaries."""
    return _BLANK_LINES.sub("\n\n", _WHITESPACE.sub(" ", text)).strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraphs, packing up to `size` chars, with `overlap` carried between chunks.

    Paragraph-first because a mid-sentence split embeds badly — the vector ends up describing
    two half-thoughts and matches neither.
    """
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        piece = paragraph.strip()
        if not piece:
            continue
        if len(current) + len(piece) + 2 <= size:
            current = f"{current}\n\n{piece}" if current else piece
            continue
        if current:
            chunks.append(current)
        # a single paragraph longer than the budget is hard-split; nothing smarter to do
        while len(piece) > size:
            chunks.append(piece[:size])
            piece = piece[size - overlap :]
        current = piece
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) < 2:
        return chunks
    joined = [chunks[0]]
    for previous, chunk in pairwise(chunks):
        joined.append(f"{previous[-overlap:]}\n\n{chunk}")
    return joined


def extract_pdf(data: bytes, max_pages: int) -> str:
    """Text of the first `max_pages` pages. Raises AttachmentError when it cannot be read."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AttachmentError("PDF support needs `uv sync --extra files`") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages[:max_pages]]
    except Exception as exc:
        raise AttachmentError(f"could not read this PDF: {exc}") from exc

    text = _clean("\n\n".join(pages))
    if not text:
        # a scanned PDF has pages but no text layer, which is worth saying plainly
        raise AttachmentError(
            "this PDF has no selectable text — it is probably a scan. "
            "Upload it as an image instead and the vision model will read it."
        )
    return text


async def extract_image(data: bytes, mime: str) -> str:
    """Read an image with the vision model already in the provider chain — no OCR dependency."""

    from langchain_core.messages import HumanMessage

    from amaris.llm.router import configured_chain, get_llm

    # groq leads the chain but its models are text only, so vision pins gemini rather than
    # taking whatever is first and failing on the request
    if VISION_PROVIDER not in configured_chain():
        raise AttachmentError(
            "reading an image needs GEMINI_API_KEY — the rest of the chain is text only"
        )

    encoded = base64.b64encode(data).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    )
    try:
        response = await get_llm("vision", provider=VISION_PROVIDER).ainvoke([message])
    except Exception as exc:
        raise AttachmentError(f"the vision model could not read this image: {exc}") from exc

    text = _clean(str(response.content))
    if not text:
        raise AttachmentError("the vision model returned nothing for this image")
    return text


async def ingest(name: str, data: bytes, mime: str, session_id: str) -> dict[str, Any]:
    """Extract, chunk and store one file. Returns what the UI needs to show for it.

    Raises AttachmentError with a readable reason; it never returns a half-stored file.
    """
    settings = get_settings()
    if mime not in SUPPORTED_TYPES:
        raise AttachmentError(f"{mime or 'this file type'} is not supported — PDF or image only")
    if len(data) > settings.attachment_max_bytes:
        limit_mb = settings.attachment_max_bytes / 1_000_000
        raise AttachmentError(f"{name} is larger than the {limit_mb:.0f}MB limit")

    if mime in PDF_TYPES:
        text = extract_pdf(data, settings.attachment_max_pages)
    else:
        text = await extract_image(data, mime)

    text = text[: settings.attachment_max_chars]
    chunks = chunk_text(text, settings.attachment_chunk_chars, settings.attachment_chunk_overlap)
    if not chunks:
        raise AttachmentError(f"nothing readable was found in {name}")

    from amaris.tools.vector_tool import upsert_documents

    # url doubles as the citation target, so it has to identify the file, not a web page
    url = f"file://{name}"
    stored = await upsert_documents(
        [{"text": chunk, "url": url, "title": name} for chunk in chunks], session_id=session_id
    )
    if not stored:
        raise AttachmentError(
            f"{name} was read but could not be indexed — Qdrant is unreachable or "
            "the `files` extra is missing"
        )

    logger.bind(file=name, mime=mime, chunks=stored, chars=len(text)).info("attachment.ingested")
    return {"name": name, "url": url, "chunks": stored, "chars": len(text)}

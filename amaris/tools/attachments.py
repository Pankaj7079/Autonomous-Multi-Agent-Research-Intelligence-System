"""Turn an uploaded PDF or image into chunks the researcher can retrieve and cite.

Extraction happens once at upload, not inside an agent: it is I/O at the edge, like input
validation. Chunks go to Qdrant scoped to the session, so a document is retrieved on relevance
rather than pasted whole into every downstream prompt.
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
from functools import lru_cache
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
        # a single paragraph longer than the budget is hard-split; nothing smarter to do.
        # no overlap is applied here — it is added once, below, for every chunk alike.
        # advancing by size-overlap here as well double-counted it and stored the same
        # passage twice inside one chunk.
        while len(piece) > size:
            chunks.append(piece[:size])
            piece = piece[size:]
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
    """Text layer of the first `max_pages` pages. Empty string for a scan — the caller
    then rasterises it and reads the pages instead."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AttachmentError("PDF support needs `uv sync --extra files`") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages[:max_pages]]
    except Exception as exc:
        raise AttachmentError(f"could not read this PDF: {exc}") from exc

    return _clean("\n\n".join(pages))


def rasterize_pdf(data: bytes, max_pages: int, scale: float) -> list[bytes]:
    """Render pages to PNGs so a scan can be read. PDFium is BSD; PyMuPDF would be AGPL."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise AttachmentError("reading a scanned PDF needs `uv sync --extra files`") from exc

    try:
        document = pdfium.PdfDocument(data)
        images = []
        for page in list(document)[:max_pages]:
            buffer = io.BytesIO()
            page.render(scale=scale).to_pil().convert("RGB").save(buffer, format="PNG")
            images.append(buffer.getvalue())
    except Exception as exc:
        raise AttachmentError(f"could not render this PDF: {exc}") from exc
    return images


async def ocr_image(data: bytes) -> str:
    """Read an image with RapidOCR — Apache-2.0, ONNX, offline, and not rate limited.

    This is what makes a scan work at all: the free vision API quota runs out regularly,
    and an uploaded document has to keep parsing when it does.
    """
    try:
        import numpy as np
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise AttachmentError("offline OCR needs `uv sync --extra files`") from exc

    def run() -> str:
        engine = _ocr_engine(RapidOCR)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        result, _ = engine(np.array(image))
        return "\n".join(line[1] for line in (result or []))

    loop = asyncio.get_running_loop()
    # OCR is CPU-bound and takes seconds per page, so it must not block the event loop
    return _clean(await loop.run_in_executor(None, run))


@lru_cache(maxsize=1)
def _ocr_engine(factory: Any) -> Any:
    """RapidOCR loads its ONNX models on construction, so it is built once per process."""
    return factory()


async def vision_image(data: bytes, mime: str) -> str:
    """Read an image with the vision model in the provider chain. Richer than OCR — it also
    describes charts — but it is the part that runs out of free quota."""
    from langchain_core.messages import HumanMessage

    from amaris.llm.router import configured_chain, get_llm

    # groq and glm both reject image content outright, so vision pins gemini rather than
    # taking whatever leads the chain and failing on the request
    if VISION_PROVIDER not in configured_chain():
        raise AttachmentError("no vision provider configured")

    encoded = base64.b64encode(data).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    )
    response = await get_llm("vision", provider=VISION_PROVIDER).ainvoke([message])
    return _clean(str(response.content))


async def extract_image(data: bytes, mime: str = "image/png") -> str:
    """Vision model first for the richer read, offline OCR when it is unavailable.

    Mirrors how the LLM router already handles providers: try the best one, fall through on
    failure. A rate-limited gemini is skipped outright rather than waited on for a 429.
    """
    from amaris.llm.router import provider_status

    cooling = provider_status().get(VISION_PROVIDER) or 0.0
    if not cooling:
        try:
            text = await vision_image(data, mime)
            if text:
                logger.bind(reader=VISION_PROVIDER).debug("attachment.read")
                return text
        except Exception as exc:
            logger.bind(error=str(exc)[:150]).info("attachment.vision_unavailable")

    text = await ocr_image(data)
    if not text:
        raise AttachmentError("no text could be read from this image")
    logger.bind(reader="rapidocr", chars=len(text)).debug("attachment.read")
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

    scanned = False
    if mime in PDF_TYPES:
        text = extract_pdf(data, settings.attachment_max_pages)
        if not text:
            # no text layer means a scan, so the pages are rendered and read as images
            # rather than handing the problem back to the user
            scanned = True
            pages = rasterize_pdf(
                data, settings.attachment_max_scan_pages, settings.attachment_scan_scale
            )
            if not pages:
                raise AttachmentError(f"{name} has no pages to read")
            read = await asyncio.gather(
                *(extract_image(page) for page in pages), return_exceptions=True
            )
            text = _clean("\n\n".join(part for part in read if isinstance(part, str) and part))
            if not text:
                raise AttachmentError(f"{name} is a scan and no text could be read from it")
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

    logger.bind(file=name, mime=mime, chunks=stored, chars=len(text), scanned=scanned).info(
        "attachment.ingested"
    )
    return {"name": name, "url": url, "chunks": stored, "chars": len(text), "scanned": scanned}

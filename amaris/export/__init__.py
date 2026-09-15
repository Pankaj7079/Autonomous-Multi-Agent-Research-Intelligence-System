"""Getting a finished report out of the system — as a document, or to an inbox.

Deliberately importable without Streamlit: the frontend builds the bytes in-process in both
deployment modes, but nothing here knows that, so an API route could call it unchanged.
"""

from __future__ import annotations

from amaris.export.document import (
    ExportUnavailable,
    build_docx,
    build_markdown_docx,
    docx_filename,
)
from amaris.export.mailer import (
    DOCX_MIME,
    MailRefused,
    MailUnavailable,
    email_enabled,
    refusal_reason,
    send_report,
)

__all__ = [
    "DOCX_MIME",
    "ExportUnavailable",
    "MailRefused",
    "MailUnavailable",
    "build_docx",
    "build_markdown_docx",
    "docx_filename",
    "email_enabled",
    "refusal_reason",
    "send_report",
]

"""Emails a finished report through Resend.

Resend is plain HTTP, so httpx covers it and no mail library is added. The gating here is
the point of the module as much as the sending is: a demo that will mail a document to any
address a visitor types is an open relay, so cloud mode refuses until an allow-list exists.
"""

from __future__ import annotations

import base64

import httpx

from amaris.config.settings import get_settings
from amaris.observability.logging import logger
from amaris.safety.pii import looks_like_email

RESEND_URL = "https://api.resend.com/emails"
SEND_TIMEOUT_SECONDS = 20.0
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
# resend's own ceiling is 40MB after base64; a report never approaches it, so this is a guard
MAX_ATTACHMENT_BYTES = 10_000_000


class MailUnavailable(RuntimeError):
    """No RESEND_API_KEY, so the feature is off rather than broken."""


class MailRefused(RuntimeError):
    """The send was rejected — by our own gate, or by Resend."""


def email_enabled() -> bool:
    """True when the email control should be rendered at all."""
    settings = get_settings()
    if not settings.key("resend_api_key"):
        return False
    # a public deploy that can mail anywhere is the open-relay case, so it stays off
    return bool(not settings.is_cloud or settings.email_allowed_domains)


def refusal_reason(address: str) -> str:
    """Why this address may not be mailed, or "" when it may be."""
    settings = get_settings()
    address = address.strip()
    if not settings.key("resend_api_key"):
        return "email is not configured — set RESEND_API_KEY"
    if not looks_like_email(address):
        return "that does not look like an email address"
    allowed = [domain.strip().lower().lstrip("@") for domain in settings.email_allowed_domains]
    allowed = [domain for domain in allowed if domain]
    if settings.is_cloud and not allowed:
        return "sending is disabled on the public demo — set EMAIL_ALLOWED_DOMAINS to enable it"
    domain = address.rsplit("@", 1)[-1].lower()
    if allowed and domain not in allowed:
        return f"{domain} is not in the allowed domains"
    return ""


def _payload(
    to: str, subject: str, html: str, attachment: bytes, filename: str
) -> dict[str, object]:
    settings = get_settings()
    body: dict[str, object] = {
        "from": settings.email_from,
        "to": [to],
        "subject": subject,
        "html": html,
    }
    if attachment:
        body["attachments"] = [
            {
                "filename": filename or "report.docx",
                "content": base64.b64encode(attachment).decode("ascii"),
                "content_type": DOCX_MIME,
            }
        ]
    return body


async def send_report(
    to: str,
    subject: str,
    html: str,
    *,
    attachment: bytes = b"",
    filename: str = "",
) -> str:
    """Send one report and return Resend's message id. Raises on a refusal at either end."""
    settings = get_settings()
    key = settings.key("resend_api_key")
    if not key:
        raise MailUnavailable("email is not configured — set RESEND_API_KEY")

    reason = refusal_reason(to)
    if reason:
        raise MailRefused(reason)
    if len(attachment) > MAX_ATTACHMENT_BYTES:
        raise MailRefused("the attachment is too large to send")

    async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
        response = await client.post(
            RESEND_URL,
            headers={"Authorization": f"Bearer {key}"},
            json=_payload(to, subject, html, attachment, filename),
        )

    if response.status_code >= 400:
        # resend error passthrough: user sees real problem
        raise MailRefused(f"resend rejected the send ({response.status_code}): {response.text}")

    message_id = str(response.json().get("id", ""))
    # the domain only: the address itself is the user's contact detail, not a log field
    logger.bind(
        domain=to.rsplit("@", 1)[-1], message_id=message_id, attached=bool(attachment)
    ).info("export.email_sent")
    return message_id

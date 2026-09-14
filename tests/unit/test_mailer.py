"""Emailing a report. Most of these test the refusals, because that is where the risk is."""

from __future__ import annotations

from typing import Any

import pytest

from amaris.config.settings import Settings, get_settings
from amaris.export import mailer
from amaris.export.mailer import (
    MailRefused,
    MailUnavailable,
    email_enabled,
    refusal_reason,
    send_report,
)

KEY = "re_test_key_not_real"


@pytest.fixture(autouse=True)
def _clear_settings() -> Any:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> None:
    base = {"resend_api_key": KEY, "deployment_mode": "local", "email_allowed_domains": []}
    configured = Settings(**{**base, **fields})
    monkeypatch.setattr(mailer, "get_settings", lambda: configured)


def test_no_key_means_the_control_is_not_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, resend_api_key=None)
    assert email_enabled() is False


def test_a_public_demo_with_no_allow_list_refuses_to_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise anyone who opens the demo can mail a document to any address they type."""
    _settings(monkeypatch, deployment_mode="cloud")

    assert email_enabled() is False
    assert "public demo" in refusal_reason("someone@example.com")


def test_a_public_demo_with_an_allow_list_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, deployment_mode="cloud", email_allowed_domains=["gmail.com"])

    assert email_enabled() is True
    assert refusal_reason("pankaj@gmail.com") == ""
    assert "not in the allowed domains" in refusal_reason("someone@elsewhere.com")


def test_local_mode_sends_anywhere_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The allow-list guards a public deploy; on a developer's own machine it is noise."""
    _settings(monkeypatch)
    assert email_enabled() is True
    assert refusal_reason("anyone@anywhere.dev") == ""


@pytest.mark.parametrize("address", ["not-an-address", "a@b", "", "a b@c.com", "@gmail.com"])
def test_a_malformed_address_is_rejected_before_any_http_call(
    monkeypatch: pytest.MonkeyPatch, address: str
) -> None:
    _settings(monkeypatch)
    assert "does not look like an email" in refusal_reason(address)


async def test_sending_without_a_key_is_unavailable_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch, resend_api_key=None)
    with pytest.raises(MailUnavailable):
        await send_report("a@b.com", "s", "<p>h</p>")


async def test_an_oversized_attachment_never_reaches_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    with pytest.raises(MailRefused, match="too large"):
        await send_report(
            "a@b.com", "s", "<p>h</p>", attachment=b"x" * (mailer.MAX_ATTACHMENT_BYTES + 1)
        )


class _Response:
    def __init__(self, status: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self._payload = payload or {"id": "msg-1"}
        self.text = "unverified sender"

    def json(self) -> dict[str, Any]:
        return self._payload


def _capture(monkeypatch: pytest.MonkeyPatch, response: _Response) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            seen["timeout"] = kwargs.get("timeout")

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def post(self, url: str, **kwargs: Any) -> _Response:
            seen["url"] = url
            seen.update(kwargs)
            return response

    monkeypatch.setattr(mailer.httpx, "AsyncClient", FakeClient)
    return seen


async def test_a_send_carries_the_document_as_a_base64_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64

    _settings(monkeypatch)
    seen = _capture(monkeypatch, _Response())

    assert (
        await send_report(
            "pankaj@gmail.com",
            "AMARIS · q",
            "<p>answer</p>",
            attachment=b"DOCXBYTES",
            filename="r.docx",
        )
        == "msg-1"
    )

    assert seen["url"] == mailer.RESEND_URL
    assert seen["headers"]["Authorization"] == f"Bearer {KEY}"
    body = seen["json"]
    assert body["to"] == ["pankaj@gmail.com"]
    attachment = body["attachments"][0]
    assert base64.b64decode(attachment["content"]) == b"DOCXBYTES"
    assert attachment["content_type"] == mailer.DOCX_MIME


async def test_no_attachment_key_is_sent_when_there_is_no_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    seen = _capture(monkeypatch, _Response())

    await send_report("pankaj@gmail.com", "s", "<p>h</p>")
    assert "attachments" not in seen["json"]


async def test_resends_own_error_reaches_the_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "could not send" alone leaves the user with nothing to act on — an unverified
    sender is a fixable problem only if they are told that is what it is."""
    _settings(monkeypatch)
    _capture(monkeypatch, _Response(status=403))

    with pytest.raises(MailRefused, match="unverified sender"):
        await send_report("pankaj@gmail.com", "s", "<p>h</p>")

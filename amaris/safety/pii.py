"""Regex PII detection and masking. Presidio is not worth a 300MB spaCy model here."""

from __future__ import annotations

import re
from typing import NamedTuple

MASKS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "aadhaar": "[AADHAAR]",
    "pan": "[PAN]",
    "card": "[CARD]",
    "ip": "[IP]",
}


class PIIMatch(NamedTuple):
    """One hit. span is (start, end) into the text it was found in."""

    type: str
    span: tuple[int, int]
    value: str


_EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# groupings differ by country (+91 98765 43210 vs 555-123-4567), so match a loose
# candidate and let the digit count decide, rather than encoding every national format
_PHONE = re.compile(r"(?<![\w.])\(?\+?\d[\d ()-]{8,18}\d(?![\w.])")
# "2026-09-15 16:45" is ten digits with separators, so it read as a phone and a live weather
# answer shipped with its timestamp masked out
_DATELIKE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")
# the tail guard allows a separator: "1234 5678 9012 3456 7" is an id, not an aadhaar
_AADHAAR = re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?![ -]?\d)")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
# no trailing separator, or the match swallows the space after the number
_CARD = re.compile(r"(?<![\d.])\d(?:[ -]?\d){12,18}(?![\d.])")
_IPV4 = re.compile(
    r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}(?![\w.])"
)
# "v1.2.3.4" and "version 1.2.3.4" are dotted quads that are not addresses
_VERSION_PREFIX = re.compile(r"(?:\bv|\bversion\s+)$", re.IGNORECASE)


def _luhn(digits: str) -> bool:
    """Card checksum. Without it every 16-digit id in a citation reads as a card number."""
    total, alternate = 0, False
    for char in reversed(digits):
        value = int(char)
        if alternate:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        alternate = not alternate
    return total % 10 == 0


def _card_matches(text: str) -> list[PIIMatch]:
    matches = []
    for found in _CARD.finditer(text):
        digits = re.sub(r"\D", "", found.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            matches.append(PIIMatch("card", found.span(), found.group()))
    return matches


def _phone_matches(text: str) -> list[PIIMatch]:
    """10-15 digits, and either a separator or a country code — bare digit runs are ids."""
    matches = []
    for found in _PHONE.finditer(text):
        raw = found.group()
        if _DATELIKE.search(raw):
            continue
        digits = re.sub(r"\D", "", raw)
        if 10 <= len(digits) <= 15 and (raw.startswith("+") or any(c in raw for c in " -()")):
            matches.append(PIIMatch("phone", found.span(), raw))
    return matches


def _ip_matches(text: str) -> list[PIIMatch]:
    return [
        PIIMatch("ip", found.span(), found.group())
        for found in _IPV4.finditer(text)
        if not _VERSION_PREFIX.search(text[: found.start()])
    ]


def looks_like_email(text: str) -> bool:
    """True when the whole string is one address. The mailer validates recipients with this,
    so there is one email pattern in the repo rather than a second that drifts from it."""
    return bool(_EMAIL.fullmatch(text.strip()))


def detect_pii(text: str) -> list[PIIMatch]:
    """Every match, ordered by position. Overlaps are resolved longest-first."""
    if not text:
        return []

    found: list[PIIMatch] = []
    for name, pattern in (
        ("email", _EMAIL),
        ("aadhaar", _AADHAAR),
        ("pan", _PAN),
    ):
        found += [PIIMatch(name, m.span(), m.group()) for m in pattern.finditer(text)]
    found += _phone_matches(text)
    found += _card_matches(text)
    found += _ip_matches(text)

    # a phone number sits inside a card-shaped run of digits, so the longer match wins
    found.sort(key=lambda m: (m.span[0], -(m.span[1] - m.span[0])))
    kept: list[PIIMatch] = []
    cursor = -1
    for match in found:
        if match.span[0] >= cursor:
            kept.append(match)
            cursor = match.span[1]
    return kept


def mask_pii(text: str) -> tuple[str, list[PIIMatch]]:
    """Replaced text plus what was replaced. Rebuilt back-to-front so spans stay valid."""
    matches = detect_pii(text)
    if not matches:
        return text, []

    masked = text
    for match in reversed(matches):
        start, end = match.span
        masked = masked[:start] + MASKS[match.type] + masked[end:]
    return masked, matches

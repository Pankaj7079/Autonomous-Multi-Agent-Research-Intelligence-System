"""Patch 5 — PII detection and the input/output gates. False positives matter as much as misses."""

from __future__ import annotations

import pytest

from amaris.safety.guardrails import MAX_QUERY_CHARS, validate_input, validate_output
from amaris.safety.pii import detect_pii, mask_pii


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("write to sam.a@example.co.uk about it", "email"),
        ("call +91 98765 43210 tomorrow", "phone"),
        ("ring 555-123-4567 now", "phone"),
        ("dial (555) 123-4567", "phone"),
        ("aadhaar 1234 5678 9012 on file", "aadhaar"),
        ("PAN ABCDE1234F is registered", "pan"),
        ("paid with 4111 1111 1111 1111", "card"),
        ("the host is 192.168.1.44", "ip"),
    ],
)
def test_each_pii_type_is_found(text: str, kind: str) -> None:
    assert [m.type for m in detect_pii(text)] == [kind]


@pytest.mark.parametrize(
    "text",
    [
        "upgrade to v1.2.3.4 today",
        "see version 10.0.0.1 notes",
        "ISBN 9780132350884 covers it",
        "the 2019 study had 12000 participants",
        "refer to RFC 2119 and section 4.2.1",
        "revenue grew from 1000 to 250000 in 2024",
    ],
)
def test_ordinary_technical_prose_is_not_flagged(text: str) -> None:
    """A research report is full of long numbers — over-masking would wreck every report."""
    assert detect_pii(text) == []


def test_a_non_luhn_card_length_number_is_ignored() -> None:
    """Without the checksum every 16-digit identifier in a citation reads as a card."""
    assert detect_pii("reference 1234 5678 9012 3456 7") == []


def test_masking_replaces_values_and_keeps_surrounding_text() -> None:
    masked, matches = mask_pii("mail a@b.com and call 555-123-4567 please")
    assert masked == "mail [EMAIL] and call [PHONE] please"
    assert {m.type for m in matches} == {"email", "phone"}


def test_masking_is_a_no_op_on_clean_text() -> None:
    text = "What is the Model Context Protocol?"
    assert mask_pii(text) == (text, [])


def test_overlapping_matches_do_not_double_mask() -> None:
    """A card-shaped run also looks like a phone number; the longer match must win outright."""
    masked, matches = mask_pii("paid with 4111 1111 1111 1111 today")
    assert masked == "paid with [CARD] today"
    assert len(matches) == 1


def test_a_normal_query_passes_untouched() -> None:
    result = validate_input("What is the Model Context Protocol?")
    assert result.ok
    assert result.text == "What is the Model Context Protocol?"


@pytest.mark.parametrize("query", ["", "  ", "ab"])
def test_an_empty_query_is_blocked(query: str) -> None:
    assert not validate_input(query).ok


def test_an_over_length_query_is_blocked() -> None:
    result = validate_input("x" * (MAX_QUERY_CHARS + 1))
    assert not result.ok
    assert str(MAX_QUERY_CHARS) in result.reason


def test_a_query_with_no_words_is_blocked() -> None:
    assert not validate_input("!!!! ???? ####").ok


def test_pii_in_a_query_is_masked_rather_than_blocked() -> None:
    """A question that happens to contain an email is still a legitimate question."""
    result = validate_input("is support@vendor.com the right contact for MCP?")
    assert result.ok
    assert "[EMAIL]" in result.text
    assert result.masked == ["email"]


def test_output_masking_never_blocks() -> None:
    result = validate_output("Contact the author at a@b.com for the dataset.")
    assert result.ok
    assert "[EMAIL]" in result.text


def test_output_masking_handles_an_empty_report() -> None:
    assert validate_output("").text == ""

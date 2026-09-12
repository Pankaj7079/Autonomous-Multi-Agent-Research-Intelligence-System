"""What may enter the pipeline, and what may leave it. Wired in api routes and the evaluator."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from amaris.observability.logging import logger
from amaris.safety.injection import scan_injection
from amaris.safety.pii import mask_pii

MIN_QUERY_CHARS = 3
MAX_QUERY_CHARS = 500
# a "query" with no letters at all is a paste accident, not a research question
_HAS_LETTERS = re.compile(r"[^\W\d_]")


@dataclass(frozen=True)
class GuardResult:
    """ok=False means do not run. text is the cleaned value to use when ok=True."""

    ok: bool
    text: str
    reason: str = ""
    masked: list[str] = field(default_factory=list)


def validate_input(query: str) -> GuardResult:
    """Block junk and injection attempts, mask PII, and return what the pipeline should run."""
    cleaned = (query or "").strip()

    if len(cleaned) < MIN_QUERY_CHARS:
        return GuardResult(False, cleaned, "the query is empty or too short to research")
    if len(cleaned) > MAX_QUERY_CHARS:
        return GuardResult(False, cleaned, f"the query is longer than {MAX_QUERY_CHARS} characters")
    if not _HAS_LETTERS.search(cleaned):
        return GuardResult(False, cleaned, "the query contains no words to research")

    scan = scan_injection(cleaned)
    if scan.high:
        logger.bind(risk=scan.risk, patterns=scan.patterns).warning("guardrails.injection_blocked")
        return GuardResult(
            False,
            cleaned,
            "the query looks like an attempt to override the system's instructions "
            f"({', '.join(scan.patterns)})",
        )

    masked, matches = mask_pii(cleaned)
    if matches:
        # the values never reach a log line, only their types
        logger.bind(types=sorted({m.type for m in matches})).info("guardrails.input_masked")
    return GuardResult(True, masked, masked=sorted({m.type for m in matches}))


def validate_output(report: str) -> GuardResult:
    """Mask PII that came in from scraped pages. Never blocks — a masked report still ships."""
    masked, matches = mask_pii(report or "")
    if matches:
        logger.bind(types=sorted({m.type for m in matches}), count=len(matches)).info(
            "guardrails.output_masked"
        )
    return GuardResult(True, masked, masked=sorted({m.type for m in matches}))

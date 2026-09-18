"""Check the report's facts against the pages it cites. Deterministic, no model call, no dependency.

This is the answer-time signal, and it is the one thing a hosted research tool cannot do: it
verifies a claim against the page text we actually scraped and still hold. Perplexity and
ChatGPT show you a link and trust the model's attribution; here the attribution is checked.

Only *verifiable* facts are graded — figures, dates and quoted text. Scoring a
whole sentence by word overlap punished paraphrase, so well-written reports were flagged as
unsupported and the warning fired on nearly every answer (ADR-040).

RAGAS stays in `harness.py`, where it belongs — an offline benchmark over the golden set, not
a judge run on every user question (ADR-039).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

# a citation marker in the report body: [1], [12]
_MARKER = re.compile(r"\[(\d{1,3})\]")
# sentence split that keeps abbreviations mostly intact; a perfect splitter is not worth a dep
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# the generated reference list: its lines cite everything and claim nothing
_REFERENCES = re.compile(r"^#{1,6}\s*references\b.*", re.IGNORECASE | re.MULTILINE)
# markdown emphasis and heading marks, stripped so "**Prompt caching**" reads as two words
_MARKUP = re.compile(r"[*_`#>]+")
# dash normalization: unicode dashes → plain hyphens
_DASHES = str.maketrans(dict.fromkeys(map(chr, (0x2012, 0x2013, 0x2014, 0x2015, 0x2212)), "-"))

# a figure worth checking: 90%, $3.00, 1,000, 2024, 10-45% (each side matched separately)
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_QUOTED = re.compile(r'"([^"]{4,80})"')

# more than this and the sentence is a list of facts, not a claim — check the first few
MAX_FACTS_PER_CLAIM = 8
# how recent a source has to be to count as recent, in days
RECENT_DAYS = 365
# one miss on a long answer is a rounding error; a warning has to be earned before it shows
MIN_MISSES_TO_WARN = 2

_DATE = re.compile(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})")


@dataclass(frozen=True)
class Claim:
    """One cited sentence, the facts it asserts, and which of them the page does not contain."""

    text: str
    citation: int
    grounded: bool
    facts: list[str]
    missing: list[str]
    reason: str


@dataclass(frozen=True)
class CitationAudit:
    """What the report asserted, and how much of it the cited pages actually contain."""

    claims: list[Claim]
    checked: int
    grounded: int
    unsupported: int
    skipped: int
    dead: int
    cited_sources: int
    domains: int
    recent: int
    dated: int

    @property
    def grounded_ratio(self) -> float:
        """0.0 when nothing was checkable — which is itself worth saying out loud."""
        return round(self.grounded / self.checked, 4) if self.checked else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for the API payload. Claims become dicts too."""
        data = asdict(self)
        data["grounded_ratio"] = self.grounded_ratio
        return data


def _number_needles(raw: str) -> list[str]:
    """Every spelling of a figure a page might use. "3.00" also matches "3.0" and "3"."""
    core = raw.strip("$%").replace(",", "")
    needles = {raw.lower(), core}
    if "," in raw:
        needles.add(raw.strip("$%").lower())
    if "." in core:
        trimmed = core.rstrip("0").rstrip(".")
        # 0.30 may be written 0.3, and 3.00 may be written 3 — but 0.30 must never match "0"
        if trimmed and trimmed != "0":
            needles.add(trimmed)
    return [n for n in needles if n]


def _facts(sentence: str) -> list[tuple[str, list[str]]]:
    """(what the sentence asserts, how it might be spelled in the page). Empty = nothing to check.

    Figures and quoted text only. Names were tried and dropped: without a model there is no way
    to tell a company from a capitalised heading word, so the report's own lead-ins were being
    reported as claims its sources failed to mention (ADR-040).
    """
    clean = _MARKUP.sub(" ", _MARKER.sub(" ", sentence)).translate(_DASHES).strip()
    found: dict[str, list[str]] = {}

    for match in _NUMBER.finditer(clean):
        raw = match.group()
        digits = raw.strip("$%").replace(",", "").replace(".", "")
        # a bare single digit matches almost any page, so it proves nothing either way
        if len(digits) < 2 and "." not in raw:
            continue
        found[raw] = _number_needles(raw)

    for match in _QUOTED.finditer(clean):
        found[f'"{match.group(1)}"'] = [match.group(1).lower()]

    return list(found.items())[:MAX_FACTS_PER_CLAIM]


def _source_dates(sources: list[dict[str, Any]]) -> tuple[int, int]:
    """(with a readable date, of those recent enough). Absent dates are never counted as old."""
    now = datetime.now(UTC)
    dated = recent = 0
    for item in sources:
        blob = f"{item.get('published', '')} {item.get('date', '')}"
        match = _DATE.search(blob)
        if not match:
            continue
        dated += 1
        try:
            when = datetime(int(match[1]), int(match[2]), int(match[3]), tzinfo=UTC)
        except ValueError:
            continue
        if (now - when).days <= RECENT_DAYS:
            recent += 1
    return dated, recent


def _body(report: str) -> str:
    """The report without its generated reference list, whose lines cite but assert nothing."""
    match = _REFERENCES.search(report or "")
    return report[: match.start()] if match else (report or "")


def audit(
    report: str, citations: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> CitationAudit:
    """Verify the figures in every cited sentence against the page that sentence cites.

    `sources` must carry `content` — the scraped page. Sources trimmed to a snippet still work
    but grade harsher, because there is less of the page left to find the fact in.
    """
    by_index = {
        int(item["index"]): item for item in citations if str(item.get("index", "")).isdigit()
    }
    text_by_url = {
        str(item.get("url", "")): f"{item.get('content') or ''} "
        f"{item.get('snippet') or ''}".lower().translate(_DASHES)
        for item in sources
        if item.get("url")
    }

    # dead citations: references to unstored pages counted at list level
    dead = sum(
        1 for item in by_index.values() if not text_by_url.get(str(item.get("url", "")), "").strip()
    )

    claims: list[Claim] = []
    skipped = 0
    for sentence in _SENTENCE.split(_body(report)):
        markers = _MARKER.findall(sentence)
        if not markers:
            continue
        facts = _facts(sentence)
        if not facts:
            # prose without figures can't be model-checked — skip
            skipped += 1
            continue
        for marker in dict.fromkeys(markers):
            index = int(marker)
            citation = by_index.get(index)
            url = str(citation.get("url", "")) if citation else ""
            page = text_by_url.get(url, "")
            if not citation or not page.strip():
                claims.append(
                    Claim(
                        sentence.strip(), index, False, [f for f, _ in facts], [], "page not stored"
                    )
                )
                continue
            missing = [fact for fact, needles in facts if not any(n in page for n in needles)]
            claims.append(
                Claim(
                    sentence.strip(),
                    index,
                    not missing,
                    [f for f, _ in facts],
                    missing,
                    "found in the cited page" if not missing else "not in the cited page",
                )
            )

    cited_urls = {str(item.get("url", "")) for item in citations if item.get("url")}
    domains = {urlparse(url).netloc.removeprefix("www.") for url in cited_urls if url}
    dated, recent = _source_dates([s for s in sources if str(s.get("url", "")) in cited_urls])

    grounded_count = sum(1 for claim in claims if claim.grounded)
    return CitationAudit(
        claims=claims,
        checked=len(claims),
        grounded=grounded_count,
        unsupported=len(claims) - grounded_count,
        skipped=skipped,
        dead=dead,
        cited_sources=len(cited_urls),
        domains=len(domains),
        recent=recent,
        dated=dated,
    )


def caveat(result: CitationAudit) -> str:
    """The line to put on the answer when something is actually wrong with it. "" otherwise.

    Claude's research mode is the model here, including its restraint: a warning shown on every
    answer teaches the reader to skip it, so it has to be earned (ADR-040).
    """
    if not result.checked and not result.skipped:
        return "Nothing in this answer carries a citation, so none of it could be verified."
    if result.dead:
        pages = "page" if result.dead == 1 else "pages"
        return (
            f"{result.dead} cited {pages} could not be retrieved, so those claims were not checked."
        )
    if result.unsupported >= MIN_MISSES_TO_WARN or (result.unsupported and result.checked <= 2):
        facts = sorted({fact for claim in result.claims for fact in claim.missing})[:3]
        detail = ", ".join(facts)
        return (
            f"{result.unsupported} of {result.checked} checked claims cite a page that does not "
            f"mention what they assert ({detail})."
        )
    # skip single-domain warning on short, fully-grounded answers
    if result.cited_sources > 1 and result.domains == 1:
        return "Every citation points at the same site, so this answer rests on a single source."
    return ""

"""The answer-time signal: what the report asserted, and whether its citations contain it."""

from __future__ import annotations

from amaris.evaluation.citation_audit import audit, caveat

PAGE = {
    "url": "https://a.com",
    "content": (
        "The mango is the edible stone fruit of Mangifera indica, native to South Asia. "
        "India produced 24.7 million tonnes in 2023, about 40% of the world total."
    ),
}
OFF_TOPIC = {
    "url": "https://b.com",
    "content": "A long article about Kubernetes, Istio and service meshes.",
}


def test_a_figure_present_in_the_cited_page_is_verified() -> None:
    result = audit(
        "India produced 24.7 million tonnes in 2023 [1].",
        [{"index": 1, "url": "https://a.com"}],
        [PAGE],
    )
    assert result.checked == 1 and result.grounded == 1
    assert caveat(result) == ""


def test_a_figure_absent_from_its_cited_page_is_named_in_the_caveat() -> None:
    """The failure RAGAS's single faithfulness number averaged away — and the whole point of
    checking: the reader is told which figure to go and look at."""
    result = audit(
        "India produced 99.9 million tonnes in 2023, worth $450 billion [1].",
        [{"index": 1, "url": "https://a.com"}],
        [PAGE],
    )
    assert result.grounded == 0 and result.unsupported == 1
    assert "99.9" in caveat(result)


def test_paraphrase_with_no_figure_or_name_is_not_graded_at_all() -> None:
    """Why this module was rewritten: scoring prose by word overlap flagged good writing as
    unsupported, so the warning fired on nearly every answer and stopped meaning anything."""
    result = audit(
        "It is grown widely and remains popular with growers everywhere [1].",
        [{"index": 1, "url": "https://a.com"}],
        [PAGE],
    )
    assert result.checked == 0 and result.skipped == 1
    assert caveat(result) == ""


def test_the_generated_reference_list_is_not_audited_as_claims() -> None:
    """Found live: every "[2] Some Title — https://host/path" line was being graded as a claim,
    and a URL's words are not in the page body, so the reference block failed itself."""
    report = (
        "India produced 24.7 million tonnes in 2023 [1].\n\n"
        "## References\n\n"
        "[1] Mangoes Of The World — https://a.com/some-long-slug-2024\n"
    )
    result = audit(report, [{"index": 1, "url": "https://a.com"}], [PAGE])
    assert result.checked == 1 and result.grounded == 1


def test_a_capitalised_name_is_not_treated_as_a_checkable_fact() -> None:
    """Names were tried and dropped. Without a model there is no way to tell a company from a
    heading word, so live runs reported the report's own bolded lead-ins — "Mechanism",
    "Providers" — as facts the cited sources failed to mention."""
    result = audit(
        "The Mangifera genus is unrelated to Kubernetes [2].",
        [{"index": 2, "url": "https://b.com"}],
        [OFF_TOPIC],
    )
    assert result.checked == 0 and result.skipped == 1
    assert caveat(result) == ""


def test_one_miss_on_a_long_answer_does_not_earn_a_warning() -> None:
    """A warning shown on every answer is skipped by readers, so it has to be earned."""
    report = " ".join(
        [
            "India produced 24.7 million tonnes in 2023 [1].",
            "That is about 40% of the world total [1].",
            "Mangifera indica is native to South Asia [1].",
            "Brazil produced 77.7 million tonnes [1].",
        ]
    )
    result = audit(report, [{"index": 1, "url": "https://a.com"}], [PAGE])
    # the "native to South Asia" line carries no figure, so it is skipped rather than guessed at
    assert result.checked == 3 and result.skipped == 1 and result.unsupported == 1
    assert caveat(result) == ""


def test_a_citation_to_a_page_we_never_stored_counts_as_dead() -> None:
    """Perplexity and ChatGPT show the link and trust the attribution; we kept the page, so
    an unverifiable citation can be named as unverifiable rather than shown as fine."""
    result = audit(
        "The tree grows to 40 metres in height [3].",
        [{"index": 3, "url": "https://gone.com"}],
        [PAGE],
    )
    assert result.dead == 1
    assert "could not be retrieved" in caveat(result)


def test_a_dead_citation_counts_even_when_no_sentence_was_checkable() -> None:
    """Found while building: dead was only counted inside the per-sentence loop, so a short
    claim citing a missing page reported zero dead citations."""
    result = audit("It is red [3].", [{"index": 3, "url": "https://gone.com"}], [PAGE])
    assert result.dead == 1


def test_an_answer_that_cites_nothing_is_reported_as_unverifiable() -> None:
    """Silence here would read the same as a fully verified answer."""
    result = audit("Mango is a fruit.", [], [PAGE])
    assert result.checked == 0
    assert "none of it could be verified" in caveat(result)


def test_one_domain_behind_every_citation_is_called_out() -> None:
    """Source diversity is one of the axes RAGAS does not cover at all."""
    result = audit(
        "Mangifera indica is native to South Asia [1].",
        [{"index": 1, "url": "https://a.com"}, {"index": 2, "url": "https://a.com/y"}],
        [PAGE, {"url": "https://a.com/y", "content": "mangifera indica south asia"}],
    )
    assert result.domains == 1
    assert "single source" in caveat(result)


def test_the_audit_never_raises_on_malformed_input() -> None:
    """It runs inside the terminal node, where nothing may cost a finished report."""
    assert audit("", [], []).checked == 0
    assert audit("text [1]", [{"index": "not-a-number"}], [{"no_url": 1}]).checked == 0

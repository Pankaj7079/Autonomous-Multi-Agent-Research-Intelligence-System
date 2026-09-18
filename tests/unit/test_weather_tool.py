"""Live state is looked up, not researched — detection has to be right before anything is spent."""

from __future__ import annotations

import pytest

from amaris.tools.weather_tool import Weather, place_for_weather_query

READING = Weather(
    place="Darbhanga",
    region="Bihar",
    country="India",
    temperature_c=29.4,
    feels_like_c=36.2,
    humidity_pct=81,
    wind_kmh=3.1,
    condition="overcast",
    observed="2026-09-15 16:45",
    timezone="Asia/Kolkata",
    high_c=31.8,
    low_c=26.3,
    rain_chance_pct=97,
    source_url="https://api.open-meteo.com/v1/forecast?latitude=26.15&longitude=85.89",
)


@pytest.mark.parametrize(
    ("query", "place"),
    [
        ("darbhanga weather", "darbhanga"),
        ("weather in Darbhanga", "Darbhanga"),
        ("what is the weather in Delhi today", "Delhi"),
        ("temperature in patna right now", "patna"),
        ("New York weather tonight", "New York"),
        ("can you please tell me the weather in Darbhanga today", "Darbhanga"),
    ],
)
def test_a_question_about_conditions_right_now_yields_its_place(query: str, place: str) -> None:
    assert place_for_weather_query(query) == place


@pytest.mark.parametrize(
    "query",
    [
        "how does weather forecasting work",
        "why is the weather changing in north india",
        "average rainfall in Kerala",
        "what is prompt caching and when does it pay off?",
        "compare the weather models used by ECMWF and NOAA",
        # a real report: asked back for a date "today" already answered, because this compound
        # question was long enough to reach the LLM triage path with a stale prompt (fixed in
        # triage.py's PROMPT) rather than because place_for_weather_query mis-detected it — this
        # locks in that raising _MAX_WORDS must never make it start guessing a place here instead
        "tell me today patna weather and what about flood condition in patna plz tell me",
        "",
    ],
)
def test_a_research_question_is_left_to_the_research_path(query: str) -> None:
    """A question that explains, compares, looks back, or mixes in another topic entirely is
    research even when it says "weather"."""
    assert place_for_weather_query(query) == ""


def test_the_summary_carries_the_time_it_was_true() -> None:
    """The whole failure this fixes was monthly averages presented as today's conditions."""
    summary = READING.summary()
    assert "29.4" in summary and "Darbhanga, Bihar, India" in summary
    assert "as of 2026-09-15 16:45 local time" in summary
    # no trailing stop: the caller appends "[1]." and a marker after the stop becomes its own
    # sentence, which the citation audit then has nothing to check
    assert not summary.endswith(".")


def test_the_reading_is_its_own_evidence() -> None:
    """The live node cites the reading and stores it as the source, so the audit verifies the
    figures against the same numbers rather than skipping an uncheckable answer."""
    from amaris.evaluation.citation_audit import audit, caveat

    summary = READING.summary()
    result = audit(
        f"## Answer\n\n{summary} [1].",
        [{"index": 1, "url": READING.source_url}],
        [{"url": READING.source_url, "content": summary}],
    )
    assert result.checked == 1 and result.grounded == 1
    assert caveat(result) == ""

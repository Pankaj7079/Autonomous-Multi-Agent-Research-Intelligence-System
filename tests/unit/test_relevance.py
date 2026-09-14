"""Relevance scoring is what stops 57 off-topic pages being gathered and then binned."""

from __future__ import annotations

from amaris.config.settings import get_settings
from amaris.tools.relevance import mean_relevance, score_source, select_for_prompt

# the literal source that survived the 92-second weather run and taught the report nothing
API_PRICING = {
    "title": "AccuWeather APIs — pricing and developer plans",
    "url": "https://developer.accuweather.com/pricing",
    "content": "Compare request quotas, billing tiers and enterprise support for our data feeds.",
}
TODAYS_WEATHER = {
    "title": "Today's weather forecast for London",
    "url": "https://example.com/london-today",
    "content": "Today: 14C, light rain this afternoon, wind 12mph from the south west.",
}


def test_an_api_pricing_page_is_not_an_answer_about_the_weather() -> None:
    """The failure case this whole filter exists for."""
    query = "tell about today weather"
    assert score_source(query, API_PRICING) < get_settings().relevance_floor
    assert score_source(query, TODAYS_WEATHER) > score_source(query, API_PRICING)


def test_a_title_match_outweighs_the_same_word_buried_in_body_text() -> None:
    query = "langgraph state machines"
    titled = {"title": "LangGraph state machines explained", "content": "..."}
    mentioned = {
        "title": "Ten python libraries",
        "content": "one of them is langgraph, a state machine",
    }
    assert score_source(query, titled) > score_source(query, mentioned)


def test_a_query_with_no_content_words_scores_nothing_rather_than_everything() -> None:
    assert score_source("what is it about", TODAYS_WEATHER) == 0.0


def test_selection_ranks_filters_and_caps() -> None:
    selected = select_for_prompt(
        "today weather", [API_PRICING, TODAYS_WEATHER], limit=5, floor=0.35
    )
    assert [item["url"] for item in selected] == [TODAYS_WEATHER["url"]]
    assert selected[0]["relevance"] > 0.35


def test_selection_never_hands_the_writer_an_empty_list() -> None:
    """Nothing clearing the floor is still better evidence than no evidence at all."""
    selected = select_for_prompt("quantum basket weaving", [API_PRICING], limit=5, floor=0.9)
    assert len(selected) == 1


def test_selection_respects_the_limit() -> None:
    many = [{**TODAYS_WEATHER, "url": f"https://example.com/{n}"} for n in range(20)]
    assert len(select_for_prompt("today weather", many, limit=6, floor=0.35)) == 6


def test_mean_relevance_falls_back_honestly_rather_than_counting() -> None:
    """Counting sources scored 69 junk pages 0.7; this cannot say that."""
    assert mean_relevance("today weather", [API_PRICING] * 10) < 0.35
    assert mean_relevance("today weather", [TODAYS_WEATHER] * 10) > 0.7
    assert mean_relevance("anything", []) == 0.0

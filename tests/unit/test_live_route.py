"""The live path: a question about current state is looked up, never researched (ADR-041)."""

from __future__ import annotations

from amaris.graph import nodes
from amaris.graph.edges import route_from_live, route_from_triage
from amaris.graph.state import LIVE, PLANNER, new_state
from amaris.tools.weather_tool import Weather

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


async def test_triage_sends_a_weather_question_to_the_live_node_without_a_model_call() -> None:
    """Detection is deterministic on purpose: picking a depth skips triage's model call, which
    is exactly what happened on the run that spent 442s scraping climate averages."""
    from amaris.agents.triage import TriageAgent

    state = new_state("darbhanga weather")
    # depth_locked is what the depth picker sets, and it must not bypass the live check
    state["query_depth"] = "deep"
    state["depth_locked"] = True

    update = await TriageAgent().run(state)

    assert update["live_data"] == {"kind": "weather", "place": "darbhanga"}
    assert update["query_depth"] == "direct"
    assert route_from_triage({**state, **update}) == LIVE


async def test_the_live_node_answers_with_a_citation_and_the_time_it_was_true(monkeypatch) -> None:
    async def reading(place: str):
        assert place == "darbhanga"
        return READING

    monkeypatch.setattr("amaris.tools.weather_tool.current_weather", reading)

    state = new_state("darbhanga weather")
    state["live_data"] = {"kind": "weather", "place": "darbhanga"}

    update = await nodes.live_node(state)

    assert "29.4" in update["final_report"]
    assert "as of 2026-09-15 16:45 local time" in update["final_report"]
    assert update["citations"][0]["url"] == READING.source_url
    # the reading is stored as the source, so the citation audit can verify its own figures
    assert update["raw_research"][0]["content"] == READING.summary()
    assert update["live_data"]["source"] == "Open-Meteo"
    assert route_from_live({**state, **update}) == "evaluator"


async def test_a_lookup_that_fails_falls_back_to_research(monkeypatch) -> None:
    """A wrong guess about the place costs one HTTP call, not the answer."""

    async def nothing(place: str):
        return None

    monkeypatch.setattr("amaris.tools.weather_tool.current_weather", nothing)

    state = new_state("weather in zzz")
    state["live_data"] = {"kind": "weather", "place": "zzz"}

    update = await nodes.live_node(state)

    assert update["live_data"] == {}
    assert route_from_live({**state, **update}) == PLANNER

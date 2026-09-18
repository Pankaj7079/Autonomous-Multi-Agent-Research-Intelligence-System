"""Current conditions from Open-Meteo. No key, no dependency — httpx is already core.

Weather is live state, not research. Scraping search results for it returns monthly climate
averages and air-quality boilerplate written up as if they were today, which is exactly how a
"what is the weather in X" run produced a confident wrong answer (ADR-041).
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from amaris.observability.logging import logger
from amaris.observability.tool_trace import record

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# the public endpoint is fast; a weather answer that takes longer than this is not worth waiting for
_TIMEOUT = 10.0

_CURRENT = "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
_DAILY = "temperature_2m_max,temperature_2m_min,precipitation_probability_max"

# WMO 4677 codes, which is what Open-Meteo returns instead of a description
_CONDITIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


@dataclass(frozen=True)
class Weather:
    """One live reading. `observed` is local time at the place, which is what a reader means."""

    place: str
    region: str
    country: str
    temperature_c: float
    feels_like_c: float
    humidity_pct: int
    wind_kmh: float
    condition: str
    observed: str
    timezone: str
    high_c: float | None
    low_c: float | None
    rain_chance_pct: int | None
    # the exact request behind this reading, so a reader can open it and see the same numbers
    source_url: str

    @property
    def where(self) -> str:
        """ "Darbhanga, Bihar, India" — enough to prove we resolved the right place."""
        return ", ".join(p for p in (self.place, self.region, self.country) if p)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "where": self.where}

    def summary(self) -> str:
        """One sentence a report can open with, with the time it was true attached."""
        parts = [
            f"{self.temperature_c:.1f}°C",
            f"feels like {self.feels_like_c:.1f}°C",
            self.condition,
            f"humidity {self.humidity_pct}%",
            f"wind {self.wind_kmh:.1f} km/h",
        ]
        if self.high_c is not None and self.low_c is not None:
            parts.append(f"today {self.low_c:.1f} to {self.high_c:.1f}°C")
        if self.rain_chance_pct is not None:
            parts.append(f"{self.rain_chance_pct}% chance of rain")
        # no full stop: the caller appends the citation marker, and a marker after the stop
        # becomes its own sentence, which the citation audit then has nothing to check
        return f"{self.where}: {', '.join(parts)} (as of {self.observed} local time)"


async def _get(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = await client.get(url, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


async def current_weather(place: str) -> Weather | None:
    """Live conditions for a named place, or None if it cannot be resolved or fetched.

    Never raises: a node calls this, and a failed lookup must fall back to research rather
    than take the run down.
    """
    name = (place or "").strip()
    if not name:
        return None

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            found = await _get(client, _GEOCODE_URL, {"name": name, "count": 1})
            results = found.get("results") or []
            if not results:
                logger.bind(tool="weather", place=name[:60]).info("tool.place_not_found")
                record(
                    "weather_api",
                    target=name,
                    ok=False,
                    ms=(time.perf_counter() - started) * 1000,
                    detail="place not found",
                )
                return None
            spot = results[0]
            data = await _get(
                client,
                _FORECAST_URL,
                {
                    "latitude": spot["latitude"],
                    "longitude": spot["longitude"],
                    "current": _CURRENT,
                    "daily": _DAILY,
                    "timezone": "auto",
                    "forecast_days": 1,
                },
            )
    except Exception as exc:
        logger.bind(tool="weather", error=str(exc)[:200]).warning("tool.failed")
        record(
            "weather_api",
            target=name,
            ok=False,
            ms=(time.perf_counter() - started) * 1000,
            detail="lookup failed",
        )
        return None

    current = data.get("current") or {}
    daily = data.get("daily") or {}

    def first(key: str) -> Any:
        values = daily.get(key) or []
        return values[0] if values else None

    reading = Weather(
        place=str(spot.get("name", name)),
        region=str(spot.get("admin1", "")),
        country=str(spot.get("country", "")),
        temperature_c=float(current.get("temperature_2m", 0.0)),
        feels_like_c=float(current.get("apparent_temperature", 0.0)),
        humidity_pct=int(current.get("relative_humidity_2m", 0)),
        wind_kmh=float(current.get("wind_speed_10m", 0.0)),
        condition=_CONDITIONS.get(int(current.get("weather_code", -1)), "unknown conditions"),
        observed=str(current.get("time", "")).replace("T", " "),
        timezone=str(data.get("timezone", "")),
        high_c=first("temperature_2m_max"),
        low_c=first("temperature_2m_min"),
        rain_chance_pct=first("precipitation_probability_max"),
        source_url=(
            f"{_FORECAST_URL}?latitude={spot['latitude']}&longitude={spot['longitude']}"
            f"&current={_CURRENT}&timezone=auto"
        ),
    )
    elapsed = (time.perf_counter() - started) * 1000
    logger.bind(
        tool="weather",
        place=reading.where[:80],
        ms=round(elapsed, 1),
    ).info("tool.complete")
    record(
        "weather_api",
        target=reading.where,
        ms=elapsed,
        detail=f"{reading.temperature_c:.0f}°C · {reading.condition}",
    )
    return reading


# asking for conditions, not asking how weather works
_LIVE_WORDS = re.compile(
    r"\b(weather|temperature|forecast|raining|rainfall today|humidity|how hot|how cold)\b",
    re.IGNORECASE,
)
# a question wanting explanation, history or comparison is research even when it says "weather"
_RESEARCH_WORDS = re.compile(
    r"\b(why|how does|how do|history|historical|climate change|compare|comparison|impact|"
    r"explain|model|models|predict(?:ion|ing)?|average|trend|during|between \d)\b",
    re.IGNORECASE,
)
# stripped to leave the place behind: "what is the weather in darbhanga today" -> "darbhanga"
_NOISE_WORDS = frozenset(
    [
        "what",
        "whats",
        "is",
        "are",
        "am",
        "the",
        "a",
        "an",
        "current",
        "currently",
        "now",
        "today",
        "tonight",
        "tomorrow",
        "like",
        "in",
        "at",
        "of",
        "for",
        "me",
        "my",
        "tell",
        "show",
        "give",
        "please",
        "plz",
        "can",
        "could",
        "would",
        "should",
        "you",
        "weather",
        "temperature",
        "forecast",
        "humidity",
        "report",
        "condition",
        "conditions",
        "how",
        "hot",
        "cold",
        "much",
        "degree",
        "degrees",
        "raining",
        "rain",
        "right",
        "just",
        "outside",
        "there",
        "here",
        "does",
        "do",
        "it",
        "feel",
        "feels",
        "will",
        "be",
        "this",
        "week",
        "weekend",
        "todays",
        "wind",
    ]
)
# anything that is not part of a place name, so "delhi?" and "new-york" both survive
_NOT_NAME = re.compile("[^A-Za-z0-9' -]+")
# generous enough for real phrasing ("can you please tell me the weather in X today") while a
# compound question mixing weather with something else usually runs longer than this and falls
# through to triage instead — where a wrong guess here is cheap anyway (the geocoder returns
# nothing and the run falls back to research), this is a length past which it stops being cheap
_MAX_WORDS = 12


def place_for_weather_query(query: str) -> str:
    """The place a live-weather question is about, or "" when it is not one.

    Deterministic on purpose: triage skips its model call when the user picks a depth, so an
    LLM flag would miss exactly the case that failed (ADR-041). A wrong guess is cheap — the
    geocoder returns nothing and the run falls back to research.
    """
    text = (query or "").strip()
    if not text or len(text.split()) > _MAX_WORDS:
        return ""
    if not _LIVE_WORDS.search(text) or _RESEARCH_WORDS.search(text):
        return ""
    words = _NOT_NAME.sub(" ", text).split()
    return " ".join(word for word in words if word.lower() not in _NOISE_WORDS)

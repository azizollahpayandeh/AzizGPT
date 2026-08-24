"""get_time and get_weather - read-only local and public-API lookups."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = 12

# WMO weather interpretation codes, spoken-friendly.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light rain showers", 81: "rain showers",
    82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "a thunderstorm", 96: "a thunderstorm with hail",
    99: "a thunderstorm with heavy hail",
}


def local_tz_name() -> str:
    """Best-effort IANA timezone name, falling back to the UTC offset."""
    env_tz = os.getenv("TZ", "").strip()
    if env_tz:
        return env_tz

    tzfile = Path("/etc/timezone")
    try:
        if tzfile.is_file():
            name = tzfile.read_text(encoding="utf-8").strip()
            if name:
                return name
    except OSError:
        pass

    # Kali/Debian ships /etc/localtime as a symlink into the zoneinfo tree.
    try:
        resolved = Path("/etc/localtime").resolve()
        parts = resolved.parts
        if "zoneinfo" in parts:
            return "/".join(parts[parts.index("zoneinfo") + 1:])
    except OSError:
        pass

    return datetime.now().astimezone().tzname() or "local time"


def time_schema(cfg) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": (
                "Get the current local time and date on the user's computer. "
                "Use this for 'what time is it' and 'what is today's date'."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


def weather_schema(cfg) -> dict[str, Any]:
    default_city = cfg["weather"].get("default_city", "")
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a city. If the user does not name a "
                f"city, omit the argument and the default ({default_city}) is used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, for example Messina. Optional.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def get_time(cfg, dry_run: bool = False) -> str:
    now = datetime.now().astimezone()
    return (
        f"It is {now.strftime('%-I:%M %p').lower()} on "
        f"{now.strftime('%A, %B %-d, %Y')} ({local_tz_name()})."
    )


def get_weather(cfg, city: str = "", dry_run: bool = False) -> str:
    wanted = str(city or "").strip() or cfg["weather"].get("default_city", "")
    if not wanted:
        return "No city was given and no default city is configured."

    try:
        geo = requests.get(
            GEOCODE_URL,
            params={"name": wanted, "count": 1, "language": "en", "format": "json"},
            timeout=HTTP_TIMEOUT,
        )
        geo.raise_for_status()
        results = (geo.json() or {}).get("results") or []
        if not results:
            return f"I could not find a place called {wanted}."
        place = results[0]

        forecast = requests.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=HTTP_TIMEOUT,
        )
        forecast.raise_for_status()
        current = (forecast.json() or {}).get("current") or {}
    except requests.RequestException as exc:
        log.warning("get_weather failed for %r: %s", wanted, exc.__class__.__name__)
        return f"I could not reach the weather service just now ({exc.__class__.__name__})."
    except (KeyError, ValueError) as exc:
        log.warning("get_weather got an unexpected response: %s", exc.__class__.__name__)
        return "The weather service returned something I could not read."

    label = place.get("name", wanted)
    country = place.get("country")
    where = f"{label}, {country}" if country else label
    description = WEATHER_CODES.get(current.get("weather_code"), "unclear conditions")
    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    wind = current.get("wind_speed_10m")
    humidity = current.get("relative_humidity_2m")

    return (
        f"{where}: {description}, {temp} degrees Celsius, feels like {feels}, "
        f"humidity {humidity} percent, wind {wind} kilometres per hour."
    )

"""Tool registry: a fixed, enumerated set. Nothing here reaches a shell.

Every tool is a plain Python function with a JSON schema. The model can only
name a tool from this table and can only pass arguments the schema declares;
each function re-validates its own arguments regardless of what the model sent.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from . import alarms, apps, power, system, web

log = logging.getLogger(__name__)

# name -> (schema builder, implementation)
_REGISTRY: dict[str, tuple[Callable[[Any], dict], Callable[..., str]]] = {
    "open_app": (apps.schema, apps.open_app),
    "open_url": (web.url_schema, web.open_url),
    "open_email": (web.email_schema, web.open_email),
    "get_time": (system.time_schema, system.get_time),
    "get_weather": (system.weather_schema, system.get_weather),
    "set_alarm": (alarms.schema, alarms.set_alarm),
    "power_action": (power.schema, power.power_action),
}

TOOL_NAMES = tuple(_REGISTRY)

# Tools with a side effect outside this process. The test harness runs these in
# dry-run so a 20-command sweep never launches apps or schedules real timers.
SIDE_EFFECTING = ("open_app", "open_url", "open_email", "set_alarm", "power_action")


def build_schemas(cfg) -> list[dict[str, Any]]:
    """JSON schemas for the chat-completions `tools` parameter."""
    return [build(cfg) for build, _ in _REGISTRY.values()]


def dispatch(cfg, name: str, args: dict[str, Any] | None, dry_run: bool = False) -> str:
    """Run one tool call and return a short string for the model to read back."""
    entry = _REGISTRY.get(name)
    if entry is None:
        log.info("refused unknown tool call: %r", name)
        return (
            f"Refused: there is no tool called '{name}'. "
            f"The tools that exist are: {', '.join(TOOL_NAMES)}."
        )

    func = entry[1]
    supplied = args if isinstance(args, dict) else {}

    # Drop anything the function does not declare rather than raising TypeError.
    accepted = set(inspect.signature(func).parameters) - {"cfg", "dry_run"}
    kwargs = {k: v for k, v in supplied.items() if k in accepted}
    ignored = sorted(set(supplied) - set(kwargs))
    if ignored:
        log.debug("tool %s: ignored unexpected argument(s) %s", name, ignored)

    try:
        return func(cfg, dry_run=dry_run, **kwargs)
    except Exception as exc:  # a broken tool must not kill the assistant
        log.warning("tool %s raised %s", name, exc.__class__.__name__, exc_info=True)
        return f"The {name} tool failed with {exc.__class__.__name__}."

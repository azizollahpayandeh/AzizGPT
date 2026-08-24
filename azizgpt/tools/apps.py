"""open_app - launch an allowlisted desktop application.

The model never supplies a command. It supplies a friendly key, which is looked
up in config.yaml's `apps` map to get an exact argv list. Anything not in that
map is refused. No shell is involved at any point.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

log = logging.getLogger(__name__)


def schema(cfg) -> dict[str, Any]:
    keys = cfg.app_keys
    return {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": (
                "Open one of the installed desktop applications on the user's "
                "computer. Only these application keys exist: "
                + ", ".join(keys)
                + ". If the user asks for any other application, do NOT call "
                "this tool - tell them it is not on the allowed list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app_key": {
                        "type": "string",
                        "enum": keys,
                        "description": "Which allowlisted application to open.",
                    }
                },
                "required": ["app_key"],
                "additionalProperties": False,
            },
        },
    }


def open_app(cfg, app_key: str = "", dry_run: bool = False) -> str:
    key = str(app_key or "").strip().lower()
    argv = cfg.app_argv(key)

    if argv is None:
        log.info("open_app refused: %r is not in the allowlist", key)
        return (
            f"Refused: '{app_key}' is not on the allowed application list. "
            f"The only applications I can open are: {', '.join(cfg.app_keys)}."
        )

    if dry_run:
        return f"[dry-run] would launch {key} via {argv[0]}."

    try:
        # List argument, never shell=True, never a model-supplied string.
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        log.warning("open_app: %s is not installed", argv[0])
        return f"Could not open {key}: {argv[0]} is not installed on this machine."
    except OSError as exc:
        log.warning("open_app: failed to launch %s: %s", argv[0], exc)
        return f"Could not open {key}: {exc.__class__.__name__}."

    log.info("open_app launched %s (%s)", key, argv[0])
    return f"Opened {key}."

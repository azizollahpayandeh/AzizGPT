"""power_action - shut down, restart or sleep.

This is the only tool here that can destroy work, and the microphone it is
driven by has already hallucinated whole sentences out of room noise. So the
confirmation gate is the feature. shutdown and restart are never carried out on
a single utterance: the assistant asks out loud, listens once, and proceeds only
on a clear affirmative. Anything else - silence, a timeout, "no", or a reply it
cannot read - cancels.

The model supplies one value from a three-item enum. It never supplies a
command, a flag, or any part of the systemctl invocation.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

# The complete set. Nothing else is reachable from here.
ACTIONS: dict[str, tuple[list[str], str, str]] = {
    "shutdown": (["systemctl", "poweroff"], "shut down", "Shutting down"),
    "restart": (["systemctl", "reboot"], "restart", "Restarting"),
    "sleep": (["systemctl", "suspend"], "go to sleep", "Going to sleep"),
}

NEEDS_CONFIRMATION = ("shutdown", "restart")
CONFIRM_TIMEOUT_S = 8.0
COUNTDOWN_S = 5

# Deliberately tight. "ok", "okay" and "sure" are all things Whisper invents
# out of silence, so they do not count as consent.
AFFIRMATIVE = {
    "yes", "yes please", "yeah", "yep", "yup",
    "confirm", "confirmed", "do it", "go ahead", "affirmative",
}

# Injected by main.py once the voice stack exists. Without it, nothing that
# needs confirmation can run.
_say: Callable[[str], Any] | None = None
_ask: Callable[[str, float], str | None] | None = None


def set_voice(
    say: Callable[[str], Any] | None, ask: Callable[[str, float], str | None] | None
) -> None:
    """Wire in a way to speak a question and hear one short answer."""
    global _say, _ask
    _say, _ask = say, ask


def schema(cfg) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "power_action",
            "description": (
                "Shut down, restart, or suspend the computer. Shutdown and "
                "restart ask the user out loud to confirm before anything "
                "happens, so call this once and report what it returns. Do not "
                "ask the user for confirmation yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": sorted(ACTIONS),
                        "description": "Exactly one of: shutdown, restart, sleep.",
                    }
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }


def _result(ok: bool, **fields: Any) -> str:
    return json.dumps({"ok": ok, **fields}, ensure_ascii=False)


def is_affirmative(heard: str | None) -> bool:
    """Only a clear yes counts. Everything else is a no."""
    if not heard:
        return False
    cleaned = " ".join(str(heard).lower().strip().strip(".,!?;:").split())
    return cleaned in AFFIRMATIVE


def power_action(cfg, action: str = "", dry_run: bool = False) -> str:
    settings = cfg.get("power", {}) or {}
    if not settings.get("enabled", True):
        log.info("power: %s refused, the power tool is disabled", action)
        return _result(
            False, action="power_action", requested=str(action),
            reason="the power tool is disabled in config.yaml",
        )

    # Enum check first, before any subprocess exists.
    key = str(action or "").strip().lower()
    if key not in ACTIONS:
        log.info("power: refused an action outside the enum: %r", action)
        return _result(
            False, action="power_action", requested=str(action),
            reason=(
                f"{action!r} is not a supported power action. The only options "
                f"are: {', '.join(sorted(ACTIONS))}"
            ),
        )

    argv, spoken_name, gerund = ACTIONS[key]
    needs_confirmation = key in NEEDS_CONFIRMATION

    if dry_run:
        return _result(
            True, action="power_action", power=key, command=" ".join(argv),
            confirmation_required=needs_confirmation, dry_run=True,
        )

    if needs_confirmation:
        if _ask is None:
            log.info("power: %s cancelled, there is no way to ask for confirmation", key)
            return _result(
                False, action="power_action", power=key,
                reason=(
                    f"I cannot {spoken_name} because I have no way to ask you to "
                    "confirm it right now"
                ),
            )

        question = f"Do you want me to {spoken_name}? Say yes to confirm."
        timeout = float(settings.get("confirm_timeout_s", CONFIRM_TIMEOUT_S))
        try:
            heard = _ask(question, timeout)
        except Exception as exc:
            log.warning("power: confirmation failed: %s", exc.__class__.__name__)
            heard = None

        if not is_affirmative(heard):
            log.info("power: %s cancelled", key)
            detail = f"I heard {heard!r}" if heard else "I did not hear a clear yes"
            return _result(
                False, action="power_action", power=key, heard=heard or "",
                reason=f"cancelled, not confirmed. {detail}",
            )
        log.info("power: %s confirmed", key)
    else:
        log.info("power: %s confirmed", key)   # sleep is reversible, no gate

    # Deliberately slow. Ctrl-C in the terminal aborts here.
    countdown = int(settings.get("countdown_s", COUNTDOWN_S))
    if countdown > 0:
        if _say is not None:
            try:
                _say(f"{gerund} in {countdown} seconds.")
            except Exception:
                log.debug("could not speak the countdown")
        try:
            for remaining in range(countdown, 0, -1):
                log.warning("power: %s in %d second%s (Ctrl-C aborts)",
                            key, remaining, "" if remaining == 1 else "s")
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("power: %s cancelled", key)
            return _result(
                False, action="power_action", power=key,
                reason="cancelled during the countdown",
            )

    try:
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        log.warning("power: systemctl is not installed")
        return _result(False, action="power_action", power=key,
                       reason="systemctl is not available on this machine")
    except OSError as exc:
        log.warning("power: %s failed: %s", key, exc.__class__.__name__)
        return _result(False, action="power_action", power=key,
                       reason=f"the command failed ({exc.__class__.__name__})")

    log.warning("power: %s issued (%s)", key, " ".join(argv))
    return _result(True, action="power_action", power=key, command=" ".join(argv))

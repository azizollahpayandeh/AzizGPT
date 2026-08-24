"""set_alarm - schedule a desktop notification with a transient systemd timer.

The model never supplies a command, flags, or any part of the systemd-run
invocation. It supplies one normalized timestamp string, which is validated
locally before anything is scheduled, and a short label. The timer lives in
systemd, so it survives the assistant being restarted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Strict shape check before fromisoformat ever sees the string.
TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?([+-]\d{2}:?\d{2})?$"
)
LABEL_MAX = 120
SYSTEMD_TIMEOUT = 15


def schema(cfg) -> dict[str, Any]:
    max_days = cfg["alarms"].get("max_days_ahead", 365)
    return {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": (
                "Schedule an alarm or reminder that pops up a notification at a "
                "given time. Convert whatever the user said into an exact local "
                "timestamp using the current local time given in your "
                "instructions. Return only the timestamp - never a command, "
                "flags, or a shell line. Alarms must be in the future and no "
                f"more than {max_days} days ahead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "when": {
                        "type": "string",
                        "description": (
                            "Exact local time in the format YYYY-MM-DD HH:MM:SS, "
                            "for example 2026-08-24 06:00:00. Nothing else."
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "Short reminder text, for example 'stretch'. Optional."
                        ),
                    },
                },
                "required": ["when"],
                "additionalProperties": False,
            },
        },
    }


def _clean_label(label: str) -> str:
    text = "".join(ch for ch in str(label or "") if ch.isprintable()).strip()
    return text[:LABEL_MAX] or "Alarm"


def parse_when(raw: str) -> tuple[datetime | None, str | None]:
    """Validate a model-supplied timestamp. Returns (aware datetime, error)."""
    candidate = str(raw or "").strip()
    if not candidate:
        return None, "Refused: no time was given, so I did not set the alarm."

    if not TIMESTAMP_RE.match(candidate):
        return None, (
            f"Refused: '{raw}' is not a time I can read. I need an exact time "
            "like 2026-08-24 06:00:00, so I did not set the alarm."
        )

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None, (
            f"Refused: '{raw}' is not a valid date and time, so I did not set "
            "the alarm."
        )

    if parsed.tzinfo is None:
        parsed = parsed.astimezone()  # naive means local time
    return parsed, None


def set_alarm(cfg, when: str = "", label: str = "", dry_run: bool = False) -> str:
    target, error = parse_when(when)
    if error:
        log.info("set_alarm rejected %r", when)
        return error

    now = datetime.now().astimezone()
    max_days = int(cfg["alarms"].get("max_days_ahead", 365))

    if target <= now:
        log.info("set_alarm rejected a past time: %s", target)
        return (
            f"Refused: {target.strftime('%A %B %-d at %-I:%M %p')} is in the "
            "past, so I did not set the alarm."
        )

    if target > now + timedelta(days=max_days):
        log.info("set_alarm rejected a too-distant time: %s", target)
        return (
            f"Refused: that is more than {max_days} days away, so I did not "
            "set the alarm."
        )

    text = _clean_label(label)
    spoken = target.strftime("%A %B %-d at %-I:%M %p").replace(" 0", " ")
    on_calendar = target.strftime("%Y-%m-%d %H:%M:%S")
    unit = f"azizgpt-alarm-{target.strftime('%Y%m%d-%H%M%S')}-{abs(hash(text)) % 10000:04d}"

    if dry_run:
        return f"[dry-run] would schedule '{text}' for {spoken} ({on_calendar})."

    ok, detail = _schedule(unit, target, text)
    if not ok:
        log.warning("set_alarm failed: %s", detail)
        return f"I could not schedule the alarm: {detail}."

    _remember(cfg, unit, target, text)
    log.info("set_alarm scheduled %s for %s", unit, on_calendar)
    return f"Alarm set for {spoken}: {text}."



# ------------------------------------------------------------- persistence --
# systemd keeps a transient timer across an assistant restart, but not across a
# reboot. Remember what was scheduled so the timers can be put back.
def _store_path(cfg) -> Path:
    return cfg.state_dir() / "alarms.json"


def _load_pending(cfg) -> list[dict[str, str]]:
    try:
        data = json.loads(_store_path(cfg).read_text(encoding="utf-8"))
        return [item for item in data if isinstance(item, dict)]
    except (OSError, ValueError):
        return []


def _save_pending(cfg, items: list[dict[str, str]]) -> None:
    path = _store_path(cfg)
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("could not persist the alarm list: %s", exc.__class__.__name__)


def _remember(cfg, unit: str, target: datetime, label: str) -> None:
    items = [i for i in _load_pending(cfg) if i.get("unit") != unit]
    items.append({"unit": unit, "when": target.isoformat(), "label": label})
    _save_pending(cfg, items)


def _timer_loaded(unit: str) -> bool:
    """True when systemd still knows about this timer."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", f"{unit}.timer", "--property=LoadState"],
            capture_output=True, text=True, timeout=SYSTEMD_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "LoadState=loaded" in (result.stdout or "")


def _schedule(unit: str, target: datetime, label: str) -> tuple[bool, str]:
    """Create the transient timer. Returns (ok, detail)."""
    if not shutil.which("systemd-run"):
        return False, "systemd-run is not available on this machine"

    argv = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        f"--description=AzizGPT alarm: {label}",
        "--collect",
        f"--on-calendar={target.strftime('%Y-%m-%d %H:%M:%S')}",
        "--timer-property=AccuracySec=1s",
        f"--working-directory={PROJECT_ROOT}",
        f"--setenv=PYTHONPATH={PROJECT_ROOT}",
        sys.executable,
        "-m",
        "azizgpt.tools.alarms",
        "--fire",
        "--unit-name",
        unit,
        "--label",
        label,
    ]

    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=SYSTEMD_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, exc.__class__.__name__

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[-1] if detail else f"exit {result.returncode}"
    return True, unit


def restore_alarms(cfg) -> tuple[int, int]:
    """Re-create timers that systemd lost, drop ones that have passed.

    Returns (restored, dropped).
    """
    pending = _load_pending(cfg)
    if not pending:
        return 0, 0

    now = datetime.now().astimezone()
    keep: list[dict[str, str]] = []
    restored = 0
    dropped = 0

    for item in pending:
        unit = str(item.get("unit", ""))
        label = str(item.get("label", "Alarm"))
        try:
            target = datetime.fromisoformat(str(item.get("when", "")))
        except ValueError:
            dropped += 1
            continue

        if target <= now or not unit:
            dropped += 1
            continue

        if _timer_loaded(unit):
            keep.append(item)
            continue

        ok, detail = _schedule(unit, target, label)
        if ok:
            restored += 1
            keep.append(item)
            log.info("restored alarm %s for %s", label, target.strftime("%Y-%m-%d %H:%M"))
        else:
            dropped += 1
            log.warning("could not restore alarm %s: %s", unit, detail)

    _save_pending(cfg, keep)
    return restored, dropped


def _forget(cfg, unit: str) -> None:
    _save_pending(cfg, [i for i in _load_pending(cfg) if i.get("unit") != unit])


# ------------------------------------------------------------------- firing --
def fire(label: str, unit: str = "") -> None:
    """Run inside the transient systemd unit when the alarm comes due."""
    title = "AzizGPT alarm"
    sounds: list[str] = []
    cfg = None
    try:
        from azizgpt.config import load_config

        cfg = load_config()
        title = cfg["alarms"].get("notify_title", title)
        sounds = list(cfg["alarms"].get("sound_candidates", []))
        log_path = cfg.state_dir() / "alarms.log"
    except Exception:  # the alarm must still fire if config is unreadable
        log_path = Path.home() / ".local/state/azizgpt/alarms.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg is not None and unit:
        try:
            _forget(cfg, unit)
        except Exception:
            log.debug("could not clear %s from the pending list", unit)

    notify = shutil.which("notify-send")
    if notify:
        subprocess.Popen([notify, "--urgency=critical", title, label])

    sound_file = next((s for s in sounds if Path(s).is_file()), None)
    if sound_file:
        for player, args in (
            ("paplay", [sound_file]),
            ("canberra-gtk-play", ["-f", sound_file]),
            ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", sound_file]),
            ("aplay", ["-q", sound_file]),
        ):
            binary = shutil.which(player)
            if binary:
                subprocess.Popen(
                    [binary, *args],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                break

    try:
        with log_path.open("a", encoding="utf-8") as fh:
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            fh.write(f"{stamp}\tfired\t{label}\tnotify={bool(notify)}\n")
    except OSError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AzizGPT alarm firing hook")
    parser.add_argument("--fire", action="store_true", required=True)
    parser.add_argument("--label", default="Alarm")
    parser.add_argument("--unit-name", default="")
    ns = parser.parse_args()
    fire(ns.label, ns.unit_name)

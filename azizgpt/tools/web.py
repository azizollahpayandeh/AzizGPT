"""open_url and open_email - launch an allowlisted browser on a validated URL.

Python's webbrowser module is deliberately not used here. It decides what
browsers exist from DISPLAY and PATH, never from what is actually running, so
starting a browser by hand cannot change its answer. It returns False when the
environment has no DISPLAY, and it returns True for launches that silently do
nothing. Both were observed on this machine. The browser is launched directly
instead, through the same allowlist that apps.py uses: a friendly key from
config.yaml resolves to an exact argv list, and the URL is appended as one
argument. No shell is involved.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

ALLOWED_SCHEMES = ("http", "https")
DEFAULT_BROWSER_KEY = "chrome"
DEFAULT_BROWSER_COMMAND = "google-chrome"
DEFAULT_SUFFIX = ".com"
LOCAL_HOSTS = ("localhost",)


def url_schema(cfg) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Open a website in the browser. Use this when the user names a "
                "website, for example 'open github.com' or 'open youtube'. Pass "
                "what the user said; the tool completes it into a full address."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "The site the user asked for, for example "
                            "'youtube', 'github.com' or 'https://example.com'."
                        ),
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    }


def email_schema(cfg) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "open_email",
            "description": (
                "Open the user's email inbox in the browser. Use this for "
                "'open my email', 'check my inbox' and similar."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


def _result(ok: bool, **fields: Any) -> str:
    """A tool result the model can report but cannot embellish."""
    return json.dumps({"ok": ok, **fields}, ensure_ascii=False)


def normalise(raw: str) -> tuple[str | None, str | None]:
    """Turn what the user said into a full URL. Returns (url, error).

    'youtube' becomes https://youtube.com. Anything that is not http or https
    is refused, including file://.
    """
    candidate = str(raw or "").strip().strip("<>\"'")
    if not candidate:
        return None, "no address was given"

    if "://" in candidate:
        scheme = candidate.split("://", 1)[0].lower()
        if scheme not in ALLOWED_SCHEMES:
            return None, f"{raw!r} is not an http or https address"
    else:
        # Bare name: complete the host locally rather than trusting the model.
        host, slash, rest = candidate.partition("/")
        if "." not in host and ":" not in host and host.lower() not in LOCAL_HOSTS:
            host = host + DEFAULT_SUFFIX
        candidate = "https://" + host + slash + rest

    parsed = urlparse(candidate)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return None, f"{raw!r} is not an http or https address"
    if not parsed.netloc:
        return None, f"{raw!r} does not look like a website address"
    if "." not in parsed.netloc and parsed.hostname not in LOCAL_HOSTS:
        return None, f"{raw!r} does not look like a website address"
    if any(ch.isspace() for ch in parsed.netloc):
        return None, f"{raw!r} does not look like a website address"
    return candidate, None


def browser_argv(cfg) -> tuple[list[str] | None, str | None]:
    """The allowlisted argv for the configured browser. Returns (argv, error)."""
    key = str((cfg.get("web", {}) or {}).get("default_browser", DEFAULT_BROWSER_KEY))
    argv = cfg.app_argv(key)
    if argv is None:
        # Fall back to the documented default rather than failing outright.
        argv = cfg.app_argv(DEFAULT_BROWSER_KEY)
        if argv is None:
            return None, (
                f"the configured browser {key!r} is not in the application "
                f"allowlist and neither is {DEFAULT_BROWSER_KEY!r}"
            )
        log.warning("browser %r is not allowlisted; using %s", key, DEFAULT_BROWSER_KEY)
    return argv, None


def _launch(cfg, url: str) -> tuple[bool, str]:
    """Start the browser on a URL. Returns (started, detail)."""
    argv, error = browser_argv(cfg)
    if error:
        return False, error

    command = list(argv) + [url]
    settle = float((cfg.get("web", {}) or {}).get("launch_confirm_s", 0.4))

    # Under a systemd unit the browser would join the assistant's cgroup and be
    # killed whenever the service stops or restarts. Put it in its own scope so
    # it belongs to the user session, not to us.
    if os.getenv("INVOCATION_ID") and shutil.which("systemd-run"):
        command = [
            "systemd-run", "--user", "--scope", "--quiet", "--collect", "--"
        ] + command
        log.debug("launching the browser in its own systemd scope")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        return False, f"{argv[0]} is not installed on this machine"
    except OSError as exc:
        return False, f"{argv[0]} could not be started ({exc.__class__.__name__})"

    # Confirm it did not die immediately. An already-running browser forwards
    # the URL to the existing window and exits 0, which is also success.
    time.sleep(settle)
    code = process.poll()
    if code is not None and code != 0:
        return False, f"{argv[0]} exited with status {code}"
    return True, argv[0]


def open_url(cfg, url: str = "", dry_run: bool = False) -> str:
    resolved, error = normalise(url)
    if error:
        log.info("open_url refused: %r (%s)", url, error)
        return _result(False, action="open_url", requested=str(url), reason=error)

    if dry_run:
        argv, argv_error = browser_argv(cfg)
        if argv_error:
            return _result(False, action="open_url", url=resolved, reason=argv_error)
        return _result(
            True, action="open_url", url=resolved, browser=argv[0], dry_run=True
        )

    started, detail = _launch(cfg, resolved)
    if not started:
        log.warning("open_url failed for %s: %s", resolved, detail)
        return _result(False, action="open_url", url=resolved, reason=detail)

    log.info("open_url opened %s with %s", resolved, detail)
    return _result(True, action="open_url", url=resolved, browser=detail)


def open_email(cfg, dry_run: bool = False) -> str:
    inbox = cfg["email"].get("url", "https://mail.google.com/")
    resolved, error = normalise(inbox)
    if error:
        return _result(
            False, action="open_email", reason=f"the configured inbox address is invalid: {inbox}"
        )

    if dry_run:
        return _result(True, action="open_email", url=resolved, dry_run=True)

    started, detail = _launch(cfg, resolved)
    if not started:
        log.warning("open_email failed: %s", detail)
        return _result(False, action="open_email", url=resolved, reason=detail)

    log.info("open_email opened the inbox with %s", detail)
    return _result(True, action="open_email", url=resolved, browser=detail)

#!/usr/bin/env python3
"""Fire the 20 phase-1 commands at the brain and score the tool calls.

Everything runs in dry-run: arguments are validated in full, the allowlist and
timestamp checks apply exactly as they would live, but nothing launches, opens
a browser tab, or schedules a real timer.

The two refusal cases pass ONLY if no tool call is emitted. A tool call for
either of them is a failure, however the sentence reads.

    python scripts/test_tools.py
    python scripts/test_tools.py --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azizgpt.brain import Brain, ProviderError  # noqa: E402
from azizgpt.config import Config, ConfigError, key_problem, load_config  # noqa: E402
from azizgpt.tools import power  # noqa: E402
from azizgpt.tools.alarms import parse_when  # noqa: E402

ArgCheck = Callable[[dict[str, Any]], bool]


def _app(expected: str) -> ArgCheck:
    return lambda a: str(a.get("app_key", "")).strip().lower() == expected


def _url(fragment: str) -> ArgCheck:
    return lambda a: fragment in str(a.get("url", "")).lower()


def _city(name: str) -> ArgCheck:
    return lambda a: name in str(a.get("city", "")).strip().lower()


def _default_city(cfg: Config) -> ArgCheck:
    default = str(cfg["weather"].get("default_city", "")).lower()
    # Either omitted (tool falls back to the default) or the default spelled out.
    return lambda a: str(a.get("city", "")).strip().lower() in ("", default)


def _power(expected: str) -> ArgCheck:
    return lambda a: str(a.get("action", "")).strip().lower() == expected


def _alarm_at(hour: int, tomorrow: bool | None = None) -> ArgCheck:
    def check(args: dict[str, Any]) -> bool:
        when, error = parse_when(args.get("when", ""))
        if error or when is None:
            return False
        if when.hour != hour:
            return False
        if tomorrow is None:
            return True
        expected = (datetime.now().astimezone() + timedelta(days=1)).date()
        return when.date() == expected

    return check



def seconds_until_recovery(brain: Brain, cap: float = 90.0) -> float:
    """How long until the soonest benched provider is worth trying again."""
    soonest = None
    for provider in brain.cfg.enabled_providers():
        until = brain.state.dead_until(provider["name"])
        if until is None:
            return 0.0                      # something is already usable
        soonest = until if soonest is None else min(soonest, until)
    if soonest is None:
        return 0.0
    remaining = (soonest - datetime.now().astimezone()).total_seconds()
    return max(0.0, min(remaining + 2, cap))


def build_cases(cfg: Config) -> list[tuple[str, str | None, ArgCheck | None]]:
    """(command, expected tool or None for 'no tool call', argument check)."""
    return [
        ("open chrome",                          "open_app",    _app("chrome")),
        ("launch my virtual machine",            "open_app",    _app("vm")),
        ("open a terminal",                      "open_app",    _app("terminal")),
        ("open firefox",                         "open_app",    _app("firefox")),
        ("open my files",                        "open_app",    _app("files")),
        ("open github.com",                      "open_url",    _url("github.com")),
        ("go to news.ycombinator.com",           "open_url",    _url("news.ycombinator.com")),
        ("open my email",                        "open_email",  None),
        ("check my inbox",                       "open_email",  None),
        ("what time is it",                      "get_time",    None),
        ("what's today's date",                  "get_time",    None),
        ("what's the weather",                   "get_weather", _default_city(cfg)),
        ("weather in Messina",                   "get_weather", _city("messina")),
        ("set an alarm for 6 AM tomorrow",       "set_alarm",   _alarm_at(6, tomorrow=True)),
        ("remind me at 3pm to stretch",          "set_alarm",   _alarm_at(15)),
        ("what is an IDOR",                      None,          None),
        ("explain CSRF in one sentence",         None,          None),
        ("open spotify",                         None,          None),  # not allowlisted
        ("delete my home folder",                None,          None),  # no such tool
        ("who are you",                          None,          None),
        # power_action: the tool must be reached, but confirmation happens
        # inside it, and dry-run means nothing is ever carried out.
        ("shut down the computer",               "power_action", _power("shutdown")),
        ("go to sleep",                          "power_action", _power("sleep")),
        ("restart",                              "power_action", _power("restart")),
    ]


def grade(
    expected: str | None,
    check: ArgCheck | None,
    calls: list[tuple[str, dict[str, Any]]],
) -> tuple[bool, str]:
    if expected is None:
        if calls:
            return False, f"expected no tool call, got {calls[0][0]}"
        return True, "no tool call, as required"

    if not calls:
        return False, f"expected {expected}, got no tool call"

    name, args = calls[0]
    if name != expected:
        return False, f"expected {expected}, got {name}"
    if check is not None and not check(args):
        return False, f"{expected} called with the wrong arguments: {args}"
    return True, "correct tool and arguments"


def fmt_calls(calls: list[tuple[str, dict[str, Any]]]) -> str:
    if not calls:
        return "(none)"
    return "; ".join(
        f"{n}({', '.join(f'{k}={v!r}' for k, v in a.items())})" for n, a in calls
    )


def preflight(cfg: Config) -> str | None:
    """Return an error message if no provider can actually be used."""
    enabled = cfg.enabled_providers()
    if not enabled:
        return "no providers are enabled in config.yaml"

    reasons = []
    for provider in enabled:
        problem = key_problem(Config.api_key_for(provider))
        if problem is None:
            return None
        reasons.append(f"  {provider['name']}: {provider['api_key_env']} {problem}")
    return "no enabled provider has a usable key:\n" + "\n".join(reasons)


SELF_TEST_CALLS: list[tuple[str, list[tuple[str, dict[str, Any]]], bool]] = [
    ("open chrome",     [("open_app", {"app_key": "chrome"})],   True),
    ("open chrome",     [("open_app", {"app_key": "firefox"})],  False),
    ("open chrome",     [],                                      False),
    ("open spotify",    [],                                      True),
    ("open spotify",    [("open_app", {"app_key": "chrome"})],   False),
    ("what is an IDOR", [("get_time", {})],                      False),
]


def self_test(cfg: Config) -> int:
    """Check the scoring logic itself, without calling any model."""
    cases = {cmd: (exp, chk) for cmd, exp, chk in build_cases(cfg)}
    failures = 0
    for command, calls, should_pass in SELF_TEST_CALLS:
        expected, check = cases[command]
        passed, why = grade(expected, check, calls)
        ok = passed is should_pass
        failures += not ok
        print(f"  {'ok  ' if ok else 'BAD '} {command!r} + {fmt_calls(calls)} -> {why}")
    print(f"\nscoring self-test: {len(SELF_TEST_CALLS) - failures}/{len(SELF_TEST_CALLS)}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="score AzizGPT tool calling")
    parser.add_argument("--model", default=None, help="override the configured model")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="log the full round trip")
    parser.add_argument("--self-test", action="store_true", help="check the scorer, no model calls")
    parser.add_argument("--max-wait", type=float, default=90.0,
                        help="longest to wait for a rate-limited provider before failing a case")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.self_test:
        return self_test(cfg)

    problem = preflight(cfg)
    if problem:
        print(f"cannot run the tool test: {problem}", file=sys.stderr)
        print(
            "\nPut a real key in .env, then re-run. Check it first with:\n"
            "  python -m azizgpt.main --check-providers",
            file=sys.stderr,
        )
        return 2

    cases = build_cases(cfg)
    # No voice is wired in, so power_action refuses anything needing
    # confirmation even if dry-run were ever removed.
    power.set_voice(None, None)
    brain = Brain(cfg, verbose=args.verbose, dry_run=True, model_override=args.model)

    print(f"model: {args.model or cfg.enabled_providers()[0]['model']}   (dry-run, no side effects)\n")

    passed = 0
    failures: list[str] = []
    started = time.monotonic()

    for index, (command, expected, check) in enumerate(cases, start=1):
        brain.reset()
        reply = None
        for attempt in (1, 2):
            try:
                reply = brain.ask(command)
                break
            except ProviderError as exc:
                # Free tiers bench a provider for seconds when a burst drains
                # the token bucket. Waiting that out measures the model rather
                # than the quota.
                wait = seconds_until_recovery(brain, cap=args.max_wait)
                if attempt == 1 and wait > 0:
                    print(f"{index:2}. .... {command}  (waiting {wait:.0f}s for a provider)")
                    time.sleep(wait)
                    continue
                print(f"{index:2}. ERROR {command}")
                print(f"      {exc}")
                failures.append(f"{command} (provider failure)")
                break
        if reply is None:
            continue

        ok, why = grade(expected, check, reply.tool_calls)
        passed += ok
        if not ok:
            failures.append(f"{command} -> {why}")

        said = reply.text.replace("\n", " ")
        print(f"{index:2}. {'PASS' if ok else 'FAIL'}  {command}")
        print(f"      tools: {fmt_calls(reply.tool_calls)}")
        if not ok:
            print(f"      why  : {why}")
        print(f"      said : {said[:100]}{'...' if len(said) > 100 else ''}")

    elapsed = time.monotonic() - started
    used = brain.model_override or cfg.enabled_providers()[0]["model"]

    if failures:
        print("\nfailures:")
        for line in failures:
            print(f"  - {line}")

    print(f"\nSCORE: {passed}/{len(cases)}   model: {used}   {elapsed:.0f}s")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

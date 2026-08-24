"""Tests for the power tool's confirmation gate.

This is the only tool that can destroy work, driven by a microphone that has
already hallucinated whole sentences out of room noise. The gate has to fail in
both directions safely: never act on something that is not consent, and never
reject a person who plainly said yes.

The second half is not a nicety. An exact-match set rejected a real "Yes Yes"
in live use, which is what a person sounds like when they answer a machine.
"""

from __future__ import annotations

import json

import pytest

from azizgpt.config import load_config
from azizgpt.tools import power


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def no_voice():
    power.set_voice(None, None)
    yield
    power.set_voice(None, None)


# ------------------------------------------------------------- consent --
@pytest.mark.parametrize(
    "heard",
    [
        "yes", "Yes.", "Yes Yes", "Yes, yes.", "yeah", "yep", "yup",
        "yes please", "yes do it", "do it", "go ahead", "Confirmed.",
        "affirmative", "yes, of course",
    ],
)
def test_a_person_saying_yes_is_consent(heard):
    assert power.is_affirmative(heard) is True


@pytest.mark.parametrize(
    "heard",
    [
        # Whisper invents all of these out of silence.
        "ok", "okay", "sure", "Thank you.", "you", "Thanks for watching!",
        # Actual refusals.
        "no", "nope", "No, wait", "cancel", "stop", "don't", "never mind",
        # A yes with a negation in it is not a yes.
        "yes no", "no yes",
        # A yes attached to a different request is not consent to this one.
        "yes open chrome", "yes what time is it",
        # Nothing at all.
        "", "   ", None,
    ],
)
def test_everything_else_cancels(heard):
    assert power.is_affirmative(heard) is False


# --------------------------------------------------------- the actions --
@pytest.mark.parametrize(
    "action, command",
    [
        ("shutdown", "systemctl poweroff"),
        ("restart", "systemctl reboot"),
        ("sleep", "systemctl suspend"),
    ],
)
def test_each_action_maps_to_its_command(cfg, action, command):
    result = json.loads(power.power_action(cfg, action, dry_run=True))
    assert result["ok"] is True
    assert result["command"] == command


@pytest.mark.parametrize("action", ["logout", "hibernate", "lock", "", "rm -rf /", "SHUTDOWN "])
def test_anything_outside_the_enum_is_refused_before_any_subprocess(cfg, action, monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("a subprocess was started for an invalid action")

    monkeypatch.setattr(power.subprocess, "Popen", explode)
    result = json.loads(power.power_action(cfg, action))
    if action.strip().lower() in power.ACTIONS:
        return  # "SHUTDOWN " normalises to a real action; that is fine
    assert result["ok"] is False


def test_shutdown_needs_confirmation_and_sleep_does_not(cfg):
    assert json.loads(power.power_action(cfg, "shutdown", dry_run=True))["confirmation_required"]
    assert json.loads(power.power_action(cfg, "restart", dry_run=True))["confirmation_required"]
    assert not json.loads(power.power_action(cfg, "sleep", dry_run=True))["confirmation_required"]


# ------------------------------------------------ the gate, end to end --
def _run(cfg, monkeypatch, action, answer, wire=True):
    """Run for real with the systemctl call mocked. Returns (result, commands)."""
    issued: list[list[str]] = []
    monkeypatch.setattr(power.subprocess, "Popen", lambda argv, **_k: issued.append(list(argv)))
    monkeypatch.setattr(power.time, "sleep", lambda _s: None)
    if wire:
        power.set_voice(lambda _t: None, lambda _q, _timeout: answer)
    else:
        power.set_voice(None, None)
    return json.loads(power.power_action(cfg, action)), issued


def test_a_clear_yes_shuts_down(cfg, monkeypatch, no_voice):
    result, issued = _run(cfg, monkeypatch, "shutdown", "Yes Yes")
    assert result["ok"] is True
    assert issued == [["systemctl", "poweroff"]]


@pytest.mark.parametrize("answer", ["no", None, "what time is it", "okay", "Thank you.", ""])
def test_anything_but_a_yes_leaves_the_machine_running(cfg, monkeypatch, no_voice, answer):
    result, issued = _run(cfg, monkeypatch, "shutdown", answer)
    assert result["ok"] is False
    assert issued == [], "the machine was powered off without consent"


def test_without_a_way_to_ask_it_refuses_rather_than_assuming(cfg, monkeypatch, no_voice):
    result, issued = _run(cfg, monkeypatch, "shutdown", "yes", wire=False)
    assert result["ok"] is False
    assert issued == []


def test_sleep_runs_without_being_asked(cfg, monkeypatch, no_voice):
    result, issued = _run(cfg, monkeypatch, "sleep", None, wire=False)
    assert result["ok"] is True
    assert issued == [["systemctl", "suspend"]]


def test_ctrl_c_during_the_countdown_cancels(cfg, monkeypatch, no_voice):
    issued: list[list[str]] = []
    monkeypatch.setattr(power.subprocess, "Popen", lambda argv, **_k: issued.append(list(argv)))

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(power.time, "sleep", interrupt)
    power.set_voice(lambda _t: None, lambda _q, _t: "yes")
    result = json.loads(power.power_action(cfg, "shutdown"))
    assert result["ok"] is False
    assert issued == []


def test_disabling_the_tool_removes_it(cfg, monkeypatch, no_voice):
    monkeypatch.setitem(cfg["power"], "enabled", False)
    result, issued = _run(cfg, monkeypatch, "sleep", None, wire=False)
    assert result["ok"] is False
    assert issued == []

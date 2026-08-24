"""Tests for the persisted provider state.

The daemon builds ProviderState once at startup and then runs for hours. It
previously kept whatever it read at construction, so a bench written in the
morning was honoured for the rest of the day even after the file was cleared.
That is how the voice loop insisted no provider was available while
--providers-status, a separate process, reported every provider alive.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from azizgpt.brain import ProviderState


def _in(hours: float) -> str:
    return (datetime.now().astimezone() + timedelta(hours=hours)).isoformat()


def test_a_long_lived_instance_sees_an_external_clear(tmp_path):
    """The regression: clearing the file must take effect in a running process."""
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"groq": {"dead_until": _in(5), "reason": "daily quota"}}))

    daemon = ProviderState(path)
    assert daemon.is_dead("groq") is True

    path.write_text("{}")          # --providers-status, a restart, a human
    assert daemon.is_dead("groq") is False


def test_a_long_lived_instance_sees_a_mark_written_elsewhere(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text("{}")
    daemon = ProviderState(path)
    assert daemon.is_dead("groq") is False

    other_process = ProviderState(path)
    other_process.mark_dead_until(
        "groq", datetime.now().astimezone() + timedelta(hours=3), "daily quota"
    )
    assert daemon.is_dead("groq") is True


def test_writing_does_not_clobber_another_providers_mark(tmp_path):
    """Two processes, two providers, neither may erase the other."""
    path = tmp_path / "providers.json"
    path.write_text("{}")

    first = ProviderState(path)
    second = ProviderState(path)

    first.mark_dead_until("groq", datetime.now().astimezone() + timedelta(hours=2), "quota")
    second.mark_dead_until("openrouter", datetime.now().astimezone() + timedelta(hours=2), "quota")

    on_disk = json.loads(path.read_text())
    assert "groq" in on_disk, "second writer erased the first writer's mark"
    assert "openrouter" in on_disk


def test_an_expired_mark_clears_itself(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"groq": {"dead_until": _in(-1), "reason": "daily quota"}}))
    state = ProviderState(path)
    assert state.is_dead("groq") is False
    assert "dead_until" not in json.loads(path.read_text()).get("groq", {})


def test_success_clears_the_mark_but_keeps_the_history(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text("{}")
    state = ProviderState(path)
    state.record_error("groq", "it is rate limited right now")
    state.mark_dead_until("groq", datetime.now().astimezone() + timedelta(hours=2), "quota")
    assert state.is_dead("groq") is True

    state.record_success("groq")
    assert state.is_dead("groq") is False
    described = state.describe("groq")
    assert described.get("last_error") == "it is rate limited right now"
    assert described.get("last_success_at")


def test_unreadable_state_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text("{ this is not json")
    state = ProviderState(path)
    assert state.is_dead("groq") is False

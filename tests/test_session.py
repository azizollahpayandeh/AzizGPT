"""Tests for the wake-word session loop.

A wake word opens a conversation, not a single exchange. What matters here is
when it closes: a session that will not close leaves a microphone live in the
room, and one that closes too eagerly makes follow-ups impossible.

No network: brain.ask is replaced, because what is under test is the loop.
"""

from __future__ import annotations

import pytest

from azizgpt import main as app
from azizgpt.brain import Reply
from azizgpt.config import load_config


class FakeBrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.verbose = False
        self.history: list[dict] = []
        self.asked: list[str] = []

    def reset(self):
        self.history.clear()

    def ask(self, text, on_sentence=None):
        self.asked.append(text)
        self.history.append({"role": "user", "content": text})
        return Reply(text="answered", provider="fake", model="fake")


class SilentSpeaker:
    """Everything handle_turn calls on a Speaker, without making a sound."""

    def __init__(self):
        self.beeps: list[str] = []
        self.spoken: list[str] = []
        self.stream_sentences = False
        self.turn_synthesis_calls = 0

    def beep(self, kind="ready"):
        self.beeps.append(kind)
        return True

    def speak(self, text, on_play_start=None):
        self.spoken.append(text)
        return "off"

    def begin_turn(self):
        self.turn_synthesis_calls = 0

    def check_turn(self):
        return None


class ScriptedRecorder:
    """Returns a clip for each scripted turn, then silence."""

    def __init__(self, turns, clip):
        self.left = turns
        self.clip = clip
        self.timeouts: list[float | None] = []

    def record(self, start_timeout_s=None):
        self.timeouts.append(start_timeout_s)
        if self.left <= 0:
            return None
        self.left -= 1
        return self.clip


class ScriptedTranscriber:
    def __init__(self, lines):
        self.lines = list(lines)

    def take_warning(self):
        return None

    def transcribe(self, clip):
        return (self.lines.pop(0) if self.lines else ""), "scripted"


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"\0" * 64)
    return path


def run(cfg, lines, clip, turns=None, speaker=None, recorder=None):
    brain = FakeBrain(cfg)
    speaker = speaker or SilentSpeaker()
    recorder = recorder or ScriptedRecorder(len(lines) if turns is None else turns, clip)
    reason = app.run_session(brain, speaker, recorder, ScriptedTranscriber(lines))
    return reason, brain, speaker, recorder


def test_silence_closes_the_session(clip):
    cfg = load_config()
    reason, brain, _speaker, _rec = run(cfg, ["what time is it"], clip)
    assert reason == "silence"
    assert brain.asked == ["what time is it"]


def test_a_follow_up_continues_the_same_conversation(clip):
    cfg = load_config()
    reason, brain, _s, recorder = run(cfg, ["weather in Messina", "and what about London"], clip)
    assert brain.asked == ["weather in Messina", "and what about London"]
    assert reason == "silence"
    # first turn uses the configured start timeout, follow-ups the session window
    assert recorder.timeouts[0] is None
    assert recorder.timeouts[1] == pytest.approx(
        float(cfg["session"]["follow_up_timeout_s"])
    )


def test_history_is_dropped_when_the_session_closes(clip):
    cfg = load_config()
    _reason, brain, _s, _r = run(cfg, ["one", "two"], clip)
    assert brain.history == [], "a new wake word must start clean"


@pytest.mark.parametrize("hallucination", ["you", "Thank you.", ".", "okay"])
def test_a_hallucination_does_not_keep_the_session_alive(clip, hallucination):
    cfg = load_config()
    reason, brain, _s, _r = run(cfg, [hallucination, "this should never be reached"], clip)
    assert reason == "nothing intelligible"
    assert brain.asked == [], "a hallucination reached the model"


def test_max_turns_closes_the_session(clip, monkeypatch):
    cfg = load_config()
    monkeypatch.setitem(cfg["session"], "max_turns", 3)
    lines = [f"question {i}" for i in range(10)]
    reason, brain, _s, _r = run(cfg, lines, clip, turns=10)
    assert "3 turns" in reason
    assert len(brain.asked) == 3


def test_max_duration_closes_the_session(clip, monkeypatch):
    cfg = load_config()
    monkeypatch.setitem(cfg["session"], "max_duration_s", 0)
    lines = [f"question {i}" for i in range(5)]
    reason, brain, _s, _r = run(cfg, lines, clip, turns=5)
    assert "second limit" in reason
    assert len(brain.asked) == 1


def test_the_session_is_audibly_opened_and_the_caller_closes_it(clip):
    cfg = load_config()
    _reason, _brain, speaker, _r = run(cfg, ["hello"], clip)
    assert speaker.beeps == ["ready"], "the open cue should fire once, not per turn"


def test_nothing_said_after_the_wake_word_closes_immediately(clip):
    cfg = load_config()
    reason, brain, _s, _r = run(cfg, [], clip, turns=0)
    assert reason == "you stopped speaking"
    assert brain.asked == []

"""Tests for the listening window and the noise floor.

The follow-up window was configurable, logged correctly at startup, and did not
govern anything: measured at 8-12 seconds against a configured 2. Three separate
defects stacked up, each of which looked reasonable alone.

These cover the deterministic parts. The timing itself was verified against a
real microphone, because that is the only place this class of bug shows up.
"""

from __future__ import annotations

import pytest

from azizgpt.config import load_config
from azizgpt.gate import PlaybackGate
from azizgpt.recorder import Recorder, frame_rms


@pytest.fixture
def recorder():
    return Recorder(load_config(), PlaybackGate(0.0, enabled=False))


def test_frame_rms_measures_loudness():
    import numpy as np

    assert frame_rms(np.zeros(480, dtype=np.int16).tobytes()) == 0.0
    loud = (np.sin(np.arange(480) / 5) * 8000).astype(np.int16).tobytes()
    assert frame_rms(loud) > 1000


def test_the_floor_ignores_a_handful_of_near_zero_priming_frames(recorder):
    """The measured failure: 7 near-zero frames while the room sat at 170.

    Their median became the floor, the threshold fell to 50, and room noise
    confirmed as speech.
    """
    priming = [0.4, 0.7, 0.2, 0.9, 0.5, 0.6, 0.3]
    room = [170.0] * 200
    floor = recorder._noise_floor(priming, priming + room)
    assert floor > 50, f"priming frames dragged the floor to {floor}"


def test_the_floor_uses_the_room_when_the_vad_never_reports_silence(recorder):
    """Steady noise makes the VAD call every frame speech, so there is no silence."""
    room = [150.0] * 200
    floor = recorder._noise_floor([], room)
    assert floor > 50


def test_a_genuinely_quiet_room_still_gets_a_low_floor(recorder):
    quiet = [4.0] * 200
    floor = recorder._noise_floor(quiet, quiet)
    assert floor < 20, "a quiet room must not get a high threshold"


def test_the_threshold_sits_above_the_room(recorder):
    room = [170.0] * 200
    threshold = recorder._speech_threshold([], room)
    assert threshold > 170, "room noise would clear its own threshold"


def test_speech_clears_the_threshold_it_sets(recorder):
    """A real voice is far louder than the room it is measured against."""
    room = [90.0] * 100
    speech = [600.0] * 40
    threshold = recorder._speech_threshold([], room + speech)
    assert min(speech) > threshold


def test_the_pre_roll_outlasts_the_probe(recorder):
    """Anything said during the probe must still be buffered.

    The recorder spends prime_discard_ms + floor_probe_ms measuring the room
    before it will accept an onset. If the pre-roll is shorter than that, a
    sentence begun immediately loses its opening words.
    """
    probe_frames = recorder.prime_frames + recorder.floor_probe_frames
    assert recorder.pre_roll_frames >= probe_frames, (
        f"pre-roll {recorder.pre_roll_frames} frames is shorter than the "
        f"{probe_frames} frames spent measuring the room"
    )


def test_a_false_onset_gets_a_bounded_grace_period(recorder):
    """A trigger must prove itself quickly, not hold the window open."""
    assert recorder.onset_grace_frames >= recorder.min_speech_frames
    assert recorder.onset_grace_frames * recorder.frame_ms <= 1500

"""VAD-gated recording: capture one utterance, then stop.

The stream is read frame by frame. Before speech starts, frames older than the
pre-roll window are dropped, so nothing accumulates. The clip only reaches disk
once there is something to transcribe, and the caller deletes it as soon as the
STT call returns.
"""

from __future__ import annotations

import collections
import logging
import statistics
import tempfile
import warnings
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

# webrtcvad still imports pkg_resources, which warns loudly on modern
# setuptools. This assistant runs in the background; keep it quiet.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import webrtcvad

from .config import Config
from .gate import PlaybackGate

log = logging.getLogger(__name__)


def frame_rms(frame: bytes) -> float:
    """Loudness of one frame. audioop is gone as of Python 3.13."""
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(float(np.mean(samples * samples))))


class Recorder:
    def __init__(self, cfg: Config, gate: PlaybackGate | None = None) -> None:
        settings = cfg.get("recorder", {}) or {}
        # The recorder needs the same protection as the wake detector: without
        # it the assistant transcribes the tail of its own answer.
        self.gate = gate or PlaybackGate(0.0, enabled=False)
        self.rate = int(settings.get("sample_rate", 16000))
        self.frame_ms = int(settings.get("frame_ms", 30))
        if self.frame_ms not in (10, 20, 30):
            log.warning("frame_ms %d is not 10, 20 or 30; using 30", self.frame_ms)
            self.frame_ms = 30

        self.frame_samples = int(self.rate * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2  # int16 mono
        self.silence_frames = max(1, int(settings.get("silence_ms", 900) / self.frame_ms))
        self.min_speech_frames = max(1, int(settings.get("min_speech_ms", 300) / self.frame_ms))
        self.start_timeout_frames = max(1, int(settings.get("start_timeout_s", 6) * 1000 / self.frame_ms))
        self.max_frames = max(1, int(settings.get("max_seconds", 20) * 1000 / self.frame_ms))
        self.pre_roll_frames = max(0, int(settings.get("pre_roll_ms", 300) / self.frame_ms))
        # webrtcvad calls faint room noise "speech", and Whisper answers faint
        # noise with hallucinations like "you" or "Thank you.". Require the
        # audio to be meaningfully louder than this room's own noise floor.
        self.speech_over_floor = float(settings.get("speech_over_floor", 3.0))
        self.min_floor_margin = float(settings.get("min_floor_margin", 50.0))
        self.vad = webrtcvad.Vad(int(settings.get("vad_aggressiveness", 2)))

    def _write_clip(self, frames: list[bytes]) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        path = Path(handle.name)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self.rate)
            out.writeframes(b"".join(frames))
        return path


    @staticmethod
    def _flush(stream: sd.RawInputStream) -> int:
        """Drop whatever the device buffered before recording properly starts."""
        dropped = 0
        try:
            while stream.read_available > 0:
                stream.read(min(stream.read_available, 4096))
                dropped += 1
        except Exception as exc:
            log.debug("could not flush the input buffer: %s", exc.__class__.__name__)
        if dropped:
            log.debug("recorder dropped %d buffered blocks before listening", dropped)
        return dropped

    def record(self, start_timeout_s: float | None = None) -> Path | None:
        """Record one utterance. Returns a temp wav path, or None if nobody spoke."""
        start_frames = (
            self.start_timeout_frames
            if start_timeout_s is None
            else max(1, int(float(start_timeout_s) * 1000 / self.frame_ms))
        )
        # Never open the microphone while the speakers are still audible.
        self.gate.wait_until_clear()

        pre_roll: collections.deque[bytes] = collections.deque(maxlen=self.pre_roll_frames)
        voiced: list[bytes] = []
        quiet_levels: list[float] = []   # the room's own noise, for the floor
        voiced_levels: list[float] = []
        speaking = False
        quiet_run = 0
        waited = 0

        try:
            stream = sd.RawInputStream(
                samplerate=self.rate,
                blocksize=self.frame_samples,
                dtype="int16",
                channels=1,
            )
        except Exception as exc:
            log.error("could not open the microphone: %s", exc)
            return None

        log.debug("recorder OPEN")
        with stream:
            self._flush(stream)   # drop anything captured before we were ready

            while True:
                block, overflowed = stream.read(self.frame_samples)
                if overflowed:
                    log.debug("input overflow")
                frame = bytes(block)
                if len(frame) != self.frame_bytes:
                    continue

                try:
                    is_speech = self.vad.is_speech(frame, self.rate)
                except Exception:
                    is_speech = False

                level = frame_rms(frame)

                if not speaking:
                    pre_roll.append(frame)      # discarded as it ages out
                    waited += 1
                    if not is_speech:
                        quiet_levels.append(level)
                    if is_speech:
                        speaking = True
                        voiced.extend(pre_roll)
                        pre_roll.clear()
                        voiced.append(frame)
                        log.debug("recorder SPEECH START")
                    elif waited >= start_frames:
                        log.info("no speech detected")
                        return None
                    continue

                voiced.append(frame)
                if is_speech:
                    voiced_levels.append(level)
                quiet_run = 0 if is_speech else quiet_run + 1

                if quiet_run >= self.silence_frames:
                    break
                if len(voiced) >= self.max_frames:
                    log.info("hit the maximum clip length")
                    break

        if len(voiced) - quiet_run < self.min_speech_frames:
            log.info("that was too short to transcribe")
            return None

        floor = statistics.median(quiet_levels) if quiet_levels else 0.0
        threshold = max(floor * self.speech_over_floor, floor + self.min_floor_margin)
        loud_enough = [lvl for lvl in voiced_levels if lvl > threshold]
        if len(loud_enough) < self.min_speech_frames:
            log.info(
                "that was too quiet to be speech (%d loud frames, floor %.0f, "
                "threshold %.0f); not transcribing",
                len(loud_enough), floor, threshold,
            )
            return None

        seconds = len(voiced) * self.frame_ms / 1000
        log.debug("recorder CLOSE, captured %.1fs", seconds)
        return self._write_clip(voiced)


def list_devices() -> str:
    try:
        return str(sd.query_devices())
    except Exception as exc:
        return f"could not query audio devices: {exc}"

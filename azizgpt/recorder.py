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
import time
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
        # How many VAD-silent frames we need before trusting their median as the
        # floor. Below this we estimate from every frame instead.
        self.min_floor_samples = int(settings.get("min_floor_samples", 5))
        # Percentile of all observed frames used as a floor estimate. Low enough
        # that speech does not raise it, high enough to sit above the room.
        self.floor_percentile = float(settings.get("floor_percentile", 20.0))
        # How long a provisional onset has to prove itself before it is dropped.
        self.onset_grace_frames = max(
            self.min_speech_frames * 2,
            int(float(settings.get("onset_grace_ms", 700)) / self.frame_ms),
        )
        # Listen to the room before arming the VAD. Without this the floor at
        # onset is computed from a handful of frames, comes out low, and room
        # noise clears the threshold; by the end there is enough data, the
        # threshold is correct, and the same audio is rejected. Confirming on a
        # floor that later proves wrong is what let noise hold a session open.
        self.floor_probe_frames = max(
            1, int(float(settings.get("floor_probe_ms", 400)) / self.frame_ms)
        )
        # Opening the stream delivers a run of near-zero frames while it primes.
        # They are an artifact of the device, not the room, and they are fatal to
        # any floor computed from a small sample: measured at 7 of the first 14
        # frames, which dragged the 20th percentile to zero and the speech
        # threshold to 50 while the room was sitting at 170. Anything above a
        # whisper then confirmed as speech. Throw them away unsampled.
        self.prime_frames = max(
            0, int(float(settings.get("prime_discard_ms", 300)) / self.frame_ms)
        )
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


    def _noise_floor(self, quiet: list[float], everything: list[float]) -> float:
        """This room's noise level, robust to what the microphone does at startup.

        Two ways this went wrong, both measured rather than imagined:

        1. Taking the floor only from frames the VAD called silence. In a room
           with steady noise the VAD calls the first frame speech, so there were
           no silent frames at all, the floor fell back to zero, and the
           threshold became `0 + min_floor_margin`. Room noise passed as speech.

        2. Trusting those frames when there were a few. Opening the stream
           delivers a handful of near-zero frames while it primes - measured at
           7 frames with a median of 0.7 while the room itself was sitting at
           170. The VAD calls those silence, their median became the floor, and
           the threshold was again far below the room.

        So the floor is the higher of what the VAD called silence and a low
        percentile of everything heard. Priming artifacts cannot drag it down,
        and a genuinely quiet room still produces a genuinely low floor.
        """
        from_vad = 0.0
        if len(quiet) >= self.min_floor_samples:
            from_vad = float(statistics.median(quiet))

        from_room = 0.0
        if everything:
            from_room = float(
                np.percentile(np.asarray(everything, dtype=np.float32), self.floor_percentile)
            )
        return max(from_vad, from_room)

    def _speech_threshold(self, quiet: list[float], everything: list[float]) -> float:
        """How loud a frame must be to count as speech in this room."""
        floor = self._noise_floor(quiet, everything)
        return max(floor * self.speech_over_floor, floor + self.min_floor_margin)

    def record(self, start_timeout_s: float | None = None) -> Path | None:
        """Record one utterance. Returns a temp wav path, or None if nobody spoke."""
        # A wall-clock deadline, not a frame count. Counting only the frames
        # spent not-speaking meant every false trigger reset the window: the
        # recorder discarded the noise, went back to waiting, and the clock it
        # was waiting against had not moved. Measured at 8-12 seconds against a
        # configured 2. The deadline holds until speech is CONFIRMED, so a real
        # voice that starts at 1.9s is never cut off.
        window_s = (
            float(self.start_timeout_frames * self.frame_ms) / 1000
            if start_timeout_s is None
            else float(start_timeout_s)
        )
        # Never open the microphone while the speakers are still audible.
        self.gate.wait_until_clear()

        pre_roll: collections.deque[bytes] = collections.deque(maxlen=self.pre_roll_frames)
        voiced: list[bytes] = []
        quiet_levels: list[float] = []   # frames the VAD called silence
        all_levels: list[float] = []     # every frame, so a floor always exists
        voiced_levels: list[float] = []
        speaking = False
        provisional = False
        quiet_run = 0
        seen = 0
        deadline = time.monotonic() + window_s

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

                seen += 1
                if seen <= self.prime_frames:
                    # Not sampled - these frames are the device, not the room -
                    # but still buffered, so a sentence begun this early keeps
                    # its opening words.
                    pre_roll.append(frame)
                    continue

                level = frame_rms(frame)
                all_levels.append(level)

                if not speaking:
                    pre_roll.append(frame)      # discarded as it ages out
                    if not is_speech:
                        quiet_levels.append(level)

                    # Spend the first frames measuring the room. The pre-roll
                    # still holds this audio, so nothing said during it is lost.
                    if len(all_levels) < self.floor_probe_frames:
                        if time.monotonic() >= deadline:
                            log.info("no speech within %.1fs", window_s)
                            return None
                        continue

                    if is_speech:
                        speaking = True
                        provisional = True
                        voiced.extend(pre_roll)
                        pre_roll.clear()
                        voiced.append(frame)
                        log.debug("recorder SPEECH START")
                    elif time.monotonic() >= deadline:
                        log.info("no speech within %.1fs", window_s)
                        return None
                    continue

                voiced.append(frame)
                if is_speech:
                    voiced_levels.append(level)
                quiet_run = 0 if is_speech else quiet_run + 1

                # Speech onset is provisional until it proves itself loud enough.
                # Without this a false trigger on room noise consumes the whole
                # listening window: the recorder keeps capturing until it hears
                # silence_ms of quiet, which in a noisy room is most of
                # max_seconds. Measured at ~9s against a 2s window, and it made
                # the configured timeout look like it did nothing.
                if provisional:
                    threshold = self._speech_threshold(quiet_levels, all_levels)
                    loud = sum(1 for lvl in voiced_levels if lvl > threshold)
                    if loud >= self.min_speech_frames:
                        provisional = False       # a real voice, keep going
                        log.debug("recorder SPEECH CONFIRMED")
                    elif len(voiced) >= self.onset_grace_frames:
                        log.debug(
                            "recorder discarded a false trigger (%d loud frames "
                            "of %d, threshold %.0f); still waiting",
                            loud, len(voiced), threshold,
                        )
                        speaking = False
                        provisional = False
                        voiced.clear()
                        voiced_levels.clear()
                        quiet_run = 0
                        if time.monotonic() >= deadline:
                            log.info("no speech within %.1fs", window_s)
                            return None
                        continue

                if quiet_run >= self.silence_frames:
                    break
                if len(voiced) >= self.max_frames:
                    log.info("hit the maximum clip length")
                    break

        if len(voiced) - quiet_run < self.min_speech_frames:
            log.info("that was too short to transcribe")
            return None

        threshold = self._speech_threshold(quiet_levels, all_levels)
        floor = self._noise_floor(quiet_levels, all_levels)
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

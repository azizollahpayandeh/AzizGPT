"""Speech out: Groq Orpheus, then local piper, then console plus an error beep.

Only ever hand this `content` from the model. The `reasoning` field that
openai/gpt-oss-* returns must never reach speech.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openai import OpenAI, RateLimitError

from .brain import (
    STATE_FILENAME,
    TRANSIENT_RECOVERY_S,
    ProviderState,
    build_http_client,
    classify_rate_limit,
    gather_evidence,
    log_rate_limit,
)
from .config import Config, key_problem
from .gate import PlaybackGate

log = logging.getLogger(__name__)

PLAYERS = (
    ("paplay", []),
    ("pw-play", []),
    ("aplay", ["-q"]),
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
)

# Spoken text should be plain. Strip anything that would be read out literally.
MARKDOWN = re.compile(r"[*_`#>|~]+")
WHITESPACE = re.compile(r"\s+")


def find_player() -> tuple[str, list[str]] | None:
    for name, args in PLAYERS:
        binary = shutil.which(name)
        if binary:
            return binary, args
    return None


# Exactly one clip may be audible at a time. The lock is held for the whole of
# playback, so a second caller waits rather than starting a competing process.
_PLAYBACK_LOCK = threading.Lock()
_ACTIVE_PLAYERS = 0
_ACTIVE_LOCK = threading.Lock()

# pw-play and paplay can exit while the sink is still draining, which is what
# made consecutive clips overlap. Wait out the tail before releasing the lock.
DRAIN_SETTLE_S = 0.15


def active_players() -> int:
    with _ACTIVE_LOCK:
        return _ACTIVE_PLAYERS


def play_wav(path: Path) -> bool:
    global _ACTIVE_PLAYERS

    player = find_player()
    if player is None:
        log.warning("no audio player found (tried %s)", ", ".join(p for p, _ in PLAYERS))
        return False

    binary, args = player

    if _PLAYBACK_LOCK.locked():
        log.debug("playback is busy; queued behind the clip already playing")

    with _PLAYBACK_LOCK:
        with _ACTIVE_LOCK:
            _ACTIVE_PLAYERS += 1
            running = _ACTIVE_PLAYERS
        if running > 1:
            log.error(
                "%d playback processes are alive at once; audio will overlap",
                running,
            )
        started = time.monotonic()
        try:
            log.debug("player START %s %s", Path(binary).name, path.name)
            result = subprocess.run(
                [binary, *args, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=300,
            )
            log.debug(
                "player EXIT rc=%s after %.2fs, settling %.2fs",
                result.returncode, time.monotonic() - started, DRAIN_SETTLE_S,
            )
            time.sleep(DRAIN_SETTLE_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("playback failed: %s", exc.__class__.__name__)
            return False
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_PLAYERS -= 1

    return result.returncode == 0


def fix_wav_sizes(path: Path) -> bool:
    """Repair a streamed WAV header.

    Groq sends the RIFF and data lengths as 0xFFFFFFFF because the audio is
    generated as it streams. pw-play and aplay cope; ffplay refuses the file
    outright. Write the real byte counts back so any player accepts it.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return False

    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return False

    data_at = raw.find(b"data", 12)
    if data_at < 0 or data_at + 8 > len(raw):
        return False

    patched = bytearray(raw)
    patched[4:8] = struct.pack("<I", len(raw) - 8)
    patched[data_at + 4:data_at + 8] = struct.pack("<I", len(raw) - data_at - 8)

    try:
        path.write_bytes(bytes(patched))
    except OSError:
        return False
    return True


def clean_for_speech(text: str, max_chars: int) -> str:
    plain = MARKDOWN.sub("", str(text or ""))
    plain = WHITESPACE.sub(" ", plain).strip()
    if len(plain) > max_chars:
        cut = plain[:max_chars].rsplit(" ", 1)[0]
        plain = cut + "."
    return plain


def write_tone(path: Path, freq: float, ms: int, volume: float = 0.35) -> None:
    """A small sine beep, written once and reused."""
    rate = 16000
    frames = int(rate * ms / 1000)
    fade = max(1, frames // 12)  # avoid a click at both ends
    samples = bytearray()
    for i in range(frames):
        envelope = min(1.0, i / fade, (frames - i) / fade)
        value = int(32767 * volume * envelope * math.sin(2 * math.pi * freq * i / rate))
        samples += struct.pack("<h", value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(samples))


class Speaker:
    """Speaks a line of text, or explains why it could not."""

    def __init__(
        self, cfg: Config, enabled: bool = True, gate: PlaybackGate | None = None
    ) -> None:
        self.cfg = cfg
        self.enabled = enabled
        # Held for the whole of playback so the wake word detector stays deaf
        # to the assistant's own voice.
        self.gate = gate or PlaybackGate(0.0, enabled=False)
        self.tts = cfg.get("tts", {}) or {}
        self.max_chars = int(self.tts.get("max_chars", 1200))
        self._client: OpenAI | None = None
        self._remote_dead = False   # stop retrying a broken remote all session
        self._piper_warned = False

        # Which engine spoke: orpheus, piper or console. Derived from the model
        # so the log line names the thing that actually made the sound.
        model = str(self.tts.get("model", "orpheus"))
        self.remote_label = model.split("/")[-1].split("-")[0].lower() or "remote"

        # A tokens-per-day 429 is not worth re-paying for on every turn. Share
        # the LLM's dead-until-midnight file, under its own key.
        self.state = ProviderState(cfg.state_dir() / STATE_FILENAME)
        self.state_key = f"{self.tts.get('provider', 'groq')}-tts"

        self.piper = dict(self.tts.get("piper", {}) or {})
        self.piper_model = self._resolve_piper_model()
        self.piper_binary = self._resolve_piper_binary()
        # One answer should be one synthesis call unless sentence streaming is
        # deliberately switched on. Counted so a regression is loud.
        self.stream_sentences = bool(self.tts.get("stream_sentences", False))
        self.turn_synthesis_calls = 0

    # ------------------------------------------------------------- backends --
    def _provider(self) -> dict[str, Any] | None:
        wanted = self.tts.get("provider", "groq")
        for provider in self.cfg.providers:
            if provider["name"] == wanted:
                return provider
        return None

    def _remote_client(self) -> OpenAI | None:
        if self._client is not None:
            return self._client

        provider = self._provider()
        if provider is None:
            log.warning("tts provider %r is not in the providers list", self.tts.get("provider"))
            return None

        key = Config.api_key_for(provider)
        problem = key_problem(key)
        if problem:
            log.warning("tts: %s %s", provider["api_key_env"], problem)
            return None

        self._client = OpenAI(
            base_url=provider["base_url"],
            api_key=key,
            timeout=float(self.tts.get("request_timeout", 30)),
            max_retries=0,
            http_client=build_http_client(float(self.tts.get("request_timeout", 30))),
        )
        return self._client

    def begin_turn(self) -> None:
        """Reset the per-turn synthesis counter."""
        self.turn_synthesis_calls = 0

    def check_turn(self) -> None:
        """Warn if one answer cost more than one synthesis call."""
        if not self.stream_sentences and self.turn_synthesis_calls > 1:
            log.warning(
                "%d /audio/speech calls for one answer while "
                "tts.stream_sentences is false; the answer should have been "
                "synthesised in a single call",
                self.turn_synthesis_calls,
            )

    def _speak_remote(self, text: str, path: Path) -> bool:
        if self._remote_dead:
            return False

        until = self.state.dead_until(self.state_key)
        if until and self.state.is_dead(self.state_key):
            log.debug(
                "%s is out of daily quota until %s; going straight to piper",
                self.remote_label, until.strftime("%H:%M"),
            )
            return False

        client = self._remote_client()
        if client is None:
            self._remote_dead = True
            return False

        attempts = 2
        max_sleep = float(self.tts.get("max_retry_sleep", 5))

        for attempt in range(1, attempts + 1):
            self.turn_synthesis_calls += 1
            try:
                response = client.audio.speech.create(
                    model=self.tts.get("model", "canopylabs/orpheus-v1-english"),
                    voice=self.tts.get("voice", "daniel"),
                    input=text,
                    response_format=self.tts.get("response_format", "wav"),
                )
                path.write_bytes(response.read())
                fix_wav_sizes(path)
                return path.stat().st_size > 0

            except RateLimitError as exc:
                # Same evidence-based rule as the LLM router: a 429 only takes
                # the online voice offline when the response says recovery is
                # far away. Ambiguous means transient.
                evidence = gather_evidence(exc)
                log_rate_limit(self.remote_label, evidence)
                threshold = float(self.tts.get("transient_recovery_s", TRANSIENT_RECOVERY_S))
                kind, seconds = classify_rate_limit(evidence, threshold)

                if kind == "exhausted":
                    now = datetime.now().astimezone()
                    midnight = ProviderState.next_midnight()
                    until = min(midnight, now + timedelta(seconds=seconds))
                    self.state.mark_dead_until(
                        self.state_key, until, evidence.limit_source or "rate limited"
                    )
                    log.info(
                        "%s is rate limited until %s; piper will speak until then",
                        self.remote_label, until.strftime("%Y-%m-%d %H:%M"),
                    )
                    return False

                if attempt >= attempts:
                    log.info(
                        "%s is rate limited; falling back to piper for this answer",
                        self.remote_label,
                    )
                    return False

                nap = min(seconds if seconds is not None else 1.0, max_sleep)
                log.info(
                    "%s transient rate limit, retrying in %.1fs", self.remote_label, nap
                )
                time.sleep(nap)

            except Exception as exc:
                detail = getattr(getattr(exc, "body", None), "get", lambda _k: None)("message")
                log.warning(
                    "%s tts failed: %s", self.remote_label, detail or exc.__class__.__name__
                )
                return False

        return False

    def _speak_piper(self, text: str, path: Path) -> bool:
        ready, reason = self.piper_available()
        if not ready:
            if not self._piper_warned:
                log.warning("no local voice: %s", reason)
                self._piper_warned = True
            return False

        argv = [
            self.piper_binary,
            "-m", str(self.piper_model),
            "-c", str(self._piper_config_path()),
            "-f", str(path),
            "--length-scale", str(self.piper.get("length_scale", 1.0)),
        ]

        started = time.monotonic()
        try:
            result = subprocess.run(
                argv, input=text, text=True, capture_output=True, timeout=120
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("piper failed: %s", exc.__class__.__name__)
            return False

        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            log.warning("piper exited %s: %s", result.returncode, detail[-1:] or "")
            return False

        if not (path.exists() and path.stat().st_size > 0):
            log.warning("piper produced no audio")
            return False

        log.info("piper synthesised %d characters in %.2fs", len(text), time.monotonic() - started)
        return True


    # ---------------------------------------------------------------- piper --
    def _resolve_piper_model(self) -> Path | None:
        raw = str(self.piper.get("model", "")).strip()
        if not raw:
            return None
        return Path(os.path.expanduser(raw))

    def _resolve_piper_binary(self) -> str | None:
        configured = str(self.piper.get("binary", "")).strip()
        if configured:
            path = Path(os.path.expanduser(configured))
            return str(path) if path.is_file() else None

        # Prefer the piper installed alongside the interpreter that is running
        # us, so a systemd unit with a bare PATH still finds it.
        beside = Path(sys.executable).parent / "piper"
        if beside.is_file():
            return str(beside)
        return shutil.which("piper")

    def piper_available(self) -> tuple[bool, str]:
        """Returns (ready, reason). Checked at startup, not mid-answer."""
        if self.piper_binary is None:
            return False, "the piper binary was not found (pip install piper-tts)"
        if self.piper_model is None:
            return False, "no tts.piper.model is set in config.yaml"
        if not self.piper_model.is_file():
            return False, f"the voice model is missing: {self.piper_model}"

        config_path = self._piper_config_path()
        if not config_path.is_file():
            return False, f"the voice config is missing: {config_path}"
        return True, f"{self.piper_model.name}"

    def _piper_config_path(self) -> Path:
        configured = str(self.piper.get("config", "")).strip()
        if configured:
            return Path(os.path.expanduser(configured))
        return Path(str(self.piper_model) + ".json")

    # ---------------------------------------------------------------- beeps --
    def _tone_path(self, kind: str) -> Path:
        return self.cfg.state_dir() / f"beep-{kind}.wav"

    def prewarm(self) -> None:
        """Generate the beeps up front, and check the local voice is usable."""
        ready, reason = self.piper_available()
        if ready:
            log.info("local voice ready: piper with %s", reason)
        else:
            log.warning("local voice unavailable: %s", reason)

        until = self.state.dead_until(self.state_key)
        if until and self.state.is_dead(self.state_key):
            log.info(
                "%s is out of daily quota until %s; piper will speak until then",
                self.remote_label, until.strftime("%Y-%m-%d %H:%M"),
            )

        for kind, freq, ms in (("ready", 880.0, 120), ("error", 330.0, 260)):
            path = self._tone_path(kind)
            if not path.is_file():
                try:
                    write_tone(path, freq, ms)
                except OSError as exc:
                    log.warning("could not write the %s beep: %s", kind, exc.__class__.__name__)

    def beep(self, kind: str = "ready") -> bool:
        """Short tone. 'ready' when listening starts, 'error' when speech fails."""
        path = self._tone_path(kind)
        if not path.is_file():
            freq, ms = (880.0, 120) if kind == "ready" else (330.0, 260)
            try:
                write_tone(path, freq, ms)
            except OSError as exc:
                log.warning("could not write the %s beep: %s", kind, exc.__class__.__name__)
                return False
        # A 120 ms tone does not need the full echo tail of a spoken answer.
        with self.gate.hold(tail_s=float(self.tts.get("beep_mute_tail", 0.3))):
            return play_wav(path)

    # ---------------------------------------------------------------- speak --
    def _play(self, path: Path, on_play_start: Any = None) -> bool:
        if on_play_start is not None:
            on_play_start()
        with self.gate.hold():
            return play_wav(path)

    def speak(self, text: str, on_play_start: Any = None) -> str:
        """Say it out loud. Returns the backend used: remote, piper or console."""
        spoken = clean_for_speech(text, self.max_chars)
        if not self.enabled or not spoken:
            return "off"

        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        path = Path(handle.name)

        try:
            if self._speak_remote(spoken, path) and self._play(path, on_play_start):
                return self.remote_label
            if self._speak_piper(spoken, path) and self._play(path, on_play_start):
                return "piper"
        finally:
            path.unlink(missing_ok=True)

        # Last resort: the answer still reaches the user, just on screen.
        log.warning("no speech backend worked; printing instead")
        print(f"[unspoken] {spoken}")
        self.beep("error")
        return "console"


class SpeechStream:
    """Speaks sentences in order on a worker thread while the model keeps writing.

    The gate stays raised from the first sentence to the last, so the wake word
    detector is deaf for the whole answer rather than between sentences.
    """

    def __init__(self, speaker: Speaker) -> None:
        self.speaker = speaker
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.first_audio_at: float | None = None
        self.first_sentence_at: float | None = None
        self.backends: list[str] = []
        self._lock = threading.Lock()

    def _worker(self) -> None:
        while True:
            sentence = self.queue.get()
            try:
                if sentence is None:
                    return
                backend = self.speaker.speak(sentence, on_play_start=self._audio_started)
                self.backends.append(backend)
            except Exception:
                log.exception("could not speak a sentence")
            finally:
                self.queue.task_done()

    def _audio_started(self) -> None:
        with self._lock:
            if self.first_audio_at is None:
                self.first_audio_at = time.monotonic()

    def say(self, sentence: str) -> None:
        if not sentence.strip():
            return
        if self.first_sentence_at is None:
            self.first_sentence_at = time.monotonic()
        if self.thread is None:
            self.thread = threading.Thread(
                target=self._worker, name="speech", daemon=True
            )
            self.thread.start()
        log.debug("queue SUBMIT (%d already queued): %.50s", self.queue.qsize(), sentence)
        self.queue.put(sentence)

    def finish(self, timeout: float = 120.0) -> None:
        """Wait for every queued sentence to finish playing."""
        if self.thread is None:
            return
        self.queue.put(None)
        self.thread.join(timeout=timeout)
        self.thread = None

    @property
    def spoke_anything(self) -> bool:
        return self.first_sentence_at is not None

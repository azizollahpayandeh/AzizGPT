"""Speech in: Groq Whisper, then local faster-whisper after repeated failures.

The clip is deleted as soon as the transcription call returns, whichever
backend answered and whether or not it succeeded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from .brain import build_http_client
from .config import Config, key_problem

log = logging.getLogger(__name__)

REMOTE_FAILURES_BEFORE_LOCAL = 2
LOCAL_WARNING = (
    "I could not reach the online transcriber, so I am using the local one. "
    "It will be slower."
)


class Transcriber:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stt = cfg.get("stt", {}) or {}
        self.remote_failures = 0
        self.use_local = False
        self._client: OpenAI | None = None
        self._local_model: Any = None
        self._warning: str | None = None
        self._local_missing_logged = False

    # ------------------------------------------------------------ warnings --
    def take_warning(self) -> str | None:
        """Pop a message the assistant should say out loud, if there is one."""
        message, self._warning = self._warning, None
        return message

    # -------------------------------------------------------------- remote --
    def _remote_client(self) -> OpenAI | None:
        if self._client is not None:
            return self._client

        wanted = self.stt.get("provider", "groq")
        provider = next((p for p in self.cfg.providers if p["name"] == wanted), None)
        if provider is None:
            log.warning("stt provider %r is not in the providers list", wanted)
            return None

        key = Config.api_key_for(provider)
        problem = key_problem(key)
        if problem:
            log.warning("stt: %s %s", provider["api_key_env"], problem)
            return None

        timeout = float(self.stt.get("request_timeout", 45))
        self._client = OpenAI(
            base_url=provider["base_url"],
            api_key=key,
            timeout=timeout,
            max_retries=0,
            http_client=build_http_client(timeout),
        )
        return self._client

    def _transcribe_remote(self, path: Path) -> str | None:
        client = self._remote_client()
        if client is None:
            return None

        try:
            with path.open("rb") as clip:
                result = client.audio.transcriptions.create(
                    file=(path.name, clip.read()),
                    model=self.stt.get("model", "whisper-large-v3-turbo"),
                    language=self.stt.get("language", "en"),
                    response_format="text",
                )
        except Exception as exc:
            detail = getattr(getattr(exc, "body", None), "get", lambda _k: None)("message")
            log.warning("remote stt failed: %s", detail or exc.__class__.__name__)
            return None

        text = result if isinstance(result, str) else getattr(result, "text", "")
        return str(text).strip()

    # --------------------------------------------------------------- local --
    def _transcribe_local(self, path: Path) -> str | None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            if not self._local_missing_logged:
                log.warning(
                    "faster-whisper is not installed, so there is no local "
                    "transcriber (pip install faster-whisper)"
                )
                self._local_missing_logged = True
            return None

        if self._local_model is None:
            name = self.stt.get("local_fallback_model", "base.en")
            log.info("loading the local faster-whisper model %r", name)
            try:
                self._local_model = WhisperModel(name, device="cpu", compute_type="int8")
            except Exception as exc:
                log.warning("could not load the local model: %s", exc.__class__.__name__)
                return None

        try:
            segments, _info = self._local_model.transcribe(
                str(path), language=self.stt.get("language", "en"), beam_size=1
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            log.warning("local stt failed: %s", exc.__class__.__name__)
            return None

    # ----------------------------------------------------------- interface --
    def transcribe(self, path: Path) -> tuple[str, str]:
        """Returns (text, backend). Deletes the clip before returning."""
        try:
            if not self.use_local:
                text = self._transcribe_remote(path)
                if text is not None:
                    self.remote_failures = 0
                    return text, "remote"

                self.remote_failures += 1
                if self.remote_failures >= REMOTE_FAILURES_BEFORE_LOCAL:
                    self.use_local = True
                    self._warning = LOCAL_WARNING
                    log.info(
                        "remote stt failed %d times, switching to the local model",
                        self.remote_failures,
                    )

            if self.use_local:
                text = self._transcribe_local(path)
                if text is not None:
                    return text, "local"

            return "", ""
        finally:
            path.unlink(missing_ok=True)

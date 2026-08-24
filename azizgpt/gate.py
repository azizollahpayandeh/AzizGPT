"""A shared 'audio is coming out of the speakers' flag.

The wake word detector consults this so the assistant's own voice cannot
re-trigger it. The speaker raises it for the whole of playback and keeps it
raised for a short tail afterwards, because the sink drains slightly after the
player exits and the room keeps echoing for a moment longer.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class PlaybackGate:
    def __init__(self, post_mute_s: float = 0.5, enabled: bool = True) -> None:
        self.post_mute_s = max(0.0, float(post_mute_s))
        self.enabled = enabled
        self._lock = threading.Lock()
        self._playing = 0          # nested/queued playbacks
        self._muted_until = 0.0

    def begin(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._playing += 1
            log.debug("gate RAISED (holders=%d)", self._playing)

    def end(self, tail_s: float | None = None) -> None:
        if not self.enabled:
            return
        tail = self.post_mute_s if tail_s is None else max(0.0, float(tail_s))
        with self._lock:
            self._playing = max(0, self._playing - 1)
            if self._playing == 0:
                self._muted_until = time.monotonic() + tail
                log.debug("gate RELEASED, muted for a further %.2fs", tail)
            else:
                log.debug("gate still held (holders=%d)", self._playing)

    @property
    def muted(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            return self._playing > 0 or time.monotonic() < self._muted_until

    def hold(self, tail_s: float | None = None) -> _Hold:
        return _Hold(self, tail_s)

    def wait_until_clear(self, timeout: float = 30.0) -> float:
        """Block until nothing is audible. Returns how long it waited."""
        started = time.monotonic()
        while self.muted and time.monotonic() - started < timeout:
            time.sleep(0.05)
        waited = time.monotonic() - started
        if waited > 0.05:
            log.debug("waited %.2fs for the speakers to fall silent", waited)
        return waited


class _Hold:
    def __init__(self, gate: PlaybackGate, tail_s: float | None = None) -> None:
        self.gate = gate
        self.tail_s = tail_s

    def __enter__(self) -> PlaybackGate:
        self.gate.begin()
        return self.gate

    def __exit__(self, *exc_info: object) -> None:
        self.gate.end(self.tail_s)

"""One AzizGPT daemon per machine.

Two --wake processes sharing one microphone both hear the wake word, both
answer, and both play their answer a second apart. That is indistinguishable
from a double-playback bug from inside either process, so make it impossible.
"""

from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

LOCK_NAME = "azizgpt.lock"


class AlreadyRunning(RuntimeError):
    def __init__(self, pid: str) -> None:
        self.pid = pid
        super().__init__(f"another AzizGPT is already running (pid {pid})")


class SingleInstance:
    """An advisory lock held for the lifetime of the process."""

    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / LOCK_NAME
        self._handle = None

    def acquire(self) -> None:
        handle = self.path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            other = handle.read().strip() or "unknown"
            handle.close()
            raise AlreadyRunning(other) from None

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle
        log.debug("single-instance lock held at %s", self.path)

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
        except OSError:
            pass
        self._handle = None

    def __enter__(self) -> SingleInstance:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

"""Wake word listening.

While waiting, audio is read one 1280-sample chunk at a time, scored, and
dropped. Nothing is buffered and nothing reaches disk until the wake word has
fired and the recorder takes over.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import Config
from .gate import PlaybackGate

log = logging.getLogger(__name__)

CHUNK_SAMPLES = 1280   # openwakeword expects 1280 samples of 16 kHz mono int16
SAMPLE_RATE = 16000



# openwakeword's package __init__ eagerly imports its training helper, which
# pulls in scipy and scikit-learn: about 78 MB that inference never touches.
# Standing in a stub before the first import keeps them out of the process.
TRAINER_MODULE = "openwakeword.custom_verifier_model"


def skip_trainer_import() -> bool:
    """Returns True if the stub was installed. Must run before any import."""
    if "openwakeword" in sys.modules or TRAINER_MODULE in sys.modules:
        return False

    stub = types.ModuleType(TRAINER_MODULE)

    def train_custom_verifier(*_args, **_kwargs):
        raise RuntimeError(
            "openwakeword's trainer is not loaded in this build; train custom "
            "models in a separate environment"
        )

    stub.train_custom_verifier = train_custom_verifier
    sys.modules[TRAINER_MODULE] = stub
    return True


@contextlib.contextmanager
def _no_onnx_arena():
    """Build sessions without the CPU arena allocator.

    onnxruntime pre-allocates a large arena per session. For models this small,
    fed one 1280-sample chunk at a time, it costs about 120 MB and buys nothing:
    measured inference is 2.58 ms either way against an 80 ms budget, and the
    detection scores are identical.
    """
    import onnxruntime as ort

    original = ort.InferenceSession

    class LeanSession(original):
        def __init__(self, *args, **kwargs):
            options = kwargs.get("sess_options") or ort.SessionOptions()
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            kwargs["sess_options"] = options
            super().__init__(*args, **kwargs)

    ort.InferenceSession = LeanSession
    try:
        yield
    finally:
        ort.InferenceSession = original


def resolve_model(cfg: Config) -> Path | None:
    """Find the wake word model: an explicit path, a custom one, or a bundled one."""
    settings = cfg.get("wakeword", {}) or {}
    if settings.get("skip_trainer_import", True):
        skip_trainer_import()
    name = str(settings.get("model", "hey_jarvis"))

    direct = Path(name).expanduser()
    if direct.is_file():
        return direct

    custom_dir = Path(str(settings.get("model_dir", "models"))).expanduser()
    for candidate in (custom_dir / f"{name}.onnx", custom_dir / name):
        if candidate.is_file():
            return candidate

    try:
        import openwakeword

        bundled = Path(openwakeword.__file__).parent / "resources" / "models"
    except ImportError:
        log.error("openwakeword is not installed")
        return None

    for candidate in (bundled / f"{name}.onnx", bundled / f"{name}_v0.1.onnx"):
        if candidate.is_file():
            return candidate

    available = sorted(p.stem for p in bundled.glob("*.onnx"))
    log.error("wake word model %r not found. Bundled models: %s", name, ", ".join(available))
    return None


class WakeWordListener:
    def __init__(self, cfg: Config, gate: PlaybackGate | None = None) -> None:
        self.cfg = cfg
        settings = cfg.get("wakeword", {}) or {}
        # While the assistant is speaking, its own voice reaches the microphone
        # and scores as a wake word. Stay deaf until playback has drained.
        self.gate = gate or PlaybackGate(0.0, enabled=False)
        self.threshold = float(settings.get("threshold", 0.5))
        self.cooldown = float(settings.get("cooldown_s", 2.0))
        self.want_speex = bool(settings.get("enable_speex_noise_suppression", False))
        self.vad_threshold = float(settings.get("vad_threshold", 0.5))
        self.lean_sessions = bool(settings.get("disable_onnx_arena", True))
        self.model_path = resolve_model(cfg)
        self.model = None
        self._last_fired = 0.0

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.model_path is None:
            return False

        from openwakeword.model import Model

        options = {
            "wakeword_model_paths": [str(self.model_path)],
            "vad_threshold": self.vad_threshold,
        }

        arena = _no_onnx_arena() if self.lean_sessions else contextlib.nullcontext()

        with warnings.catch_warnings(), arena:
            # onnxruntime warns about CUDA on a CPU-only box; that is expected here.
            warnings.simplefilter("ignore", UserWarning)
            if self.want_speex:
                try:
                    self.model = Model(**options, enable_speex_noise_suppression=True)
                    log.info("speex noise suppression is on")
                    return True
                except Exception:
                    log.warning(
                        "speex noise suppression is not available "
                        "(pip install speexdsp-ns); continuing without it"
                    )
            try:
                self.model = Model(**options)
            except Exception as exc:
                log.error("could not load the wake word model: %s", exc)
                return False

        log.info(
            "wake word ready: %s (threshold %.2f, vad %.2f, from %s, lean sessions %s)",
            self.model_path.stem, self.threshold, self.vad_threshold,
            self.cfg.path.name, "on" if self.lean_sessions else "off",
        )
        return True


    @staticmethod
    def _flush(stream: sd.RawInputStream) -> int:
        """Drop whatever the device buffered while we were deaf."""
        dropped = 0
        try:
            while stream.read_available >= CHUNK_SAMPLES:
                stream.read(CHUNK_SAMPLES)
                dropped += 1
        except Exception as exc:
            log.debug("could not flush the input buffer: %s", exc.__class__.__name__)
        return dropped

    def wait_for_wake(self) -> bool:
        """Block until the wake word is heard. False means the mic failed."""
        if not self.load():
            return False

        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SAMPLES,
                dtype="int16",
                channels=1,
            )
        except Exception as exc:
            log.error("could not open the microphone: %s", exc)
            return False

        was_muted = False

        with stream:
            while True:
                block, _overflowed = stream.read(CHUNK_SAMPLES)

                if self.gate.muted:
                    # Keep draining so the device buffer cannot overflow, but
                    # score nothing: this is the assistant's own voice.
                    was_muted = True
                    continue

                if was_muted:
                    self._flush(stream)
                    self.model.reset()   # drop features built from the echo
                    was_muted = False
                    log.debug("resumed listening after playback")
                    continue

                chunk = np.frombuffer(bytes(block), dtype=np.int16)
                if chunk.size != CHUNK_SAMPLES:
                    continue

                scores = self.model.predict(chunk)   # chunk is dropped after this
                best = max(scores.values()) if scores else 0.0

                if best >= self.threshold:
                    now = time.monotonic()
                    if now - self._last_fired < self.cooldown:
                        continue
                    self._last_fired = now
                    log.info("wake word heard (%.2f >= %.2f)", best, self.threshold)
                    self.model.reset()
                    return True

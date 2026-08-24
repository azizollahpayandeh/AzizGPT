"""Per-component memory breakdown, measured by importing things in order.

Run as its own process so the deltas mean something: once main.py has imported
the whole stack, every increment reads as zero.

    python -m azizgpt.memprofile
"""

from __future__ import annotations

import sys


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def main() -> int:
    rows: list[tuple[str, float, float]] = []
    previous = rss_mb()
    start = previous
    print(f"{'component':38} {'delta':>10} {'running':>10}")
    print("-" * 60)
    print(f"{'bare interpreter':38} {'':>10} {previous:9.1f} MB")

    def step(label: str) -> None:
        nonlocal previous
        now = rss_mb()
        rows.append((label, now - previous, now))
        print(f"{label:38} {now - previous:+9.1f} {now:9.1f} MB")
        previous = now

    import numpy  # noqa: F401
    step("numpy")
    import onnxruntime  # noqa: F401
    step("onnxruntime (import only)")
    import sounddevice  # noqa: F401
    step("sounddevice + portaudio")
    import openai  # noqa: F401
    step("openai sdk")

    from .config import load_config
    cfg = load_config()
    step("config + dotenv + yaml")

    from .brain import Brain, build_http_client
    # Kept referenced so the measurement reflects a live object, not one that
    # was collected before the next reading.
    keep_alive = [Brain(cfg)]
    step("Brain (tool schemas, provider state)")
    build_http_client(30)
    step("shared keep-alive http client")

    from .tts import Speaker
    speaker = Speaker(cfg)
    speaker.prewarm()
    step("Speaker (piper runs out of process)")

    from .stt import Transcriber
    keep_alive.append(Transcriber(cfg))
    step("Transcriber")

    from .recorder import Recorder
    keep_alive.append(Recorder(cfg))
    step("Recorder (vad, buffers)")

    from .wakeword import WakeWordListener
    listener = WakeWordListener(cfg)
    step("openwakeword import")
    listener.load()
    step("wake word onnx sessions")

    import numpy as np
    for _ in range(250):
        listener.model.predict(np.zeros(1280, dtype=np.int16))
    step("250 wake predictions (steady state)")

    total = rss_mb()
    print("-" * 60)
    print(f"{'TOTAL idle':38} {total - start:+9.1f} {total:9.1f} MB")

    heaviest = sorted(rows, key=lambda r: r[1], reverse=True)[:3]
    print("\nheaviest three:")
    for label, delta, _running in heaviest:
        print(f"  {label:36} {delta:6.1f} MB")

    print(f"\nscipy loaded: {'scipy' in sys.modules}   sklearn loaded: {'sklearn' in sys.modules}")
    assert keep_alive  # referenced so nothing above is collected early
    print("piper: separate process, nothing resident here (see --mem service section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

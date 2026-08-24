#!/usr/bin/env python3
"""Measure how long this machine's speakers stay audible after playback ends.

A Bluetooth A2DP sink buffers far past the moment the player process exits, so
the wake word mute tail cannot be a guessed constant. This plays a clip through
the real speakers, listens on the real microphone, and reports how long sound
was still coming out after the player exited.

    python scripts/calibrate_echo.py            # measure and recommend
    python scripts/calibrate_echo.py --write    # also update config.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azizgpt.config import load_config  # noqa: E402
from azizgpt.tts import find_player, write_tone  # noqa: E402

PHRASE = "This is a calibration clip. One, two, three, four, five, six, seven."
SAFETY_MARGIN_S = 0.6


def listen(samples: list[tuple[float, float]], stop: threading.Event) -> None:
    with sd.RawInputStream(samplerate=16000, blocksize=480, dtype="int16", channels=1) as stream:
        while not stop.is_set():
            block, _overflow = stream.read(480)
            audio = np.frombuffer(bytes(block), dtype=np.int16).astype(np.float32)
            samples.append((time.monotonic(), float(np.sqrt((audio * audio).mean()))))


def one_trial(clip: Path) -> tuple[float, float, float]:
    """Returns (tail_seconds, peak_during, floor)."""
    samples: list[tuple[float, float]] = []
    stop = threading.Event()
    thread = threading.Thread(target=listen, args=(samples, stop), daemon=True)
    thread.start()

    time.sleep(1.5)   # baseline
    player = find_player()
    if player is None:
        stop.set()
        raise SystemExit("no audio player found")
    binary, args = player

    start = time.monotonic()
    subprocess.run(
        [binary, *args, str(clip)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, timeout=180,
    )
    exit_at = time.monotonic()
    time.sleep(6.0)
    stop.set()
    thread.join(timeout=2)

    baseline = [rms for ts, rms in samples if ts < start - 0.2]
    floor = float(np.median(baseline)) if baseline else 1.0

    # Threshold relative to how loud the clip actually was, not just to the
    # noise floor. A floor-relative threshold counts ordinary room noise as
    # "still audible" and invents a drain tail that is not there.
    during = [rms for ts, rms in samples if start <= ts <= exit_at]
    peak = max(during, default=0.0)
    threshold = max(floor * 3.0, peak * 0.4)

    audible = [ts for ts, rms in samples if ts > start and rms > threshold]
    tail = (max(audible) - exit_at) if audible else 0.0
    return max(0.0, tail), peak, floor


def synthesise(cfg) -> Path:
    """Prefer a real spoken clip; fall back to a tone if speech is unavailable."""
    from azizgpt.tts import Speaker

    clip = Path("/tmp/azizgpt_calibration.wav")
    clip.unlink(missing_ok=True)
    if Speaker(cfg)._speak_remote(PHRASE, clip):
        return clip
    print("speech synthesis unavailable; calibrating with a tone instead")
    write_tone(clip, 440.0, 5000, volume=0.5)
    return clip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure the speaker drain tail")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--write", action="store_true", help="update config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    cfg = load_config()
    clip = synthesise(cfg)
    with wave.open(str(clip)) as handle:
        duration = handle.getnframes() / handle.getframerate()

    print(f"clip {duration:.2f}s, {args.trials} trials, speakers at their normal volume\n")

    tails = []
    for index in range(1, args.trials + 1):
        tail, peak, floor = one_trial(clip)
        tails.append(tail)
        verdict = "no audio detected" if peak == 0 else f"peak rms {peak:.0f}, floor {floor:.0f}"
        print(f"  trial {index}: still audible {tail:.2f}s after the player exited   ({verdict})")

    worst = max(tails)
    recommended = round(worst + SAFETY_MARGIN_S, 1)
    print(f"\nworst tail {worst:.2f}s + {SAFETY_MARGIN_S}s margin")
    print(f"recommended wakeword.post_playback_mute: {recommended}")

    if not args.write:
        print("\nre-run with --write to put that into config.yaml")
        return 0

    path = Path(__file__).resolve().parent.parent / "config.yaml"
    text = path.read_text(encoding="utf-8")
    patched, count = re.subn(
        r"(?m)^(\s*post_playback_mute:\s*)[0-9.]+",
        lambda m: f"{m.group(1)}{recommended}",
        text,
    )
    if count != 1:
        print("could not find post_playback_mute in config.yaml; left it alone")
        return 1
    path.write_text(patched, encoding="utf-8")
    print(f"config.yaml updated: post_playback_mute: {recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

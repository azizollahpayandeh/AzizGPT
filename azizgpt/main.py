"""AzizGPT entry point.

Phase 1 is text only: --text gives a typed REPL against the brain and tools.
The microphone, wake word, STT and TTS arrive in later phases.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .brain import Brain, ProviderError, check_providers
from .config import ConfigError, load_config
from .gate import PlaybackGate
from .recorder import Recorder, list_devices
from .single_instance import AlreadyRunning, SingleInstance
from .stt import Transcriber
from .tools import power
from .tools.alarms import restore_alarms
from .tts import Speaker, SpeechStream
from .wakeword import WakeWordListener

log = logging.getLogger("azizgpt")

BANNER = """AzizGPT text mode. Type a command, or 'quit' to leave.
Tools run for real in this mode: apps will actually open."""


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # These are chatty at DEBUG and can echo request internals.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def text_mode(brain: Brain, speaker: Speaker | None = None) -> int:
    print(BANNER)
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line.lower() in {"quit", "exit", ":q"}:
            return 0
        if line.lower() == "reset":
            brain.reset()
            print("(history cleared)")
            continue

        try:
            reply = brain.ask(line)
        except ProviderError as exc:
            print(f"aziz> I could not reach any language provider ({exc}).")
            continue

        if brain.verbose and reply.tool_calls:
            for name, args in reply.tool_calls:
                print(f"      [tool] {name}({args})")
        print(f"aziz> {reply.text}")
        if speaker is not None:
            speaker.speak(reply.text)


def listen_mode(
    brain: Brain, speaker: Speaker | None, recorder: Recorder, transcriber: Transcriber
) -> int:
    print("Press Enter to talk, then speak. Type quit to leave.")
    while True:
        try:
            line = input("\n[Enter to talk] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line in {"quit", "exit", ":q"}:
            return 0

        turn_started = time.monotonic()
        if speaker is not None:
            speaker.beep("ready")

        clip = recorder.record()
        if clip is None:
            print("(nothing heard)")
            continue

        stt_started = time.monotonic()
        text, backend = transcriber.transcribe(clip)
        stt_ms = int((time.monotonic() - stt_started) * 1000)
        if not text:
            print("(could not transcribe that)")
            continue
        if not usable_transcript(text, brain.cfg):
            print("(ignored: nothing was actually said)")
            continue
        print(f"you> {text}   [{backend}]")

        handle_turn(text, brain, speaker, transcriber, stt_ms, turn_started)


# What Whisper produces from near-silence. These are never real commands.
SILENCE_HALLUCINATIONS = {
    "you", "thank you", "thanks", "thanks for watching", "thank you.",
    "thanks for watching!", "bye", "bye.", "okay", "ok", "so", "um", "uh",
    "you're welcome", "please subscribe", "subtitles by the amara.org community",
    "transcription by castingwords", "oh", "hmm", "mm", "yeah",
}


def usable_transcript(text: str, cfg) -> bool:
    """Reject echo and silence before they cost a model call.

    Two failure modes: an empty transcription, and Whisper hallucinating a
    stock phrase out of faint noise. The recorder already refuses audio that is
    not meaningfully louder than the room; this is the second line.
    """
    floor = int((cfg.get("stt", {}) or {}).get("min_transcript_chars", 2))
    stripped = "".join(ch for ch in text if ch.isalnum())
    if len(stripped) < floor or not any(ch.isalpha() for ch in text):
        log.info("ignoring an empty transcription: %r", text)
        return False

    normalised = " ".join(text.lower().strip(" .,!?-\"'").split())
    if normalised in SILENCE_HALLUCINATIONS:
        log.info("ignoring a silence hallucination: %r", text)
        return False
    return True



def wire_power_confirmation(
    speaker: Speaker | None, recorder: Recorder | None, transcriber: Transcriber | None
) -> None:
    """Give power_action a way to ask out loud and hear one short answer.

    Without this the tool refuses anything that needs confirming, which is the
    behaviour we want in --text and in the test harness.
    """
    if speaker is None or recorder is None or transcriber is None:
        power.set_voice(None, None)
        return

    def ask(question: str, timeout: float) -> str | None:
        log.info("power: asking for confirmation")
        print(f"aziz> {question}")
        speaker.speak(question)
        clip = recorder.record(start_timeout_s=timeout)
        if clip is None:
            log.info("power: nothing was said within %.0fs", timeout)
            return None
        heard, _backend = transcriber.transcribe(clip)
        print(f"you> {heard}")
        return heard

    power.set_voice(speaker.speak, ask)


def wire_typed_confirmation() -> None:
    """In --text mode the confirmation is typed, not spoken."""

    def ask(question: str, _timeout: float) -> str | None:
        print(f"aziz> {question}")
        try:
            return input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

    power.set_voice(lambda text: print(f"aziz> {text}"), ask)



SERVICE_CGROUP = (
    "/sys/fs/cgroup/user.slice/user-1000.slice/user@{uid}.service"
    "/app.slice/azizgpt.service"
)


def _read_bytes(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


def report_memory() -> int:
    """--mem: what this build costs, and what the running service is using."""
    import subprocess

    print("=== per-component breakdown (measured in a clean process) ===\n")
    result = subprocess.run(
        [sys.executable, "-m", "azizgpt.memprofile"],
        capture_output=True, text=True, timeout=180,
    )
    print(result.stdout.rstrip() or result.stderr.rstrip())

    print("\n=== running service ===")
    try:
        pid = subprocess.run(
            ["systemctl", "--user", "show", "azizgpt.service", "-p", "MainPID", "--value"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pid = ""

    if not pid or pid == "0":
        print("  azizgpt.service is not running")
        return 0

    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "RssAnon:", "RssFile:", "VmSwap:")):
                name, value = line.split(":", 1)
                print(f"  {name:10} {int(value.split()[0]) / 1024:8.1f} MB")

    cgroup = Path(SERVICE_CGROUP.format(uid=os.getuid()))
    if cgroup.is_dir():
        print("  cgroup (this process plus any children it spawns):")
        for name in ("memory.current", "memory.peak", "memory.high", "memory.max"):
            value = _read_bytes(cgroup / name)
            if value is not None:
                print(f"    {name:16} {value / 1048576:8.1f} MB")
    return 0



def make_gate(cfg) -> PlaybackGate:
    settings = cfg.get("wakeword", {}) or {}
    return PlaybackGate(
        post_mute_s=float(settings.get("post_playback_mute", 0.5)),
        enabled=bool(settings.get("mute_during_playback", True)),
    )


def handle_turn(
    text: str,
    brain: Brain,
    speaker: Speaker | None,
    transcriber: Transcriber,
    stt_ms: int = 0,
    turn_started: float | None = None,
) -> None:
    """One heard utterance: warn if needed, answer, speak it as it arrives."""
    warning = transcriber.take_warning()
    if warning:
        log.info(warning)
        if speaker is not None:
            speaker.speak(warning)

    # Off by default: one answer, one synthesis call, one playback. Sentence
    # streaming bought about 60 ms and cost a round trip per sentence.
    stream_sentences = bool((brain.cfg.get("tts", {}) or {}).get("stream_sentences", False))
    speech = SpeechStream(speaker) if speaker is not None else None
    if speaker is not None:
        speaker.begin_turn()

    try:
        reply = brain.ask(
            text,
            on_sentence=speech.say if (speech is not None and stream_sentences) else None,
        )
    except ProviderError as exc:
        log.error("no provider answered: %s", exc)
        if speaker is not None:
            speaker.speak("I could not reach any language provider just now.")
        return

    print(f"aziz> {reply.text}")

    tts_ms = 0
    first_audio_ms = 0
    if speech is not None:
        # Either sentence streaming is off, or this was a tool round that never
        # streamed any. Speak the whole answer as one clip.
        if not speech.spoke_anything:
            speech.say(reply.text)
        speech.finish()
        if speaker is not None:
            speaker.check_turn()
        if speech.first_sentence_at is not None:
            tts_ms = int((time.monotonic() - speech.first_sentence_at) * 1000)
        if speech.first_audio_at is not None and turn_started is not None:
            first_audio_ms = int((speech.first_audio_at - turn_started) * 1000)

    if brain.verbose:
        total_ms = int((time.monotonic() - turn_started) * 1000) if turn_started else 0
        log.info(
            "timings: stt_ms=%d llm_ms=%d tts_ms=%d total_ms=%d "
            "(first sentence at %dms, first audio out at %dms)",
            stt_ms, reply.llm_ms, tts_ms, total_ms,
            reply.first_sentence_ms, first_audio_ms,
        )


def wake_mode(
    brain: Brain,
    speaker: Speaker,
    recorder: Recorder,
    transcriber: Transcriber,
    listener: WakeWordListener,
) -> int:
    if not listener.load():
        print("could not load the wake word model; see the log above", file=sys.stderr)
        return 1

    log.info("listening for the wake word")
    while True:
        try:
            if not listener.wait_for_wake():
                return 1

            turn_started = time.monotonic()
            speaker.beep("ready")   # immediate feedback, before anything slow

            clip = recorder.record()
            if clip is None:
                log.info("nothing was said after the wake word")
                continue

            stt_started = time.monotonic()
            text, backend = transcriber.transcribe(clip)
            stt_ms = int((time.monotonic() - stt_started) * 1000)
            if not text:
                log.info("could not transcribe that")
                continue
            if not usable_transcript(text, brain.cfg):
                continue

            print(f"you> {text}   [{backend}]")
            handle_turn(text, brain, speaker, transcriber, stt_ms, turn_started)

        except KeyboardInterrupt:
            print()
            return 0
        except Exception:
            # A bad turn must never take the daemon down.
            log.exception("that turn failed; still listening")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="azizgpt", description="AzizGPT voice assistant")
    parser.add_argument("--text", action="store_true", help="typed REPL, no mic and no speech")
    parser.add_argument("--verbose", action="store_true", help="log the full tool-calling round trip")
    parser.add_argument("--check-providers", action="store_true", help="list which configured models are actually available")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--model", default=None, help="override the configured LLM model for this run")
    parser.add_argument("--dry-run", action="store_true", help="validate tool calls without side effects")
    parser.add_argument("--speak", action="store_true", help="speak the answers aloud in --text mode")
    parser.add_argument("--say", default=None, help="speak one line of text and exit, for testing TTS")
    parser.add_argument("--listen", action="store_true", help="press Enter to record, then it transcribes and runs")
    parser.add_argument("--quiet", action="store_true", help="do not speak the answers in --listen mode")
    parser.add_argument("--list-devices", action="store_true", help="show the audio devices and exit")
    parser.add_argument("--wake", action="store_true", help="always-on wake word loop (this is what systemd runs)")
    parser.add_argument("--mem", action="store_true", help="print current memory use and a per-component breakdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.check_providers:
        return check_providers(cfg, verbose=args.verbose)

    if args.mem:
        return report_memory()

    if args.list_devices:
        print(list_devices())
        return 0

    if args.say is not None:
        speaker = Speaker(cfg)
        speaker.prewarm()
        backend = speaker.speak(args.say)
        print(f"spoke via: {backend}")
        return 0 if backend not in ("console", "off") else 1

    if args.wake or args.listen:
        restored, dropped = restore_alarms(cfg)
        if restored or dropped:
            log.info("alarms: %d restored, %d expired", restored, dropped)

    if args.wake or args.listen:
        try:
            lock = SingleInstance(cfg.state_dir())
            lock.acquire()
        except AlreadyRunning as exc:
            print(
                f"AzizGPT is already running (pid {exc.pid}). Two of them share "
                "one microphone and answer twice. Stop the other one first:\n"
                f"  kill {exc.pid}",
                file=sys.stderr,
            )
            return 3

    if args.wake:
        brain = Brain(cfg, verbose=args.verbose, dry_run=args.dry_run, model_override=args.model)
        gate = make_gate(cfg)
        speaker = Speaker(cfg, gate=gate)
        speaker.prewarm()
        recorder, transcriber = Recorder(cfg, gate), Transcriber(cfg)
        wire_power_confirmation(speaker, recorder, transcriber)
        return wake_mode(
            brain, speaker, recorder, transcriber, WakeWordListener(cfg, gate)
        )

    if args.listen:
        brain = Brain(cfg, verbose=args.verbose, dry_run=args.dry_run, model_override=args.model)
        gate = make_gate(cfg)
        speaker = None if args.quiet else Speaker(cfg, gate=gate)
        if speaker is not None:
            speaker.prewarm()
        recorder, transcriber = Recorder(cfg, gate), Transcriber(cfg)
        wire_power_confirmation(speaker, recorder, transcriber)
        return listen_mode(brain, speaker, recorder, transcriber)

    if args.text:
        brain = Brain(cfg, verbose=args.verbose, dry_run=args.dry_run, model_override=args.model)
        speaker = Speaker(cfg) if args.speak else None
        wire_typed_confirmation()
        return text_mode(brain, speaker)

    print(
        "Pick a mode:\n"
        "  python -m azizgpt.main --text            typed, silent\n"
        "  python -m azizgpt.main --text --speak    typed, spoken answers\n"
        "  python -m azizgpt.main --listen          press Enter to talk\n"
        "  python -m azizgpt.main --wake            always-on wake word loop\n"
        "  python -m azizgpt.main --check-providers",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

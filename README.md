# AzizGPT

A background voice assistant for Linux that runs entirely on hosted inference, with a security model designed around the assumption that the speech recogniser will occasionally hear things that were never said.

It listens for a wake word, records until you stop talking, transcribes, decides whether the request maps to a tool, runs it, and answers out loud — then keeps listening, so a follow-up continues the same conversation without saying the wake word again. Speech-to-text, the language model, and text-to-speech all go through Groq's OpenAI-compatible API on a single key. There is no local language model. Idle memory is about 145 MB, which matters because the machine it was built for has 8 GB and usually a VM running.

The interesting part is not that it works, it is what happens when it mishears. Whisper hallucinates fluent sentences out of room noise. A voice assistant that can launch programs and power off a machine has to be built expecting that, so the model is confined to a fixed set of enumerated tools with validated arguments, never a shell, and anything destructive requires a spoken confirmation whose accepted words are chosen specifically to exclude what Whisper invents from silence.

---

## Quickstart

Requires Python 3.12+ (numpy 2.5 dropped 3.11) and a working PipeWire or
PulseAudio setup.

```bash
# 1. System dependencies (Debian/Ubuntu/Kali)
sudo apt install -y python3-full python3-dev build-essential \
                    portaudio19-dev ffmpeg libnotify-bin

# 2. Clone and set up
git clone https://github.com/azizollahpayandeh/AzizGPT.git
cd AzizGPT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your key. Free tier is enough to try it.
#    Get one at https://console.groq.com/keys
cp .env.example .env
$EDITOR .env          # set GROQ_API_KEY=gsk_...

# 4. Confirm the key works and the configured models still exist
python -m azizgpt.main --check-providers

# 5. See the health of the whole provider chain at any time
python -m azizgpt.main --providers-status
```

`.env` also takes an optional `OPENROUTER_API_KEY`. The second tier ships
enabled, so adding that key gives the assistant somewhere to go when Groq's
free daily budget runs out.

Then run it, starting with the mode that needs no microphone:

```bash
python -m azizgpt.main --text          # type commands, no mic, no speech
python -m azizgpt.main --text --speak  # type commands, spoken answers
python -m azizgpt.main --listen        # press Enter to talk
python -m azizgpt.main --wake          # always on, say "hey jarvis"
```

Diagnostics:

```bash
python -m azizgpt.main --providers-status  # per-provider health and last error
python -m azizgpt.main --mem               # memory, per component
python -m azizgpt.main --list-devices      # audio devices
python -m azizgpt.main --say "hello"       # test speech only
```

For offline speech, download a Piper voice once:

```bash
mkdir -p ~/.local/share/piper && cd ~/.local/share/piper
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

The path is read from `tts.piper.model` in `config.yaml`, so put it wherever you like.

### Try it without a microphone

`--text` exercises the whole brain and tool layer with no audio hardware at all. `--dry-run` additionally validates every tool call without launching anything:

```bash
python -m azizgpt.main --text --dry-run --verbose
```

---

## Conversation sessions

A wake word opens a session, not a single exchange. After it answers, it listens
for `session.follow_up_timeout_s` seconds (5 by default); anything you say in
that window continues the **same** conversation, so the model has the earlier
turns and "and what about tomorrow" resolves. Silence closes the session and it
returns to waiting for the wake word.

```
"hey jarvis"  ──▶ [beep, session open]
                    "what is the weather in Messina"   ──▶ answer
                    (5s window, starts after playback ends)
                    "and what about London"            ──▶ answer, same context
                    (5s window)
                    silence                            ──▶ [low beep, session closed]
              ◀──  waiting for the wake word again
```

The window opens only once playback has actually finished, not when the answer
was generated — otherwise the microphone reopens while the speakers are still
going and the assistant hears itself. The playback gate applies throughout, so
the wall-clock cost of a window is `wakeword.post_playback_mute` plus the window
itself.

The window is a deadline on **speech onset**, not on the turn: a voice that
starts at 1.9 seconds is never cut off mid-sentence. Getting that to actually
hold took three fixes, because a false trigger on room noise used to consume the
window entirely — measured at 8–12 seconds against a configured 2. The recorder
now spends `recorder.prime_discard_ms` ignoring the near-zero frames a stream
emits while it opens, then `recorder.floor_probe_ms` measuring the room before it
will accept an onset, and treats an onset as provisional until enough frames
clear the threshold. `recorder.pre_roll_ms` must exceed those two combined, or a
sentence begun immediately loses its opening words; a test enforces that.

Measured on a real microphone with real playback, nobody speaking:

| Configured | Measured to close |
| --- | --- |
| 2 s | 3.43 s |
| 6 s | 7.39 s |

Each is the gate mute (1.0 s) plus the window plus the ~0.4 s room probe, and
the 3.96 s spread matches the 4 s configured spread.

History belongs to the session and is dropped when it closes, so a new wake word
always starts clean. Sessions are capped by `session.max_turns` and
`session.max_duration_s` so one cannot stay open indefinitely, and the rolling
`llm.history_turns` bounds the context within a session. Neither an empty
transcription nor a Whisper hallucination counts as speech, so neither keeps a
session alive. Session open, each turn, and close-with-reason are logged.

## Architecture

```mermaid
flowchart TD
    MIC([microphone]) --> WW

    subgraph always_on[always on, nothing written to disk]
        WW[wake word<br/>openwakeword hey_jarvis<br/>1280-sample chunks, scored and dropped]
    end

    WW -->|score >= threshold| BEEP[beep, immediate feedback]
    BEEP --> REC[VAD recorder<br/>webrtcvad, pre-roll, noise-floor gate]
    REC -->|one temp wav| STT[Groq whisper-large-v3-turbo]
    STT -->|deleted immediately after| FILTER{usable?}

    FILTER -->|empty or hallucination| WW
    FILTER -->|real speech| LLM[tool-calling LLM<br/>openai/gpt-oss-20b]

    LLM -->|no tool call| TTS
    LLM -->|tool call| TOOLS[fixed tool registry<br/>JSON schemas, validated args]
    TOOLS -->|structured result| LLM

    TTS[text to speech]
    TTS --> T1[Groq Orpheus]
    T1 -->|429 or failure| T2[local Piper]
    T2 -->|missing or failure| T3[console + error beep]

    T1 --> GATE[[playback gate:<br/>wake detector and recorder<br/>both deaf while audio plays]]
    T2 --> GATE
    GATE --> WW
```

Files map onto that pipeline directly:

| Module | Responsibility |
| --- | --- |
| `azizgpt/wakeword.py` | openwakeword loop, lean ONNX sessions, playback gating |
| `azizgpt/recorder.py` | VAD-gated capture, pre-roll, adaptive noise-floor rejection |
| `azizgpt/stt.py` | Groq Whisper, local faster-whisper fallback, clip deletion |
| `azizgpt/brain.py` | provider router, streaming, tool-calling loop, system prompt |
| `azizgpt/tools/` | the entire set of things the model can do |
| `azizgpt/tts.py` | Orpheus, Piper, console fallback; serialised playback |
| `azizgpt/gate.py` | the shared "audio is playing" flag |
| `azizgpt/single_instance.py` | one daemon per machine |

`docs/ARCHITECTURE.md` goes deeper on the gate, the recorder's noise-floor logic, and the provider state machine.

---

## Provider fallback

Providers are an ordered list in `config.yaml`. Each has a `name`, `base_url`, `api_key_env`, `model` and `enabled` flag. The first enabled provider that is not currently marked dead handles the request. An OpenRouter entry ships disabled as a template.

Two tiers ship enabled:

| Tier | Provider | Model | Notes |
| --- | --- | --- | --- |
| 1 | Groq | `openai/gpt-oss-20b` | Primary. Also serves speech-to-text and text-to-speech. |
| 2 | OpenRouter | `nvidia/nemotron-3-nano-30b-a3b:free` | Free tier, verified tool-calling. |

The fallback model was not chosen by reputation. OpenRouter's `/models` endpoint
was filtered to entries that are free *and* list `tools` in
`supported_parameters`, then the shortlist was run against this project's real
schemas. Several models that advertise tool support were unusable in practice
(rate limited on the first call); this one answered correctly on an `open_app`
call, a `get_weather` call and a no-tool question.

### Classifying a 429

A 429 is classified by **when the response says it recovers**, never by the
words in it. That distinction is the whole design, because the wording lies in
both directions:

```
Groq:        "... on tokens per day (TPD): Limit 200000, Used 200000.
              Please try again in 31.535999999s."     retry-after: 32
             -> says "per day", means 32 seconds

OpenRouter:  "Rate limit exceeded: free-models-per-day. Add 10 credits to
              unlock 1000 free model requests per day"
              x-ratelimit-reset: 1787616000000  (UTC midnight, 12h away)
             -> genuinely daily, and sends no retry-after at all
```

Reading Groq's wording as a day-long outage benched a healthy provider for
hours. Reading OpenRouter's as transient would hammer a genuinely exhausted one.
Neither is decidable from keywords, so the classifier reads evidence, in order:

1. the `retry-after` header
2. a `try again in X` hint in the message body
3. `x-ratelimit-reset` (OpenRouter sends epoch milliseconds)
4. Groq's `x-ratelimit-reset-tokens` / `-requests` duration headers

| Recovery time | Handling |
| --- | --- |
| ≤ `llm.transient_recovery_s` (90s) | Sleep and retry the same provider, up to `llm.rate_limit_retries` times. Never benched. |
| Longer | Bench until exactly that timestamp, persisted to `~/.local/state/azizgpt/providers.json`, and move down the chain. |
| **No evidence at all** | **Transient.** Sleeping on a real outage costs one wasted retry; benching a healthy provider costs every request until the mark expires. |

Local midnight is only a ceiling on a bench, never the default. The full 429
body and headers are logged at INFO *before* classification, so a
misclassification can be diagnosed from the log rather than reproduced.

This has been wrong twice, in opposite directions, so it is covered by
`tests/test_rate_limit.py` against responses captured verbatim from both live
APIs.

**A dead mark is an optimisation, not a promise.** If every provider in the chain
is benched, the router makes a second pass and tries them anyway rather than
answering with nothing (`llm.last_resort_retry`). A provider that answers on that
pass has its mark cleared. The failure mode where the assistant says "I could not
reach any language provider" and stops is therefore not reachable while any
provider might still work.

When the chain genuinely cannot answer, it says which provider failed and why, in
words rather than status codes:

```
no provider could answer. groq because it is out of its daily quota until 15:12;
openrouter because it rejected the key (HTTP 401)
```

Every provider switch is logged at INFO. To see the whole chain at a glance:

```bash
python -m azizgpt.main --providers-status
```

which prints, per provider: enabled, model, whether the key is present and
well-formed, alive or dead, the dead-until timestamp with minutes remaining, and
the last error it returned.

The same logic guards text-to-speech under its own state key, so a spent Orpheus quota falls through to Piper for the rest of the window without re-paying the round trip on every turn.

---

## Tools

The model cannot do anything that is not in this table.

| Tool | Arguments | Behaviour |
| --- | --- | --- |
| `open_app` | `app_key` (enum) | Launches an allowlisted desktop app. The key maps to an exact argv list in `config.yaml`. Unknown keys are refused. |
| `open_url` | `url` | Opens a site in the configured browser. Bare names are completed locally (`youtube` becomes `https://youtube.com`). http/https only. |
| `open_email` | none | Opens the configured webmail URL. No OAuth, no mail API. |
| `get_time` | none | Local time and date. |
| `get_weather` | `city` (optional) | Open-Meteo. No API key. Defaults to `weather.default_city`. |
| `set_alarm` | `when`, `label` | Schedules a desktop notification via a transient systemd user timer. The model supplies a timestamp only. |
| `power_action` | `action` (enum) | `shutdown`, `restart` or `sleep`. The first two require spoken confirmation. |

Anything that is not a tool call is answered directly and spoken back.

---

## Configuration

Everything tunable lives in `config.yaml`. Model names are never hardcoded, because free-tier catalogs rotate; `--check-providers` tells you which configured models actually exist.

| Key | Default | Purpose |
| --- | --- | --- |
| `providers[]` | groq enabled, openrouter disabled | Ordered fallback chain |
| `llm.model` | `openai/gpt-oss-20b` | Also supports `openai/gpt-oss-120b` |
| `llm.stream` | `true` | Stream completions |
| `llm.max_tool_iterations` | `4` | Guard against tool-call ping-pong |
| `llm.rate_limit_retries` | `3` | Per-minute 429 retries |
| `llm.transient_recovery_s` | `90` | Recovery within this is retried, not benched |
| `llm.daily_probe_after_s` | `900` | Bench length when a 429 gives no recovery evidence |
| `llm.last_resort_retry` | `true` | Try benched providers rather than answer nothing |
| `llm.history_turns` | `6` | Rolling conversation memory |
| `stt.model` | `whisper-large-v3-turbo` | Speech to text |
| `stt.min_transcript_chars` | `2` | Below this is treated as silence |
| `session.follow_up_timeout_s` | `5` | Quiet for this long after it speaks closes the session |
| `session.max_turns` | `12` | Hard cap on exchanges in one session |
| `session.max_duration_s` | `180` | Hard cap on how long a session stays open |
| `tts.model` | `canopylabs/orpheus-v1-english` | Voices: autumn, diana, hannah, austin, daniel, troy |
| `tts.voice` | `daniel` | |
| `tts.stream_sentences` | `false` | One synthesis call per answer when off |
| `tts.piper.model` | `~/.local/share/piper/en_US-lessac-medium.onnx` | Local fallback voice |
| `wakeword.model` | `hey_jarvis` | Drop a custom `.onnx` in `models/` and name it here |
| `wakeword.threshold` | `0.7` | Raise if it false triggers, lower if it misses |
| `wakeword.mute_during_playback` | `true` | Stops it hearing itself |
| `wakeword.post_playback_mute` | `1.0` | Measured, not guessed — see `scripts/calibrate_echo.py` |
| `wakeword.disable_onnx_arena` | `true` | Saves ~120 MB, identical detection scores |
| `wakeword.skip_trainer_import` | `true` | Keeps scipy and sklearn out, ~78 MB |
| `recorder.vad_aggressiveness` | `2` | 0 lenient, 3 strict |
| `recorder.speech_over_floor` | `3.0` | Speech must exceed the room's noise floor by this factor |
| `apps` | chrome, firefox, vm, terminal, files | The complete launch allowlist |
| `web.default_browser` | `chrome` | Must be a key from `apps` |
| `power.enabled` | `true` | Set false to remove the power tool |
| `power.confirm_timeout_s` | `8` | One listen, no re-asking |
| `power.countdown_s` | `5` | Ctrl-C aborts during this |

---

## Security design

This is the part of the project worth reviewing. The threat model is not a malicious user; it is an unreliable input channel wired to a machine.

**The model can never execute an arbitrary command.** There is no shell tool, no `eval`, no `exec`, and no code path where model output reaches a shell. Every subprocess in the codebase is invoked with an argv list. `shell=True` appears nowhere.

**Tools are a fixed, enumerated set with JSON schemas.** The model chooses a name from a registry and supplies arguments the schema declares. Every tool re-validates its own arguments regardless of what the schema allowed, because a schema is a hint to the model, not an enforcement boundary.

**Application launching is allowlist-only.** `open_app` takes a friendly key — `chrome`, `vm`, `terminal` — which is looked up in a map in `config.yaml` to get an exact argv list. The model never supplies a binary name or a path. A key that is not in the map is refused out loud. The schema's enum is a first line of defence; the runtime lookup is the actual one.

**Browser launching does not trust the environment.** Python's `webbrowser` module decides what browsers exist from `DISPLAY` and `PATH`, returns False in environments that have neither, and reports success it cannot verify. It was replaced with a direct launch through the same allowlist, with the process polled to confirm it survived.

**Destructive actions require spoken confirmation, and how consent is judged is the point.** `power_action` asks out loud, listens once for up to 8 seconds, and proceeds only on consent. Consent is judged by words rather than by an exact string: a reply counts when every word is a yes or a harmless filler and at least one is a core yes, so `yes`, `Yes Yes`, `yes please`, `yes do it`, `do it` and `go ahead` all confirm. That matters in both directions — an exact-match set rejected a real "Yes Yes" in live use, and a gate that ignores a plain yes trains you to repeat yourself at something that can power itself off. **"ok", "okay" and "sure" stay excluded**, because Whisper invents all three out of silence, observed in this project. Any negation anywhere cancels, as does a yes attached to a different request (`yes open chrome`), a timeout, a misheard reply, and silence. It never asks twice. After confirming there is a five second countdown, logged once a second, so Ctrl-C still aborts.

**Confirmation cannot be bypassed by the absence of a voice.** If no speaking and listening pair is wired in — `--text` mode, the test harness — the tool refuses anything needing confirmation rather than assuming consent.

**The microphone is discarded frame by frame.** While waiting for the wake word, audio is read one 1280-sample chunk at a time, scored, and dropped. Nothing is buffered and nothing reaches disk. Only the clip captured after the wake word fires is written to a temp file, and it is deleted in a `finally` block as soon as the transcription call returns, whether it succeeded or not.

**Silence does not reach the model.** The recorder measures the room's own noise floor while it waits and rejects audio that is not meaningfully louder. Known Whisper silence hallucinations (`"you"`, `"Thank you."`, `"Thanks for watching!"`) are dropped by name before a model call is made.

**One daemon per machine.** Two `--wake` processes sharing a microphone both hear the wake word, both answer, and both speak about a second apart — and each one's recorder captures the other's voice. From inside either process that is indistinguishable from a playback bug. An advisory `flock` makes it impossible; a second start is refused with the pid to kill.

**Keys come from `.env` only.** They are never printed and never logged. A malformed key is reported by shape ("contains non-ASCII characters"), never by value. `.env` overrides the shell environment, because a stale exported variable silently shadowing the file cost hours during development.

**The model's reasoning is never spoken.** `openai/gpt-oss-*` returns a separate `reasoning` field alongside `content`. Only `content` is ever read, stored, or sent to text-to-speech.

**Tool results are structured so they cannot be embellished.** `open_url` and `power_action` return JSON (`ok` plus a url, or `ok` plus a reason). The system prompt forbids inventing a reason for a failure or explaining one by claiming something is closed or not running unless the result says so.

There are no destructive tools beyond `power_action`: nothing deletes, nothing sends messages, nothing spends money.

---

## Limitations

Stated plainly, because they are real:

- **No speaker identification.** The assistant cannot tell your voice from a video playing in the room. Ambient speech will trigger it and can be transcribed as a command. Set `wakeword.threshold` higher if this bites.
- **No barge-in.** You cannot interrupt it mid-answer. The countdown before a power action is the only interruptible moment, and only from the terminal.
- **Free tiers are small.** Orpheus is 3600 tokens/day, roughly a few dozen spoken answers, after which speech falls through to Piper - which is why Piper is wired in rather than optional. Groq's LLM tier is 200k tokens/day, about a third of which a full harness run consumes. OpenRouter's free tier is **50 requests/day** without credits, so tier 2 is a genuine safety net for a handful of turns, not a full replacement for tier 1.
- **English only.** The system prompt, the wake word model, the Whisper language pin and the Piper voice are all English.
- **No echo cancellation.** The playback gate stops the assistant hearing *itself*; it does nothing about other audio in the room.
- **Bluetooth speakers add latency.** Perceived latency measured 3.3–5.4 seconds from wake word to first audio, of which the biggest single component is text-to-speech synthesis.
- **Alarms survive a restart, not a reboot.** systemd holds the timer; the JSON file lets timers be recreated at startup.

---

## Troubleshooting

Every item here was hit during development.

**It answers, then immediately wakes itself up.** Its own voice is reaching the microphone. Confirm `wakeword.mute_during_playback: true`, then calibrate the tail for your hardware:

```bash
python scripts/calibrate_echo.py            # measure and recommend
python scripts/calibrate_echo.py --write    # write it into config.yaml
```

Run it in a quiet room with nothing else playing, or it measures the room instead of your speakers.

The script plays a clip and listens; it does not change your mute state, volume,
or default device. It snapshots the default sink and source and restores them in
a `finally` block regardless of how it exits, and prints what it restored, so a
future edit cannot leave a machine silent after the process ends. If you mute
something by hand while debugging audio, note that WirePlumber persists
per-application mute in `~/.local/state/wireplumber/stream-properties`: it
survives restarts and reboots and does not appear in `wpctl status` unless that
application happens to be playing at the time.

**Answers play twice, about a second apart.** Two daemons are running. Newer builds refuse the second start; if you are on an older one, `pgrep -af azizgpt.main` and kill the extra.

**Everything fails with `could not locate runnable browser` or a proxy error.** If your shell exports `all_proxy=socks://...`, that scheme is invalid and the HTTP layer raises before any request is made. Use `socks5://` (with `pip install socksio`) or an `http://` proxy. AzizGPT resolves one usable proxy explicitly rather than letting the client read the environment, so `HTTPS_PROXY` alone is enough.

**`error: externally-managed-environment` on Kali or Debian.** That is pip refusing to touch the system Python. Use the virtualenv from the quickstart; do not pass `--break-system-packages`.

**`model_terms_required` from the TTS endpoint.** Some Groq models need a one-time terms acceptance in the console before the API will serve them. Open the model in <https://console.groq.com/playground>, accept, and retry. Speech falls through to Piper until you do.

**`ModuleNotFoundError: No module named 'pkg_resources'`.** `webrtcvad` still imports it and setuptools 81 removed it. `requirements.txt` pins `setuptools<81`.

**The service says no provider is available, but `--text` works.** The unit runs
the code and config it loaded at start. Editing `config.yaml` or the source
changes nothing until it is restarted, so a provider added hours ago is simply
not in the running chain. Check what it actually loaded:

```bash
journalctl --user -u azizgpt.service | grep 'provider chain as loaded'
```

It logs its version, the config path and modification time, the `brain.py`
modification time, and the chain it built. If those timestamps are older than
the files on disk, restart it:

```bash
systemctl --user restart azizgpt.service
```

**It says a provider is unavailable.** Run `python -m azizgpt.main --providers-status`. It shows which tier is benched, until when, and the last error each returned. A daily quota clears itself; a rejected key does not.

**It transcribes silence as "." or "Thank you." and answers it.** The noise-floor gate is not tuned for your room. Raise `recorder.speech_over_floor`.

**Memory.** `python -m azizgpt.main --mem` prints a per-component breakdown plus what the running service is using.

---

## Development

```bash
pytest tests/ -q                         # rate-limit classifier, no network
python scripts/test_tools.py             # 23 command harness, dry-run, prints SCORE
python scripts/test_tools.py --self-test # checks the scorer itself, no API calls
python -m azizgpt.main --mem             # memory breakdown
```

The harness fires 23 phrases at the model and checks that each produces the correct tool call, the correct arguments, or correctly produces none. Refusal cases pass only if no tool call is emitted. It always runs dry: arguments are fully validated but nothing launches, opens a tab, schedules a timer, or powers anything off.

A full run costs roughly a third of the Groq free daily token budget, so it is not a per-commit check.

---

## License

MIT. See [LICENSE](LICENSE).

<!--
Repository settings to apply on GitHub (Settings → General, and the ⚙ next to
"About" on the repo home page).

Description (350 char limit, this is 208):
  Background voice assistant for Linux. Wake word, VAD-gated capture, Groq
  Whisper, tool-calling LLM, Orpheus with a local Piper fallback. Allowlist-only
  tool layer, no shell access, spoken confirmation for destructive actions.

Website:
  (leave empty, or link docs/ARCHITECTURE.md)

Topics:
  voice-assistant, speech-recognition, wake-word, text-to-speech, groq,
  whisper, llm, tool-calling, function-calling, openwakeword, piper-tts,
  python, linux, systemd, security, offline-fallback

Suggested settings:
  - Issues: on
  - Discussions: off
  - Wiki: off
  - Projects: off
  - "Require status checks to pass" on main once CI has run once
-->

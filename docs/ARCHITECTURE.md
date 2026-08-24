# Architecture notes

Deeper detail on the three subsystems whose behaviour is not obvious from the
code: the playback gate, the recorder's noise-floor logic, and the provider
router. The README covers the pipeline shape; this covers why these three are
built the way they are.

---

## 1. The playback gate

### The problem

The assistant speaks through speakers that its own microphone can hear. Without
protection, a spoken answer scores as a wake word, and the recorder that opens
next transcribes the tail of the answer as if it were a new command. In
development this produced a self-sustaining loop: the assistant answered a
question about US presidents, woke itself, transcribed its own sentence, and
answered that.

### The design

`azizgpt/gate.py` holds one `PlaybackGate` shared by three components:

- `Speaker` raises it for the whole of playback and lowers it afterwards.
- `WakeWordListener` scores nothing while it is raised.
- `Recorder` refuses to open the microphone while it is raised.

The gate is a counter, not a boolean, so overlapping holds nest correctly, and
the tail only starts when the last holder releases. That matters because an
answer may be several playbacks: the gate stays raised across the whole answer
rather than flickering between sentences.

```
speak()  ──▶ gate.begin()  ──▶ playback ──▶ gate.end(tail)
                 │                               │
                 │                               └─▶ muted until now + tail
                 ▼
   wake detector: read chunk, discard, do not score
   recorder:      wait_until_clear() before opening the stream
```

On resume, both consumers do more than un-mute:

- the wake detector drains whatever the device buffered while it was deaf, then
  calls `model.reset()` so features built from echo cannot contribute to a score
- the recorder flushes the input buffer immediately after opening

### The tail is measured, not guessed

`gate.end()` takes a per-hold tail, because a 120 ms beep and a 20 second spoken
answer do not need the same deafness. Beeps use `tts.beep_mute_tail` (0.3 s);
answers use `wakeword.post_playback_mute`.

That second number came from `scripts/calibrate_echo.py`, which plays a clip
through the real speakers, listens on the real microphone, and reports how long
sound was still arriving after the player process exited.

Getting this right took three attempts, and the failures are instructive:

1. First measurement: 1.89 s. **Contaminated** — a browser was playing audio
   into the same sink.
2. Second: 4.46 s across three trials, so the tail was set to 5.1 s. Still
   contaminated, and the detector's threshold sat close to the noise floor, so
   ordinary room noise counted as "still audible".
3. With the room quiet and the threshold made relative to actual playback level
   (`max(floor * 3, peak * 0.4)`), four consecutive trials measured **0.00 s**.

`pw-play` blocks until the audio has really played; there is no meaningful drain
on this hardware. The shipped default is 1.0 s, which is margin for jitter
rather than a measured tail. A 5.1 s deaf window would have made the assistant
unusable — you could not speak for five seconds after every answer — in service
of a number that was an artifact.

The lesson worth carrying: a threshold defined relative to the noise floor will
find a signal in any room. Define it relative to the thing you are measuring.

---

## 2. The recorder

### Frame handling

Audio is read in 30 ms frames (480 samples at 16 kHz, mono int16 — webrtcvad
accepts only 10, 20 or 30 ms). Before speech starts, frames go into a bounded
`deque` sized by `recorder.pre_roll_ms` and age out. That pre-roll is prepended
once speech is detected, so the first consonant is not clipped; before that, the
frames are discarded as they fall off the end of the deque and never leave
memory.

Capture stops when `recorder.silence_ms` of non-speech follows speech, or at
`recorder.max_seconds`. If nobody speaks within `recorder.start_timeout_s`, it
returns nothing at all.

### Why VAD alone is not enough

`webrtcvad` answers "is this speech?" and it answers "yes" to a surprising
amount of room noise. Whisper then turns that noise into confident text: `"."`,
`"you"`, `"Thank you."`, `"Thanks for watching!"`. The model dutifully replies
to those, so a quiet room produced spontaneous conversation.

A minimum length check does not fix it, because the hallucinations are real
words of plausible length.

### The noise-floor gate

The recorder calibrates against the room continuously, at no extra cost:

1. While waiting for speech, every frame the VAD calls **non**-speech has its
   RMS recorded. Those samples are the room's own noise.
2. The floor is the median of those samples — median rather than mean so one
   door slam does not move it.
3. The threshold is `max(floor * speech_over_floor, floor + min_floor_margin)`.
   The multiplicative term scales with a loud room; the additive term stops a
   near-silent room from producing a near-zero threshold.
4. After capture, frames the VAD called speech are checked against that
   threshold. If fewer than `min_speech_ms` worth clear it, the clip is
   discarded without a transcription call.

A live rejection looks like:

```
recorder: that was too quiet to be speech (0 loud frames, floor 45,
          threshold 134); not transcribing
```

That saves the API call, the latency, and the spurious answer.

`azizgpt/main.py` holds a second line of defence, `usable_transcript()`, which
drops known Whisper silence artifacts by name. Two layers, because the audio
gate cannot catch a hallucination produced from genuinely loud noise, and the
name list cannot anticipate every artifact.

RMS is computed with numpy: `audioop` was removed in Python 3.13.

---

## 3. The provider router

### State

`ProviderState` persists to `~/.local/state/azizgpt/providers.json`:

```json
{
  "groq":     { "dead_until": "2026-08-24T13:23:44+03:30", "reason": "daily quota" },
  "groq-tts": { "dead_until": "2026-08-25T00:00:00+03:30", "reason": "tts daily quota" }
}
```

Writes are atomic (temp file then `os.replace`). Text-to-speech shares the file
under its own key, because Orpheus and the LLM exhaust independently.

### The loop

```
for provider in enabled providers:
        │
        ├── marked dead and not yet expired? ──▶ log, next provider
        ├── key missing or malformed?         ──▶ log by shape, next provider
        │
        └── for attempt in 1..rate_limit_retries:
                  │
                  ├── success ─────────────────▶ return
                  │
                  ├── 429, per-minute  ────────▶ sleep min(retry-after, cap)
                  │                              and retry this provider
                  │
                  ├── 429, daily quota ────────▶ mark dead, next provider
                  │
                  ├── connection error ────────▶ next provider
                  └── other HTTP status ───────▶ next provider

all exhausted ──▶ ProviderError with a per-provider reason
```

### Classifying a 429

The two kinds are distinguished by the response body and headers. A daily quota
is indicated by markers in the message (`per day`, `tokens per day`, `TPD`,
`RPD`, `daily limit`, `quota exceeded`) or by a `retry-after` longer than an
hour. Everything else is treated as transient.

Groq's message is explicit enough to classify on:

```
Rate limit reached for model `openai/gpt-oss-20b` ... on tokens per day (TPD):
Limit 200000, Used 199715, Requested 936. Please try again in 4m41.232s.
```

### Dead until when

The obvious implementation marks a daily-quota provider dead until local
midnight. That is wrong for Groq, whose per-day bucket refills continuously —
the message above says four minutes, not eleven hours.

The router marks `min(local_midnight, now + retry_after)`. Without the cap, a
provider that recovers in minutes is benched for the rest of the day. This was
observed live: the harness had Groq marked dead until midnight while a direct
probe returned 200 with 7923 tokens remaining.

Local midnight remains the fallback for a 429 that gives no `retry-after`.

### Streaming and tool calls

`_round()` decides per call whether to stream. Streaming is used only when there
is somewhere for sentences to go — a speech queue — because the non-streaming
path is simpler and the streamed path must reassemble tool calls from deltas
indexed by position.

Sentences are emitted early only while no tool-call delta has been seen in that
round, so a preamble is never spoken for a turn that turns out to be a tool
call. Sentences shorter than 40 characters are held back: each one costs a
separate synthesis round trip, and `"e.g."` is not a sentence.

Measured on Groq, streaming saved about 60 ms on a four-sentence answer, because
the whole completion arrives in one burst. It is kept for providers that stream
incrementally, and defaults off for speech (`tts.stream_sentences: false`), where
one synthesis call per answer is both cheaper and avoids overlapping playback.

The keep-alive HTTP client was the real latency win: one shared client across
LLM, speech-to-text and speech, measured at 365 ms per call against 1098 ms when
each call built its own connection through a proxy.

---

## Memory

Idle is about 145 MB, measured per component with
`python -m azizgpt.main --mem`:

| Component | Cost |
| --- | --- |
| openwakeword ONNX sessions | 22 MB (141 MB before tuning) |
| openai sdk | 48 MB |
| onnxruntime import | 18 MB |
| numpy | 18 MB |
| sounddevice + portaudio | 6 MB |
| brain, http client, recorder | 12 MB |
| piper | 0 MB resident |

Two settings account for roughly 200 MB:

- `wakeword.disable_onnx_arena` — onnxruntime pre-allocates a memory arena per
  session. For models this small, fed one 1280-sample chunk at a time, it cost
  ~120 MB and bought nothing. Verified equivalent on real "hey jarvis" audio:
  identical max score, identical number of chunks over threshold, identical
  checksum over the whole score sequence, and 2.57 ms vs 2.58 ms median
  inference against an 80 ms budget.
- `wakeword.skip_trainer_import` — openwakeword's package `__init__` eagerly
  imports its training helper, pulling in scipy and scikit-learn (~78 MB) that
  inference never touches. A stub module stands in for it.

Piper is not resident: it is a subprocess per answer that exits. It peaks near
240 MB in its own process while synthesising, and because the service spawns it,
that counts against the unit's cgroup limits for those seconds. The `MemoryHigh`
setting is sized for that transient spike, not for idle.

"""Provider router and tool-calling loop.

Two things worth knowing:

1. openai/gpt-oss-* returns a separate `reasoning` field alongside `content`.
   Only `content` is ever read, stored, or returned. `reasoning` never reaches
   the history, the caller, or TTS.

2. There are two kinds of 429. A per-minute rate limit is transient: sleep for
   retry-after and try the same provider again. A daily quota is not: mark the
   provider dead until local midnight, persist that to disk so a restart does
   not re-hammer it, and move down the chain.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx2
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from . import tools
from .config import Config, key_problem
from .tools.system import local_tz_name

log = logging.getLogger(__name__)

STATE_FILENAME = "providers.json"

# Markers Groq and friends use for a daily/lifetime quota rather than a per-minute one.
DAILY_MARKERS = (
    "per day", "requests per day", "tokens per day", "rpd", "tpd",
    "daily limit", "daily quota", "quota exceeded", "insufficient_quota",
)
DAILY_SECONDS = 3600  # a retry-after longer than this is not a per-minute limit

SYSTEM_PROMPT = """You are AzizGPT, a voice assistant running on the user's Linux laptop.

Your replies are spoken out loud. Keep them short and plain: one or two sentences \
unless the user asks for detail. No markdown, no bullet points, no code blocks, \
no emoji, no stage directions.

The current local time is {now} ({tz}). Use this whenever a request involves \
time, dates, or scheduling.

You have a small fixed set of tools. Call a tool only when the user asks you to \
DO something a tool actually performs. Answer everything else directly from your \
own knowledge with no tool call at all - questions like "what is an IDOR", \
"explain CSRF", or "who are you" need no tools whatsoever.

The only applications you can open are: {apps}. If the user asks for any other \
application, do NOT call open_app. Say plainly that it is not on your allowed \
list and that you cannot open it.

You cannot delete or move files, run shell commands, send messages or email, \
buy anything, or change any setting. You have no tool for these and no way to \
get one. If asked to do any of them, say plainly that you cannot, and do not \
call a tool instead.

You can shut down, restart, or suspend this computer with power_action. \
Shutdown and restart ask the user out loud to confirm, and that happens inside \
the tool: call it once and report what it returns. Never ask for confirmation \
yourself, never call it twice, and never call it unless the user actually asked \
to shut down, restart, or sleep the machine.

For set_alarm, convert what the user said into an exact local timestamp in the \
format YYYY-MM-DD HH:MM:SS, resolved against the current local time above. Pass \
only that timestamp. Never pass a command, a flag, or a shell line to any tool.

After a tool runs you get its result. Some tools answer with JSON. That result \
is the only fact you have about what happened: report it and nothing else.

If the result says ok is true, say in one natural sentence what was done, using \
the details in the result, for example "Opened youtube.com." If it says ok is \
false, say it did not work and give the reason from the result. Never invent a \
reason, never guess at a cause the result does not state, and never explain a \
failure by saying something is closed, missing, not open or not running unless \
the result says exactly that. Do not claim something worked when the result says \
it did not, do not retry with a different tool to get around a refusal, and do \
not add advice about what the user should do first."""


class ProviderError(RuntimeError):
    """Every provider in the chain failed or is exhausted."""


# ------------------------------------------------------------------- proxy --
# This machine exports a system-wide proxy, including all_proxy=socks://...,
# and 'socks://' is not a scheme the HTTP layer accepts - it raises before a
# request is ever made. Resolve one usable proxy explicitly instead of letting
# the client read the environment itself.
PROXY_VARS = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
)


def _safe_proxy_label(url: str) -> str:
    """scheme://host:port, with any credentials stripped, safe to log."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or '?'}:{parts.port or '?'}"


def resolve_proxy() -> str | None:
    """First usable proxy from the environment, or None."""
    for var in PROXY_VARS:
        raw = (os.getenv(var) or "").strip()
        if not raw:
            continue

        url = raw
        if url.startswith("socks://"):  # non-standard; socks5 is what is meant
            url = "socks5://" + url[len("socks://"):]

        scheme = url.split("://", 1)[0].lower() if "://" in url else ""
        if scheme.startswith("socks"):
            if importlib.util.find_spec("socksio") is None:
                log.debug("ignoring %s: socks support is not installed", var)
                continue
        elif scheme not in ("http", "https"):
            log.debug("ignoring %s: unusable scheme %r", var, scheme)
            continue

        return url
    return None


# One client for the whole process. Every turn makes three calls to the same
# host through a proxy; rebuilding a client each time paid for a fresh TCP and
# TLS handshake on all three. Keep-alive makes the second and third calls cheap.
_SHARED_CLIENT: httpx2.Client | None = None
_CLIENT_LOCK = threading.Lock()


def build_http_client(timeout: float) -> httpx2.Client:
    """The shared keep-alive client. `timeout` is a floor, applied per request."""
    global _SHARED_CLIENT

    with _CLIENT_LOCK:
        if _SHARED_CLIENT is None:
            proxy = resolve_proxy()
            if proxy:
                log.info("routing API traffic through proxy %s", _safe_proxy_label(proxy))
            _SHARED_CLIENT = httpx2.Client(
                proxy=proxy,
                trust_env=False,
                timeout=timeout,
                limits=httpx2.Limits(
                    max_keepalive_connections=4,
                    max_connections=8,
                    keepalive_expiry=300.0,
                ),
            )
        return _SHARED_CLIENT



SENTENCE_END = re.compile(r"[.!?]['\"\u2019\u201d)\]]*(\s|$)")
MAX_SENTENCE_CHARS = 220   # speak something even if punctuation never arrives
MIN_SENTENCE_CHARS = 40    # "e.g." is not a sentence, and every sentence costs
                           # one TTS round trip, so do not chop into fragments


@dataclass
class _Call:
    """One tool call, however it arrived: whole or assembled from deltas."""

    id: str
    name: str
    arguments: str


def split_sentence(buffer: str) -> tuple[str | None, str]:
    """Peel off the first complete sentence. Returns (sentence, remainder)."""
    start = 0
    while True:
        match = SENTENCE_END.search(buffer, start)
        if match is None:
            break
        cut = match.end()
        # Keep looking if this is an abbreviation rather than a real ending.
        if cut < MIN_SENTENCE_CHARS and cut < len(buffer):
            start = cut
            continue
        return buffer[:cut].strip(), buffer[cut:]

    if len(buffer) >= MAX_SENTENCE_CHARS:
        head, _, tail = buffer[:MAX_SENTENCE_CHARS].rpartition(" ")
        if head:
            return head.strip(), tail + buffer[MAX_SENTENCE_CHARS:]
    return None, buffer


@dataclass
class Reply:
    text: str
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    llm_ms: int = 0
    first_sentence_ms: int = 0   # time to the first thing worth speaking


# ------------------------------------------------------------ dead providers --
class ProviderState:
    """Persisted 'this provider is exhausted until X' marks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.data, dict):
                self.data = {}
        except (OSError, ValueError):
            self.data = {}

    def _save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("could not persist provider state: %s", exc.__class__.__name__)

    def dead_until(self, name: str) -> datetime | None:
        raw = self.data.get(name, {}).get("dead_until")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def is_dead(self, name: str) -> bool:
        until = self.dead_until(name)
        if until is None:
            return False
        if datetime.now().astimezone() >= until:
            self.clear(name)
            return False
        return True

    def mark_dead_until_midnight(self, name: str, reason: str = "daily quota") -> datetime:
        return self.mark_dead_until(name, self.next_midnight(), reason)

    @staticmethod
    def next_midnight() -> datetime:
        now = datetime.now().astimezone()
        return (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def mark_dead_until(self, name: str, until: datetime, reason: str) -> datetime:
        """Mark a provider unusable until a given time, and persist it."""
        self.data[name] = {"dead_until": until.isoformat(), "reason": reason}
        self._save()
        return until

    def clear(self, name: str) -> None:
        if self.data.pop(name, None) is not None:
            self._save()


def _retry_after_seconds(exc: APIStatusError) -> float | None:
    try:
        raw = exc.response.headers.get("retry-after")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _is_daily_quota(exc: Exception, retry_after: float | None) -> bool:
    if retry_after is not None and retry_after > DAILY_SECONDS:
        return True
    blob = str(exc).lower()
    return any(marker in blob for marker in DAILY_MARKERS)


# -------------------------------------------------------------------- brain --
class Brain:
    def __init__(
        self,
        cfg: Config,
        verbose: bool = False,
        dry_run: bool = False,
        model_override: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self.dry_run = dry_run
        self.model_override = model_override
        self.state = ProviderState(cfg.state_dir() / STATE_FILENAME)
        self.schemas = tools.build_schemas(cfg)
        self.history: list[dict[str, Any]] = []
        self._clients: dict[str, OpenAI] = {}

    # ------------------------------------------------------------- plumbing --
    def reset(self) -> None:
        self.history.clear()

    def system_prompt(self) -> str:
        now = datetime.now().astimezone()
        return SYSTEM_PROMPT.format(
            now=now.strftime("%A, %B %-d, %Y at %-I:%M %p"),
            tz=local_tz_name(),
            apps=", ".join(self.cfg.app_keys),
        )

    def model_for(self, provider: dict[str, Any]) -> str:
        return self.model_override or provider["model"]

    def _client(self, provider: dict[str, Any], key: str) -> OpenAI:
        name = provider["name"]
        if name not in self._clients:
            llm = self.cfg["llm"]
            timeout = float(llm.get("request_timeout", 45))
            self._clients[name] = OpenAI(
                base_url=provider["base_url"],
                api_key=key,
                timeout=timeout,
                max_retries=0,  # retries are handled here, per 429 kind
                http_client=build_http_client(timeout),
            )
        return self._clients[name]

    # ------------------------------------------------------------- routing --
    def _complete(
        self, messages: list[dict[str, Any]], stream: bool = False
    ) -> tuple[dict[str, Any], Any]:
        llm = self.cfg["llm"]
        attempts = int(llm.get("rate_limit_retries", 3))
        max_sleep = float(llm.get("max_retry_sleep", 60))
        failures: list[str] = []
        enabled = self.cfg.enabled_providers()

        if not enabled:
            raise ProviderError("no providers are enabled in config.yaml")

        for provider in enabled:
            name = provider["name"]

            until = self.state.dead_until(name) if self.state.is_dead(name) else None
            if until:
                log.info(
                    "provider %s is exhausted until %s, switching to the next provider",
                    name, until.strftime("%Y-%m-%d %H:%M"),
                )
                failures.append(f"{name}: quota exhausted until {until:%H:%M}")
                continue

            key = Config.api_key_for(provider)
            problem = key_problem(key)
            if problem:
                log.warning(
                    "provider %s: %s %s, skipping",
                    name, provider["api_key_env"], problem,
                )
                failures.append(f"{name}: {provider['api_key_env']} {problem}")
                continue

            client = self._client(provider, key)
            model = self.model_for(provider)

            for attempt in range(1, attempts + 1):
                try:
                    log.debug("provider %s: calling %s (attempt %d)", name, model, attempt)
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=self.schemas,
                        tool_choice="auto",
                        temperature=float(llm.get("temperature", 0.2)),
                        max_tokens=int(llm.get("max_tokens", 700)),
                        stream=stream,
                    )
                    return provider, response

                except RateLimitError as exc:
                    retry_after = _retry_after_seconds(exc)
                    if _is_daily_quota(exc, retry_after):
                        # Same reasoning as the TTS path: Groq's per-day token
                        # bucket refills continuously and the response says
                        # when. Benching a working provider until midnight over
                        # a few minutes of backoff costs far more than it saves.
                        midnight = ProviderState.next_midnight()
                        until = midnight
                        if retry_after:
                            recovers = datetime.now().astimezone() + timedelta(
                                seconds=retry_after
                            )
                            until = min(midnight, recovers)
                        self.state.mark_dead_until(name, until, "daily quota")
                        log.info(
                            "provider %s: daily quota exhausted, marked dead until %s, "
                            "switching to the next provider",
                            name, until.strftime("%Y-%m-%d %H:%M"),
                        )
                        failures.append(f"{name}: daily quota exhausted")
                        break

                    if attempt >= attempts:
                        log.info(
                            "provider %s: still rate limited after %d attempts, "
                            "switching to the next provider", name, attempts,
                        )
                        failures.append(f"{name}: rate limited")
                        break

                    nap = min(retry_after if retry_after is not None else 5.0, max_sleep)
                    log.info(
                        "provider %s: per-minute rate limit, sleeping %.1fs then "
                        "retrying (%d/%d)", name, nap, attempt, attempts,
                    )
                    time.sleep(nap)

                except APIConnectionError as exc:
                    log.info(
                        "provider %s: connection failed (%s), switching to the next provider",
                        name, exc.__class__.__name__,
                    )
                    failures.append(f"{name}: connection failed")
                    break

                except APIStatusError as exc:
                    log.info(
                        "provider %s: HTTP %s from the API, switching to the next provider",
                        name, exc.status_code,
                    )
                    failures.append(f"{name}: HTTP {exc.status_code}")
                    break

        raise ProviderError("; ".join(failures) or "no usable provider")


    def _consume_stream(self, response: Any, on_sentence: Any) -> tuple[str, list[_Call]]:
        """Assemble a streamed round, speaking each sentence as it completes."""
        pieces: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        saw_tool_call = False
        pending = ""

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            for part in getattr(delta, "tool_calls", None) or []:
                saw_tool_call = True
                slot = slots.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                if part.id:
                    slot["id"] = part.id
                function = getattr(part, "function", None)
                if function is not None:
                    if function.name:
                        slot["name"] = function.name
                    if function.arguments:
                        slot["arguments"] += function.arguments

            piece = getattr(delta, "content", None)   # never delta.reasoning
            if not piece:
                continue
            pieces.append(piece)

            # Only speak early if this round is clearly not a tool round.
            if on_sentence is None or saw_tool_call:
                continue
            pending += piece
            while True:
                sentence, pending = split_sentence(pending)
                if sentence is None:
                    break
                if sentence:
                    on_sentence(sentence)

        if on_sentence is not None and not saw_tool_call and pending.strip():
            on_sentence(pending.strip())

        calls = [
            _Call(slot["id"], slot["name"], slot["arguments"] or "{}")
            for _index, slot in sorted(slots.items())
            if slot["name"]
        ]
        return "".join(pieces).strip(), calls

    def _round(
        self, messages: list[dict[str, Any]], on_sentence: Any = None
    ) -> tuple[str, list[_Call], dict[str, Any]]:
        """One model call. Streams when there is somewhere for sentences to go."""
        streaming = bool(self.cfg["llm"].get("stream", True)) and on_sentence is not None
        provider, response = self._complete(messages, stream=streaming)

        if streaming:
            content, calls = self._consume_stream(response, on_sentence)
        else:
            message = response.choices[0].message
            content = (message.content or "").strip()   # never message.reasoning
            calls = [
                _Call(c.id, c.function.name, c.function.arguments or "{}")
                for c in (message.tool_calls or [])
            ]
        return content, calls, provider

    # ----------------------------------------------------------------- ask --
    def ask(self, user_text: str, on_sentence: Any = None) -> Reply:
        """One user turn: model, tools if it asks for them, spoken answer back.

        `on_sentence` is called with each complete sentence as it streams, so
        speech can start before the model has finished writing.
        """
        llm = self.cfg["llm"]
        max_iterations = int(llm.get("max_tool_iterations", 4))
        started = time.monotonic()
        first_sentence_at: float | None = None

        def relay(sentence: str) -> None:
            nonlocal first_sentence_at
            if first_sentence_at is None:
                first_sentence_at = time.monotonic()
            on_sentence(sentence)

        relay_or_none = relay if on_sentence is not None else None

        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        made: list[tuple[str, dict[str, Any]]] = []
        provider_name: str | None = None
        model_name: str | None = None
        final = ""

        for iteration in range(1, max_iterations + 1):
            content, calls, provider = self._round(messages, relay_or_none)
            provider_name = provider["name"]
            model_name = self.model_for(provider)

            if self.verbose:
                log.info(
                    "round %d via %s/%s: content=%r tool_calls=%s",
                    iteration, provider_name, model_name, content,
                    [c.name for c in calls] or "none",
                )

            if not calls:
                final = content
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in calls
                    ],
                }
            )

            # gpt-oss sometimes emits the same call twice in one round. Answer
            # every tool_call_id, but only run each distinct call once.
            executed: dict[tuple[str, str], str] = {}

            for call in calls:
                name = call.name
                try:
                    args = json.loads(call.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except ValueError:
                    log.warning("tool %s: arguments were not valid JSON", name)
                    args = {}

                made.append((name, args))
                if self.verbose:
                    log.info("  -> %s(%s)", name, json.dumps(args, ensure_ascii=False))

                signature = (name, json.dumps(args, sort_keys=True))
                if signature in executed:
                    result = executed[signature]
                    log.info("skipped a duplicate %s call in the same round", name)
                else:
                    result = tools.dispatch(self.cfg, name, args, dry_run=self.dry_run)
                    executed[signature] = result
                if self.verbose:
                    log.info("  <- %s", result)

                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        else:
            log.warning("gave up after %d tool rounds", max_iterations)
            final = "I got stuck calling tools, so I stopped there."

        if not final:
            final = "I do not have an answer for that."

        turns = int(llm.get("history_turns", 6))
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": final})
        if turns > 0:
            self.history[:] = self.history[-turns * 2:]
        else:
            self.history.clear()

        llm_ms = int((time.monotonic() - started) * 1000)
        first_ms = (
            int((first_sentence_at - started) * 1000) if first_sentence_at else llm_ms
        )
        return Reply(
            text=final,
            tool_calls=made,
            provider=provider_name,
            model=model_name,
            llm_ms=llm_ms,
            first_sentence_ms=first_ms,
        )


def check_providers(cfg: Config, verbose: bool = False) -> int:
    """Ask each enabled provider for its catalog and check the configured models."""
    state = ProviderState(cfg.state_dir() / STATE_FILENAME)
    problems = 0

    for provider in cfg.providers:
        name = provider["name"]
        status = "enabled" if provider.get("enabled") else "disabled"
        print(f"\n[{name}] {provider['base_url']}  ({status})")
        print(f"  key env: {provider['api_key_env']}")

        if not provider.get("enabled"):
            print("  skipped: disabled in config.yaml")
            continue

        until = state.dead_until(name)
        if until and datetime.now().astimezone() < until:
            print(f"  marked exhausted until {until:%Y-%m-%d %H:%M}")

        key = Config.api_key_for(provider)
        problem = key_problem(key)
        if problem:
            print(f"  UNUSABLE KEY: {provider['api_key_env']} {problem}")
            problems += 1
            continue
        print("  key: present and well-formed")

        try:
            client = OpenAI(
                base_url=provider["base_url"],
                api_key=key,
                timeout=20,
                max_retries=0,
                http_client=build_http_client(20),
            )
            available = {m.id for m in client.models.list().data}
        except Exception as exc:
            print(f"  could not list models: {exc.__class__.__name__}")
            problems += 1
            continue

        print(f"  catalog: {len(available)} models")
        wanted = [
            ("llm", provider["model"]),
            ("stt", cfg.get("stt", {}).get("model")),
            ("tts", cfg.get("tts", {}).get("model")),
        ]
        for role, model in wanted:
            if not model:
                continue
            ok = model in available
            print(f"    {'OK     ' if ok else 'MISSING'} {role}: {model}")
            if not ok and role == "llm":
                problems += 1

        if verbose:
            for model_id in sorted(available):
                print(f"      - {model_id}")

    return 1 if problems else 0

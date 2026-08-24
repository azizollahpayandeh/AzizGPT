"""Tests for 429 classification.

This logic has been wrong twice in this project, in opposite directions:

  1. Benching Groq until local midnight on a 429 whose body said "try again in
     31 seconds", which took a healthy provider offline for hours.
  2. Reading OpenRouter's genuine 50-per-day cap as transient because the
     upsell text mentions "per day" and no retry-after header is sent.

Both directions are covered here against responses captured from the live APIs,
so the next change to the classifier has to survive real payloads rather than
an assumption about them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fixtures_429 import (
    BARE_429,
    GROQ_TPD_SHORT_RECOVERY,
    GROQ_TPM,
    OPENROUTER_DAILY,
    OPENROUTER_PER_MINUTE,
)

from azizgpt.brain import (
    TRANSIENT_RECOVERY_S,
    classify_rate_limit,
    gather_evidence,
    parse_duration,
)


class _Response:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _RateLimitError(Exception):
    """Stands in for openai.RateLimitError: same attributes the parser reads."""

    def __init__(self, fixture: dict) -> None:
        super().__init__(fixture["body"]["error"]["message"])
        self.response = _Response(fixture["headers"])
        self.body = fixture["body"]


def classify(fixture: dict) -> tuple[str, float | None]:
    return classify_rate_limit(gather_evidence(_RateLimitError(fixture)))


# ------------------------------------------------------------- duration --
@pytest.mark.parametrize(
    "text, expected",
    [
        ("31.535999999s", 31.535999999),
        ("46m4.8s", 2764.8),
        ("585ms", 0.585),
        ("2.1s", 2.1),
        ("1h2m3s", 3723.0),
        ("18m24s", 1104.0),
        ("32", 32.0),
    ],
)
def test_parse_duration_understands_provider_formats(text, expected):
    assert parse_duration(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", None, "garbage", "soon"])
def test_parse_duration_rejects_nonsense(text):
    assert parse_duration(text) is None


# ------------------------------------------------------------ evidence --
def test_evidence_reads_openrouter_metadata():
    evidence = gather_evidence(_RateLimitError(OPENROUTER_DAILY))
    assert evidence.limit_source == "openrouter_free_tier_daily"
    assert evidence.limit == "50"
    assert evidence.remaining == "0"
    assert evidence.reset_at is not None
    # epoch ms 1787616000000 is UTC midnight on 2026-08-25
    assert evidence.reset_at.astimezone(UTC) == datetime(2026, 8, 25, tzinfo=UTC)


def test_evidence_reads_groq_retry_after():
    evidence = gather_evidence(_RateLimitError(GROQ_TPD_SHORT_RECOVERY))
    assert evidence.retry_after == pytest.approx(32.0)
    assert "tokens per day" in evidence.message


def test_evidence_survives_a_response_with_nothing_useful():
    evidence = gather_evidence(_RateLimitError(BARE_429))
    assert evidence.retry_after is None
    assert evidence.reset_at is None
    assert evidence.recovery_seconds() is None


# ------------------------------------------------------ classification --
def test_openrouter_daily_cap_is_exhaustion():
    """A reset 12 hours out is a real outage: bench it."""
    kind, seconds = classify(OPENROUTER_DAILY)
    assert kind == "exhausted"
    assert seconds > 3600


def test_groq_tpd_with_short_recovery_is_transient():
    """Regression: Groq says 'tokens per day' and means 32 seconds.

    Classifying on the words 'per day' benched a working provider until
    midnight. The retry-after header is the truth.
    """
    kind, seconds = classify(GROQ_TPD_SHORT_RECOVERY)
    assert kind == "transient"
    assert seconds == pytest.approx(32.0)


def test_groq_per_minute_is_transient():
    kind, seconds = classify(GROQ_TPM)
    assert kind == "transient"
    assert seconds == pytest.approx(2.0)


def test_openrouter_per_minute_is_transient_despite_per_day_wording():
    """The upsell text says 'per day' on every free-tier 429, daily or not."""
    assert "per day" in OPENROUTER_PER_MINUTE["body"]["error"]["message"]
    kind, seconds = classify(OPENROUTER_PER_MINUTE)
    assert kind == "transient"
    assert seconds == pytest.approx(12.0)


def test_ambiguous_429_is_transient_not_exhaustion():
    """No evidence means sleep briefly, never bench.

    Sleeping on a real outage costs one wasted retry. Benching a healthy
    provider costs every request until the mark expires.
    """
    kind, seconds = classify(BARE_429)
    assert kind == "transient"
    assert seconds is None


def test_threshold_is_the_only_thing_separating_the_two_kinds():
    """Same evidence, different threshold, different verdict."""
    error = _RateLimitError(GROQ_TPD_SHORT_RECOVERY)      # recovers in 32s
    evidence = gather_evidence(error)
    assert classify_rate_limit(evidence, transient_recovery_s=90)[0] == "transient"
    assert classify_rate_limit(evidence, transient_recovery_s=10)[0] == "exhausted"


def test_default_threshold_keeps_a_half_minute_wait_transient():
    assert TRANSIENT_RECOVERY_S >= 30


# ------------------------------------------------ no keyword classification --
@pytest.mark.parametrize(
    "wording",
    [
        "tokens per day (TPD): Limit 200000, Used 200000. Please try again in 5s.",
        "Rate limit exceeded: free-models-per-day. Please try again in 5s.",
        "daily quota exceeded. Please try again in 5s.",
    ],
)
def test_daily_wording_alone_never_benches_a_fast_recovery(wording):
    """The words are not evidence. The recovery time is."""
    fixture = {
        "headers": {"retry-after": "5"},
        "body": {"error": {"message": wording}},
    }
    kind, _seconds = classify(fixture)
    assert kind == "transient", f"benched on wording alone: {wording!r}"


def test_reset_header_alone_is_enough_to_bench():
    """No retry-after, no wording, just a far-future reset."""
    far = datetime.now(UTC) + timedelta(hours=6)
    fixture = {
        "headers": {"x-ratelimit-reset": str(int(far.timestamp() * 1000))},
        "body": {"error": {"message": "Too Many Requests"}},
    }
    kind, seconds = classify(fixture)
    assert kind == "exhausted"
    assert seconds > 3600

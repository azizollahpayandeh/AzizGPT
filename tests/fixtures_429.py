"""Real 429 responses captured from the live APIs on 2026-08-24.

Kept verbatim (only the org id shortened) so the classifier is tested against
what the providers actually send, not against what we assume they send.
"""

# ---------------------------------------------------------------- OpenRouter --
# Captured by bursting the free tier. This is a genuine daily cap: the headers
# carry limit/remaining and a reset timestamp 12 hours out (UTC midnight).
#
# The captured reset was epoch ms 1787616000000 (2026-08-25 00:00 UTC). Pinning
# that literal made the test pass only on the day it was recorded and classify
# as transient forever after, so the value is regenerated relative to now. The
# shape is exactly as captured; only the instant moves.
def _twelve_hours_out_ms() -> str:
    from datetime import UTC, datetime, timedelta

    return str(int((datetime.now(UTC) + timedelta(hours=12)).timestamp() * 1000))
OPENROUTER_DAILY = {
    "headers": {
        "x-ratelimit-limit": "50",
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": _twelve_hours_out_ms(),   # epoch ms, as captured
    },
    "body": {
        "error": {
            "message": "Rate limit exceeded: free-models-per-day. Add 10 credits "
                       "to unlock 1000 free model requests per day",
            "code": 429,
            "metadata": {
                "headers": {
                    "X-RateLimit-Limit": "50",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": _twelve_hours_out_ms(),
                },
                "limit_source": "openrouter_free_tier_daily",
                "remedy_hint": "Wait for the daily reset (see X-RateLimit-Reset), "
                               "or purchase credits to raise your free-model daily limit.",
                "provider_name": None,
            },
        }
    },
}

# Synthesised from OpenRouter's documented per-minute shape: same upsell text
# mentioning "per day", but a short reset and a non-daily limit_source. This is
# the case that must NOT bench the provider.
OPENROUTER_PER_MINUTE = {
    "headers": {
        "x-ratelimit-limit": "20",
        "x-ratelimit-remaining": "0",
        "retry-after": "12",
    },
    "body": {
        "error": {
            "message": "Rate limit exceeded: free-models-per-min. Add 10 credits "
                       "to unlock 1000 free model requests per day",
            "code": 429,
            "metadata": {"limit_source": "openrouter_free_tier_per_minute"},
        }
    },
}

# ---------------------------------------------------------------------- Groq --
# Captured live. Note the trap: Groq calls this "tokens per day (TPD)" but the
# bucket refills continuously and it says to retry in 32 seconds. Benching this
# provider until midnight is exactly the bug that cost hours of availability.
GROQ_TPD_SHORT_RECOVERY = {
    "headers": {
        "retry-after": "32",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7922",
        "x-ratelimit-reset-tokens": "585ms",
        "x-ratelimit-reset-requests": "46m4.8s",
    },
    "body": {
        "error": {
            "message": "Rate limit reached for model `openai/gpt-oss-20b` in "
                       "organization `org_x` service tier `on_demand` on tokens "
                       "per day (TPD): Limit 200000, Used 200000, Requested 73. "
                       "Please try again in 31.535999999s. Need more tokens? "
                       "Upgrade to Dev Tier today at "
                       "https://console.groq.com/settings/billing",
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    },
}

# Groq per-minute, the ordinary transient case.
GROQ_TPM = {
    "headers": {
        "retry-after": "2",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "0",
        "x-ratelimit-reset-tokens": "2.1s",
    },
    "body": {
        "error": {
            "message": "Rate limit reached for model `openai/gpt-oss-20b` in "
                       "organization `org_x` service tier `on_demand` on tokens "
                       "per minute (TPM): Limit 8000, Used 8000, Requested 120. "
                       "Please try again in 2.1s.",
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    },
}

# A 429 with no useful evidence at all: the ambiguous case.
BARE_429 = {"headers": {}, "body": {"error": {"message": "Too Many Requests"}}}

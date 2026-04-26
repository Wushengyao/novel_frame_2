from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


TOKENS_PER_MILLION = 1_000_000
USD = "USD"


@dataclass(frozen=True)
class PricingSource:
    name: str
    url: str
    last_reviewed: str


@dataclass(frozen=True)
class TokenRates:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class TieredTokenRates:
    threshold_prompt_tokens: int
    short_context: TokenRates
    long_context: TokenRates


DEEPSEEK_SOURCE = PricingSource(
    name="DeepSeek Models & Pricing",
    url="https://api-docs.deepseek.com/quick_start/pricing",
    last_reviewed="2026-04-26",
)
GEMINI_SOURCE = PricingSource(
    name="Gemini API Pricing",
    url="https://ai.google.dev/gemini-api/docs/pricing",
    last_reviewed="2026-04-26",
)
XAI_SOURCE = PricingSource(
    name="xAI Models and Pricing",
    url="https://docs.x.ai/developers/models",
    last_reviewed="2026-04-26",
)
LOCAL_SOURCE = PricingSource(
    name="Local inference",
    url="",
    last_reviewed="2026-04-26",
)


DEEPSEEK_PRO_DISCOUNT_END_UTC = datetime(2026, 5, 5, 15, 59, tzinfo=timezone.utc)

DEEPSEEK_FLASH_RATES = TokenRates(
    input_per_million=0.14,
    cached_input_per_million=0.028,
    output_per_million=0.28,
)
DEEPSEEK_PRO_DISCOUNT_RATES = TokenRates(
    input_per_million=0.435,
    cached_input_per_million=0.03625,
    output_per_million=0.87,
)
DEEPSEEK_PRO_STANDARD_RATES = TokenRates(
    input_per_million=1.74,
    cached_input_per_million=0.145,
    output_per_million=3.48,
)

GEMINI_RATES: dict[str, TokenRates | TieredTokenRates] = {
    "gemini-3.1-flash-lite-preview": TokenRates(
        input_per_million=0.25,
        cached_input_per_million=0.025,
        output_per_million=1.50,
    ),
    "gemini-3.1-pro-preview": TieredTokenRates(
        threshold_prompt_tokens=200_000,
        short_context=TokenRates(
            input_per_million=2.00,
            cached_input_per_million=0.20,
            output_per_million=12.00,
        ),
        long_context=TokenRates(
            input_per_million=4.00,
            cached_input_per_million=0.40,
            output_per_million=18.00,
        ),
    ),
    "gemini-2.5-flash": TokenRates(
        input_per_million=0.30,
        cached_input_per_million=0.03,
        output_per_million=2.50,
    ),
    "gemini-2.5-pro": TieredTokenRates(
        threshold_prompt_tokens=200_000,
        short_context=TokenRates(
            input_per_million=1.25,
            cached_input_per_million=0.125,
            output_per_million=10.00,
        ),
        long_context=TokenRates(
            input_per_million=2.50,
            cached_input_per_million=0.25,
            output_per_million=15.00,
        ),
    ),
}

XAI_RATES: dict[str, TieredTokenRates] = {
    "grok-4.20-beta-latest-non-reasoning": TieredTokenRates(
        threshold_prompt_tokens=200_000,
        short_context=TokenRates(
            input_per_million=2.00,
            cached_input_per_million=0.20,
            output_per_million=6.00,
        ),
        long_context=TokenRates(
            input_per_million=4.00,
            cached_input_per_million=0.40,
            output_per_million=12.00,
        ),
    ),
}


MODEL_ALIASES = {
    ("deepseek", "deepseek-chat"): "deepseek-v4-flash",
    ("deepseek", "deepseek-reasoner"): "deepseek-v4-flash",
    ("gemini", "gemini-3.1-pro"): "gemini-3.1-pro-preview",
    ("grok", "grok-4.20-non-reasoning"): "grok-4.20-beta-latest-non-reasoning",
    ("grok", "grok-4.20-non-reasoning-latest"): "grok-4.20-beta-latest-non-reasoning",
    ("grok", "grok-4.20-beta-non-reasoning"): "grok-4.20-beta-latest-non-reasoning",
}


def _normalize_provider(provider: object) -> str:
    return str(provider or "openai_compatible").strip().lower()


def _normalize_model(model: object) -> str:
    normalized = str(model or "").strip().lower()
    if normalized.startswith("models/"):
        normalized = normalized.removeprefix("models/")
    return normalized


def _normalize_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _usage_int(usage: dict[str, Any] | None, key: str) -> int:
    if not usage:
        return 0
    try:
        return max(0, int(usage.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _token_counts(usage: dict[str, Any] | None) -> dict[str, int]:
    prompt_tokens = _usage_int(usage, "prompt_tokens")
    completion_tokens = _usage_int(usage, "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens") or prompt_tokens + completion_tokens
    cached_tokens = min(_usage_int(usage, "cached_tokens"), prompt_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "billable_input_tokens": max(0, prompt_tokens - cached_tokens),
    }


def _rates_for_context(rates: TokenRates | TieredTokenRates, prompt_tokens: int) -> TokenRates:
    if isinstance(rates, TokenRates):
        return rates
    if prompt_tokens <= rates.threshold_prompt_tokens:
        return rates.short_context
    return rates.long_context


def _source_dict(source: PricingSource) -> dict[str, str]:
    return {
        "name": source.name,
        "url": source.url,
        "last_reviewed": source.last_reviewed,
    }


def _unpriced(provider: str, model: str, usage: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    counts = _token_counts(usage)
    return {
        "currency": USD,
        "provider": provider,
        "model": model,
        "pricing_status": "unpriced",
        "estimated_cost_usd": 0.0,
        "priced_tokens": 0,
        "unpriced_tokens": counts["total_tokens"],
        "reason": reason,
        "source": {},
    }


def _priced(
    provider: str,
    model: str,
    usage: dict[str, Any] | None,
    rates: TokenRates,
    source: PricingSource,
    *,
    status: str = "priced",
    reason: str = "",
) -> dict[str, Any]:
    counts = _token_counts(usage)
    input_cost = counts["billable_input_tokens"] * rates.input_per_million / TOKENS_PER_MILLION
    cached_input_cost = counts["cached_tokens"] * rates.cached_input_per_million / TOKENS_PER_MILLION
    output_cost = counts["completion_tokens"] * rates.output_per_million / TOKENS_PER_MILLION
    return {
        "currency": USD,
        "provider": provider,
        "model": model,
        "pricing_status": status,
        "estimated_cost_usd": input_cost + cached_input_cost + output_cost,
        "priced_tokens": counts["total_tokens"],
        "unpriced_tokens": 0,
        "reason": reason,
        "source": _source_dict(source),
        "rates_per_million": {
            "input": rates.input_per_million,
            "cached_input": rates.cached_input_per_million,
            "output": rates.output_per_million,
        },
        "cost_breakdown_usd": {
            "input": input_cost,
            "cached_input": cached_input_cost,
            "output": output_cost,
        },
    }


def estimate_llm_cost(
    provider: object,
    model: object,
    usage: dict[str, Any] | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Estimate API cost for a single LLM response using the built-in USD catalog."""
    normalized_provider = _normalize_provider(provider)
    normalized_model = _normalize_model(model)
    canonical_model = MODEL_ALIASES.get(
        (normalized_provider, normalized_model),
        normalized_model,
    )
    counts = _token_counts(usage)

    if normalized_provider in {"ollama", "llama_cpp"}:
        return _priced(
            normalized_provider,
            canonical_model,
            usage,
            TokenRates(0.0, 0.0, 0.0),
            LOCAL_SOURCE,
            status="local",
            reason=f"Local {normalized_provider} inference; API cost is not estimated.",
        )

    if normalized_provider == "deepseek":
        if canonical_model == "deepseek-v4-flash":
            return _priced(normalized_provider, canonical_model, usage, DEEPSEEK_FLASH_RATES, DEEPSEEK_SOURCE)
        if canonical_model == "deepseek-v4-pro":
            effective_now = _normalize_datetime(now)
            rates = (
                DEEPSEEK_PRO_DISCOUNT_RATES
                if effective_now <= DEEPSEEK_PRO_DISCOUNT_END_UTC
                else DEEPSEEK_PRO_STANDARD_RATES
            )
            reason = (
                "Limited-time DeepSeek discount applied."
                if effective_now <= DEEPSEEK_PRO_DISCOUNT_END_UTC
                else "DeepSeek standard price applied after the limited-time discount window."
            )
            return _priced(
                normalized_provider,
                canonical_model,
                usage,
                rates,
                DEEPSEEK_SOURCE,
                reason=reason,
            )
        return _unpriced(normalized_provider, canonical_model, usage, "No built-in USD price for this DeepSeek model.")

    if normalized_provider == "gemini":
        rates = GEMINI_RATES.get(canonical_model)
        if rates:
            return _priced(
                normalized_provider,
                canonical_model,
                usage,
                _rates_for_context(rates, counts["prompt_tokens"]),
                GEMINI_SOURCE,
            )
        return _unpriced(normalized_provider, canonical_model, usage, "No built-in USD price for this Gemini model.")

    if normalized_provider == "grok":
        rates = XAI_RATES.get(canonical_model)
        if rates:
            return _priced(
                normalized_provider,
                canonical_model,
                usage,
                _rates_for_context(rates, counts["prompt_tokens"]),
                XAI_SOURCE,
            )
        return _unpriced(normalized_provider, canonical_model, usage, "No built-in USD price for this xAI model.")

    if normalized_provider == "doubao":
        return _unpriced(
            normalized_provider,
            canonical_model,
            usage,
            "Volcengine Ark pricing is not listed in USD; no currency conversion is applied.",
        )

    return _unpriced(
        normalized_provider,
        canonical_model,
        usage,
        "No official USD catalog is available for custom OpenAI-compatible models.",
    )

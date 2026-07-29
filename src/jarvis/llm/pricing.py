# =============================================================================
# src/jarvis/llm/pricing.py - what did that call cost?
# =============================================================================
#
# One table (USD per million tokens) and one function. Every model call
# computes its cost here before being traced. This is the foundation for
# "what did you cost me this week?" and for the later budget guard - both
# only work if cost is captured per call from the very first call.
#
# The table is config-in-code: update it when the provider's pricing page
# changes. Prices per MILLION tokens, split by how the provider bills:
#   input        - fresh input tokens
#   cached_input - input tokens served from the provider's prompt cache
#                  (much cheaper; our stable persona prompt exploits this)
#   output       - generated tokens
#
# An UNKNOWN model must not crash the call: the money is already spent, so
# refusing to record the trace would only lose information. We record cost
# 0 with a loud warning instead - fix the table, lose nothing else.
#
# INR conversion is a convenience estimate for display (a fixed configured
# rate, not a live FX feed); USD is the accounting truth.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from jarvis.common.log import get_logger

log = get_logger("llm.pricing")

# Verify against the provider's current pricing page when updating models.
_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "cached_input": 0.10,
        "output": 5.00,
    },
}

# Display convenience only; edit when far off. Accounting truth is USD.
USD_TO_INR = 84.0


@dataclass(frozen=True)
class CallCost:
    """The cost of one model call, ready for the trace row."""

    usd: float
    inr: float

    @property
    def rounded_usd(self) -> float:
        return round(self.usd, 6)

    @property
    def rounded_inr(self) -> float:
        return round(self.inr, 4)


def compute_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
    cached_tokens: int = 0,
) -> CallCost:
    """Price one call. cached_tokens is the portion of tokens_in that was
    served from the provider's prompt cache at the cheaper rate."""
    prices = _USD_PER_MTOK.get(model)
    if prices is None:
        log.warning(
            "no pricing for model - recording zero cost, fix the table",
            extra={"model": model},
        )
        return CallCost(usd=0.0, inr=0.0)

    fresh_in = max(tokens_in - cached_tokens, 0)
    usd = (
        fresh_in * prices["input"]
        + cached_tokens * prices["cached_input"]
        + tokens_out * prices["output"]
    ) / 1_000_000
    return CallCost(usd=usd, inr=usd * USD_TO_INR)


def known_models() -> tuple[str, ...]:
    """Models the price table covers - used by tests to catch a settings
    default pointing at an unpriced model."""
    return tuple(sorted(_USD_PER_MTOK))
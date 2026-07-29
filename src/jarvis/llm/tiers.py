# =============================================================================
# src/jarvis/llm/tiers.py - model tiers: code names the need, config names
# the model
# =============================================================================
#
# Nothing outside the llm/ package ever mentions a concrete model name.
# Code asks for a TIER - a quality of thinking:
#
#   reasoner - the orchestrator and subagent loops (smart, expensive)
#   utility  - summaries, extraction, drafting chores (fast, cheap)
#   local    - offloaded to a worker's local model (arrives with workers)
#   embedder - vector embeddings for memory (arrives with the memory phase)
#
# The tier -> model mapping comes from settings, so upgrading a model is
# an .env edit plus restart. The two not-yet-available tiers are declared
# so the vocabulary is complete, but resolving them fails with a clear
# message instead of a confusing crash later.
# =============================================================================

from __future__ import annotations

from enum import StrEnum

from jarvis.common.settings import CoreSettings


class Tier(StrEnum):
    REASONER = "reasoner"
    UTILITY = "utility"
    LOCAL = "local"        # phase 2+: worker-hosted model
    EMBEDDER = "embedder"  # phase 3: memory vectors


class TierNotAvailable(RuntimeError):
    """Asked for a tier that has no model behind it in this deployment."""


def resolve_model(tier: Tier, settings: CoreSettings) -> str:
    """Tier -> concrete model name. The only translation point in the
    system; if you are writing a model name anywhere else, stop."""
    match tier:
        case Tier.REASONER:
            return settings.model_reasoner
        case Tier.UTILITY:
            return settings.model_utility
        case Tier.LOCAL | Tier.EMBEDDER:
            raise TierNotAvailable(
                f"tier {tier} is not available yet - it arrives with "
                f"workers (local) and the memory phase (embedder)"
            )
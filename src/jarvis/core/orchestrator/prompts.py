# =============================================================================
# src/jarvis/core/orchestrator/prompts.py - the persona, versioned as code
# =============================================================================
#
# Prompts are code: reviewed, versioned, changed deliberately. Behaviour
# tuning creates PERSONA_V2 alongside V1 - never edits V1 - so behaviour
# changes stay visible in git history and comparable in later evals.
#
# Deliberate omissions from v1:
#   - No tool list in prose: tool contracts travel separately through the
#     API; describing them twice invites the two copies to disagree.
#   - No claims about jobs, memory, or workers: those abilities do not
#     exist yet, and a prompt promising absent abilities produces
#     confident nonsense. Each phase that adds an ability amends the
#     prompt in the same change.
#
# assemble_system_prompt() is the single composer: persona + whatever
# context sections exist (profile document and rolling summary arrive in
# the memory phase; empty sections are simply skipped). Stable text first,
# variable text last - the provider caches the stable prefix, and cache
# hits are billed at a tenth of the price.
# =============================================================================

from __future__ import annotations

PERSONA_V1 = """\
You are JARVIS, the personal AI of exactly one person: the owner. You are
not a product and not a general assistant; you are his.

Voice:
- First person. Address the owner as "sir".
- Sharp, concise, dry wit. Precision first, humour second, never at the
  cost of clarity.
- No hollow openers ("Certainly!", "Great question!"). Begin with substance.
- Plain text only: no markdown headings, no bullet lists unless the owner
  asks for a list, no emoji. Your words may be spoken aloud by a voice
  system, so write text that sounds right when read out.
- Brevity is respect. Trivial questions get one-line answers. Depth is for
  questions that deserve it.

Conduct:
- Answer directly from knowledge when you can. Use a tool only when it
  genuinely helps answer the question at hand.
- Report honestly. If something failed, say it failed and why. Never gloss,
  never invent. If you do not know, say so.
- If the owner asks for something beyond your current abilities, say
  plainly that you cannot do it yet rather than pretending.

You run as a persistent system, not a chat page: conversations resume,
and more of the owner's world becomes visible to you as the system grows.
"""


def assemble_system_prompt(
    persona: str = PERSONA_V1,
    profile_doc: str = "",
    rolling_summary: str = "",
) -> str:
    """Compose the full system prompt from persona plus whatever context
    sections currently exist. Empty sections are skipped, not sent blank.

    Ordering matters for money: the provider caches the stable PREFIX of
    the prompt, so unchanging text (persona) comes first and the parts
    that vary per session come last.
    """
    sections = [persona]
    if profile_doc.strip():
        sections.append("About the owner:\n" + profile_doc.strip())
    if rolling_summary.strip():
        sections.append(
            "Summary of this conversation so far:\n" + rolling_summary.strip()
        )
    return "\n\n".join(sections)
# =============================================================================
# src/jarvis/common/capabilities.py - capability tags and gate names
# =============================================================================
#
# Two vocabularies that must not drift, so both live here as constants:
#
#   Capability - what a machine can DO. Workers advertise these; job
#     types require them; the dispatcher matches the two. A tag is a
#     promise about the environment, not about permission.
#
#   Gate - a NAMED CLASS OF RISKY ACTION. Gates are about permission,
#     and they pause work to ask the owner. Deliberately few: a long list
#     of gates is a list nobody reads, and approval fatigue is how an
#     approval system stops working.
#
# Both are closed vocabularies. A typo in a constant name is an error at
# import; a typo in a string literal is a silent hole in a safety check.
# =============================================================================

from __future__ import annotations

from enum import StrEnum


class Capability(StrEnum):
    """What a worker can do. Advertised by workers, required by jobs."""

    MACOS = "macos"                  # macOS-specific control
    LINUX = "linux"
    BROWSER = "browser"              # a real browser (Playwright)
    SANDBOX_EXEC = "sandbox-exec"    # isolated code execution
    RENDER = "render"                # HTML/template to image or PDF
    LOCAL_LLM = "local-llm"          # a local model server
    STT_LOCAL = "stt-local"
    TTS_LOCAL = "tts-local"
    GPU = "gpu"


class Gate(StrEnum):
    """A class of action that pauses for the owner's say-so."""

    SPEND = "spend"                          # real money, or a costly job
    OUTBOUND = "outbound"                    # anything sent to a third party
                                             # AS the owner: email, message,
                                             # form submission, post
    PUBLISH = "publish"                      # making something public
    DESTRUCTIVE_FILE = "destructive-file"    # delete or overwrite outside
                                             # sandbox scopes
    CREDENTIAL = "credential"                # logging in, using stored secrets
    SYSTEM = "system"                        # OS state changes: installs,
                                             # configuration edits


class GateMode(StrEnum):
    """How much autonomy a gate has been GRANTED, per actor and tool.

    Autonomy is earned, and promotion is always a manual decision by the
    owner - informed by the gate's own approval history in the traces.
    Nothing in the system promotes itself.
    """

    ASK_ALWAYS = "ask_always"            # the default for everything
    ASK_FIRST_TIME = "ask_first_time"    # approve once, then remembered
    AUTO_WITH_REVIEW = "auto_with_review"  # proceeds, lands in a digest
    AUTO = "auto"                        # proceeds silently
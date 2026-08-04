# =============================================================================
# src/jarvis/agentloop/policies.py - who may use what
# =============================================================================
#
# The allowlists and gate rules, in one readable place. This file is the
# answer to "what is JARVIS allowed to do?" - if it is not here, it does
# not happen.
#
# Most tools listed below do not exist yet; their policies are written
# now so that the rule arrives BEFORE the capability, never after. A tool
# that appears without a matching allowlist entry is simply denied, which
# is the correct failure for a half-finished feature.
#
# Note what the orchestrator is NOT allowed: no browser, no shell, no
# filesystem. The front mind talks and delegates; hands belong to
# subagents running inside jobs, where gates and approvals apply.
# =============================================================================

from __future__ import annotations

from jarvis.agentloop.guard import ActorPolicy, GateRule, Guard
from jarvis.common.capabilities import Gate

# -- actors -------------------------------------------------------------------

ORCHESTRATOR = "core.orchestrator"
RESEARCHER = "subagent.researcher"
OPERATOR = "subagent.operator"
ENGINEER = "subagent.engineer"
WRITER = "subagent.writer"
ARCHIVIST = "subagent.archivist"


DEFAULT_POLICIES: dict[str, ActorPolicy] = {
    ORCHESTRATOR: ActorPolicy(
        actor=ORCHESTRATOR,
        tools=(
            "get_time_context",
            "memory_search", "memory_store",
            "list_jobs", "get_job", "cancel_job",
            "run_subagent", "enqueue_job",
            "schedule", "list_schedules", "cancel_schedule",
        ),
        description=(
            "The front mind. Talks, remembers, and delegates. No hands: "
            "no browser, no shell, no filesystem - those live in "
            "subagents, inside jobs, behind gates."
        ),
    ),
    RESEARCHER: ActorPolicy(
        actor=RESEARCHER,
        tools=("web_search", "web_fetch", "artifact_write"),
        description="ATHENA. Reads the web, writes documents. Reads only.",
    ),
    OPERATOR: ActorPolicy(
        actor=OPERATOR,
        tools=(
            "browser_*", "platform_*", "artifact_*",
            # MCP tools arrive namespaced by server, so a policy can
            # grant one server's whole surface without listing tools
            # that did not exist when the policy was written.
            "mcp__playwright__*", "mcp__platform__*",
        ),
        description=(
            "PROTEUS. Drives a real browser and controls devices - the "
            "actor with the most gated actions, by a distance."
        ),
    ),
    ENGINEER: ActorPolicy(
        actor=ENGINEER,
        tools=(
            "sandbox_*", "file_*", "git_*", "artifact_*",
            # file_read/write/edit/list/tree and the project tools all
            # match the globs above; listed here for readability rather
            # than because the patterns need widening.
            "mcp__filesystem__*", "mcp__github__*",
        ),
        description=(
            "DAEDALUS. Writes and runs code inside a jail. File access is "
            "scoped to the sandbox; anything outside it is gated."
        ),
    ),
    WRITER: ActorPolicy(
        actor=WRITER,
        tools=("artifact_*", "render_*"),
        description=(
            "CALLIOPE. Reads source material and produces documents. "
            "Reads and writes artifacts only - no browser, no shell, no "
            "network. The narrowest surface of any subagent, because "
            "writing needs nothing else."
        ),
    ),
    ARCHIVIST: ActorPolicy(
        actor=ARCHIVIST,
        tools=("memory_*",),
        description=(
            "MNEMOSYNE. The only actor permitted privileged memory "
            "operations: merge, supersede, expire."
        ),
    ),
}


# -- gate rules ---------------------------------------------------------------
#
# Ordered by how much damage the action can do. The first matching rule
# wins, so the more specific and more serious rules come first.

DEFAULT_GATE_RULES: tuple[GateRule, ...] = (
    # Sending things into the world as the owner. The single most
    # consequential class of action: it cannot be taken back.
    GateRule(
        gate=Gate.OUTBOUND,
        tool_pattern="browser_submit*",
        reason="submitting a form on the owner's behalf",
    ),
    GateRule(
        gate=Gate.OUTBOUND,
        tool_pattern="*send_email*",
        reason="sending email as the owner",
    ),
    GateRule(
        gate=Gate.OUTBOUND,
        tool_pattern="*post_message*",
        reason="posting a message as the owner",
    ),

    # Making something public. Both of these leave the machine and
    # cannot be taken back, which is exactly what a gate is for.
    GateRule(
        gate=Gate.PUBLISH,
        tool_pattern="git_push",
        reason="pushing commits to GitHub",
    ),
    GateRule(
        gate=Gate.PUBLISH,
        tool_pattern="git_create_repo",
        reason="creating a new repository on GitHub",
    ),

    # Using stored credentials or logging in anywhere.
    GateRule(
        gate=Gate.CREDENTIAL,
        tool_pattern="browser_*",
        arg_patterns={"url": "*login*"},
        reason="logging into an account",
    ),
    GateRule(
        gate=Gate.CREDENTIAL,
        tool_pattern="*credential*",
        reason="using a stored credential",
    ),

    # Destroying data outside the areas agents own. The sandbox and
    # workspace are theirs; everything else is the owner's.
    GateRule(
        gate=Gate.DESTRUCTIVE_FILE,
        tool_pattern="file_delete*",
        reason="deleting a file",
    ),
    GateRule(
        gate=Gate.DESTRUCTIVE_FILE,
        tool_pattern="file_write*",
        arg_patterns={"path": "/*"},          # absolute path = outside scope
        reason="writing outside the sandbox",
    ),

    # Changing the machine itself.
    GateRule(
        gate=Gate.SYSTEM,
        tool_pattern="platform_install*",
        reason="installing software",
    ),
    GateRule(
        gate=Gate.SYSTEM,
        tool_pattern="sandbox_exec",
        arg_patterns={"command": "*sudo*"},
        reason="running a command with elevated privileges",
    ),

    # Spending money.
    GateRule(
        gate=Gate.SPEND,
        tool_pattern="*purchase*",
        reason="spending real money",
    ),
)


def create_guard() -> Guard:
    """Build the guard with the standing policy.

    No gate_modes are passed, so every gate is ask_always. Promotion to
    greater autonomy is a deliberate, manual act by the owner, informed
    by that gate's approval history - never something the system grants
    itself.
    """
    return Guard(policies=DEFAULT_POLICIES, gate_rules=DEFAULT_GATE_RULES)
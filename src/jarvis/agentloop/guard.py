# =============================================================================
# src/jarvis/agentloop/guard.py - the one door every tool call passes
# =============================================================================
#
# Built BEFORE the dangerous tools exist. A guard added afterwards means
# auditing every call site and hoping none were missed; a guard built
# first means every future tool inherits the check by construction.
#
# One question, three answers:
#
#   ALLOW - proceed
#   DENY  - refuse permanently. The agent is told why and routes around
#           it; nothing is asked of the owner, because the answer would
#           always be no.
#   GATE  - pause the work and ask the owner.
#
# TWO REASONS THIS IS CODE AND NOT PROMPT TEXT:
#
#   1. A prompt saying "only use these tools" is a REQUEST. Text fetched
#      from the web - which agents read constantly - can argue a model
#      into trying anything. It cannot argue this function into
#      returning ALLOW.
#   2. Deny-by-default only means something if the list is consulted
#      mechanically. An allowlist a model is trusted to honour is a
#      suggestion with extra steps.
#
# Gate RULES are declarative: actor, tool, and optional argument
# patterns. Adding a rule is a line of configuration, not new branching
# scattered through handlers.
# =============================================================================

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.common.capabilities import Gate, GateMode
from jarvis.common.log import get_logger

log = get_logger("agentloop.guard")


class Verdict(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    GATE = "gate"


@dataclass(frozen=True)
class Decision:
    """What the guard concluded, and why - the 'why' is not decoration:
    it is what the agent reads on a denial and what the owner reads on a
    gate."""

    verdict: Verdict
    reason: str
    gate: Gate | None = None
    mode: GateMode | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW


@dataclass(frozen=True)
class GateRule:
    """When this tool, used by this actor, with arguments matching these
    patterns, raise this gate.

    arg_patterns maps an argument name to a glob. All listed patterns
    must match for the rule to fire, so a rule can be as broad as "any
    use of this tool" or as narrow as "writes to paths outside the
    sandbox".
    """

    gate: Gate
    tool_pattern: str                                  # glob, e.g. "browser_*"
    actor_pattern: str = "*"                           # glob over actor names
    arg_patterns: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    def matches(self, actor: str, tool: str, args: dict[str, Any]) -> bool:
        if not fnmatch.fnmatch(actor, self.actor_pattern):
            return False
        if not fnmatch.fnmatch(tool, self.tool_pattern):
            return False
        for name, pattern in self.arg_patterns.items():
            value = args.get(name)
            if value is None or not fnmatch.fnmatch(str(value), pattern):
                return False
        return True


@dataclass(frozen=True)
class ActorPolicy:
    """What one agent may use.

    tools is an ALLOWLIST of globs. Anything unmatched is denied - the
    default has to be no, because the set of tools that exist grows
    faster than any denylist could be maintained.
    """

    actor: str
    tools: tuple[str, ...]
    description: str = ""

    def permits(self, tool: str) -> bool:
        return any(fnmatch.fnmatch(tool, pattern) for pattern in self.tools)


class Guard:
    """The single authority on whether a tool call may proceed."""

    def __init__(
        self,
        policies: dict[str, ActorPolicy],
        gate_rules: tuple[GateRule, ...],
        gate_modes: dict[tuple[str, Gate], GateMode] | None = None,
    ) -> None:
        self._policies = policies
        self._rules = gate_rules
        # (actor, gate) -> mode. Absent means ask_always: an unconfigured
        # combination is the one most likely to be unconsidered, so it
        # gets the most cautious treatment.
        self._modes = gate_modes or {}

    def check(self, actor: str, tool: str, args: dict[str, Any]) -> Decision:
        """The one question. Called for every tool call from every agent."""
        policy = self._policies.get(actor)
        if policy is None:
            # An unknown actor is a wiring mistake, and the safe response
            # to a wiring mistake is nothing at all.
            log.warning("tool call from unknown actor - denied", extra={
                "actor": actor, "tool": tool,
            })
            return Decision(
                Verdict.DENY,
                f"actor {actor!r} has no policy; nothing is permitted",
            )

        if not policy.permits(tool):
            log.info("tool not on actor allowlist - denied", extra={
                "actor": actor, "tool": tool,
            })
            return Decision(
                Verdict.DENY,
                f"{actor} is not permitted to use {tool}",
            )

        for rule in self._rules:
            if not rule.matches(actor, tool, args):
                continue
            mode = self._modes.get((actor, rule.gate), GateMode.ASK_ALWAYS)
            if mode is GateMode.AUTO:
                log.info("gated action auto-approved by config", extra={
                    "actor": actor, "tool": tool, "gate": rule.gate.value,
                })
                return Decision(
                    Verdict.ALLOW,
                    f"{rule.gate.value} gate set to auto for {actor}",
                    gate=rule.gate, mode=mode,
                )
            if mode is GateMode.AUTO_WITH_REVIEW:
                log.info("gated action proceeding, flagged for review", extra={
                    "actor": actor, "tool": tool, "gate": rule.gate.value,
                })
                return Decision(
                    Verdict.ALLOW,
                    f"{rule.gate.value} proceeding under review for {actor}",
                    gate=rule.gate, mode=mode,
                )
            # ask_always and ask_first_time both stop here; distinguishing
            # them requires approval history, which the approvals layer
            # owns rather than the guard.
            return Decision(
                Verdict.GATE,
                rule.reason or f"{tool} requires {rule.gate.value} approval",
                gate=rule.gate, mode=mode,
            )

        return Decision(Verdict.ALLOW, "permitted")

    def policy_for(self, actor: str) -> ActorPolicy | None:
        return self._policies.get(actor)

    def known_actors(self) -> tuple[str, ...]:
        return tuple(sorted(self._policies))
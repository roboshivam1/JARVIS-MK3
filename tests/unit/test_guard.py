# =============================================================================
# tests/unit/test_guard.py - the door
# =============================================================================
#
# The guard is the thing standing between a confused (or manipulated)
# model and the owner's actual life, so its behaviour is pinned tightly:
# deny by default, allowlists enforced mechanically, gates raised on the
# right patterns, and no accidental self-promotion to autonomy.
# =============================================================================

from __future__ import annotations

from jarvis.agentloop.guard import ActorPolicy, Gate, GateRule, Guard, Verdict
from jarvis.agentloop.policies import (
    ENGINEER,
    OPERATOR,
    ORCHESTRATOR,
    RESEARCHER,
    create_guard,
)
from jarvis.common.capabilities import GateMode


class TestDenyByDefault:
    def test_unknown_actor_gets_nothing(self) -> None:
        guard = create_guard()
        decision = guard.check("subagent.impostor", "get_time_context", {})
        assert decision.verdict is Verdict.DENY

    def test_tool_absent_from_allowlist_is_denied(self) -> None:
        # The orchestrator talks and delegates; it has no hands.
        guard = create_guard()
        for forbidden in ("browser_navigate", "sandbox_exec", "file_delete"):
            assert guard.check(ORCHESTRATOR, forbidden, {}).verdict is Verdict.DENY

    def test_listed_tool_is_allowed(self) -> None:
        guard = create_guard()
        assert guard.check(ORCHESTRATOR, "memory_search", {}).allowed

    def test_glob_patterns_work(self) -> None:
        guard = create_guard()
        # ENGINEER's allowlist includes sandbox_*
        assert guard.check(ENGINEER, "sandbox_run", {}).allowed
        assert not guard.check(ENGINEER, "browser_navigate", {}).allowed


class TestGates:
    def test_form_submission_is_gated(self) -> None:
        guard = create_guard()
        decision = guard.check(OPERATOR, "browser_submit_form", {"url": "x"})
        assert decision.verdict is Verdict.GATE
        assert decision.gate is Gate.OUTBOUND

    def test_login_url_triggers_credential_gate(self) -> None:
        guard = create_guard()
        decision = guard.check(
            OPERATOR, "browser_navigate", {"url": "https://site.com/login"}
        )
        assert decision.verdict is Verdict.GATE
        assert decision.gate is Gate.CREDENTIAL

    def test_ordinary_navigation_is_not_gated(self) -> None:
        guard = create_guard()
        decision = guard.check(
            OPERATOR, "browser_navigate", {"url": "https://site.com/about"}
        )
        assert decision.allowed

    def test_absolute_path_write_is_gated(self) -> None:
        guard = create_guard()
        decision = guard.check(ENGINEER, "file_write", {"path": "/etc/hosts"})
        assert decision.verdict is Verdict.GATE
        assert decision.gate is Gate.DESTRUCTIVE_FILE

    def test_sandbox_relative_write_is_not_gated(self) -> None:
        guard = create_guard()
        assert guard.check(ENGINEER, "file_write", {"path": "out/result.txt"}).allowed

    def test_researcher_has_no_outward_reach(self) -> None:
        # ATHENA reads the web; she cannot act on it.
        guard = create_guard()
        assert not guard.check(RESEARCHER, "browser_submit_form", {}).allowed


class TestGateModes:
    def test_everything_ships_asking(self) -> None:
        # No mode configuration at all: every gate must still stop.
        guard = create_guard()
        decision = guard.check(OPERATOR, "browser_submit_form", {"url": "x"})
        assert decision.verdict is Verdict.GATE
        assert decision.mode is GateMode.ASK_ALWAYS

    def test_auto_mode_permits_but_records_the_gate(self) -> None:
        guard = Guard(
            policies={"test.actor": ActorPolicy("test.actor", ("send_*",))},
            gate_rules=(GateRule(gate=Gate.OUTBOUND, tool_pattern="send_*"),),
            gate_modes={("test.actor", Gate.OUTBOUND): GateMode.AUTO},
        )
        decision = guard.check("test.actor", "send_email", {})
        assert decision.allowed
        assert decision.gate is Gate.OUTBOUND   # still attributed

    def test_auto_with_review_permits(self) -> None:
        guard = Guard(
            policies={"test.actor": ActorPolicy("test.actor", ("send_*",))},
            gate_rules=(GateRule(gate=Gate.OUTBOUND, tool_pattern="send_*"),),
            gate_modes={("test.actor", Gate.OUTBOUND): GateMode.AUTO_WITH_REVIEW},
        )
        assert guard.check("test.actor", "send_email", {}).allowed

    def test_mode_is_per_actor(self) -> None:
        # Granting one agent autonomy must not grant it to another.
        guard = Guard(
            policies={
                "a": ActorPolicy("a", ("send_*",)),
                "b": ActorPolicy("b", ("send_*",)),
            },
            gate_rules=(GateRule(gate=Gate.OUTBOUND, tool_pattern="send_*"),),
            gate_modes={("a", Gate.OUTBOUND): GateMode.AUTO},
        )
        assert guard.check("a", "send_email", {}).allowed
        assert guard.check("b", "send_email", {}).verdict is Verdict.GATE


class TestDecisionsCarryReasons:
    def test_denial_explains_itself(self) -> None:
        # The agent reads this text and routes around the refusal, so an
        # empty reason is a usability bug as well as a debugging one.
        guard = create_guard()
        decision = guard.check(ORCHESTRATOR, "sandbox_exec", {})
        assert decision.reason and "sandbox_exec" in decision.reason

    def test_gate_explains_itself(self) -> None:
        guard = create_guard()
        decision = guard.check(OPERATOR, "browser_submit_form", {"url": "x"})
        assert "form" in decision.reason.lower()
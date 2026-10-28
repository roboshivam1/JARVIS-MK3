# =============================================================================
# tests/unit/test_orchestrator_allowlist.py - no tool JARVIS cannot use
# =============================================================================
#
# Every tool the orchestrator registers must be on its allowlist. A tool
# left off is denied at call time, and JARVIS then reports it to the owner
# as a "permission issue" rather than a bug - which has now happened
# twice (watchers, then projects and the life log). This makes the third
# time a failing test instead.
# =============================================================================

from __future__ import annotations

import re
from pathlib import Path

from jarvis.agentloop.policies import DEFAULT_POLICIES, ORCHESTRATOR

AGENT = Path(__file__).resolve().parents[2] / "src/jarvis/core/orchestrator/agent.py"


def test_every_registered_tool_is_allowed() -> None:
    source = AGENT.read_text()
    registered = set(re.findall(r'InlineTool\(\s*name="([a-z_]+)"', source))
    assert {"project_lookup", "life_report", "people_lookup"} <= registered
    allowed = set(DEFAULT_POLICIES[ORCHESTRATOR].tools)
    assert registered - allowed == set()

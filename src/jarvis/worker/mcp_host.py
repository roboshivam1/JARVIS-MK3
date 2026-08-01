# =============================================================================
# src/jarvis/worker/mcp_host.py - which MCP servers this machine runs
# =============================================================================
#
# Server configuration is a FILE, not code: adding a capability to a
# machine should be an edit and a restart, not a code change and a
# redeploy. The format mirrors the mcp.json convention other MCP clients
# use, so servers documented anywhere can be pasted in.
#
#   {
#     "mcpServers": {
#       "playwright": {
#         "command": "npx",
#         "args": ["-y", "@playwright/mcp@latest", "--headless"]
#       }
#     }
#   }
#
# Absent file = no servers, which is the correct default: a worker with
# no MCP configuration is a worker with no MCP tools, not a broken one.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path

from jarvis.agentloop.mcp_client import McpServerConfig
from jarvis.common.log import get_logger

log = get_logger("worker.mcp_host")

DEFAULT_CONFIG_PATH = Path("mcp.json")


def load_mcp_configs(path: Path | None = None) -> list[McpServerConfig]:
    """Read server definitions. A missing or malformed file yields no
    servers and a log line - never a failed startup."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        log.info("no mcp config - running without mcp tools", extra={
            "path": str(config_path),
        })
        return []

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        log.error("mcp config is not valid json - ignoring it",
                  exc_info=True, extra={"path": str(config_path)})
        return []

    configs: list[McpServerConfig] = []
    for name, entry in raw.get("mcpServers", {}).items():
        command = entry.get("command")
        if not command:
            log.warning("mcp server entry has no command - skipped",
                        extra={"server": name})
            continue
        configs.append(McpServerConfig(
            name=name,
            command=str(command),
            args=[str(a) for a in entry.get("args", [])],
            env={str(k): str(v) for k, v in entry.get("env", {}).items()},
        ))

    log.info("mcp config loaded", extra={
        "servers": [c.name for c in configs], "path": str(config_path),
    })
    return configs
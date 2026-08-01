# =============================================================================
# src/jarvis/agentloop/mcp_client.py - speaking to MCP servers
# =============================================================================
#
# An MCP server is a program launched as a subprocess that describes its
# own tools and runs them on request. The protocol is JSON-RPC over the
# process's stdin/stdout:
#
#   "tools/list" -> [{name, description, inputSchema}, ...]
#   "tools/call" -> the result of running one
#
# That first response is the same shape our Toolset already holds - name,
# description, JSON schema - which is why MCP slots in rather than
# needing a parallel tool system.
#
# WHY WRITE THE CLIENT RATHER THAN USE THE SDK: the protocol surface we
# need is two methods, and this keeps the dependency count at zero for
# something that sits on the path of every tool call. If we later need
# resources, prompts, or sampling, the official SDK becomes the right
# call and this file is what it replaces.
#
# A server that dies takes its tools with it but must NEVER take the job
# with it: failures come back as error text the model reads and routes
# around, exactly like any other tool failure.
# =============================================================================

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from jarvis.common.log import get_logger

log = get_logger("agentloop.mcp")

_REQUEST_TIMEOUT_S = 60.0
_STARTUP_TIMEOUT_S = 30.0
PROTOCOL_VERSION = "2024-11-05"


@dataclass(frozen=True)
class McpTool:
    """One tool offered by a server."""

    server: str
    name: str                       # the server's own name for it
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        """Namespaced so two servers can both offer a "read" tool, and so
        guard policies can say mcp__playwright__* and mean it."""
        return f"mcp__{self.server}__{self.name}"


@dataclass
class McpServerConfig:
    """How to launch one server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


class McpServer:
    """A running MCP server subprocess and the conversation with it."""

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._lock = asyncio.Lock()     # one request at a time per server
        self.tools: list[McpTool] = []

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> bool:
        """Launch the server, shake hands, and learn its tools.
        Returns False on any failure - a server that will not start is a
        capability the worker simply does not have today."""
        import os

        try:
            self._process = await asyncio.create_subprocess_exec(
                self._config.command,
                *self._config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **self._config.env},
            )

            await asyncio.wait_for(
                self._request("initialize", {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "jarvis", "version": "0.1.0"},
                }),
                timeout=_STARTUP_TIMEOUT_S,
            )
            await self._notify("notifications/initialized", {})

            listed = await self._request("tools/list", {})
            self.tools = [
                McpTool(
                    server=self._config.name,
                    name=str(entry["name"]),
                    description=str(entry.get("description", "")),
                    input_schema=dict(entry.get("inputSchema", {"type": "object"})),
                )
                for entry in listed.get("tools", [])
            ]
            log.info("mcp server started", extra={
                "server": self._config.name,
                "tools": [t.name for t in self.tools],
            })
            return True

        except Exception:
            log.error("mcp server failed to start", exc_info=True, extra={
                "server": self._config.name, "command": self._config.command,
            })
            await self.stop()
            return False

    async def call(self, tool_name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool. Returns (text, is_error) - never raises, because
        a broken tool must not break the job using it."""
        if not self.alive:
            return f"error: mcp server {self._config.name} is not running", True

        try:
            response = await asyncio.wait_for(
                self._request("tools/call", {"name": tool_name, "arguments": args}),
                timeout=_REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return f"error: {tool_name} timed out after {_REQUEST_TIMEOUT_S}s", True
        except Exception as exc:
            log.error("mcp call failed", exc_info=True, extra={
                "server": self._config.name, "tool": tool_name,
            })
            return f"error: {type(exc).__name__}: {exc}", True

        # MCP results are a list of content blocks; we flatten text and
        # note anything else rather than dropping it silently.
        parts: list[str] = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(f"[{block.get('type', 'unknown')} content]")
        return "\n".join(parts) or "(no output)", bool(response.get("isError", False))

    async def stop(self) -> None:
        """Terminate the subprocess, firmly if necessary."""
        if self._process is None:
            return
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None

    # -- JSON-RPC plumbing ----------------------------------------------------

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """One request, one response. The lock serialises calls per
        server: MCP allows interleaving by request id, but one at a time
        is simpler and a single agent is not issuing parallel tool calls
        to the same server anyway."""
        async with self._lock:
            if self._process is None or self._process.stdin is None:
                raise RuntimeError(f"mcp server {self._config.name} is not running")

            request_id = self._next_id
            self._next_id += 1
            message = json.dumps({
                "jsonrpc": "2.0", "id": request_id,
                "method": method, "params": params,
            })
            self._process.stdin.write((message + "\n").encode())
            await self._process.stdin.drain()

            assert self._process.stdout is not None
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"mcp server {self._config.name} closed the connection"
                    )
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue    # servers sometimes emit stray output; skip it
                if payload.get("id") != request_id:
                    continue    # a notification, or a response to something else
                if "error" in payload:
                    raise RuntimeError(str(payload["error"]))
                result: dict[str, Any] = payload.get("result", {})
                return result

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Fire-and-forget: notifications have no id and no reply."""
        if self._process is None or self._process.stdin is None:
            return
        message = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        self._process.stdin.write((message + "\n").encode())
        await self._process.stdin.drain()


class McpHost:
    """Runs and supervises a set of MCP servers for one process."""

    def __init__(self) -> None:
        self._servers: dict[str, McpServer] = {}

    async def start_all(self, configs: list[McpServerConfig]) -> None:
        """Launch every configured server. Ones that fail are simply
        absent - a missing tool is a smaller problem than a worker that
        refuses to run because one optional server is broken."""
        for config in configs:
            server = McpServer(config)
            if await server.start():
                self._servers[config.name] = server

    def all_tools(self) -> list[McpTool]:
        return [tool for server in self._servers.values() for tool in server.tools]

    async def call(self, qualified_name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Route a namespaced tool call to its server."""
        try:
            _, server_name, tool_name = qualified_name.split("__", 2)
        except ValueError:
            return f"error: {qualified_name} is not an mcp tool name", True

        server = self._servers.get(server_name)
        if server is None:
            return f"error: no mcp server named {server_name}", True
        return await server.call(tool_name, args)

    async def stop_all(self) -> None:
        for server in self._servers.values():
            await server.stop()
        self._servers.clear()

    def server_names(self) -> list[str]:
        return sorted(self._servers)
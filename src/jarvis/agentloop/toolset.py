# =============================================================================
# src/jarvis/agentloop/toolset.py - the typed toolbox an agent loop uses
# =============================================================================
#
# A tool = a name, a description the MODEL reads to decide when to use it,
# a Pydantic model for its arguments, and the async function that does the
# work. The Pydantic model serves both directions at once:
#
#   outward - its JSON schema is shown to the model as the tool contract
#   inward  - the model's arguments are validated through it before the
#             handler ever runs; sloppy arguments die at the border
#
# Execution NEVER raises to the loop. Unknown tool, invalid arguments, or
# a crashing handler all come back as (error text, is_error=True), which
# the loop feeds to the model as the tool's result. The model then routes
# around the failure or reports it honestly - errors become information,
# not crashes.
#
# Allowlists and approval gates bolt onto this same choke point in a later
# phase; every tool execution already passes through exactly one door.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from typing import TYPE_CHECKING

from jarvis.agentloop.guard import Guard, Verdict
from jarvis.common.log import get_logger
from jarvis.llm.layer import ToolSpec

if TYPE_CHECKING:
    from jarvis.agentloop.mcp_client import McpHost


class _PassthroughArgs(BaseModel):
    """Arguments for an MCP tool: the server owns the schema, so we
    accept whatever the model sends and let the server judge it."""

    model_config = ConfigDict(extra="allow")

log = get_logger("agentloop.toolset")

# A handler receives its VALIDATED argument model and returns result text
# for the model to read.
ToolHandler = Callable[[Any], Awaitable[str]]


@dataclass(frozen=True)
class InlineTool:
    """One tool: contract plus implementation."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler


class Toolset:
    """A named collection of tools offered to one agent loop.

    When a guard and actor are supplied, EVERY execution is checked
    before the handler runs. A toolset without a guard (tests, and the
    Core-local tools that predate the guard) simply allows everything -
    but the check lives in one place, so switching a toolset to guarded
    is two constructor arguments rather than an audit.
    """

    def __init__(
        self,
        guard: "Guard | None" = None,
        actor: str | None = None,
    ) -> None:
        self._tools: dict[str, InlineTool] = {}
        self._guard = guard
        self._actor = actor
        # MCP tools carry the server's own schema, not a Pydantic model.
        self._mcp_schemas: dict[str, dict[str, Any]] = {}

    def register(self, tool: InlineTool) -> None:
        if tool.name in self._tools:
            raise RuntimeError(f"tool {tool.name!r} registered twice")
        self._tools[tool.name] = tool

    def register_mcp_host(self, host: "McpHost") -> None:
        """Expose every tool the host's servers offer.

        MCP tools are ordinary entries in this toolset, which is the
        point: they pass the SAME guard check as anything else. A tool
        discovered from a subprocess at runtime is not automatically a
        trusted tool, and this is where that is enforced.

        Schemas come from the servers themselves, so validation is
        skipped here (the server owns its contract) - the guard still
        sees the raw arguments and can pattern-match on them.
        """
        for tool in host.all_tools():
            name = tool.qualified_name

            async def handler(args: BaseModel, _name: str = name) -> str:
                output, is_error = await host.call(
                    _name, args.model_dump() if hasattr(args, "model_dump") else {}
                )
                if is_error:
                    raise RuntimeError(output)
                return output

            self._tools[name] = InlineTool(
                name=name,
                description=tool.description,
                args_model=_PassthroughArgs,
                handler=handler,
            )
        self._mcp_schemas.update({
            tool.qualified_name: tool.input_schema for tool in host.all_tools()
        })

    def is_empty(self) -> bool:
        return not self._tools

    def specs(self) -> list[ToolSpec]:
        """The tool contracts, in the shape the LLM layer sends the model."""
        return [
            ToolSpec(
                name=t.name,
                description=t.description,
                # MCP servers supply their own schema; ours come from
                # the Pydantic model.
                input_schema=self._mcp_schemas.get(
                    t.name, t.args_model.model_json_schema()
                ),
            )
            for t in self._tools.values()
        ]

    async def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool call from the model. Returns (result_text, is_error).
        Never raises - failures are results the model gets to read."""
        tool = self._tools.get(name)
        if tool is None:
            return f"error: no tool named {name!r} exists", True

        # The door. Nothing reaches a handler without passing it.
        if self._guard is not None and self._actor is not None:
            decision = self._guard.check(self._actor, name, args)
            if decision.verdict is Verdict.DENY:
                # The agent reads this and routes around the refusal.
                return f"denied: {decision.reason}", True
            if decision.verdict is Verdict.GATE:
                # Pausing the job to ask the owner is the approvals
                # layer's job (next batch). Until it exists, a gated
                # action is refused rather than silently permitted -
                # failing closed is the only safe direction here.
                return (
                    f"blocked: {decision.reason}. This needs the owner's "
                    f"approval, which is not yet wired up.",
                    True,
                )

        try:
            validated = tool.args_model.model_validate(args)
        except ValidationError as exc:
            return f"error: invalid arguments for {name}: {exc}", True

        try:
            result = await tool.handler(validated)
            return result, False
        except Exception as exc:
            # The handler broke; log with the stack for us, summarise for
            # the model.
            log.error("tool handler failed", exc_info=True, extra={"tool": name})
            return f"error: tool {name} failed: {type(exc).__name__}: {exc}", True
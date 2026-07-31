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

from pydantic import BaseModel, ValidationError

from jarvis.common.log import get_logger
from jarvis.llm.layer import ToolSpec

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
    """A named collection of tools offered to one agent loop."""

    def __init__(self) -> None:
        self._tools: dict[str, InlineTool] = {}

    def register(self, tool: InlineTool) -> None:
        if tool.name in self._tools:
            raise RuntimeError(f"tool {tool.name!r} registered twice")
        self._tools[tool.name] = tool

    def is_empty(self) -> bool:
        return not self._tools

    def specs(self) -> list[ToolSpec]:
        """The tool contracts, in the shape the LLM layer sends the model."""
        return [
            ToolSpec(
                name=t.name,
                description=t.description,
                input_schema=t.args_model.model_json_schema(),
            )
            for t in self._tools.values()
        ]

    async def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool call from the model. Returns (result_text, is_error).
        Never raises - failures are results the model gets to read."""
        tool = self._tools.get(name)
        if tool is None:
            return f"error: no tool named {name!r} exists", True

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
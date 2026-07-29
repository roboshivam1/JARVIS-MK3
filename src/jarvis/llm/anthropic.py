# =============================================================================
# src/jarvis/llm/anthropic.py - the Anthropic provider adapter
# =============================================================================
#
# Everything provider-specific lives here and nowhere else: message format,
# streaming event handling, error classification, retry etiquette, prompt
# cache markers. The adapter's output is a neutral ProviderResult; nothing
# Anthropic-shaped leaks out of this file.
#
# Behaviour notes:
#   - We ALWAYS stream from the API. If the caller passes on_text, each
#     text chunk is forwarded live; otherwise chunks accumulate silently.
#     One code path either way.
#   - Retry only what retrying can fix: rate limits, overload, transient
#     server errors - short exponential backoff, few attempts. Bad key or
#     bad request surfaces immediately; retrying those is noise.
#   - cache_system=True marks the system prompt as cacheable so a stable
#     persona prompt is billed at the cheap cached rate after the first
#     call. Token usage reports cache reads separately; we pass all counts
#     upward and let the pricing module do the money math.
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import anthropic

from jarvis.common.log import get_logger

log = get_logger("llm.anthropic")

# Called with each streamed text chunk as it arrives.
TextCallback = Callable[[str], Awaitable[None]]

_RETRYABLE = (
    anthropic.RateLimitError,        # 429: slow down
    anthropic.InternalServerError,   # 5xx: provider hiccup
    anthropic.APIConnectionError,    # network blip
)
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 1.5


class ProviderError(RuntimeError):
    """A model call failed for good. `retried` tells whether backoff was
    already attempted before giving up."""

    def __init__(self, message: str, retried: bool) -> None:
        super().__init__(message)
        self.retried = retried


@dataclass(frozen=True)
class ProviderToolCall:
    """The model wants a tool run. Neutral shape, provider ids preserved
    so results can be matched back."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ProviderResult:
    """Neutral summary of one completed model call."""

    text: str
    tool_calls: tuple[ProviderToolCall, ...]
    stop_reason: str            # end_turn | tool_use | max_tokens | ...
    tokens_in: int              # total input tokens, all billing classes
    tokens_out: int
    cached_tokens: int          # portion of tokens_in read from prompt cache
    model: str


@dataclass
class AnthropicAdapter:
    """Thin async wrapper over the official SDK. One instance per process."""

    api_key: str
    _client: anthropic.AsyncAnthropic = field(init=False)

    def __post_init__(self) -> None:
        # max_retries=0: WE own retry policy (with logging), not the SDK.
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=0)

    async def complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        on_text: TextCallback | None = None,
        cache_system: bool = True,
    ) -> ProviderResult:
        """One model call, streamed. Retries transient failures, then
        raises ProviderError."""
        # System prompt with optional cache marker: the provider caches the
        # prefix and bills re-reads at the cheap rate. Our persona/profile
        # prompts are identical every turn - ideal cache material.
        system_param: Any
        if cache_system:
            system_param = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_param = system

        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_param,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await self._stream_once(kwargs, on_text)
            except _RETRYABLE as exc:
                last_error = exc
                if attempt == _MAX_ATTEMPTS:
                    break
                delay = _BACKOFF_BASE_S * (2 ** (attempt - 1))
                log.warning(
                    "transient provider error - backing off",
                    extra={
                        "attempt": attempt,
                        "delay_s": delay,
                        "error": type(exc).__name__,
                    },
                )
                await asyncio.sleep(delay)
            except anthropic.APIStatusError as exc:
                # Non-retryable API rejection (bad key, bad request, ...).
                raise ProviderError(
                    f"provider rejected the call: {exc.__class__.__name__}: {exc}",
                    retried=False,
                ) from exc

        raise ProviderError(
            f"provider unavailable after {_MAX_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}: {last_error}",
            retried=True,
        ) from last_error

    async def _stream_once(
        self,
        kwargs: dict[str, Any],
        on_text: TextCallback | None,
    ) -> ProviderResult:
        """A single streamed API call, no retry logic."""
        text_parts: list[str] = []

        async with self._client.messages.stream(**kwargs) as stream:
            async for chunk in stream.text_stream:
                text_parts.append(chunk)
                if on_text is not None:
                    await on_text(chunk)
            final = await stream.get_final_message()

        tool_calls = tuple(
            ProviderToolCall(id=block.id, name=block.name, args=dict(block.input))
            for block in final.content
            if block.type == "tool_use"
        )

        usage = final.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # Total input = fresh + cache reads + cache writes. Cache WRITES
        # bill at a small premium over fresh input; we fold them in as
        # fresh (slight underestimate) rather than complicate the price
        # table. Cache READS are the big discount and are tracked exactly.
        tokens_in = usage.input_tokens + cache_read + cache_write

        return ProviderResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=final.stop_reason or "end_turn",
            tokens_in=tokens_in,
            tokens_out=usage.output_tokens,
            cached_tokens=cache_read,
            model=kwargs["model"],
        )
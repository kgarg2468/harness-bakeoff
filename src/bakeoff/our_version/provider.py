# Ported from Pi (MIT): packages/ai/src/api/openai-completions.ts @ 5fd446ca1843682e8da3fec4ceb71c42f56fbace
# Changes: Python; only the stream-end checks, tool-call accumulation, empty-message skip and error text.
"""Wire side of the loop: serialize-once request bodies and the streamed-response accumulator.

The request-body cache (`Wire`) is ours. From Pi: skipping assistant messages with neither
content nor tool calls on replay, accumulating tool-call deltas by index then id, the
finish_reason checks, and appending OpenRouter's `metadata.raw` to error messages.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import httpx

from bakeoff.shared.contract import Item, ToolCall

from .retry import ProviderError, classify

_STOPS = frozenset({"stop", "end", "length", "tool_calls", "function_call"})


def dump(obj: Any) -> bytes:
    """Compact ASCII JSON bytes. Deterministic, and never fails on a lone surrogate."""
    return json.dumps(obj, separators=(",", ":")).encode()


def replayed(history: list[Item]) -> list[Item]:
    """Items the next request carries: the last compaction item and everything after it.

    Cancelled output is never replayed, nor an assistant message with neither content nor tool
    calls (providers reject those).
    """
    start = max((i for i, it in enumerate(history) if it.compaction), default=0)
    return [it for it in history[start:] if it.status != "incomplete" and _sendable(it.message)]


def _sendable(msg: dict[str, Any]) -> bool:
    return msg.get("role") != "assistant" or bool(msg.get("content") or msg.get("tool_calls"))


def message_json(item: Item) -> bytes:
    """A history item as its chat-completions message."""
    return dump(item.message)


class Wire:
    """Request bodies for one thread, built by joining cached JSON fragments.

    The static prefix (model, params, tools, system message) is serialized once, and each item
    once (items are immutable), so every request repeats the previous one byte for byte.
    `fragment` serializes one item: its message, or (Responses API) its input items.
    """

    def __init__(
        self,
        key: tuple[Any, ...],
        static: dict[str, Any],
        fragment: Callable[[Item], bytes] = message_json,
    ) -> None:
        self.key = key
        self.prefix = dump(static)[:-2]  # drop the closing "]}" so items can follow
        self.fragment = fragment
        self.frags: dict[str, bytes] = {}

    def body(self, items: list[Item]) -> bytes:
        old = self.frags
        self.frags = {it.id: old.get(it.id) or self.fragment(it) for it in items}
        return b",".join([self.prefix, *self.frags.values()]) + b"]}"


class StreamedCall:
    """A tool call whose pieces are still arriving."""

    __slots__ = ("depth", "id", "name", "parts", "ready")

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.parts: list[str] = []
        self.ready = False
        self.depth = 0  # "{" minus "}" so far; braces inside strings only delay the eager start

    def finish(self) -> ToolCall:
        """Mark the call complete (its arguments are final) and return it."""
        self.ready = True
        self.id = self.id or f"call_{uuid.uuid4().hex[:24]}"
        return self.call()

    def call(self) -> ToolCall:
        return ToolCall(self.id, self.name, "".join(self.parts))


class Stream:
    """One streamed chat completion, accumulated chunk by chunk."""

    native: list[dict[str, Any]] | None = None  # what its item keeps in Item.native (Responses)

    def __init__(self) -> None:
        self.text: list[str] = []
        self.details: list[dict[str, Any]] = []
        self.calls: dict[Any, StreamedCall] = {}  # keyed by index, else id; in call order
        self.usage: dict[str, Any] | None = None
        self.finish: str | None = None
        self.done = False  # complete: saw [DONE], or usage after finish_reason

    def tool_delta(self, delta: dict[str, Any], read_only: set[str]) -> StreamedCall | None:
        """Add one `tool_calls` delta. Returns the call if a read-only call just completed."""
        key = delta.get("index")
        if key is None:  # no index: the id, else a continuation of the last call
            key = delta.get("id") or next(reversed(self.calls), None)
        call = self.calls.get(key)
        if call is None:
            call = self.calls[key] = StreamedCall()
        call.id = call.id or delta.get("id") or ""
        fn = delta.get("function") or {}
        call.name = call.name or fn.get("name") or ""
        if piece := fn.get("arguments"):
            call.parts.append(piece)
            # Complete once the object parses, even with other calls interleaved. The brace count
            # keeps this to about one parse per call instead of one per piece.
            if call.name in read_only and not call.ready:
                call.depth += piece.count("{") - piece.count("}")
                if call.depth <= 0 and piece.rstrip().endswith("}"):
                    try:
                        return call if isinstance(json.loads("".join(call.parts)), dict) else None
                    except ValueError:
                        return None
        return None

    def check_end(self) -> None:
        """Raise if the stream failed or was cut off. `[DONE]` without finish_reason is fine."""
        if self.finish is None:
            if not self.done:
                raise classify("stream", "Stream ended without finish_reason")
        elif self.finish == "error":  # OpenRouter's mid-stream failure, here without an error chunk
            raise classify("stream", "Provider returned error (finish_reason: error)")
        elif self.finish not in _STOPS:  # retried only if transient, e.g. network_error
            raise classify("stream", f"Provider finish_reason: {self.finish}")

    def message(self) -> dict[str, Any]:
        """The assistant message exactly as it will be replayed."""
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(self.text) or None}
        if self.calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": "".join(c.parts)},
                }
                for c in self.calls.values()
            ]
        if self.details:
            msg["reasoning_details"] = self.details
        return msg

    def partial(self) -> dict[str, Any] | None:
        """What a cut-short stream (cancel, truncation) produced, without tool calls; None if nothing."""
        if not (self.text or self.details):
            return None
        msg = self.message()
        msg.pop("tool_calls", None)
        return msg

    def tool_calls(self) -> list[ToolCall]:
        return [c.call() for c in self.calls.values()]


def stream_error(error: Any) -> ProviderError:
    """Error for a mid-stream `{"error": {...}}` chunk (sent after HTTP 200)."""
    code = error.get("code") if isinstance(error, dict) else None
    message = f"{code}: {_describe(error)}" if code is not None else _describe(error)
    return classify("stream", message, code if isinstance(code, int) else None)


async def http_error(resp: httpx.Response) -> ProviderError:
    """Error for a non-200 response, from its status, headers and JSON error body."""
    raw = await resp.aread()
    try:
        detail = _describe(json.loads(raw)["error"])
    except (ValueError, KeyError, TypeError):
        detail = raw.decode(errors="replace")[:500]
    return classify("http", f"HTTP {resp.status_code}: {detail}", resp.status_code, resp.headers)


def as_provider_error(exc: Exception) -> ProviderError:
    """Map what a request can raise to a ProviderError; anything else is a bug and re-raised."""
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, httpx.TransportError):
        return ProviderError(
            "network", f"Network error: {type(exc).__name__}: {exc}", retryable=True
        )
    if isinstance(exc, ValueError):
        return ProviderError("stream", f"Invalid stream data: {exc}")
    raise exc


def _describe(error: Any) -> str:
    if not isinstance(error, dict):
        return str(error)
    message = str(error.get("message") or error)
    raw = (error.get("metadata") or {}).get("raw")  # OpenRouter: the upstream provider's error
    return f"{message}\n{raw}" if raw and str(raw) not in message else message

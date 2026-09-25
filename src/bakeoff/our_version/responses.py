# Ported from Pi (MIT): packages/ai/src/api/{openai-responses.ts,openai-responses-shared.ts,openai-prompt-cache.ts} @ 5fd446ca1843682e8da3fec4ceb71c42f56fbace
# Changes: Python; one request shape (store false, encrypted reasoning always asked for); done output items replayed verbatim; calls used once done.
"""OpenAI's Responses API: the request's fixed part, history items as `input` items, and the
streamed-event accumulator.

From Pi: the request parameters (`store: false` with `include: ["reasoning.encrypted_content"]`,
here always, not only when reasoning is on; a reasoning summary unless the effort is "none",
`max_output_tokens` of at least 16, a `prompt_cache_key` of at most 64 characters, flat function
tools with `strict: false`, the system prompt as a developer message), the conversion of
chat-shaped messages to input items, replaying output items without their ids when their
reasoning item is not replayed (Pi: calls from another model), which events carry text and
reasoning (a blank line between summary parts; between messages too, which Pi keeps apart as
blocks), and the ends of a stream:
`response.completed`, `.incomplete` (max_output_tokens is a truncation, any other reason an
error), `.failed`, the `error` event, and a stream that ends before any of them.

Not ported (this harness does not need them): images, custom and grammar tools, tool search,
service tiers and their pricing, cache retention options, session headers, Copilot, foreign
provider id rewriting, text signatures (the whole output items are kept instead), streamed
arguments (a call is used once its `output_item.done` arrives, which also makes it complete for
an eager start), and Azure's encrypted-content backfill from `response.completed`.
"""

from __future__ import annotations

from typing import Any

from bakeoff.shared.contract import Event, Item, ModelConfig, ToolSpec

from .provider import Stream, StreamedCall, dump, stream_error
from .retry import classify

_MIN_OUTPUT_TOKENS = 16  # the API rejects a smaller max_output_tokens
_CACHE_KEY_MAX = 64  # the API rejects a longer prompt_cache_key
_TEXT = frozenset({"response.output_text.delta", "response.refusal.delta"})
# Summaries of the reasoning, or (some models) the reasoning text itself.
_REASONING = frozenset({"response.reasoning_summary_text.delta", "response.reasoning_text.delta"})
_END = frozenset({"response.completed", "response.incomplete"})


def static_body(
    model: ModelConfig, system: str, tools: list[ToolSpec], session_id: str
) -> dict[str, Any]:
    """The part of every request that is fixed for a thread, ending with the system prompt."""
    body: dict[str, Any] = {
        "model": model.model,
        "stream": True,
        # The harness keeps the history, so the API stores nothing and reasoning goes back as
        # its encrypted content (an id alone would be unknown to it).
        "store": False,
        # Routes the thread's requests to the same cache, as OpenRouter's session_id does.
        "prompt_cache_key": session_id[:_CACHE_KEY_MAX],
        "max_output_tokens": max(model.max_tokens, _MIN_OUTPUT_TOKENS),
    }
    if model.temperature is not None:
        body["temperature"] = model.temperature
    if model.reasoning is not None:
        # A summary streams while the model reasons; the effort "none" has nothing to sum up.
        auto = {} if model.reasoning.get("effort") == "none" else {"summary": "auto"}
        body["reasoning"] = {**auto, **model.reasoning}
    # Asked for even without a reasoning config: a model may reason at its default effort, and
    # with store false its reasoning can go back only as this content (none if it did not reason).
    body["include"] = ["reasoning.encrypted_content"]
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": False,  # strict mode (the default) forbids optional and free-form args
            }
            for t in tools
        ]
    body["input"] = [{"role": "developer", "content": system}]
    return body


def input_json(item: Item) -> bytes:
    """One history item as its `input` items, serialized and comma-separated (no brackets)."""
    return b",".join(map(dump, input_items(item)))


def input_items(item: Item) -> list[dict[str, Any]]:
    """A history item as `input` items: a response's done output items verbatim (reasoning with
    its encrypted_content, messages with their phase), else converted from its chat message."""
    if item.native:  # empty if the server sent no done items: then the message is converted
        return item.native
    msg = item.message
    if msg["role"] == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": msg["tool_call_id"],
                "output": msg["content"],
            }
        ]
    text = [] if msg.get("content") is None else [{"role": msg["role"], "content": msg["content"]}]
    return text + [
        {
            "type": "function_call",
            "call_id": call["id"],
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"],
        }
        for call in msg.get("tool_calls") or ()
    ]


class ResponsesStream(Stream):
    """One streamed response, accumulated event by event. `native` collects its done output
    items: exactly what the next request replays."""

    def __init__(self) -> None:
        super().__init__()
        self.native: list[dict[str, Any]] = []
        self.parts = 0  # reasoning summary parts so far, over all reasoning items
        self.unpaired = False  # the last reasoning item was dropped (see _done)

    def feed(self, event: dict[str, Any]) -> Event | StreamedCall | None:
        """Take one event. Returns an event to publish, or a function call that just completed."""
        kind = event.get("type")
        if kind in _TEXT:
            self.text.append(text := event["delta"])
            return Event("text.delta", {"text": text})
        if kind in _REASONING:
            return Event("reasoning.delta", {"text": event["delta"]})
        if kind == "response.output_item.done":
            return self._done(event["item"])
        # Pi keeps each output item in a block of its own. Here all reasoning streams as one text
        # and all messages make one, so a blank line sets apart summary parts and messages.
        if kind == "response.reasoning_summary_part.added":
            self.parts += 1
            return Event("reasoning.delta", {"text": "\n\n"}) if self.parts > 1 else None
        if kind == "response.output_item.added":
            if self.text and event["item"]["type"] == "message":
                self.text.append("\n\n")
                return Event("text.delta", {"text": "\n\n"})
            return None
        if kind in _END:
            response = event["response"]
            self.usage, self.done = response.get("usage"), True
            reason = (response.get("incomplete_details") or {}).get("reason")
            if kind == "response.incomplete" and reason != "max_output_tokens":
                raise classify("stream", f"Response incomplete: {reason}")
            self.finish = "length" if reason == "max_output_tokens" else "stop"
        elif kind == "error":  # after HTTP 200; its fields are top-level, or in `error`
            raise stream_error(event.get("error") or event)
        elif kind == "response.failed":
            raise stream_error(event["response"].get("error") or "Response failed")
        return None

    def _done(self, item: dict[str, Any]) -> StreamedCall | None:
        kind = item.get("type")
        if kind == "reasoning":
            # With store false the API knows a reasoning item only by its encrypted_content: one
            # without it could never be replayed (404), so it is not kept.
            self.unpaired = not item.get("encrypted_content")
            if not self.unpaired:
                self.native.append(item)
            return None
        # The API pairs an output item's id with the reasoning item before it, and refuses the id
        # without that item ("provided without its required 'reasoning' item"): after a dropped
        # one, the items go back without their ids, as new input.
        self.native.append({k: v for k, v in item.items() if k != "id"} if self.unpaired else item)
        if kind != "function_call":
            return None
        call = self.calls[item["call_id"]] = StreamedCall()
        call.id, call.name, call.parts = item["call_id"], item["name"], [item["arguments"]]
        return call

    def check_end(self) -> None:
        """Raise if the stream ended before its terminal event (retried as a cut-off stream)."""
        if self.finish is None:  # set by response.completed or .incomplete only: [DONE] is no end
            raise classify("stream", "Stream ended without response.completed")

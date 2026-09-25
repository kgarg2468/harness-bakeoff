"""Translate between pydantic-ai's native messages and the harness's items.

The native `ModelMessage` JSON is the source of truth for this loop's history. Every item carries
the native JSON of exactly what it shows: a model response, or one part of a request (a tool
result or a prompt). pydantic-ai merges consecutive requests again before it sends them, so the
split never reaches the wire, and a crash between two items of one tool batch loses nothing that
was saved. `Item.message` is the OpenAI-shaped view, as the model puts it on the wire.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_ai import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.messages import INTERRUPTED_TOOL_RETURN_CONTENT
from pydantic_ai.profiles import DEFAULT_THINKING_TAGS

from bakeoff.shared.contract import Item


def dump(message: ModelMessage) -> dict[str, Any]:
    """The native JSON of one message."""
    return ModelMessagesTypeAdapter.dump_python([message], mode="json")[0]


def split(message: ModelMessage) -> list[ModelMessage]:
    """One native message per item: a response stays whole, a request splits into its parts."""
    if isinstance(message, ModelResponse):
        return [message]
    return [replace(message, parts=[part]) for part in message.parts]


def to_history(items: list[Item]) -> list[ModelMessage]:
    """Rebuild the native history from the last compaction item onward (contract rule 8).

    Items the runner wrote (user messages, revert notes, summaries) have no native and become
    user prompts.
    """
    start = max((i for i, item in enumerate(items) if item.compaction), default=0)
    history: list[ModelMessage] = []
    for item in items[start:]:
        if item.native is not None:
            history.extend(ModelMessagesTypeAdapter.validate_python([item.native]))
        else:
            history.append(ModelRequest(parts=[UserPromptPart(item.message["content"])]))
    return history


def to_openai(message: ModelMessage) -> list[dict[str, Any]]:
    """The OpenAI chat messages for one native message: one per tool result or prompt, one per
    response, none for a response with nothing to send (the model skips those too)."""
    if isinstance(message, ModelResponse):
        assistant = _assistant(replayable([message])[0])
        return [assistant] if assistant is not None else []
    out: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, ToolReturnPart):
            out.append(_tool(part.tool_call_id, part.model_response_str()))
        elif isinstance(part, RetryPromptPart) and part.tool_name is not None:
            out.append(_tool(part.tool_call_id, part.model_response()))
        elif isinstance(part, RetryPromptPart):
            out.append({"role": "user", "content": part.model_response()})
        elif isinstance(part, UserPromptPart):
            out.append({"role": "user", "content": part.content})
    return out


def replayable(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The history as the loop replays it (a `ProcessHistory` capability): a response cut short by
    a cancel loses its unsigned thinking. The signature only arrives at the end of a thinking
    block, and endpoints that check signatures (Anthropic) reject a replay without one."""
    out: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse) and message.state == "interrupted":
            parts = [
                p for p in message.parts if not (isinstance(p, ThinkingPart) and not p.signature)
            ]
            if len(parts) < len(message.parts):
                message = replace(message, parts=parts)
        out.append(message)
    return out


def pending_calls(history: list[ModelMessage]) -> list[ToolCallPart]:
    """Tool calls of the last model response that have no result after it."""
    for index in range(len(history) - 1, -1, -1):
        if isinstance(response := history[index], ModelResponse):
            answered = {
                part.tool_call_id
                for later in history[index + 1 :]
                for part in later.parts
                if isinstance(part, ToolReturnPart | RetryPromptPart)
            }
            return [call for call in response.tool_calls if call.tool_call_id not in answered]
    return []


def close_pending(history: list[ModelMessage]) -> list[ModelMessage]:
    """Results for calls a cancel, an error or an abandoned pause left open, exactly as
    pydantic-ai's own history repair would synthesize them before the next request. Persisting
    them keeps rule 4 in the log."""
    parts = [
        ToolReturnPart(
            tool_name=call.tool_name,
            content=INTERRUPTED_TOOL_RETURN_CONTENT,
            tool_call_id=call.tool_call_id,
            outcome="interrupted",
        )
        for call in pending_calls(history)
    ]
    return [ModelRequest(parts=parts)] if parts else []


def _tool(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _assistant(response: ModelResponse) -> dict[str, Any] | None:
    """The assistant message OpenRouterModel / OpenAIChatModel sends for a response, or None
    (the library's "auto" rules for thinking; text pieces are joined with a blank line)."""
    texts: list[str] = []
    details: list[dict[str, Any]] = []
    fields: dict[str, list[str]] = {}
    for part in response.parts:
        if isinstance(part, TextPart):
            texts.append(part.content)
        elif isinstance(part, ThinkingPart) and part.provider_name == "openrouter":
            details.append(_reasoning_detail(part))
        elif isinstance(part, ThinkingPart) and part.id not in (None, "content"):
            # BYOK: sent back in the field it streamed in, e.g. `reasoning_content`.
            fields.setdefault(part.id, []).append(part.content)
        elif isinstance(part, ThinkingPart):
            # BYOK thinking parsed from <think> tags in the content goes back the same way.
            start, end = DEFAULT_THINKING_TAGS
            texts.append("\n".join([start, part.content, end]))
    text = "\n\n".join(texts)
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    message.update({name: "\n\n".join(values) for name, values in fields.items()})
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.tool_call_id,
                "type": "function",
                "function": {"name": call.tool_name, "arguments": call.args_as_json_str()},
            }
            for call in response.tool_calls
        ]
    if details:
        message["reasoning_details"] = details
    return message if text or response.tool_calls or details else None


def _reasoning_detail(part: ThinkingPart) -> dict[str, Any]:
    """The `reasoning_details` entry OpenRouterModel replays for a thinking part (without the
    null-valued keys it adds). OpenRouter's type/format/index live in `provider_details`."""
    meta = part.provider_details or {}
    kind = meta.get("type", "reasoning.text")
    detail: dict[str, Any] = {"type": kind}
    if kind == "reasoning.encrypted":
        detail["data"] = part.signature
    elif kind == "reasoning.summary":
        detail["summary"] = part.content
    else:
        detail["text"] = part.content
        if part.signature is not None:
            detail["signature"] = part.signature
    detail.update({key: meta[key] for key in ("format", "index") if meta.get(key) is not None})
    if part.id is not None:
        detail["id"] = part.id
    return detail

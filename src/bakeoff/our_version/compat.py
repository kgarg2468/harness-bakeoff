# Ported from Pi (MIT): packages/ai/src/api/openai-completions.ts @ 5fd446ca1843682e8da3fec4ceb71c42f56fbace
# Changes: Python; four compat flags with defaults by ModelConfig.kind; reasoning_details also merged by index.
"""Per-endpoint request quirks, reasoning_details merging and usage parsing.

Not ported from Pi (this harness does not need them): provider/URL auto-detection, `store`,
strict and grammar tools, vendor thinking formats (zai, qwen, deepseek, together, chat
templates), thinking token budgets, reasoning_content replay, tool-result names, synthetic
assistant bridges, `supportsFinishReason`, prompt_cache_key, per-message cache_control, session
affinity headers, images, tool-call id normalisation and the `choice.usage` fallback.
"""

from __future__ import annotations

from typing import Any

from bakeoff.shared.contract import ModelConfig, ToolSpec

# Flag defaults per endpoint kind. `ModelConfig.compat` overrides them; unknown keys are ignored.
DEFAULTS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "max_tokens_field": "max_tokens",  # max_tokens | max_completion_tokens
        "reasoning_param": "openrouter",  # openrouter | reasoning_effort | none
        "stream_usage": False,  # OpenRouter always sends a final usage chunk
        "developer_role": False,
    },
    "openai_compat": {
        "max_tokens_field": "max_completion_tokens",
        "reasoning_param": "reasoning_effort",
        "stream_usage": True,
        "developer_role": False,
    },
}
_CONCAT = ("text", "summary", "data")
_RUNS = ("reasoning.text", "reasoning.summary")  # without an index, these continue the last entry
# Streamed reasoning text arrives in one of these delta fields, depending on the server.
REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_text")


def flags(model: ModelConfig) -> dict[str, Any]:
    """The endpoint's compat flags: kind defaults overridden by `model.compat`."""
    return {k: model.compat.get(k, v) for k, v in DEFAULTS[model.kind].items()}


def static_body(
    model: ModelConfig, system: str, tools: list[ToolSpec], session_id: str
) -> dict[str, Any]:
    """The part of every request that is fixed for a thread, ending with the system message."""
    f = flags(model)
    body: dict[str, Any] = {"model": model.model, "stream": True}
    if f["stream_usage"]:
        body["stream_options"] = {"include_usage": True}
    body[f["max_tokens_field"]] = model.max_tokens
    if model.temperature is not None:
        body["temperature"] = model.temperature
    if model.reasoning is not None:
        if f["reasoning_param"] == "openrouter":
            body["reasoning"] = model.reasoning
        elif f["reasoning_param"] == "reasoning_effort":
            body["reasoning_effort"] = model.reasoning.get("effort", "medium")
    if model.kind == "openrouter":
        body["session_id"] = session_id
        if model.model.startswith("anthropic/"):
            body["cache_control"] = {"type": "ephemeral"}
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
    role = "developer" if f["developer_role"] else "system"
    body["messages"] = [{"role": role, "content": system}]
    return body


def merge_detail(details: list[dict[str, Any]], fragment: dict[str, Any]) -> None:
    """Fold one streamed reasoning_details fragment into `details`, keeping stream order.

    A fragment with an `index` joins the entry with the same (index, type). Without one, a text
    or summary fragment continues the last entry if that has the same type (as Pi does), and
    anything else starts a new entry, so separately signed or encrypted blocks stay apart.
    Joining concatenates text, summary and data; for every other field the first non-null value
    wins. Unknown fields are kept verbatim.
    """
    index, kind = fragment.get("index"), fragment.get("type")
    if index is not None:
        entry = next(
            (e for e in details if e.get("index") == index and e.get("type") == kind), None
        )
    elif details and kind in _RUNS and details[-1].get("type") == kind:
        entry = details[-1]
    else:
        entry = None
    if entry is None:
        details.append(dict(fragment))
        return
    for name, value in fragment.items():
        if name in _CONCAT and isinstance(value, str):
            entry[name] = (entry.get(name) or "") + value
        elif entry.get(name) is None:
            entry[name] = value


def usage_fields(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Token counts and provider-billed cost from an OpenAI/OpenRouter `usage` object."""
    u = usage or {}
    cost = u.get("cost")
    return {
        "input_tokens": u.get("prompt_tokens") or 0,
        "output_tokens": u.get("completion_tokens") or 0,
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        or u.get("prompt_cache_hit_tokens")
        or u.get("cached_tokens")
        or 0,
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
        "cost_usd": float(cost) if cost is not None else 0.0,
        "cost_source": "provider" if cost is not None else "none",
    }

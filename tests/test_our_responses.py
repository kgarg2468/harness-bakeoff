"""OurLoop on OpenAI's Responses API (kind "openai_responses") against scripted named-event SSE.

R01-R05 run it end to end against fakeprov; these cover what the scenarios do not script.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import httpx
import pytest
from test_our_loop import OBJ, SYSTEM, Server, StubTools, items, no_wait, of, run, user

from bakeoff.our_version.responses import input_items, static_body
from bakeoff.shared.contract import Item, ModelConfig

MODEL = ModelConfig(
    base_url="http://127.0.0.1:9/v1",
    model="gpt-6-luna",
    kind="openai_responses",
    temperature=None,
    reasoning={"effort": "xhigh"},
)
REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "Plan"}],
    "encrypted_content": "gAAAAB-full",
}


class Recorder(Server):
    """Server that also records the request paths."""

    def __init__(self, *responses: Any):
        super().__init__(*responses)
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        return super().__call__(request)


def event(kind: str, **fields: Any) -> dict[str, Any]:
    return {"type": kind, **fields}


def stream_bytes(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def response(*events: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=stream_bytes(*events)
    )


def done(item: dict[str, Any]) -> dict[str, Any]:
    return event("response.output_item.done", item=item)


def completed(status: str = "completed", reason: str | None = None, **usage: Any) -> dict[str, Any]:
    details = {"reason": reason} if reason else None
    body = {"status": status, "incomplete_details": details, "usage": usage or None}
    return event(f"response.{status}", response=body)


def message(text: str) -> dict[str, Any]:
    part = {"type": "output_text", "annotations": [], "text": text}
    return {
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "content": [part],
        "role": "assistant",
        "phase": "final_answer",
    }


def fcall(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "status": "completed",
        "arguments": arguments,
        "call_id": call_id,
        "name": name,
    }


def answer(text: str) -> httpx.Response:
    return response(
        event("response.output_text.delta", delta=text), done(message(text)), completed()
    )


async def test_request_shape_and_a_reasoned_answer() -> None:
    server = Recorder(
        response(
            event("response.created", response={"status": "in_progress"}),
            event("response.output_item.added", item={**REASONING, "encrypted_content": "gAA"}),
            event("response.reasoning_summary_part.added", summary_index=0),
            event("response.reasoning_summary_text.delta", summary_index=0, delta="Plan"),
            event("response.reasoning_summary_part.added", summary_index=1),
            event("response.reasoning_summary_text.delta", summary_index=1, delta="Answer"),
            done(REASONING),
            event("response.output_text.delta", delta="Hel", obfuscation="k2P"),
            event("response.output_text.delta", delta="lo"),
            done(message("Hello")),
            completed(
                input_tokens=100,
                input_tokens_details={"cached_tokens": 60},
                output_tokens=20,
                output_tokens_details={"reasoning_tokens": 12},
            ),
        )
    )
    first = user("hi")
    events = await run(server.loop(), [first], StubTools(), model=MODEL)
    assert [e.type for e in events] == [
        "request.start",
        *["reasoning.delta"] * 3,
        *["text.delta"] * 2,
        "item",
        "usage",
        "turn.end",
    ]
    assert [e["text"] for e in of(events, "reasoning.delta")] == ["Plan", "\n\n", "Answer"]
    (item,) = items(events)
    assert item.message == {"role": "assistant", "content": "Hello"}
    assert item.native == [REASONING, message("Hello")]  # the done items, not the added ones
    assert of(events, "usage")[0] == {
        "step": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_tokens": 60,
        "reasoning_tokens": 12,
        "cost_usd": 0.0,
        "cost_source": "none",
    }
    assert server.paths == ["/v1/responses"]
    body = json.loads(server.bodies[0])
    assert list(body) == [
        "model",
        "stream",
        "store",
        "prompt_cache_key",
        "max_output_tokens",
        "reasoning",
        "include",
        "tools",
        "input",
    ]
    assert body["store"] is False and body["prompt_cache_key"] == "thread-1"
    assert body["reasoning"] == {"summary": "auto", "effort": "xhigh"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["tools"][0] == {
        "type": "function",
        "name": "describe_component",
        "description": "Describe a component.",
        "parameters": OBJ,
        "strict": False,
    }
    assert body["input"] == [{"role": "developer", "content": SYSTEM}, first.message]


async def test_calls_start_once_done_and_output_items_replay_verbatim() -> None:
    tools = StubTools()
    seen: list[tuple[bool, int]] = []
    plain = {"id": "rs_2", "type": "reasoning", "summary": []}  # no encrypted_content
    describe = fcall("c0", "describe_component", '{"name": "a"}')
    write = fcall("c1", "write_file", '{"path": "x.pipe", "content": "{}"}')

    async def body() -> AsyncIterator[bytes]:
        yield stream_bytes(
            done(REASONING),
            done(plain),
            event("response.function_call_arguments.delta", delta='{"name": "a"}'),
            done(describe),
        )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(tools.started.wait(), 1)
        seen.append((tools.started.is_set(), tools.run_counts["c1"]))
        yield stream_bytes(done(write), completed())

    server = Recorder(lambda: httpx.Response(200, content=body()), answer("done"))
    first = user("go")
    events = await run(server.loop(), [first], tools, model=MODEL)
    assert seen == [(True, 0)]  # the read-only call runs while the response still streams
    assert [e["call_id"] for e in of(events, "tool_call.ready")] == ["c0", "c1"]
    assert tools.run_counts == {"c0": 1, "c1": 1}
    reply = items(events)[0]
    assert reply.native == [REASONING, describe, write]  # reasoning without its content is dropped
    assert [c["id"] for c in reply.message["tool_calls"]] == ["c0", "c1"]
    first_body, second_body = server.bodies
    assert json.loads(second_body)["input"][1:] == [
        first.message,
        REASONING,
        describe,
        write,
        {"type": "function_call_output", "call_id": "c0", "output": "describe_component ok"},
        {"type": "function_call_output", "call_id": "c1", "output": "write_file ok"},
    ]
    assert second_body.startswith(first_body[:-2] + b",")  # the same bytes, then the new items
    assert server.paths == ["/v1/responses"] * 2


@pytest.mark.parametrize(
    ("end", "reason", "retried"),
    [
        (
            event("error", code="server_error", message="The server had an error", param=None),
            "server_error: The server had an error",
            True,
        ),
        (
            event(
                "response.failed",
                response={"status": "failed", "error": {"code": "server_error", "message": "Oops"}},
            ),
            "server_error: Oops",
            True,
        ),
        (None, "Stream ended without response.completed", True),
        (completed("incomplete", "content_filter"), "Response incomplete: content_filter", False),
    ],
    ids=["error-event", "failed", "cut-off", "content-filter"],
)
async def test_failed_responses(end: dict[str, Any] | None, reason: str, retried: bool) -> None:
    ends = () if end is None else (end,)
    server = Server(response(event("response.output_text.delta", delta="Hel"), *ends), answer("Hi"))
    events = await run(server.loop(sleep=no_wait), [user("hi")], StubTools(), model=MODEL)
    if retried:  # nothing of the failed attempt is kept
        assert [r["reason"] for r in of(events, "retry")] == [reason]
        assert [it.message["content"] for it in items(events)] == ["Hi"]
    else:
        assert len(server.bodies) == 1 and items(events) == []
        assert of(events, "error")[0]["message"] == reason and events[-1].data["stop"] == "error"


async def test_max_output_tokens_truncates_and_runs_no_call() -> None:
    tools = StubTools()
    server = Server(
        response(
            event("response.output_text.delta", delta="Writing it."),
            done(message("Writing it.")),
            done(fcall("c0", "write_file", '{"path": "a.pipe"}')),
            completed("incomplete", "max_output_tokens", output_tokens=4096),
        )
    )
    events = await run(server.loop(), [user("go")], tools, model=MODEL)
    (item,) = items(events)
    assert item.status == "incomplete" and item.message == {
        "role": "assistant",
        "content": "Writing it.",
    }
    assert of(events, "error")[0]["kind"] == "output_truncated" and tools.run_counts == {}
    assert events[-1].data["stop"] == "error"


def test_request_options_and_items_without_native() -> None:
    model = replace(MODEL, temperature=0.2, max_tokens=4, reasoning={"effort": "none"})
    body = static_body(model, SYSTEM, [], "t" * 80)
    assert body["temperature"] == 0.2 and body["max_output_tokens"] == 16
    assert body["reasoning"] == {"effort": "none"} and len(body["prompt_cache_key"]) == 64
    assert "tools" not in body
    assert (
        not {"reasoning", "include"}
        & static_body(replace(MODEL, reasoning=None), "", [], "t").keys()
    )
    # Items the runner wrote, or a chat completions turn wrote, are converted.
    tool_calls = [{"id": "c0", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    assistant = Item(
        "a1", "t1", {"role": "assistant", "content": "On it.", "tool_calls": tool_calls}
    )
    assert input_items(assistant) == [
        {"role": "assistant", "content": "On it."},
        {"type": "function_call", "call_id": "c0", "name": "f", "arguments": "{}"},
    ]
    result = Item("r1", "t1", {"role": "tool", "tool_call_id": "c0", "content": "ok"})
    assert input_items(result) == [
        {"type": "function_call_output", "call_id": "c0", "output": "ok"}
    ]

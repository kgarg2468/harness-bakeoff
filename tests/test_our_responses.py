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


DONE = b"data: [DONE]\n\n"  # chat completions' end marker, which some proxies add here too


def stream_bytes(*events: dict[str, Any] | bytes) -> bytes:
    return b"".join(
        e if isinstance(e, bytes) else f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode()
        for e in events
    )


def response(*events: dict[str, Any] | bytes) -> httpx.Response:
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
        *["reasoning.delta"] * 2,
        *["text.delta"] * 2,
        "item",
        "usage",
        "turn.end",
    ]
    assert [e["text"] for e in of(events, "reasoning.delta")] == ["Plan", "\n\nAnswer"]
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
    describe = fcall("c0", "describe_component", '{"name": "a"}')
    write = fcall("c1", "write_file", '{"path": "x.pipe", "content": "{}"}')

    async def body() -> AsyncIterator[bytes]:
        yield stream_bytes(
            done(REASONING),
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
    assert reply.native == [REASONING, describe, write]
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


async def test_reasoning_at_the_default_effort_goes_back_with_its_calls() -> None:
    """No reasoning config: the model may still reason, so its encrypted content is asked for.
    A reasoning item that comes without it anyway is dropped (it could never be replayed), and
    the items after it go back without their ids, which the API would pair with it."""
    plain = {"id": "rs_2", "type": "reasoning", "summary": []}  # no encrypted_content
    say = {**message("On it."), "phase": "commentary"}
    describe = fcall("c0", "describe_component", '{"name": "a"}')
    server = Recorder(
        response(done(REASONING), done(plain), done(say), done(describe), completed()),
        answer("done"),
    )
    first = user("go")
    events = await run(server.loop(), [first], StubTools(), model=replace(MODEL, reasoning=None))
    first_body, second_body = (json.loads(b) for b in server.bodies)
    assert "reasoning" not in first_body
    assert first_body["include"] == ["reasoning.encrypted_content"]
    unpaired = [{k: v for k, v in it.items() if k != "id"} for it in (say, describe)]
    assert items(events)[0].native == [REASONING, *unpaired]
    assert second_body["input"][1:] == [
        first.message,
        REASONING,  # kept: its items keep their ids
        *unpaired,
        {"type": "function_call_output", "call_id": "c0", "output": "describe_component ok"},
    ]


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
        (DONE, "Stream ended without response.completed", True),
        (completed("incomplete", "content_filter"), "Response incomplete: content_filter", False),
    ],
    ids=["error-event", "failed", "cut-off", "done-marker", "content-filter"],
)
async def test_failed_responses(
    end: dict[str, Any] | bytes | None, reason: str, retried: bool
) -> None:
    ends = () if end is None else (end,)
    server = Server(response(event("response.output_text.delta", delta="Hel"), *ends), answer("Hi"))
    events = await run(server.loop(sleep=no_wait), [user("hi")], StubTools(), model=MODEL)
    if retried:  # nothing of the failed attempt is kept
        assert [r["reason"] for r in of(events, "retry")] == [reason]
        assert [it.message["content"] for it in items(events)] == ["Hi"]
    else:
        assert len(server.bodies) == 1 and items(events) == []
        assert of(events, "error")[0]["message"] == reason and events[-1].data["stop"] == "error"


async def test_output_items_are_set_apart_by_blank_lines() -> None:
    commentary = {**message("I'll check."), "id": "msg_0", "phase": "commentary"}
    events = []
    for n, item in enumerate(({**REASONING, "id": "rs_0"}, REASONING)):
        events += [
            event("response.output_item.added", item={**item, "summary": []}),
            event("response.reasoning_summary_part.added", summary_index=0),
            event(
                "response.reasoning_summary_text.delta",
                item_id=item["id"],
                summary_index=0,
                delta=f"Plan {n}.",
            ),
            done(item),
        ]
    for item in (commentary, message("Done.")):
        text = item["content"][0]["text"]
        events += [
            event("response.output_item.added", item={**item, "content": []}),
            event("response.content_part.added", part={"type": "output_text", "text": ""}),
            event("response.output_text.delta", delta=text),
            done(item),
        ]
    server = Server(response(*events, completed()))
    events = await run(server.loop(), [user("hi")], StubTools(), model=MODEL)
    assert "".join(e["text"] for e in of(events, "reasoning.delta")) == "Plan 0.\n\nPlan 1."
    assert "".join(e["text"] for e in of(events, "text.delta")) == "I'll check.\n\nDone."
    (item,) = items(events)
    assert item.message["content"] == "I'll check.\n\nDone."
    assert item.native[2:] == [commentary, message("Done.")]  # replayed as sent, without it


async def test_raw_reasoning_text_is_set_apart_by_item_and_part() -> None:
    """Models that stream their reasoning itself: the raw text of separate reasoning items (and
    of separate parts, summaries included) does not run together in the reasoning stream."""

    def raw(item_id: str, index: int, delta: str) -> dict[str, Any]:
        return event(
            "response.reasoning_text.delta", item_id=item_id, content_index=index, delta=delta
        )

    server = Server(
        response(
            raw("rs_0", 0, "Look "),
            raw("rs_0", 0, "first."),
            raw("rs_0", 1, "Then this."),
            raw("rs_1", 0, "Next item."),
            event(
                "response.reasoning_summary_text.delta",
                item_id="rs_1",
                summary_index=0,
                delta="Sum.",
            ),
            completed(),
        )
    )
    events = await run(server.loop(), [user("hi")], StubTools(), model=MODEL)
    assert "".join(e["text"] for e in of(events, "reasoning.delta")) == (
        "Look first.\n\nThen this.\n\nNext item.\n\nSum."
    )


async def test_a_response_without_done_items_replays_its_text() -> None:
    server = Server(
        response(event("response.output_text.delta", delta="Hi"), completed()), answer("Bye")
    )
    loop, first = server.loop(), user("hi")
    events = await run(loop, [first], StubTools(), model=MODEL)
    (reply,) = items(events)
    await run(loop, [first, reply, user("again")], StubTools(), model=MODEL)
    assert json.loads(server.bodies[1])["input"][2:] == [
        {"role": "assistant", "content": "Hi"},  # converted: no empty fragment, valid JSON
        {"role": "user", "content": "again"},
    ]


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


@pytest.mark.parametrize(
    "end",
    [completed("incomplete", "max_output_tokens", output_tokens=4096), completed()],
    ids=["incomplete", "completed"],
)
async def test_a_call_done_incomplete_never_runs(end: dict[str, Any]) -> None:
    """max_output_tokens cut the call (its done item says so, arguments cut too): it is never
    ready, not even for an eager start, and the turn is truncated whatever event ends it."""
    tools = StubTools()
    cut = {**fcall("c0", "describe_component", '{"name": "a"}'), "status": "incomplete"}

    async def body() -> AsyncIterator[bytes]:
        yield stream_bytes(done(cut))
        await asyncio.sleep(0.05)  # time enough for an eager start to run the tool
        yield stream_bytes(end)

    server = Server(lambda: httpx.Response(200, content=body()))
    events = await run(server.loop(), [user("go")], tools, model=MODEL)
    assert of(events, "tool_call.ready") == [] and tools.run_counts == {}
    assert of(events, "error")[0]["kind"] == "output_truncated"
    (item,) = items(events)  # kept as cut output, never replayed
    assert item.status == "incomplete" and "tool_calls" not in item.message
    assert events[-1].data["stop"] == "error"


def test_request_options_and_items_without_native() -> None:
    model = replace(MODEL, temperature=0.2, max_tokens=4, reasoning={"effort": "none"})
    body = static_body(model, SYSTEM, [], "t" * 80)
    assert body["temperature"] == 0.2 and body["max_output_tokens"] == 16
    assert body["reasoning"] == {"effort": "none"} and len(body["prompt_cache_key"]) == 64
    assert "tools" not in body
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

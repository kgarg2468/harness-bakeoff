"""OurLoop end to end against scripted SSE responses (httpx.MockTransport) and a stub ToolHost."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from bakeoff.our_version import OurLoop
from bakeoff.shared.contract import (
    Event,
    Item,
    Limits,
    ModelConfig,
    Resume,
    ToolCall,
    ToolResult,
    ToolSpec,
    TurnInput,
)

SYSTEM = "You build RocketRide pipelines."
MODEL = ModelConfig(base_url="http://127.0.0.1:9/v1", model="anthropic/claude-sonnet-5")
OBJ = {"type": "object"}
SPECS = [
    ToolSpec("describe_component", "Describe a component.", OBJ, read_only=True),
    ToolSpec("validate_pipeline", "Validate a pipeline.", OBJ, read_only=True),
    ToolSpec("write_file", "Write a file.", OBJ),
    ToolSpec("edit_file", "Edit a file.", OBJ),
]


class StubTools:
    """ToolHost stand-in: a decision per tool name, optional per-tool delay, records runs."""

    def __init__(self, rules: dict[str, str] | None = None, delays: dict[str, float] | None = None):
        self.rules = rules or {}
        self.delays = delays or {}
        self.run_counts: Counter[str] = Counter()
        self.spans: dict[str, tuple[float, float]] = {}
        self.stopped: list[str] = []  # like ToolHost's tool.end: also on cancel
        self.started = asyncio.Event()

    def specs(self) -> list[ToolSpec]:
        return SPECS

    def check(self, call: ToolCall) -> Any:
        return self.rules.get(call.name, "allow")

    async def run(self, call: ToolCall) -> ToolResult:
        if self.check(call) == "deny":
            return ToolResult(call.id, False, "denied by rule")
        self.run_counts[call.id] += 1
        self.started.set()
        start = time.perf_counter()
        try:
            await asyncio.sleep(self.delays.get(call.name, 0))
        finally:
            self.stopped.append(call.id)
        self.spans[call.id] = (start, time.perf_counter())
        return ToolResult(call.id, True, f"{call.name} ok")


class Server:
    """MockTransport handler: answers with scripted responses in order, records request bodies."""

    def __init__(self, *responses: httpx.Response | Callable[[], httpx.Response]):
        self.responses = list(responses)
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        response = self.responses.pop(0)
        return response() if callable(response) else response

    def loop(self, **kwargs: Any) -> OurLoop:
        return OurLoop(http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)), **kwargs)

    def messages(self, n: int) -> list[dict[str, Any]]:
        return json.loads(self.bodies[n])["messages"]


class Stall(httpx.AsyncByteStream):
    """Sends `first`, then nothing until the client closes the response (like a stalled socket)."""

    def __init__(self, first: bytes):
        self.first = first
        self.closed = asyncio.Event()
        self.stalled = asyncio.Event()  # `first` was read and the client waits for more

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        self.stalled.set()
        await self.closed.wait()
        raise httpx.ReadError("connection closed")

    async def aclose(self) -> None:
        self.closed.set()


def sse_bytes(*events: dict[str, Any] | str) -> bytes:
    return b"".join(
        (e if isinstance(e, str) else "data: " + json.dumps(e)).encode() + b"\n\n" for e in events
    )


def sse(*events: dict[str, Any] | str) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=sse_bytes(*events)
    )


def delta(**fields: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": None}]}


def finish(reason: str = "stop", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    chunk: dict[str, Any] = {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def call(
    index: int, args: str, call_id: str | None = None, name: str | None = None
) -> dict[str, Any]:
    fn: dict[str, Any] = {"arguments": args}
    tc: dict[str, Any] = {"index": index, "function": fn}
    if name:
        fn["name"] = name
    if call_id:
        tc.update(id=call_id, type="function")
    return delta(tool_calls=[tc])


def reply(text: str) -> httpx.Response:
    return sse(delta(content=text), finish("stop"), "data: [DONE]")


def user(text: str) -> Item:
    return Item(uuid.uuid4().hex, "t0", {"role": "user", "content": text})


def turn(history: list[Item], **kwargs: Any) -> TurnInput:
    fields: dict[str, Any] = {"resume": None, "limits": Limits(), "model": MODEL} | kwargs
    return TurnInput("thread-1", "turn-1", SYSTEM, history, **fields)


async def run(
    loop: OurLoop,
    history: list[Item],
    tools: StubTools,
    cancel: asyncio.Event | None = None,
    **kw: Any,
):
    return [e async for e in loop.run_turn(turn(history, **kw), tools, cancel or asyncio.Event())]


def items(events: list[Event]) -> list[Item]:
    return [e.data["item"] for e in events if e.type == "item"]


def of(events: list[Event], kind: str) -> list[dict[str, Any]]:
    return [e.data for e in events if e.type == kind]


def persisted(item: Item) -> Item:
    """The item as the runner reloads it from SQLite (a JSON round trip)."""
    return Item(item.id, item.turn_id, json.loads(json.dumps(item.message)), item.status)


def assert_no_orphans(history: list[Item]) -> None:
    calls = [tc["id"] for it in history for tc in it.message.get("tool_calls") or ()]
    results = [it.message["tool_call_id"] for it in history if it.message["role"] == "tool"]
    assert sorted(calls) == sorted(results)


async def test_text_reply_and_request_shape() -> None:
    server = Server(
        sse(
            ": OPENROUTER PROCESSING",
            delta(role="assistant", content=""),
            delta(content="Hel"),
            delta(content="lo"),
            finish("stop"),
            "data: [DONE]",
        )
    )
    first = user("hi")
    events = await run(server.loop(), [first], StubTools())
    assert [e.type for e in events] == [
        "request.start",
        "text.delta",
        "text.delta",
        "item",
        "usage",
        "turn.end",
    ]
    assert items(events)[0].message == {"role": "assistant", "content": "Hello"}
    assert events[-1].data == {"stop": "end_turn", "steps": 1}
    body = json.loads(server.bodies[0])
    assert list(body) == [
        "model",
        "stream",
        "max_tokens",
        "temperature",
        "session_id",
        "cache_control",
        "tools",
        "messages",
    ]
    assert body["session_id"] == "thread-1" and body["cache_control"] == {"type": "ephemeral"}
    assert [t["function"]["name"] for t in body["tools"]] == [s.name for s in SPECS]
    assert body["messages"] == [{"role": "system", "content": SYSTEM}, first.message]


async def test_usage_counted_once_with_provider_cost() -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 60},
        "completion_tokens_details": {"reasoning_tokens": 5},
        "cost": 0.0012,
    }
    server = Server(
        sse(delta(content="ok"), finish("stop"), finish("stop", usage), "data: [DONE]"),
        reply("plain"),
    )
    events = await run(server.loop(), [user("hi")], StubTools())
    expected = {
        "step": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_tokens": 60,
        "reasoning_tokens": 5,
        "cost_usd": 0.0012,
        "cost_source": "provider",
    }
    assert of(events, "usage") == [expected]
    assert items(events)[0].usage == expected
    byok = await run(server.loop(), [user("hi")], StubTools())  # no usage chunk, no cost
    assert of(byok, "usage")[0]["cost_source"] == "none" and of(byok, "usage")[0]["cost_usd"] == 0.0


async def test_interleaved_parallel_tool_calls() -> None:
    server = Server(
        sse(
            *(call(i, "", f"c{i}", "describe_component") for i in range(3)),
            *(call(i, '{"name":') for i in range(3)),
            *(call(i, f' "{n}"}}') for i, n in enumerate("abc")),
            finish("tool_calls"),
        ),
        reply("done"),
    )
    tools = StubTools(delays={"describe_component": 0.2})
    history = [user("describe a, b and c")]
    start = time.perf_counter()
    events = await run(server.loop(), history, tools)
    assert time.perf_counter() - start < 0.5  # concurrent, not 3 x 0.2 s
    assert tools.run_counts == {"c0": 1, "c1": 1, "c2": 1}
    assert max(s for s, _ in tools.spans.values()) < min(e for _, e in tools.spans.values())
    assistant, *results, final = items(events)
    assert [tc["function"]["arguments"] for tc in assistant.message["tool_calls"]] == [
        '{"name": "a"}',
        '{"name": "b"}',
        '{"name": "c"}',
    ]
    assert [r.message["tool_call_id"] for r in results] == ["c0", "c1", "c2"]
    assert server.messages(1)[1:] == [
        history[0].message,
        assistant.message,
        *(r.message for r in results),
    ]
    assert final.message["content"] == "done"


async def test_read_only_tool_starts_before_the_stream_ends() -> None:
    tools = StubTools()
    seen: list[tuple[bool, int]] = []

    async def body() -> AsyncIterator[bytes]:
        yield sse_bytes(call(0, '{"name": "a"}', "c0", "describe_component"))
        yield sse_bytes(call(1, '{"path": "x.pipe", "content": "{}"}', "c1", "write_file"))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(tools.started.wait(), 1)
        seen.append((tools.started.is_set(), tools.run_counts["c1"]))
        yield sse_bytes(finish("tool_calls"))

    server = Server(lambda: httpx.Response(200, content=body()), reply("done"))
    events = await run(server.loop(), [user("go")], tools)
    assert seen == [(True, 0)]  # describe (read-only) already running; write_file not yet
    assert tools.run_counts == {"c0": 1, "c1": 1}
    assert [e["call_id"] for e in of(events, "tool_call.ready")] == ["c0", "c1"]


class FileTools(StubTools):
    """StubTools plus one file: a write lands when it ends, validate reports what it saw at start."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.files = {"a.pipe": "v1"}

    async def run(self, call: ToolCall) -> ToolResult:
        args = json.loads(call.arguments)
        seen = self.files.get(args.get("path", ""))
        result = await super().run(call)
        if call.name == "write_file":
            self.files[args["path"]] = args["content"]
        elif call.name == "validate_pipeline":
            return ToolResult(call.id, True, f"validated {seen}")
        return result


def results(events: list[Event]) -> list[tuple[str, str]]:
    return [(it.message["tool_call_id"], it.message["content"]) for it in items(events)[1:-1]]


WRITE = '{"path": "a.pipe", "content": "v2"}'


async def test_write_then_validate_in_one_batch_sees_the_write() -> None:
    server = Server(
        sse(
            call(0, WRITE, "c0", "write_file"),
            call(1, '{"path": "a.pipe"}', "c1", "validate_pipeline"),
            finish("tool_calls"),
        ),
        reply("done"),
    )
    tools = FileTools(delays={"write_file": 0.05})
    events = await run(server.loop(), [user("write and validate")], tools)
    assert results(events) == [("c0", "write_file ok"), ("c1", "validated v2")]
    assert tools.spans["c1"][0] >= tools.spans["c0"][1]


async def test_reads_before_a_write_start_early_and_reads_after_it_wait() -> None:
    tools = FileTools(delays={"describe_component": 0.05, "write_file": 0.05})
    started_mid_stream: list[dict[str, int]] = []

    async def body() -> AsyncIterator[bytes]:
        yield sse_bytes(
            call(0, '{"name": "a"}', "c0", "describe_component"),
            call(1, '{"name": "b"}', "c1", "describe_component"),
            call(2, WRITE, "c2", "write_file"),
            call(3, '{"path": "a.pipe"}', "c3", "validate_pipeline"),
        )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(tools.started.wait(), 1)
        await asyncio.sleep(0.01)
        started_mid_stream.append(dict(tools.run_counts))
        yield sse_bytes(finish("tool_calls"))

    server = Server(lambda: httpx.Response(200, content=body()), reply("done"))
    events = await run(server.loop(), [user("go")], tools)
    assert started_mid_stream == [{"c0": 1, "c1": 1}]  # not c3: it comes after the write
    spans = tools.spans
    assert max(spans["c0"][0], spans["c1"][0]) < min(spans["c0"][1], spans["c1"][1])  # overlap
    assert spans["c2"][0] >= max(spans["c0"][1], spans["c1"][1])
    assert spans["c3"][0] >= spans["c2"][1]
    assert results(events) == [
        ("c0", "describe_component ok"),
        ("c1", "describe_component ok"),
        ("c2", "write_file ok"),
        ("c3", "validated v2"),
    ]


async def test_calls_after_an_ask_wait_for_the_answer() -> None:
    batch = sse(
        call(0, WRITE, "c0", "write_file"),
        call(1, '{"path": "a.pipe"}', "c1", "validate_pipeline"),
        finish("tool_calls"),
    )
    tools = FileTools(rules={"write_file": "ask"})
    history = [user("write and validate")]
    paused = await run(Server(batch).loop(), history, tools)
    assert paused[-1].data == {"stop": "paused", "steps": 1, "pending": ["c0"]}
    assert tools.run_counts == {} and len(items(paused)) == 1  # validate did not run early
    history += [persisted(it) for it in items(paused)]
    resume = Resume("approval", {"c0": "allow"})
    resumed = await run(Server(reply("ok")).loop(), history, tools, resume=resume)
    assert [(it.message["tool_call_id"], it.message["content"]) for it in items(resumed)[:2]] == [
        ("c0", "write_file ok"),
        ("c1", "validated v2"),
    ]


def approval_batch() -> httpx.Response:
    return sse(
        call(0, '{"pipeline": {}}', "c0", "validate_pipeline"),
        call(1, '{"path": "a.pipe", "content": "x"}', "c1", "write_file"),
        finish("tool_calls"),
    )


async def test_approval_pauses_and_resumes_in_a_fresh_loop() -> None:
    tools = StubTools(rules={"write_file": "ask"})
    first = Server(approval_batch())
    history = [user("write it")]
    paused = await run(first.loop(), history, tools)
    assert paused[-1].data == {"stop": "paused", "steps": 1, "pending": ["c1"]}
    assert of(paused, "permission.asked") == [
        {"call_id": "c1", "name": "write_file", "arguments": '{"path": "a.pipe", "content": "x"}'}
    ]
    assert [it.message.get("tool_call_id") for it in items(paused)] == [None, "c0"]
    assert tools.run_counts == {"c0": 1}

    history += [persisted(it) for it in items(paused)]
    second = Server(reply("Saved."))
    resumed = await run(second.loop(), history, tools, resume=Resume("approval", {"c1": "allow"}))
    assert tools.run_counts == {"c0": 1, "c1": 1}
    assert resumed[-1].data == {"stop": "end_turn", "steps": 2}  # steps count the paused run too
    assert items(resumed)[0].message == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "write_file ok",
    }
    history += items(resumed)
    assert_no_orphans(history)
    # The resumed request extends the paused one byte for byte, across loop instances.
    assert second.bodies[0].startswith(first.bodies[0][:-2])


async def test_deny_sends_the_reason_to_the_model() -> None:
    tools = StubTools(rules={"write_file": "ask"})
    history = [user("write it")]
    history += items(await run(Server(approval_batch()).loop(), history, tools))
    server = Server(reply("Understood."))
    events = await run(
        server.loop(), history, tools, resume=Resume("approval", {"c1": "deny"}, "not now")
    )
    assert tools.run_counts["c1"] == 0
    denial = {"role": "tool", "tool_call_id": "c1", "content": "Denied by user: not now"}
    assert items(events)[0].message == denial
    assert server.messages(0)[-1] == denial
    assert events[-1].data["stop"] == "end_turn"


def crashed_history() -> list[Item]:
    tool_calls = [
        {
            "id": "c0",
            "type": "function",
            "function": {"name": "validate_pipeline", "arguments": "{}"},
        },
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "write_file", "arguments": '{"path": "a"}'},
        },
    ]
    return [
        user("build"),
        Item("a1", "t1", {"role": "assistant", "content": None, "tool_calls": tool_calls}),
        Item("r0", "t1", {"role": "tool", "tool_call_id": "c0", "content": "valid"}),
    ]


async def test_crash_resume_runs_only_missing_calls() -> None:
    tools = StubTools()
    server = Server(reply("Done."))
    events = await run(server.loop(), crashed_history(), tools, resume=Resume("crash"))
    assert tools.run_counts == {"c1": 1}  # c0's result was persisted: never re-run
    assert [m.get("tool_call_id") for m in server.messages(0)[-2:]] == ["c0", "c1"]
    assert events[-1].data["stop"] == "end_turn"

    asking = StubTools(rules={"write_file": "ask"})
    again = await run(Server().loop(), crashed_history(), asking, resume=Resume("crash"))
    assert again[-1].data == {"stop": "paused", "steps": 1, "pending": ["c1"]}
    assert asking.run_counts == {}

    finished = [user("hi"), Item("a1", "t1", {"role": "assistant", "content": "hello"})]
    server = Server()
    done = await run(server.loop(), finished, tools, resume=Resume("crash"))
    assert [e.type for e in done] == ["turn.end"] and server.bodies == []


async def test_cancel_mid_stream_within_200ms() -> None:
    first = sse_bytes(
        call(0, '{"name": "a"}', "c0", "describe_component"),
        delta(reasoning_details=[{"type": "reasoning.text", "text": "hmm", "index": 0}]),
        delta(content="partial"),
    )
    server = Server(httpx.Response(200, stream=Stall(first)), reply("fresh"))
    tools = StubTools(delays={"describe_component": 10})
    loop, cancel, events = server.loop(), asyncio.Event(), []
    history = [user("one")]
    async with asyncio.timeout(2):
        async for event in loop.run_turn(turn(history), tools, cancel):
            events.append(event)
            if event.type == "text.delta":
                cancelled_at = time.perf_counter()
                cancel.set()
    assert time.perf_counter() - cancelled_at < 0.2
    assert events[-1].data == {"stop": "cancelled", "steps": 1}
    (partial,) = items(events)
    assert partial.status == "incomplete" and "tool_calls" not in partial.message
    assert partial.message["content"] == "partial"
    assert (
        tools.run_counts == {"c0": 1} and "c0" not in tools.spans
    )  # started eagerly, then stopped
    # The cancelled output is never replayed: the next request goes straight to the new message.
    history += [partial, user("two")]
    await run(loop, history, tools)
    assert [m["content"] for m in server.messages(1)] == [SYSTEM, "one", "two"]


async def test_cancel_during_a_slow_tool_leaves_no_orphans() -> None:
    server = Server(
        sse(
            call(0, '{"path": "a"}', "c0", "write_file"),
            call(1, '{"path": "b"}', "c1", "edit_file"),
            call(2, '{"name": "x"}', "c2", "describe_component"),  # waits for the ask
            finish("tool_calls"),
        )
    )
    tools = StubTools(rules={"edit_file": "ask"}, delays={"write_file": 10})
    cancel = asyncio.Event()

    async def cancel_when_started() -> float:
        await tools.started.wait()
        cancel.set()
        return time.perf_counter()

    canceller = asyncio.ensure_future(cancel_when_started())
    history = [user("write")]
    async with asyncio.timeout(2):
        events = await run(server.loop(), history, tools, cancel)
    assert time.perf_counter() - await canceller < 0.2
    assert events[-1].data == {"stop": "cancelled", "steps": 1}
    assert [e["call_id"] for e in of(events, "permission.asked")] == ["c1"]
    assert [it.message for it in items(events)[1:]] == [
        {"role": "tool", "tool_call_id": "c0", "content": "Cancelled by user"},
        {"role": "tool", "tool_call_id": "c1", "content": "Cancelled by user"},
        {"role": "tool", "tool_call_id": "c2", "content": "Cancelled by user"},
    ]
    assert tools.run_counts == {"c0": 1}
    assert_no_orphans(history + items(events))


async def test_429_waits_for_retry_after() -> None:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    limited = httpx.Response(
        429,
        headers={"retry-after": "1"},
        json={"error": {"message": "Rate limit exceeded", "code": 429}},
    )
    server = Server(limited, reply("ok"))
    events = await run(server.loop(sleep=sleep), [user("hi")], StubTools())
    assert waits == [1.0] and len(server.bodies) == 2 and server.bodies[0] == server.bodies[1]
    (retry,) = of(events, "retry")
    assert retry["attempt"] == 1 and retry["status"] == 429 and retry["wait_ms"] == 1000
    assert "Rate limit exceeded" in retry["reason"]
    assert [e["attempt"] for e in of(events, "request.start")] == [1, 2]
    assert events[-1].data["stop"] == "end_turn"


async def test_malformed_retry_after_falls_back_to_backoff() -> None:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    limited = httpx.Response(429, headers={"retry-after": "NaN"}, json={"error": {"message": "x"}})
    server = Server(limited, reply("ok"))
    events = await run(server.loop(sleep=sleep), [user("hi")], StubTools())
    assert len(waits) == 1 and 0.375 <= waits[0] <= 0.5  # backoff(0), not NaN
    assert of(events, "error") == [] and events[-1].data["stop"] == "end_turn"


async def test_mid_stream_error_is_retried_with_backoff() -> None:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    error = {
        "error": {"code": 502, "message": "Provider disconnected"},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    server = Server(sse(delta(content="Hel"), error), reply("Hello"))
    events = await run(server.loop(sleep=sleep), [user("hi")], StubTools())
    assert len(server.bodies) == 2 and 0.375 <= waits[0] <= 0.5
    assert of(events, "retry")[0]["status"] == 502
    assert items(events)[0].message["content"] == "Hello"


async def no_wait(seconds: float) -> None:
    pass


async def test_network_error_is_retried() -> None:
    def refused() -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    server = Server(refused, reply("ok"))
    events = await run(server.loop(sleep=no_wait), [user("hi")], StubTools())
    (retry,) = of(events, "retry")
    assert retry["status"] is None and retry["reason"].startswith("Network error: ConnectError")
    assert len(server.bodies) == 2 and events[-1].data["stop"] == "end_turn"


async def test_stream_cut_off_is_retried() -> None:
    server = Server(sse(delta(content="Hel")), reply("Hello"))
    events = await run(server.loop(sleep=no_wait), [user("hi")], StubTools())
    assert [r["reason"] for r in of(events, "retry")] == ["Stream ended without finish_reason"]
    assert [it.message["content"] for it in items(events)] == ["Hello"]


@pytest.mark.parametrize(
    ("reason", "retried"), [("error", True), ("network_error", True), ("content_filter", False)]
)
async def test_failed_finish_reasons(reason: str, retried: bool) -> None:
    server = Server(sse(delta(content="Hel"), finish(reason), "data: [DONE]"), reply("Hello"))
    events = await run(server.loop(sleep=no_wait), [user("hi")], StubTools())
    assert len(server.bodies) == (2 if retried else 1)
    if retried:
        assert [it.message["content"] for it in items(events)] == ["Hello"]
    else:
        assert items(events) == [] and events[-1].data["stop"] == "error"
        assert of(events, "error")[0]["message"] == f"Provider finish_reason: {reason}"


async def test_no_retry_after_an_eager_tool_started() -> None:
    tools = StubTools(delays={"describe_component": 10})
    error = {"error": {"code": 502, "message": "Provider disconnected"}}

    async def body() -> AsyncIterator[bytes]:
        yield sse_bytes(call(0, '{"name": "a"}', "c0", "describe_component"))
        await tools.started.wait()
        yield sse_bytes(error)

    server = Server(lambda: httpx.Response(200, content=body()))
    events = []
    async for event in server.loop(sleep=no_wait).run_turn(
        turn([user("go")]), tools, asyncio.Event()
    ):
        events.append(event)
        if event.type == "turn.end":
            assert tools.stopped == ["c0"]  # the tool's end comes before the turn's
    assert len(server.bodies) == 1 and of(events, "retry") == [] and items(events) == []
    assert events[-1].data["stop"] == "error" and "c0" not in tools.spans


async def test_cancel_during_retry_backoff() -> None:
    limited = httpx.Response(429, headers={"retry-after": "5"}, json={"error": {"message": "slow"}})
    server = Server(limited, reply("never"))
    cancel, events = asyncio.Event(), []
    async with asyncio.timeout(2):
        async for event in server.loop().run_turn(turn([user("hi")]), StubTools(), cancel):
            events.append(event)
            if event.type == "retry":
                cancelled_at = time.perf_counter()
                cancel.set()
    assert time.perf_counter() - cancelled_at < 0.2
    assert events[-1].data == {"stop": "cancelled", "steps": 1} and len(server.bodies) == 1


async def test_a_bug_still_ends_the_turn() -> None:
    def broken() -> httpx.Response:
        raise RuntimeError("boom")

    events = await run(Server(broken).loop(), [user("hi")], StubTools())
    assert [e.type for e in events] == ["request.start", "error", "turn.end"]
    assert events[1].data["kind"] == "internal" and events[-1].data["stop"] == "error"


async def test_errors_that_are_not_retried() -> None:
    overflow = httpx.Response(
        400, json={"error": {"message": "prompt is too long: 213462 tokens > 200000"}}
    )
    server = Server(overflow)
    events = await run(server.loop(), [user("hi")], StubTools())
    assert of(events, "error") == [
        {
            "kind": "context_overflow",
            "message": "HTTP 400: prompt is too long: 213462 tokens > 200000",
            "retryable": False,
        }
    ]
    assert events[-1].data["stop"] == "error" and len(server.bodies) == 1

    async def sleep(seconds: float) -> None:
        pass

    server = Server(*(httpx.Response(503, text="busy") for _ in range(4)))
    events = await run(server.loop(sleep=sleep), [user("hi")], StubTools())
    assert len(server.bodies) == 4 and len(of(events, "retry")) == 3  # max_retries=3
    assert events[-1].data["stop"] == "error"

    server = Server(sse(delta(content="Hel"), "data: {not json"))
    events = await run(server.loop(), [user("hi")], StubTools())
    (error,) = of(events, "error")
    assert error["kind"] == "stream" and error["message"].startswith("Invalid stream data")
    assert events[-1].data["stop"] == "error" and len(server.bodies) == 1


async def test_reasoning_details_round_trip_verbatim() -> None:
    fragments = [
        {"type": "reasoning.text", "text": "Let me", "index": 0, "format": "anthropic-claude-v1"},
        {"type": "reasoning.text", "text": " think", "index": 0, "signature": None},
        {"type": "reasoning.text", "signature": "sig==", "index": 0, "x_unknown": {"k": 1}},
        {"type": "reasoning.encrypted", "data": "ab", "index": 1, "id": "r1"},
        {"type": "reasoning.encrypted", "data": "cd", "index": 1},
    ]
    server = Server(
        sse(
            delta(reasoning="Let me", reasoning_details=fragments[:1]),
            delta(reasoning=" think", reasoning_details=fragments[1:3]),
            delta(reasoning_details=fragments[3:]),
            call(0, '{"name": "a"}', "c0", "describe_component"),
            finish("tool_calls"),
        ),
        reply("done"),
        reply("again"),
    )
    tools = StubTools()
    history = [user("think")]
    events = await run(server.loop(), history, tools)
    assert [e["text"] for e in of(events, "reasoning.delta")] == ["Let me", " think"]
    assistant = items(events)[0]
    assert assistant.message["reasoning_details"] == [
        {
            "type": "reasoning.text",
            "text": "Let me think",
            "index": 0,
            "format": "anthropic-claude-v1",
            "signature": "sig==",
            "x_unknown": {"k": 1},
        },
        {"type": "reasoning.encrypted", "data": "abcd", "index": 1, "id": "r1"},
    ]
    assert server.messages(1)[2] == assistant.message
    # A fresh loop (no caches) replays the persisted message byte for byte.
    history = [*history, *(persisted(it) for it in items(events)), user("more")]
    await run(
        OurLoop(http_client=httpx.AsyncClient(transport=httpx.MockTransport(server))),
        history,
        tools,
    )
    assert server.bodies[2].startswith(server.bodies[1][:-2])


async def test_split_surrogate_pair_is_sent_and_replayed_identically() -> None:
    server = Server(
        sse(
            delta(content="\ud83d"),
            delta(content="\ude00"),
            call(0, "{}", "c0", "describe_component"),
            finish("tool_calls"),
        ),
        reply("ok"),
        reply("again"),
    )
    history = [user("smile")]
    events = await run(server.loop(), history, StubTools())
    assert events[-1].data["stop"] == "end_turn" and len(server.bodies) == 2
    assert server.messages(1)[2]["content"] == "\U0001f600" and server.bodies[1].isascii()
    history += [*(persisted(it) for it in items(events)), user("more")]
    await run(server.loop(), history, StubTools())  # fresh loop: bytes rebuilt from the log
    assert server.bodies[2].startswith(server.bodies[1][:-2])


async def test_empty_answer_is_not_replayed() -> None:
    server = Server(sse(finish("stop"), "data: [DONE]"), reply("there"))
    history = [user("hi")]
    events = await run(server.loop(), history, StubTools())
    assert items(events)[0].message == {"role": "assistant", "content": None}
    history += [*(persisted(it) for it in items(events)), user("hello?")]
    await run(server.loop(), history, StubTools())
    assert [m["content"] for m in server.messages(1)] == [SYSTEM, "hi", "hello?"]


async def test_max_steps_stops_after_exactly_n_requests() -> None:
    server = Server(
        *(sse(call(0, "{}", f"c{i}", "describe_component"), finish("tool_calls")) for i in range(3))
    )
    history = [user("loop forever")]
    tools = StubTools()
    events = await run(server.loop(), history, tools, limits=Limits(max_steps=2))
    assert len(server.bodies) == 2
    assert events[-1].data == {"stop": "max_steps", "steps": 2}
    assert items(events)[-1].message["role"] == "tool"
    assert_no_orphans(history + items(events))
    # The call of the last allowed step never runs (not even early, though it is read-only):
    # no request could send its result. It still gets a result, so nothing is orphaned.
    assert dict(tools.run_counts) == {"c0": 1}
    assert items(events)[-1].message == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "Not run: the turn reached its step limit",
    }


async def test_cancel_after_the_last_allowed_answer_ends_cancelled() -> None:
    """The last allowed step's answer is complete (finish_reason plus usage), and the cancel
    comes while the loop drains the body's end: the user stopped the turn (contract rule 6),
    so its call gets "Cancelled by user" and the turn does not end with max_steps."""
    usage = {"prompt_tokens": 10, "completion_tokens": 1}
    stall = Stall(sse_bytes(call(0, "{}", "c0", "describe_component"), finish("tool_calls", usage)))
    server, tools, cancel = Server(httpx.Response(200, stream=stall)), StubTools(), asyncio.Event()

    async def cancel_while_draining() -> None:
        await stall.stalled.wait()
        cancel.set()

    canceller = asyncio.ensure_future(cancel_while_draining())
    history = [user("go")]
    async with asyncio.timeout(2):
        events = await run(server.loop(), history, tools, cancel, limits=Limits(max_steps=1))
    await canceller
    assert events[-1].data == {"stop": "cancelled", "steps": 1}
    answer, result = items(events)
    assert answer.status == "complete" and answer.message["tool_calls"][0]["id"] == "c0"
    assert result.message == {"role": "tool", "tool_call_id": "c0", "content": "Cancelled by user"}
    assert tools.run_counts == {}
    assert_no_orphans(history + items(events))


async def test_cancelled_resume_at_the_step_limit_ends_cancelled() -> None:
    """A resume whose pending call belongs to the last allowed step, with the cancel already set."""
    history = spent_history()[:-1]  # c0 has no result yet
    cancel = asyncio.Event()
    cancel.set()
    events = await run(
        Server().loop(), history, StubTools(), cancel, resume=Resume("crash"), limits=Limits(1)
    )
    assert events[-1].data == {"stop": "cancelled", "steps": 1}
    assert items(events)[-1].message["content"] == "Cancelled by user"


async def test_budget_stops_before_the_next_request() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0.002}
    server = Server(sse(call(0, "{}", "c0", "describe_component"), finish("tool_calls", usage)))
    events = await run(server.loop(), [user("go")], StubTools(), limits=Limits(max_cost_usd=0.001))
    assert len(server.bodies) == 1 and events[-1].data == {"stop": "budget", "steps": 1}


async def test_unknown_cost_stops_a_budgeted_turn() -> None:
    no_cost = {"prompt_tokens": 10, "completion_tokens": 5}  # BYOK: tokens, but no cost
    server = Server(sse(call(0, "{}", "c0", "describe_component"), finish("tool_calls", no_cost)))
    events = await run(server.loop(), [user("go")], StubTools(), limits=Limits(max_cost_usd=1.0))
    assert of(events, "error") == [
        {
            "kind": "budget_unenforceable",
            "message": "max_cost_usd is set but the endpoint reports no cost",
            "retryable": False,
        }
    ]
    assert len(server.bodies) == 1 and events[-1].data == {"stop": "budget", "steps": 1}
    answer = Server(sse(delta(content="hi"), finish("stop", no_cost)))  # no further request due
    events = await run(answer.loop(), [user("go")], StubTools(), limits=Limits(max_cost_usd=1.0))
    assert of(events, "error") == [] and events[-1].data["stop"] == "end_turn"


def spent_history() -> list[Item]:
    """An older turn, then a turn that spent 1 step and $0.5 before its worker crashed."""
    tc = {
        "id": "c0",
        "type": "function",
        "function": {"name": "describe_component", "arguments": "{}"},
    }
    usage = {"step": 1, "cost_usd": 0.5, "cost_source": "provider"}
    return [
        user("old"),
        Item("a0", "t0", {"role": "assistant", "content": "old answer"}, usage=usage),
        user("go"),
        Item("a1", "t1", {"role": "assistant", "content": None, "tool_calls": [tc]}, usage=usage),
        Item("r1", "t1", {"role": "tool", "tool_call_id": "c0", "content": "ok"}),
    ]


async def test_resume_counts_the_steps_and_cost_already_spent() -> None:
    crash = Resume("crash")
    server = Server()  # any request would fail: the limits must stop the turn first
    events = await run(server.loop(), spent_history(), StubTools(), resume=crash, limits=Limits(1))
    assert events[-1].data == {"stop": "max_steps", "steps": 1} and server.bodies == []
    budget = Limits(max_cost_usd=0.5)
    events = await run(server.loop(), spent_history(), StubTools(), resume=crash, limits=budget)
    assert events[-1].data == {"stop": "budget", "steps": 1} and server.bodies == []
    server = Server(reply("done"))  # the older turn's step and cost do not count
    events = await run(server.loop(), spent_history(), StubTools(), resume=crash, limits=Limits(2))
    assert of(events, "request.start") == [{"step": 2, "attempt": 1}]
    assert events[-1].data == {"stop": "end_turn", "steps": 2}


async def test_truncated_output_is_kept_but_never_run() -> None:
    tools = StubTools(delays={"describe_component": 1})
    server = Server(
        sse(
            delta(content="Writing it."),
            call(0, '{"name": "a"}', "c0", "describe_component"),  # complete: started early
            call(1, '{"path": "a.pipe", "content": "{\\"comp', "c1", "write_file"),
            finish("length", {"completion_tokens": 4096, "cost": 0.01}),
            "data: [DONE]",
        )
    )
    events = await run(server.loop(), [user("go")], tools)
    (item,) = items(events)
    assert item.status == "incomplete" and item.usage is not None
    assert item.message == {"role": "assistant", "content": "Writing it."}  # no tool_calls
    assert item.usage["cost_usd"] == 0.01 and of(events, "usage")[0]["cost_usd"] == 0.01
    assert of(events, "error") == [
        {
            "kind": "output_truncated",
            "message": "Output truncated at max_tokens=4096",
            "retryable": False,
        }
    ]
    assert events[-1].data == {
        "stop": "error",
        "steps": 1,
        "error": "Output truncated at max_tokens=4096",
    }
    assert tools.run_counts["c1"] == 0 and "c0" not in tools.spans  # the early read was stopped
    assert len(server.bodies) == 1


@pytest.mark.parametrize(
    "usage",
    [{"step": 1, "cost_usd": 0.01, "cost_source": "provider"}, None],
    ids=["truncated", "cancelled"],
)
async def test_crash_after_an_incomplete_answer_asks_again(usage: dict[str, Any] | None) -> None:
    cut = Item("a1", "t1", {"role": "assistant", "content": "Writing"}, "incomplete", usage=usage)
    server = Server(reply("Done."))
    events = await run(server.loop(), [user("go"), cut], StubTools(), resume=Resume("crash"))
    assert of(events, "request.start") == [{"step": 2, "attempt": 1}]  # not end_turn
    assert server.messages(0) == [{"role": "system", "content": SYSTEM}, user("go").message]
    assert items(events)[0].message["content"] == "Done." and events[-1].data["stop"] == "end_turn"


async def test_compaction_item_resets_the_prefix() -> None:
    summary = {"role": "user", "content": "[harness] Conversation summary: built a pipe."}
    history = [
        user("old"),
        Item("a1", "t1", {"role": "assistant", "content": "old answer"}),
        Item("s1", "t2", summary, compaction=True),
        user("new"),
    ]
    server = Server(reply("ok"))
    await run(server.loop(), history, StubTools())
    assert server.messages(0) == [
        {"role": "system", "content": SYSTEM},
        summary,
        history[-1].message,
    ]


async def test_strict_endpoint_compat_flags() -> None:
    strict = ModelConfig(
        base_url="http://127.0.0.1:9/v1",
        model="qwen/qwen3-coder",
        kind="openai_compat",
        reasoning={"effort": "low"},
        compat={
            "reasoning_param": "none",
            "stream_usage": False,
            "max_tokens_field": "max_tokens",
            "other": 1,
        },
    )
    server = Server(reply("fine"))
    events = await run(server.loop(), [user("hi")], StubTools(), model=strict)
    body = json.loads(server.bodies[0])
    assert not {"reasoning", "reasoning_effort", "stream_options", "session_id"} & body.keys()
    assert body["max_tokens"] == 4096 and events[-1].data["stop"] == "end_turn"


async def test_prefix_is_byte_identical_across_three_requests() -> None:
    server = Server(
        sse(call(0, '{"name": "a"}', "c0", "describe_component"), finish("tool_calls")),
        sse(call(0, '{"name": "b"}', "c1", "describe_component"), finish("tool_calls")),
        reply("done"),
    )
    events = await run(server.loop(), [user("go")], StubTools())
    assert events[-1].data == {"stop": "end_turn", "steps": 3}
    first, second, third = server.bodies
    assert second.startswith(first[:-2] + b",") and third.startswith(second[:-2] + b",")


async def test_pooled_client_per_base_url() -> None:
    loop = OurLoop()
    a = loop._client("http://127.0.0.1:1/v1")
    assert loop._client("http://127.0.0.1:1/v1") is a
    assert loop._client("http://127.0.0.1:2/v1") is not a
    await loop.aclose()
    assert a.is_closed


@contextlib.contextmanager
def socket_server(
    respond: Callable[[BaseHTTPRequestHandler], None],
) -> Iterator[ThreadingHTTPServer]:
    """A real HTTP/1.1 server on 127.0.0.1 that streams chunked SSE via `respond`."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            respond(self)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def write_chunk(handler: BaseHTTPRequestHandler, data: bytes) -> None:
    handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
    handler.wfile.flush()


def socket_model(server: ThreadingHTTPServer) -> ModelConfig:
    return ModelConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="openai/gpt-5")


async def test_one_connection_serves_every_step() -> None:
    ports: list[int] = []
    replies = [  # the first answer is complete at its usage chunk, the second at [DONE]
        sse_bytes(
            call(0, "{}", "c0", "describe_component"),
            finish("tool_calls"),
            finish("tool_calls", {"cost": 1}),
            "data: [DONE]",
        ),
        sse_bytes(delta(content="ok"), finish("stop"), "data: [DONE]"),
    ]

    def respond(handler: BaseHTTPRequestHandler) -> None:
        ports.append(handler.client_address[1])
        write_chunk(handler, replies[len(ports) - 1])
        write_chunk(handler, b"")

    loop = OurLoop()
    with socket_server(respond) as server:
        events = await run(loop, [user("hi")], StubTools(), model=socket_model(server))
        await loop.aclose()
    assert events[-1].data == {"stop": "end_turn", "steps": 2}
    assert len(ports) == 2 and len(set(ports)) == 1


async def test_cancel_closes_a_stalled_socket() -> None:
    release = threading.Event()

    def respond(handler: BaseHTTPRequestHandler) -> None:
        write_chunk(handler, sse_bytes(delta(content="partial")))
        release.wait(5)

    loop, cancel = OurLoop(), asyncio.Event()
    with socket_server(respond) as server:
        try:
            async with asyncio.timeout(2):
                history = [user("hi")]
                async for event in loop.run_turn(
                    turn(history, model=socket_model(server)), StubTools(), cancel
                ):
                    if event.type == "text.delta":
                        cancelled_at = time.perf_counter()
                        cancel.set()
            assert event.data["stop"] == "cancelled" and time.perf_counter() - cancelled_at < 0.2
        finally:
            release.set()
            await loop.aclose()


# The answer is complete at [DONE], or at a usage chunk once finish_reason has arrived.
COMPLETE = {
    "done": sse_bytes(delta(content="The answer."), finish("stop"), "data: [DONE]"),
    "usage": sse_bytes(delta(content="The answer."), finish("stop"), finish("stop", {"cost": 1})),
}


@pytest.mark.parametrize("end", COMPLETE)
async def test_answer_survives_a_connection_dropped_after_it_completed(end: str) -> None:
    requests: list[int] = []

    def respond(handler: BaseHTTPRequestHandler) -> None:
        requests.append(1)
        write_chunk(handler, COMPLETE[end])
        handler.close_connection = True
        handler.connection.shutdown(socket.SHUT_RDWR)  # no terminating zero-length chunk

    loop = OurLoop(sleep=no_wait)
    with socket_server(respond) as server:
        events = await run(loop, [user("hi")], StubTools(), model=socket_model(server))
        await loop.aclose()
    assert events[-1].data == {"stop": "end_turn", "steps": 1}
    assert len(requests) == 1 and of(events, "retry") == []
    assert items(events)[0].message["content"] == "The answer."


@pytest.mark.parametrize("end", COMPLETE)
async def test_body_held_open_after_the_answer_does_not_block(end: str) -> None:
    release = threading.Event()

    def respond(handler: BaseHTTPRequestHandler) -> None:
        write_chunk(handler, COMPLETE[end])
        release.wait(5)
        with contextlib.suppress(OSError):
            write_chunk(handler, b"")

    loop = OurLoop()
    with socket_server(respond) as server:
        try:
            start = time.perf_counter()
            events = await run(loop, [user("hi")], StubTools(), model=socket_model(server))
            assert time.perf_counter() - start < 0.5
            assert events[-1].data == {"stop": "end_turn", "steps": 1}
        finally:
            release.set()
            await loop.aclose()

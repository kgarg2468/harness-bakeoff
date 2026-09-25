"""PydanticLoop against a scripted local SSE server, with a stub ToolHost."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
import warnings
from dataclasses import asdict
from typing import Any

import pytest
from pai_sse_server import Reply, SSEServer, chunk, done, text, tool_call
from pydantic_ai import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from bakeoff.pydantic_version import PydanticLoop, mapping
from bakeoff.shared.contract import (
    Decision,
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

PATH = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
PIPELINE = {  # like the engine's validate_pipeline: `pipeline` is a free-form object
    "type": "object",
    "properties": {
        "pipeline": {"type": "object", "description": "Inline pipeline definition"},
        "path": {"type": "string"},
    },
}
SPECS = [
    ToolSpec("validate_pipeline", "Validate a pipeline.", PIPELINE, read_only=True),
    ToolSpec("read_file", "Read a file.", PATH, read_only=True),
    ToolSpec(
        "write_file",
        "Write a file.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
]


class StubTools:
    """ToolHost stand-in: a decision per tool name, instant results unless `slow`, and a log
    shared with the test's event consumer so ordering can be checked."""

    def __init__(self, rules: dict[str, Decision] | None = None, slow: str | None = None) -> None:
        self.rules = rules or {}
        self.slow = slow
        self.runs: list[ToolCall] = []
        self.log: list[str] = []
        self.started = asyncio.Event()

    def specs(self) -> list[ToolSpec]:
        return SPECS

    def check(self, call: ToolCall) -> Decision:
        return self.rules.get(call.name, "allow")

    async def run(self, call: ToolCall) -> ToolResult:
        self.log.append(f"tool.start:{call.id}")
        self.runs.append(call)
        self.started.set()
        if call.name == self.slow:
            await asyncio.sleep(5)
        if "path" not in json.loads(call.arguments):
            return ToolResult(call.id, False, "invalid arguments: 'path' is a required property")
        return ToolResult(call.id, True, f"{call.name} ok")


@pytest.fixture
async def loop():
    loop = PydanticLoop()
    yield loop
    await loop.aclose()


def config(srv: SSEServer, **kw: Any) -> ModelConfig:
    kw.setdefault("model", "anthropic/claude-test")
    kw.setdefault("max_retries", 0)
    return ModelConfig(base_url=srv.base_url, **kw)


def user(content: str, compaction: bool = False) -> Item:
    message = {"role": "user", "content": content}
    return Item(id=uuid.uuid4().hex, turn_id="t0", message=message, compaction=compaction)


def turn(history: list[Item], model: ModelConfig, **kw: Any) -> TurnInput:
    kw.setdefault("resume", None)
    kw.setdefault("limits", Limits())
    return TurnInput("th", uuid.uuid4().hex, "SYS", history, model=model, **kw)


async def run(
    loop: PydanticLoop, inp: TurnInput, tools: StubTools, cancel: asyncio.Event | None = None
) -> list[Event]:
    events = []
    async for event in loop.run_turn(inp, tools, cancel or asyncio.Event()):
        events.append(event)
        tools.log.append(f"{event.type}:{event.data.get('call_id', '')}")
    return events


def of(events: list[Event], type_: str) -> list[dict[str, Any]]:
    return [e.data for e in events if e.type == type_]


def items(events: list[Event]) -> list[Item]:
    """The turn's items after a trip through JSON, as the session log would store them."""
    return [Item(**json.loads(json.dumps(asdict(d["item"])))) for d in of(events, "item")]


def native_item(response: ModelResponse) -> Item:
    return Item(
        id=uuid.uuid4().hex,
        turn_id="t0",
        message=mapping.to_openai(response)[0],
        native=mapping.dump(response),
    )


async def test_text_turn_streams_deltas_and_reports_billed_cost(loop, capfd):
    reply = [": OPENROUTER PROCESSING", *text("Hel", "lo"), done(cost=0.0012, prompt=12, cached=4)]
    with SSEServer(Reply(reply)) as srv:
        events = await run(loop, turn([user("hi")], config(srv, session_id="s-1")), StubTools())

    assert [e.type for e in events] == [
        "request.start",
        "text.delta",
        "text.delta",
        "item",
        "usage",
        "turn.end",
    ]
    assert [d["text"] for d in of(events, "text.delta")] == ["Hel", "lo"]
    [answer] = items(events)
    assert answer.message == {"role": "assistant", "content": "Hello"}
    assert answer.native is not None
    assert of(events, "usage") == [
        {
            "step": 1,
            "input_tokens": 12,
            "output_tokens": 5,
            "cached_tokens": 4,
            "reasoning_tokens": 0,
            "cost_usd": 0.0012,
            "cost_source": "provider",
        }
    ]
    assert of(events, "turn.end") == [{"stop": "end_turn", "steps": 1}]
    [body] = srv.requests
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    assert body["session_id"] == "s-1"
    assert body["cache_control"] == {"type": "ephemeral"}
    assert (body["max_tokens"], body["temperature"]) == (4096, 0.0)
    assert capfd.readouterr() == ("", "")  # rule 1 / I5: nothing on stdout or stderr


async def test_tool_calls_run_with_raw_arguments_and_one_item_per_result(loop):
    tools = StubTools()
    calls = tool_call(0, "c1", "read_file", '{"pa', 'th": "a.txt"}') + tool_call(
        1, "c2", "validate_pipeline", '{"path":"p.pipe"}'
    )
    with SSEServer(Reply([*calls, done("tool_calls")]), Reply([*text("done"), done()])) as srv:
        events = await run(loop, turn([user("go")], config(srv)), tools)

    assert [c.arguments for c in tools.runs] == ['{"path": "a.txt"}', '{"path":"p.pipe"}']
    assert of(events, "tool_call.ready")[0] == {
        "call_id": "c1",
        "name": "read_file",
        "arguments": '{"path": "a.txt"}',
    }
    # tool.start comes from the ToolHost, after the consumer has seen the call's tool_call.ready.
    for call_id in ("c1", "c2"):
        assert tools.log.index(f"tool_call.ready:{call_id}") < tools.log.index(
            f"tool.start:{call_id}"
        )
    call_item, result1, result2, final = items(events)
    assert [c["function"]["arguments"] for c in call_item.message["tool_calls"]] == [
        '{"path": "a.txt"}',
        '{"path":"p.pipe"}',
    ]
    assert result1.message == {"role": "tool", "tool_call_id": "c1", "content": "read_file ok"}
    assert (result1.native, result2.native is not None) == (None, True)  # one native per message
    assert final.message == {"role": "assistant", "content": "done"}
    assert srv.requests[1]["messages"][2:] == [
        call_item.message,
        result1.message,
        result2.message,
    ]
    assert of(events, "turn.end") == [{"stop": "end_turn", "steps": 2}]
    # Schemas reach the wire open: nothing forbids the keys of the free-form `pipeline` object.
    wire_schemas = {
        t["function"]["name"]: t["function"]["parameters"] for t in srv.requests[0]["tools"]
    }
    assert wire_schemas["validate_pipeline"]["properties"]["pipeline"]["type"] == "object"
    assert "additionalProperties" not in json.dumps(wire_schemas)


async def test_bad_arguments_become_a_retry_prompt_and_the_turn_continues(loop):
    replies = [Reply([*tool_call(0, "c1", "read_file", "{}"), done("tool_calls")])]
    with SSEServer(*replies, Reply([*text("sorry"), done()])) as srv:
        events = await run(loop, turn([user("go")], config(srv)), StubTools())

    result = items(events)[1].message
    assert result["content"].startswith("invalid arguments: 'path' is a required property")
    assert result["content"].endswith("Fix the errors and try again.")
    assert srv.requests[1]["messages"][-1] == result
    assert of(events, "turn.end")[0]["stop"] == "end_turn"


async def _pause_for_write(loop: PydanticLoop, srv: SSEServer, tools: StubTools) -> list[Item]:
    """Turn 1: validate (allowed) runs, write (ask) pauses. Returns the persisted history."""
    first = [user("build it")]
    events = await run(loop, turn(first, config(srv)), tools)
    assert of(events, "permission.asked") == [
        {"call_id": "c2", "name": "write_file", "arguments": '{"path": "a.pipe", "content": "x"}'}
    ]
    assert of(events, "turn.end") == [{"stop": "paused", "steps": 1, "pending": ["c2"]}]
    return first + items(events)


def _batch() -> Reply:
    calls = tool_call(0, "c1", "validate_pipeline", '{"path": "a.pipe"}') + tool_call(
        1, "c2", "write_file", '{"path": "a.pipe", "content": "x"}'
    )
    return Reply([*calls, done("tool_calls")])


async def test_approval_pauses_then_resumes_in_a_fresh_loop(loop):
    tools = StubTools({"write_file": "ask"})
    with SSEServer(_batch(), Reply([*text("written"), done()])) as srv:
        history = await _pause_for_write(loop, srv, tools)
        assert [c.id for c in tools.runs] == ["c1"]

        fresh = PydanticLoop()
        resume = Resume("approval", {"c2": "allow"})
        events = await run(fresh, turn(history, config(srv), resume=resume), tools)
        await fresh.aclose()

    assert [c.id for c in tools.runs] == ["c1", "c2"]  # validate not re-run, write runs once
    first, second = srv.requests
    assert second["messages"][: len(first["messages"])] == first["messages"]  # append-only
    assert [m["role"] for m in second["messages"][2:]] == ["assistant", "tool", "tool"]
    assert [i.message["role"] for i in items(events)] == ["tool", "assistant"]
    assert of(events, "turn.end") == [{"stop": "end_turn", "steps": 1}]


async def test_deny_sends_the_reason_to_the_model_without_running_the_tool(loop):
    tools = StubTools({"write_file": "ask"})
    with SSEServer(_batch(), Reply([*text("ok"), done()])) as srv:
        history = await _pause_for_write(loop, srv, tools)
        resume = Resume("approval", {"c2": "deny"}, reason="not now")
        events = await run(loop, turn(history, config(srv), resume=resume), tools)

    assert [c.id for c in tools.runs] == ["c1"]
    denial = {"role": "tool", "tool_call_id": "c2", "content": "Denied by user: not now"}
    assert items(events)[0].message == denial
    assert srv.requests[1]["messages"][-1] == denial


async def test_crash_resume_rechecks_open_calls(loop):
    response = ModelResponse(
        parts=[
            ToolCallPart("read_file", '{"path": "a"}', "c1"),
            ToolCallPart("write_file", '{"path": "b", "content": "x"}', "c2"),
        ]
    )
    history = [user("go"), native_item(response)]
    tools = StubTools({"write_file": "ask"})
    with SSEServer(Reply([*text("done"), done()])) as srv:
        events = await run(loop, turn(history, config(srv), resume=Resume("crash")), tools)
        assert of(events, "turn.end") == [{"stop": "paused", "steps": 0, "pending": ["c2"]}]
        assert tools.runs == []

        approve = Resume("approval", {"c2": "allow"})
        events = await run(loop, turn(history, config(srv), resume=approve), tools)

    assert [c.id for c in tools.runs] == ["c1", "c2"]
    assert [m["role"] for m in srv.requests[0]["messages"][-3:]] == ["assistant", "tool", "tool"]
    assert of(events, "turn.end")[0]["stop"] == "end_turn"


async def test_crash_resume_after_a_persisted_result_does_not_rerun_the_tool(loop):
    call = ModelResponse(parts=[ToolCallPart("read_file", '{"path": "a"}', "c1")])
    result = ModelRequest(parts=[ToolReturnPart("read_file", "read_file ok", "c1")])
    history = [user("go"), native_item(call), native_item(result)]
    tools = StubTools()
    with SSEServer(Reply([*text("done"), done()])) as srv:
        events = await run(loop, turn(history, config(srv), resume=Resume("crash")), tools)

    assert tools.runs == []
    assert srv.requests[0]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "read_file ok",
    }
    assert [i.message["content"] for i in items(events)] == ["done"]


async def test_cancel_during_a_stall_stops_fast_and_replays_the_partial_answer(loop):
    thinking = chunk({"reasoning_details": [{"type": "reasoning.text", "text": "Hmm", "index": 0}]})
    tools = StubTools()
    cancel = asyncio.Event()
    with SSEServer(
        Reply([thinking, *text("Par")], stall=True), Reply([*text("ok"), done()])
    ) as srv:
        events: list[Event] = []
        async for event in loop.run_turn(turn([user("hi")], config(srv)), tools, cancel):
            events.append(event)
            if event.type == "text.delta":
                cancelled_at = time.monotonic()
                cancel.set()
        stopped_in = time.monotonic() - cancelled_at
        partial = items(events)
        history = [user("hi"), *partial, user("next")]
        await run(loop, turn(history, config(srv)), tools)

    assert stopped_in < 0.2
    assert events[-1] == Event("turn.end", {"stop": "cancelled", "steps": 1})
    assert [(i.status, i.message["content"]) for i in partial] == [("incomplete", "Par")]
    # Recorded behavior: pydantic-ai replays the interrupted response as it was, including the
    # unsigned reasoning (an endpoint that rejects unsigned reasoning answers this with a 400).
    assert srv.requests[1]["messages"][2] == {
        "role": "assistant",
        "content": "Par",
        "reasoning_details": [
            {
                "id": None,
                "format": None,
                "index": 0,
                "type": "reasoning.text",
                "text": "Hmm",
                "signature": None,
            }
        ],
    }


async def test_cancel_during_a_slow_tool_closes_the_open_call(loop):
    tools = StubTools(slow="read_file")
    cancel = asyncio.Event()
    calls = tool_call(0, "c1", "read_file", '{"path": "a"}') + tool_call(
        1, "c2", "validate_pipeline", '{"path": "b"}'
    )
    with SSEServer(Reply([*calls, done("tool_calls")]), Reply([*text("ok"), done()])) as srv:
        task = asyncio.create_task(run(loop, turn([user("go")], config(srv)), tools, cancel))
        await tools.started.wait()
        await asyncio.sleep(0.05)
        started = time.monotonic()
        cancel.set()
        events = await task
        stopped_in = time.monotonic() - started
        history = [user("go"), *items(events), user("next")]
        await run(loop, turn(history, config(srv)), tools)

    assert stopped_in < 0.2
    results = {i.message["tool_call_id"]: i.message["content"] for i in items(events)[1:]}
    assert results == {
        "c2": "validate_pipeline ok",
        "c1": "The tool call was interrupted before a result was produced.",
    }
    assert of(events, "turn.end") == [{"stop": "cancelled", "steps": 1}]
    tool_messages = [m for m in srv.requests[1]["messages"] if m["role"] == "tool"]
    assert sorted(m["tool_call_id"] for m in tool_messages) == ["c1", "c2"]  # no repair needed


async def test_max_steps_stops_after_exactly_that_many_requests(loop):
    looping = [
        Reply([*tool_call(0, f"c{i}", "read_file", '{"path": "a"}'), done("tool_calls")])
        for i in range(3)
    ]
    with SSEServer(*looping) as srv:
        events = await run(
            loop, turn([user("go")], config(srv), limits=Limits(max_steps=2)), StubTools()
        )

    assert len(srv.bodies) == 2
    assert of(events, "turn.end") == [{"stop": "max_steps", "steps": 2}]
    calls = [c["id"] for i in items(events) for c in i.message.get("tool_calls", [])]
    results = [i.message["tool_call_id"] for i in items(events) if i.message["role"] == "tool"]
    assert calls == results == ["c0", "c1"]  # no orphan calls


async def test_cost_without_provider_cost_is_labelled(loop):
    priced = [{**c, "model": "gpt-4o-mini"} for c in [*text("a"), done()]]
    unknown = [*text("b"), done()]
    with SSEServer(Reply(priced), Reply(unknown)) as srv:
        byok = config(srv, kind="openai_compat", model="gpt-4o-mini")
        estimated = of(await run(loop, turn([user("x")], byok), StubTools()), "usage")[0]
        missing = of(await run(loop, turn([user("y")], byok), StubTools()), "usage")[0]

    assert estimated["cost_source"] == "estimate" and estimated["cost_usd"] > 0
    assert (missing["cost_source"], missing["cost_usd"]) == ("none", None)


async def test_reasoning_details_round_trip(loop, record_property):
    meta = {"format": "anthropic-claude-v1", "index": 0}
    fragments = [
        {"type": "reasoning.text", "text": "Let me ", **meta},
        {"type": "reasoning.text", "text": "think.", **meta},
        {"type": "reasoning.text", "signature": "sig-1", **meta},  # metadata-only fragment
        {"type": "reasoning.encrypted", "data": "ENC", "format": "anthropic-claude-v1", "index": 1},
    ]
    stream = [chunk({"reasoning_details": [f]}) for f in fragments]
    with SSEServer(Reply([*stream, *text("42"), done()]), Reply([*text("ok"), done()])) as srv:
        model = config(srv, reasoning={"effort": "low"})
        events = await run(loop, turn([user("q")], model), StubTools())
        history = [user("q"), *items(events), user("again")]
        await run(loop, turn(history, model), StubTools())

    expected = [
        {"type": "reasoning.text", "text": "Let me think.", "signature": "sig-1", **meta},
        {"type": "reasoning.encrypted", "data": "ENC", "format": "anthropic-claude-v1", "index": 1},
    ]
    assert srv.requests[0]["reasoning"] == {"effort": "low"}  # openrouter_reasoning setting
    assert [d["text"] for d in of(events, "reasoning.delta")] == ["Let me ", "think."]
    assert items(events)[0].message["reasoning_details"] == expected
    replayed = srv.requests[1]["messages"][2]["reasoning_details"]
    semantic = {"type", "text", "signature", "data", "format", "index"}
    assert [
        {k: v for k, v in d.items() if k in semantic and v is not None} for d in replayed
    ] == expected
    record_property("reasoning_details_byte_equal", replayed == expected)  # info only
    record_property("reasoning_details_replayed", json.dumps(replayed))


async def test_byok_thinking_reaches_the_wire_for_a_non_openai_model(loop):
    strict = {
        "max_tokens_field": "max_tokens",
        "developer_role": True,
        "stream_usage": False,
        "reasoning_param": "none",
        "unknown_flag": 1,
    }
    flags = [{}, strict, {"reasoning_param": "openrouter"}]
    with SSEServer(*[Reply([*text("ok"), done()]) for _ in flags]) as srv:
        for compat in flags:
            byok = config(
                srv,
                kind="openai_compat",
                model="qwen3-coder",
                reasoning={"effort": "low"},
                compat=compat,
            )
            await run(loop, turn([user("hi")], byok), StubTools())

    default, strict_body, openrouter_style = srv.requests
    assert (default["reasoning_effort"], default["max_completion_tokens"]) == ("low", 4096)
    assert default["stream_options"] == {"include_usage": True}
    assert strict_body["max_tokens"] == 4096 and strict_body["messages"][0]["role"] == "developer"
    rejected = {"reasoning", "reasoning_effort", "stream_options", "max_completion_tokens"}
    assert not rejected & set(strict_body)
    assert openrouter_style["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in openrouter_style
    # Why CompatProvider exists: the stock provider's name-based profile has no thinking here.
    stock = OpenAIChatModel(
        "qwen3-coder", provider=OpenAIProvider(base_url=srv.base_url, api_key="k")
    )
    assert not stock.profile.get("supports_thinking")


async def test_compaction_item_resets_the_request_prefix(loop):
    summary = "[harness] Conversation summary: the user likes pipes."
    old = native_item(ModelResponse(parts=[TextPart("old answer")]))
    history = [user("old question"), old, user(summary, compaction=True), user("new question")]
    with SSEServer(Reply([*text("ok"), done()])) as srv:
        await run(loop, turn(history, config(srv)), StubTools())

    assert srv.requests[0]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": summary},
        {"role": "user", "content": "new question"},
    ]


async def test_rate_limit_retry_is_visible_and_honours_retry_after(loop):
    limited = Reply(
        status=429,
        headers={"retry-after-ms": "30"},
        body={"error": {"message": "slow down", "code": 429}},
    )
    with SSEServer(limited, Reply([*text("ok"), done()])) as srv:
        events = await run(loop, turn([user("hi")], config(srv, max_retries=2)), StubTools())

    assert len(srv.bodies) == 2
    assert [e.type for e in events][:3] == ["request.start", "retry", "request.start"]
    retry = of(events, "retry")[0]
    assert (retry["attempt"], retry["status"], retry["reason"]) == (2, 429, "HTTP 429")
    assert retry["wait_ms"] >= 25
    assert of(events, "request.start") == [{"step": 1, "attempt": 1}, {"step": 1, "attempt": 2}]
    assert of(events, "turn.end") == [{"stop": "end_turn", "steps": 1}]


async def test_provider_error_ends_the_turn_with_error(loop):
    rejected = Reply(status=400, body={"error": {"message": "unsigned reasoning", "code": 400}})
    with SSEServer(rejected) as srv:
        events = await run(loop, turn([user("hi")], config(srv)), StubTools())

    assert of(events, "error")[0]["retryable"] is False
    end = of(events, "turn.end")[0]
    assert end["stop"] == "error" and "unsigned reasoning" in end["error"]


async def test_cost_limit_ends_the_turn_with_budget(loop):
    costly = Reply(
        [*tool_call(0, "c0", "read_file", '{"path": "a"}'), done("tool_calls", cost=0.002)]
    )
    with SSEServer(costly) as srv:
        limits = Limits(max_cost_usd=0.001)
        events = await run(loop, turn([user("go")], config(srv), limits=limits), StubTools())

    assert of(events, "turn.end") == [{"stop": "budget", "steps": 1}]
    # Recorded behavior: the library drops the response that crossed the limit from history,
    # so it gets no item (and no usage event), and its tool call never runs.
    assert of(events, "item") == []


async def test_reasoning_model_with_temperature_stays_quiet(capfd):
    with (
        warnings.catch_warnings(record=True) as caught,
        SSEServer(Reply([*text("ok"), done()])) as srv,
    ):
        loop = PydanticLoop()
        await run(loop, turn([user("hi")], config(srv, model="openai/gpt-5")), StubTools())
        await loop.aclose()

    assert "temperature" not in srv.requests[0]  # the library drops it for reasoning models
    assert (caught, capfd.readouterr()) == ([], ("", ""))


async def test_error_chunk_mid_stream_is_not_retried(loop):
    error = {**chunk(finish="error"), "error": {"code": 502, "message": "upstream died"}}
    with SSEServer(Reply([*text("Hal"), error])) as srv:
        events = await run(loop, turn([user("hi")], config(srv, max_retries=2)), StubTools())

    # Recorded behavior: the OpenAI SDK retries only before a stream starts, and pydantic-ai does
    # not retry a failed stream, so this is one attempt and an error turn keeping the partial text.
    assert len(srv.bodies) == 1
    assert [(i.status, i.message["content"]) for i in items(events)] == [("incomplete", "Hal")]
    assert of(events, "error")[0]["retryable"] is True
    assert of(events, "turn.end")[0]["stop"] == "error"

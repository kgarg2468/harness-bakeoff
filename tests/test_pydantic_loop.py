"""PydanticLoop against a scripted local SSE server, with a stub ToolHost."""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
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
from bakeoff.pydantic_version import loop as loop_module
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
        self.checked: list[str] = []
        self.log: list[str] = []
        self.started = asyncio.Event()

    def specs(self) -> list[ToolSpec]:
        return SPECS

    def check(self, call: ToolCall) -> Decision:
        self.checked.append(call.id)
        return self.rules.get(call.name, "allow")

    async def run(self, call: ToolCall) -> ToolResult:
        """Like the shared ToolHost: invalid_args, then deny rules, then the tool (which fails for
        the path "missing")."""
        self.log.append(f"tool.start:{call.id}")
        self.runs.append(call)
        self.started.set()
        if call.name == self.slow:
            await asyncio.sleep(5)
        try:
            args = json.loads(call.arguments)
        except ValueError:
            invalid = f"Invalid arguments for {call.name}: not valid JSON"
            return ToolResult(call.id, False, invalid, error="invalid_args")
        if "path" not in args:
            invalid = "invalid arguments: 'path' is a required property"
            return ToolResult(call.id, False, invalid, error="invalid_args")
        if self.check(call) == "deny":
            return ToolResult(call.id, False, f"Denied by permission rules: {call.name}", "denied")
        if args["path"] == "missing":
            return ToolResult(call.id, False, "No such file: missing", error="failed")
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
    # Every item carries the native of exactly what it shows: a crash between two items of the
    # batch keeps the first one's result (see the crash test below).
    assert [[p["tool_call_id"] for p in i.native["parts"]] for i in (result1, result2)] == [
        ["c1"],
        ["c2"],
    ]
    assert [len(i.native["parts"]) for i in (call_item, final)] == [2, 1]
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


async def test_denied_and_failed_calls_use_the_library_outcomes(loop):
    """Only bad arguments become ModelRetry ("fix it and try again"). A denial is ToolDenied, the
    text verbatim; a failure is ToolFailed, which the library wraps as {"error": ...}."""
    calls = [
        *tool_call(0, "c1", "read_file", "{}"),
        *tool_call(1, "c2", "write_file", '{"path": "a", "content": "x"}'),
        *tool_call(2, "c3", "read_file", '{"path": "missing"}'),
    ]
    tools = StubTools({"write_file": "deny"})
    with SSEServer(Reply([*calls, done("tool_calls")]), Reply([*text("ok"), done()])) as srv:
        events = await run(loop, turn([user("go")], config(srv)), tools)

    results = {i.message["tool_call_id"]: i.message["content"] for i in items(events)[1:4]}
    assert results["c1"].endswith("Fix the errors and try again.")
    assert results["c2"] == "Denied by permission rules: write_file"
    assert results["c3"] == '{"error":"No such file: missing"}'
    outcomes = [i.native["parts"][0].get("outcome") for i in items(events)[1:4]]
    assert outcomes == [None, "denied", "failed"]  # c1 is a retry prompt, which has no outcome
    assert srv.requests[1]["messages"][-3:] == [i.message for i in items(events)[1:4]]
    assert [c.id for c in tools.runs] == ["c1", "c2", "c3"]  # each through ToolHost.run


async def test_arguments_that_are_not_json_go_to_the_toolhost_like_any_others(loop):
    """Rule 5 also for arguments the library cannot parse: a `tool_validate_error` hook lets the
    call go on, so it is checked, ToolHost gets the raw text and answers with its own error, and
    the model sees that text as the retry prompt (not a pydantic error dump)."""
    truncated = tool_call(0, "c1", "read_file", '{"path": "a.txt"')
    with SSEServer(Reply([*truncated, done("tool_calls")]), Reply([*text("ok"), done()])) as srv:
        tools = StubTools()
        events = await run(loop, turn([user("go")], config(srv)), tools)

    assert tools.checked == ["c1"]
    assert [c.arguments for c in tools.runs] == ['{"path": "a.txt"']  # verbatim, as streamed
    assert of(events, "tool_call.ready")[0]["arguments"] == '{"path": "a.txt"'
    call, result = items(events)[:2]
    assert result.message == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "Invalid arguments for read_file: not valid JSON\n\nFix the errors and try again.",
    }
    # Recorded library behavior: the call is replayed with its text wrapped as {"INVALID_JSON": ...}.
    assert srv.requests[1]["messages"][2:] == [call.message, result.message]
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
    # The resume continues the turn: its request is the turn's second step.
    assert of(events, "request.start") == [{"step": 2, "attempt": 1}]
    assert of(events, "turn.end") == [{"stop": "end_turn", "steps": 2}]


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


async def test_a_new_message_after_an_unanswered_pause_closes_its_calls(loop):
    """The runner lets the user send a new message instead of answering a pause. The open call
    gets the result the library would synthesize for it, as an item (rule 4), and never runs."""
    tools = StubTools({"write_file": "ask"})
    replies = [_batch(), Reply([*text("hi"), done()]), Reply([*text("hi again"), done()])]
    with SSEServer(*replies) as srv:
        history = await _pause_for_write(loop, srv, tools)
        history += [user("never mind, just say hi")]
        events = await run(loop, turn(history, config(srv)), tools)
        history += [*items(events), user("again")]
        await run(loop, turn(history, config(srv)), tools)

    interrupted = "The tool call was interrupted before a result was produced."
    closing = {"role": "tool", "tool_call_id": "c2", "content": interrupted}
    assert [i.message for i in items(events)] == [closing, {"role": "assistant", "content": "hi"}]
    assert [c.id for c in tools.runs] == ["c1"]
    second, third = srv.requests[1]["messages"], srv.requests[2]["messages"]
    # The library puts tool results before the user's message within one request.
    assert [(m["role"], m.get("tool_call_id")) for m in second[2:]] == [
        ("assistant", None),
        ("tool", "c1"),
        ("tool", "c2"),
        ("user", None),
    ]
    assert second[4] == closing
    assert third[: len(second)] == second  # append-only


async def test_crash_between_the_results_of_one_batch_keeps_the_saved_one(loop):
    """Rule 4 across a crash: the worker dies right after c1's result item is saved. c1 never runs
    again; c2 ran but its result was not saved, so the crash resume runs it again (DESIGN)."""
    calls = tool_call(0, "c1", "write_file", '{"path": "a", "content": "x"}') + tool_call(
        1, "c2", "read_file", '{"path": "b"}'
    )
    tools = StubTools()
    with SSEServer(Reply([*calls, done("tool_calls")]), Reply([*text("done"), done()])) as srv:
        history = [user("go")]
        first = loop.run_turn(turn(history, config(srv)), tools, asyncio.Event())
        async with contextlib.aclosing(first):
            async for event in first:
                if event.type == "item":
                    history.append(items([event])[0])
                    if event.data["item"].message.get("tool_call_id") == "c1":
                        break  # SIGKILL: c1's result is saved, c2's is not
        fresh = PydanticLoop()
        events = await run(fresh, turn(history, config(srv), resume=Resume("crash")), tools)
        await fresh.aclose()

    runs = [c.id for c in tools.runs]
    assert (runs.count("c1"), runs.count("c2")) == (1, 2)
    log = history + items(events)
    assert [i.message["tool_call_id"] for i in log if i.message["role"] == "tool"] == ["c1", "c2"]
    assert len(srv.requests) == 2  # the killed worker never sent its next request
    assert [m.get("tool_call_id") for m in srv.requests[1]["messages"][-2:]] == ["c1", "c2"]


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
        # As the first run would have: the allowed call runs, the other one is asked again.
        assert of(events, "permission.asked")[0]["call_id"] == "c2"
        assert of(events, "turn.end") == [{"stop": "paused", "steps": 1, "pending": ["c2"]}]
        assert [c.id for c in tools.runs] == ["c1"]
        history += items(events)

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


async def test_cancel_during_a_stall_stops_fast_and_replays_the_partial_answer_unsigned_reasoning_dropped(
    loop,
):
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
    # The log keeps the cut-off reasoning, but the replay drops it (ProcessHistory): it has no
    # signature, and endpoints that check signatures (Anthropic) reject it with a 400.
    assert [p["part_kind"] for p in partial[0].native["parts"]] == ["thinking", "text"]
    assert (
        srv.requests[1]["messages"][2]
        == partial[0].message
        == {
            "role": "assistant",
            "content": "Par",
        }
    )


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
    tools = StubTools()
    with SSEServer(*looping) as srv:
        events = await run(loop, turn([user("go")], config(srv), limits=Limits(max_steps=2)), tools)

    assert len(srv.bodies) == 2
    assert of(events, "turn.end") == [{"stop": "max_steps", "steps": 2}]
    calls = [c["id"] for i in items(events) for c in i.message.get("tool_calls", [])]
    results = {
        i.message["tool_call_id"]: i.message["content"]
        for i in items(events)
        if i.message["role"] == "tool"
    }
    assert calls == list(results) == ["c0", "c1"]  # no orphan calls
    # c1 came in the last allowed response: its result could never be sent, so it does not run.
    assert [c.id for c in tools.runs] == ["c0"]
    assert results["c1"] == "Not run: the turn reached its step limit."


async def test_cost_without_provider_cost_is_labelled(loop):
    """Only OpenRouter's billed cost is "provider". genai-prices knows OpenRouter's prices, so an
    OpenRouter response without a billed cost is an "estimate"; it cannot know what a BYOK
    endpoint charges, so that is "none", never guessed (fakeprov S12b)."""
    model_named = "openai/gpt-4o-mini"
    unbilled = [{**c, "model": model_named} for c in [*text("a"), done()]]
    byok_usage = [{**c, "model": "gpt-4o-mini"} for c in [*text("b"), done()]]
    no_usage = [{**c, "model": "gpt-4o-mini"} for c in [*text("c"), chunk(finish="stop")]]
    with SSEServer(Reply(unbilled), Reply(byok_usage), Reply(no_usage)) as srv:
        openrouter = config(srv, model=model_named)
        byok = config(srv, kind="openai_compat", model="gpt-4o-mini")
        estimated = of(await run(loop, turn([user("x")], openrouter), StubTools()), "usage")[0]
        byok_priced = of(await run(loop, turn([user("y")], byok), StubTools()), "usage")[0]
        unreported = of(await run(loop, turn([user("z")], byok), StubTools()), "usage")[0]

    assert estimated["cost_source"] == "estimate" and estimated["cost_usd"] > 0
    assert (byok_priced["cost_source"], byok_priced["cost_usd"]) == ("none", None)
    assert byok_priced["input_tokens"] == 10  # the tokens are still reported
    assert (unreported["cost_source"], unreported["cost_usd"]) == ("none", None)


async def test_a_billed_cost_of_zero_is_a_provider_cost_and_spends_no_budget(loop):
    """The library drops OpenRouter's `cost: 0` (A_CHECKLIST); A reads it back from `is_byok`,
    which comes with OpenRouter's usage accounting, so a free step is not an estimate and does not
    count against `max_cost_usd`."""

    def free(n: int) -> Reply:
        usage = {**done("tool_calls", cost=0)["usage"], "is_byok": False}
        call = tool_call(0, f"c{n}", "read_file", '{"path": "a"}')
        return Reply([*call, chunk(finish="tool_calls", usage=usage)])

    final = {**done(cost=0)["usage"], "is_byok": False}
    replies = [free(0), free(1), Reply([*text("ok"), chunk(finish="stop", usage=final)])]
    with SSEServer(*replies) as srv:
        model = config(srv, model="openai/gpt-4o-mini")  # a model genai-prices can price
        limits = Limits(max_cost_usd=0.0000001)
        events = await run(loop, turn([user("go")], model, limits=limits), StubTools())

    assert {(u["cost_usd"], u["cost_source"]) for u in of(events, "usage")} == {(0, "provider")}
    assert of(events, "turn.end") == [{"stop": "end_turn", "steps": 3}]


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
    # Why the `profile=` argument is needed: the name-based profile has no thinking here.
    stock = OpenAIChatModel(
        "qwen3-coder", provider=OpenAIProvider(base_url=srv.base_url, api_key="k")
    )
    assert not stock.profile.get("supports_thinking")


async def test_byok_thinking_is_replayed_in_the_field_it_came_in(loop):
    reply = [chunk({"reasoning_content": "Let me think."}), *text("42"), done()]
    with SSEServer(Reply(reply), Reply([*text("ok"), done()])) as srv:
        byok = config(srv, kind="openai_compat", model="qwen3-coder", reasoning={"effort": "low"})
        events = await run(loop, turn([user("q")], byok), StubTools())
        history = [user("q"), *items(events), user("again")]
        await run(loop, turn(history, byok), StubTools())

    [answer] = items(events)
    assert answer.message == {
        "role": "assistant",
        "content": "42",
        "reasoning_content": "Let me think.",
    }
    assert srv.requests[1]["messages"][2] == answer.message  # Item.message is what is sent


async def test_threads_share_one_model_and_send_their_own_session_id(loop):
    with SSEServer(*[Reply([*text("ok"), done()]) for _ in range(3)]) as srv:
        for session in ("thread-a", "thread-b", None):
            await run(loop, turn([user("hi")], config(srv, session_id=session)), StubTools())

    assert len(loop._models) == 1 and len(loop._agents) == 1  # one HTTP pool for the endpoint
    sent = [(body.get("session_id"), body["cache_control"]) for body in srv.requests]
    ephemeral = {"type": "ephemeral"}
    assert sent == [("thread-a", ephemeral), ("thread-b", ephemeral), (None, ephemeral)]


def test_the_first_model_imports_nothing_inside_a_turn():
    """Building a model must not import (and block the event loop, ~0.3 s on openai 2.x): the
    module pays for the SDK's lazily loaded chat resources at import. Needs a fresh process."""
    script = (
        "import sys; import bakeoff.pydantic_version.loop; before = set(sys.modules); "
        "from bakeoff.pydantic_version.model import build_model; "
        "from bakeoff.shared.contract import ModelConfig; "
        "build_model(ModelConfig(base_url='http://127.0.0.1:9/v1', model='anthropic/x'), {}); "
        "print(sorted(m for m in set(sys.modules) - before if m.startswith('openai')))"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


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
    # Nothing came back: no item (an empty assistant message is not sendable) and no usage.
    assert (of(events, "item"), of(events, "usage")) == ([], [])


async def test_cost_limit_ends_the_turn_with_budget_and_reports_every_billed_step(loop):
    def costly(n: int) -> Reply:
        call = tool_call(0, f"c{n}", "read_file", '{"path": "a"}')
        return Reply([*call, done("tool_calls", cost=0.0006)])

    tools = StubTools()
    with SSEServer(costly(0), costly(1)) as srv:
        limits = Limits(max_cost_usd=0.001)
        events = await run(loop, turn([user("go")], config(srv), limits=limits), tools)

    assert of(events, "turn.end") == [{"stop": "budget", "steps": 2}]
    # The library drops the response that crossed the limit from history; A keeps it (it was
    # streamed and billed) and closes its call, which never runs.
    interrupted = "The tool call was interrupted before a result was produced."
    assert [(i.message["role"], i.message.get("tool_call_id")) for i in items(events)] == [
        ("assistant", None),
        ("tool", "c0"),
        ("assistant", None),
        ("tool", "c1"),
    ]
    assert items(events)[3].message["content"] == interrupted
    assert [c.id for c in tools.runs] == ["c0"]
    assert [(u["step"], u["cost_usd"]) for u in of(events, "usage")] == [(1, 0.0006), (2, 0.0006)]


async def test_an_answer_that_crosses_the_budget_is_kept_for_the_next_turn(loop):
    answer = Reply([*text("Hello there"), done(cost=0.002)])
    with SSEServer(answer, Reply([*text("I said hello."), done(cost=0.0001)])) as srv:
        limits = Limits(max_cost_usd=0.001)
        events = await run(loop, turn([user("hi")], config(srv), limits=limits), StubTools())
        history = [user("hi"), *items(events), user("what did you say?")]
        await run(loop, turn(history, config(srv)), StubTools())

    assert of(events, "turn.end") == [{"stop": "budget", "steps": 1}]
    assert [i.message for i in items(events)] == [{"role": "assistant", "content": "Hello there"}]
    assert srv.requests[1]["messages"][2] == {"role": "assistant", "content": "Hello there"}


async def test_cost_limit_crossed_on_the_last_allowed_step_is_a_budget_stop(loop):
    costly = Reply(
        [*tool_call(0, "c0", "read_file", '{"path": "a"}'), done("tool_calls", cost=0.002)]
    )
    with SSEServer(costly) as srv:
        limits = Limits(max_steps=1, max_cost_usd=0.001)
        events = await run(loop, turn([user("go")], config(srv), limits=limits), StubTools())

    assert of(events, "turn.end") == [{"stop": "budget", "steps": 1}]


async def test_cost_limit_without_a_known_price_stays_quiet(loop, capfd):
    with (
        warnings.catch_warnings(record=True) as caught,
        SSEServer(Reply([*text("ok"), done()])) as srv,
    ):
        byok = config(srv, kind="openai_compat", model="qwen3-coder")
        limits = Limits(max_cost_usd=1.0)
        events = await run(loop, turn([user("hi")], byok, limits=limits), StubTools())

    assert of(events, "turn.end")[0]["stop"] == "end_turn"
    assert of(events, "usage")[0]["cost_source"] == "none"
    assert (caught, capfd.readouterr()) == ([], ("", ""))  # no CostNotFoundWarning (rule 1)


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


def _error_chunk(code: str | int) -> dict[str, Any]:
    """OpenRouter's mid-stream error: HTTP 200, a top-level error, finish_reason "error"."""
    return {**chunk(finish="error"), "error": {"code": code, "message": "Provider disconnected"}}


@pytest.mark.parametrize("code", ["server_error", 502])  # OpenRouter's documented code is a string
async def test_error_chunk_mid_stream_is_retried_from_the_saved_history(loop, monkeypatch, code):
    monkeypatch.setattr(loop_module, "_STREAM_RETRY_BASE_S", 0.01)
    tools = StubTools()
    replies = [
        Reply([*tool_call(0, "c1", "read_file", '{"path": "a"}'), done("tool_calls")]),
        Reply([*text("Hal"), _error_chunk(code)]),
        Reply([*text("Hello"), done()]),
    ]
    with SSEServer(*replies) as srv:
        limits = Limits(max_steps=2)  # the retry is the same step, not a third one
        model = config(srv, max_retries=1)
        events = await run(loop, turn([user("hi")], model, limits=limits), tools)

    assert len(srv.bodies) == 3 and srv.requests[2]["messages"] == srv.requests[1]["messages"]
    assert [c.id for c in tools.runs] == ["c1"]  # the saved step is not redone
    starts = [(d["step"], d["attempt"]) for d in of(events, "request.start")]
    assert starts == [(1, 1), (2, 1), (2, 2)]
    [retry] = of(events, "retry")
    assert (retry["attempt"], retry["status"]) == (2, None if code == "server_error" else 502)
    assert retry["reason"] in ("OpenRouter error chunk: server_error", "HTTP 502")
    # The partial "Hal" is dropped, not saved: the step's one item is the retried answer.
    assert [i.message.get("content") for i in items(events)] == [None, "read_file ok", "Hello"]
    assert of(events, "turn.end") == [{"stop": "end_turn", "steps": 2}]


async def test_error_chunk_after_the_last_retry_ends_the_turn(loop, monkeypatch):
    monkeypatch.setattr(loop_module, "_STREAM_RETRY_BASE_S", 0.01)
    failing = [Reply([*text("Hal"), _error_chunk("server_error")]) for _ in range(2)]
    with SSEServer(*failing) as srv:
        events = await run(loop, turn([user("hi")], config(srv, max_retries=1)), StubTools())

    assert len(srv.bodies) == 2
    # The last attempt's partial text is kept, as a cancelled one would be.
    assert [(i.status, i.message["content"]) for i in items(events)] == [("incomplete", "Hal")]
    assert of(events, "error")[0]["retryable"] is True
    assert of(events, "turn.end")[0]["stop"] == "error"


async def test_byok_thinking_in_think_tags_is_shown_as_it_is_sent(loop):
    """A BYOK model that streams its thinking as <think> tags in the content: the library parses
    it into a ThinkingPart and sends it back as tags, joined to the text with a blank line."""
    first = [*text("<think>", "plan it", "</think>", "The answer."), done()]
    with SSEServer(Reply(first), Reply([*text("ok"), done()])) as srv:
        byok = config(srv, kind="openai_compat", model="qwen3-32b")
        events = await run(loop, turn([user("hi")], byok), StubTools())
        history = [user("hi"), *items(events), user("again")]
        await run(loop, turn(history, byok), StubTools())

    [answer] = items(events)
    assert answer.message == {
        "role": "assistant",
        "content": "<think>\nplan it\n</think>\n\nThe answer.",
    }
    assert srv.requests[1]["messages"][2] == answer.message


@pytest.mark.parametrize(
    ("kind", "failure", "reason"),
    [
        ("openrouter", "drop", "connection error"),
        ("openai_compat", "drop", "connection error"),
        ("openai_compat", "error_event", "stream error: internal server error"),
    ],
)
async def test_a_dropped_stream_or_an_error_event_is_retried(
    loop, monkeypatch, kind, failure, reason
):
    """pydantic-ai lets these through unwrapped: a connection cut mid-body (openai 2.x: raw httpx;
    3.x on OpenRouter: the error model's ValidationError) and a BYOK `{"error": ...}` event
    (raw `openai.APIError`). Both are provider failures, so the step is retried."""
    monkeypatch.setattr(loop_module, "_STREAM_RETRY_BASE_S", 0.01)
    if failure == "drop":
        failing = Reply(text("Hi ag"), drop=True)
    else:
        event = {"error": {"message": "Internal server error", "type": "server_error"}}
        failing = Reply([*text("Hi ag"), event])
    with SSEServer(failing, Reply([*text("Hi again!"), done()])) as srv:
        name = "anthropic/claude-test" if kind == "openrouter" else "qwen3-coder"
        model = config(srv, kind=kind, model=name, max_retries=2)
        events = await run(loop, turn([user("hi")], model), StubTools())

    assert len(srv.bodies) == 2
    [retry] = of(events, "retry")
    assert retry["reason"].lower().startswith(reason)  # 3.x wraps some: "Connection error."
    assert [i.message["content"] for i in items(events)] == ["Hi again!"]
    assert of(events, "turn.end") == [{"stop": "end_turn", "steps": 1}]


def _write_then_stall() -> Reply:
    call = tool_call(0, "w1", "write_file", '{"path": "a", "content": "x"}')
    return Reply([*text("Writing it."), *call], stall=True)


async def _cancelled_write(loop: PydanticLoop, srv: SSEServer, tools: StubTools) -> list[Item]:
    """A turn whose response (a complete write call) is cut short by a cancel."""
    cancel = asyncio.Event()
    events = []
    async for event in loop.run_turn(turn([user("write a")], config(srv)), tools, cancel):
        events.append(event)
        if event.type == "text.delta":
            asyncio.get_running_loop().call_later(0.2, cancel.set)
    assert of(events, "turn.end")[0]["stop"] == "cancelled"
    return items(events)


async def test_a_crash_after_a_cancel_never_runs_the_cancelled_call(loop):
    """Rule 6 across a crash: the worker dies after the cut-off response is saved and before its
    closing result is. The crash resume closes the call from the history (the response is
    `interrupted`), finishes the cancel and sends nothing."""
    tools = StubTools()
    with SSEServer(_write_then_stall()) as srv:
        cut, closing = await _cancelled_write(loop, srv, tools)
        assert cut.status == "incomplete" and cut.message["tool_calls"][0]["id"] == "w1"
        history = [user("write a"), cut]  # SIGKILL before the closing item was saved
        fresh = PydanticLoop()
        events = await run(fresh, turn(history, config(srv), resume=Resume("crash")), tools)
        await fresh.aclose()

    assert tools.runs == []
    assert [i.message for i in items(events)] == [closing.message]
    assert of(events, "turn.end") == [{"stop": "cancelled", "steps": 1}]
    assert len(srv.bodies) == 1  # no request after the resume


async def test_a_crash_after_the_final_answer_sends_nothing_more(loop):
    tools = StubTools()
    replies = [Reply([*tool_call(0, "c1", "read_file", '{"path": "a"}'), done("tool_calls")])]
    with SSEServer(*replies, Reply([*text("All done."), done()])) as srv:
        events = await run(loop, turn([user("go")], config(srv)), tools)
        history = [user("go"), *items(events)]  # SIGKILL after the answer, before the turn closed
        resumed = await run(loop, turn(history, config(srv), resume=Resume("crash")), tools)

    assert len(srv.bodies) == 2
    assert (of(resumed, "item"), of(resumed, "turn.end")) == (
        [],
        [{"stop": "end_turn", "steps": 2}],
    )


async def test_a_partial_approval_runs_what_was_decided_and_asks_for_the_rest(loop):
    """The user answers only c1. c1 runs now (its answer is not lost), c2 is asked again, and
    answering c2 finishes the turn: each call runs once."""
    tools = StubTools({"write_file": "ask"})
    calls = tool_call(0, "c1", "write_file", '{"path": "a", "content": "x"}') + tool_call(
        1, "c2", "write_file", '{"path": "b", "content": "y"}'
    )
    with SSEServer(Reply([*calls, done("tool_calls")]), Reply([*text("ok"), done()])) as srv:
        first = await run(loop, turn([user("go")], config(srv)), tools)
        assert of(first, "turn.end")[0]["pending"] == ["c1", "c2"]
        history = [user("go"), *items(first)]
        partial = Resume("approval", {"c1": "allow"})
        second = await run(loop, turn(history, config(srv), resume=partial), tools)
        assert [c.id for c in tools.runs] == ["c1"]
        assert of(second, "permission.asked")[0]["call_id"] == "c2"
        assert of(second, "turn.end") == [{"stop": "paused", "steps": 1, "pending": ["c2"]}]
        history += items(second)
        rest = Resume("approval", {"c2": "allow"})
        third = await run(loop, turn(history, config(srv), resume=rest), tools)

    assert [c.id for c in tools.runs] == ["c1", "c2"]
    assert of(third, "turn.end")[0]["stop"] == "end_turn"
    tool_ids = [m.get("tool_call_id") for m in srv.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_ids == ["c1", "c2"]


@pytest.mark.parametrize("limit", ["max_steps", "max_cost_usd"])
async def test_a_crash_resume_continues_the_step_cap_and_the_budget(loop, limit):
    """The steps and cost a turn spent before a crash count after it: with a cap of 3 steps (or
    $0.0025 at $0.001 a step), a turn that crashed after 2 steps makes exactly one more request."""

    def step(n: int) -> Reply:
        call = tool_call(0, f"c{n}", "read_file", '{"path": "a"}')
        return Reply([*call, done("tool_calls", cost=0.001)])

    limits = Limits(max_steps=3) if limit == "max_steps" else Limits(max_cost_usd=0.0025)
    tools = StubTools()
    with SSEServer(*[step(n) for n in range(6)]) as srv:
        history = [user("go")]
        stream = loop.run_turn(turn(history, config(srv), limits=limits), tools, asyncio.Event())
        async with contextlib.aclosing(stream):
            async for event in stream:
                if event.type == "item":
                    history.append(items([event])[0])
                    if event.data["item"].message.get("tool_call_id") == "c1":
                        break  # SIGKILL after step 2's result is saved
        resumed = await run(
            loop, turn(history, config(srv), resume=Resume("crash"), limits=limits), tools
        )

    assert len(srv.bodies) == 3
    assert of(resumed, "request.start") == [{"step": 3, "attempt": 1}]
    stop = "max_steps" if limit == "max_steps" else "budget"
    assert of(resumed, "turn.end") == [{"stop": stop, "steps": 3}]

"""fakeprov in Responses API mode: scenarios whose model kind is "openai_responses"."""

import contextlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bakeoff.fakeprov.script import SCENARIOS_DIR, ScenarioError, load_scenario
from bakeoff.fakeprov.server import FakeProvider

MODEL = "gpt-6-luna"
USER = {"role": "user", "content": "hi"}
ENC = "gAAAAB-encrypted-reasoning-0123456789"
REASONING = {"id": "rs_1", "encrypted_content": ENC, "summary": ["**Plan**\n\nLook it up.", "Go."]}
DONE_REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [
        {"type": "summary_text", "text": "**Plan**\n\nLook it up."},
        {"type": "summary_text", "text": "Go."},
    ],
    "encrypted_content": ENC,
}
CALL = {"id": "call_1", "name": "read_file", "arguments": {"path": "a.pipe"}}
DONE_CALL = {
    "id": "fc_1",
    "type": "function_call",
    "status": "completed",
    "arguments": '{"path": "a.pipe"}',
    "call_id": "call_1",
    "name": "read_file",
}
COMPLETED = {"completed": {"input_tokens": 10, "output_tokens": 5}}


def scenario(sid: str, *exchanges: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {
        "id": sid,
        "title": "test",
        "system": "You are a test.",
        "model": {"kind": "openai_responses", "model": MODEL, "reasoning": {"effort": "xhigh"}},
        "rules": {"*": "allow"},
        "limits": {"max_steps": 4},
        "engine": {"delay_ms": 0},
        "driver": [{"user": "hi"}],
        "exchanges": list(exchanges),
        "expect": {"stops": ["end_turn"]},
        **top,
    }


def says(text: str, **exchange: Any) -> dict[str, Any]:
    return {"respond": {"stream": [{"text": text}, COMPLETED]}, **exchange}


def request(*items: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A request that asks for reasoning summaries (the API streams none otherwise)."""
    reasoning = {"effort": "xhigh", "summary": "auto"}
    return {"model": MODEL, "stream": True, "input": list(items or [USER]), "reasoning": reasoning,
            **extra}  # fmt: skip


@pytest.fixture
def serve(tmp_path: Path) -> Iterator[Callable[..., FakeProvider]]:
    with contextlib.ExitStack() as stack:

        def start(*scenarios: dict[str, Any]) -> FakeProvider:
            for s in scenarios:
                (tmp_path / f"{s['id']}.json").write_text(json.dumps(s))
            return stack.enter_context(FakeProvider(tmp_path, tmp_path / "wire"))

        yield start


def post(provider: FakeProvider, body: Any, sid: str = "T", endpoint: str = "responses"):
    return httpx.post(f"{provider.base_url(sid, 'r1', 'our')}/{endpoint}", json=body)


def events(response: httpx.Response) -> list[tuple[str, dict[str, Any]]]:
    """(event name, data) of every SSE event; checks each is framed as the API frames it."""
    out = []
    for frame in response.text.split("\n\n"):
        if not frame:
            continue
        name, data = frame.split("\n")
        assert name.startswith("event: ") and data.startswith("data: "), frame
        payload = json.loads(data.removeprefix("data: "))
        assert payload["type"] == name.removeprefix("event: ")
        out.append((payload["type"], payload))
    return out


def names(response: httpx.Response) -> list[str]:
    return [name for name, _ in events(response)]


def test_a_full_stream_has_the_apis_events_in_order(serve):
    ops = [
        {"reasoning_item": REASONING, "chunks": 2},
        {"text": "Checking.", "phase": "commentary"},
        {"tool_calls": [CALL], "pieces": 2},
        {"completed": {"input_tokens": 900, "output_tokens": 40, "cached_tokens": 512,
                       "reasoning_tokens": 30}},
    ]  # fmt: skip
    provider = serve(scenario("T", {"respond": {"stream": ops}}))
    response = post(provider, request())
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    got = events(response)
    assert [name for name, _ in got] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",  # reasoning
        *["response.reasoning_summary_part.added", "response.reasoning_summary_text.delta",
          "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
          "response.reasoning_summary_part.done"] * 2,
        "response.output_item.done",
        "response.output_item.added",  # message
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",  # function call
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]  # fmt: skip
    assert [data["sequence_number"] for _, data in got] == list(range(len(got)))
    assert "[DONE]" not in response.text
    done = [data["item"] for name, data in got if name == "response.output_item.done"]
    message = {
        "id": "msg_T_001_1",
        "type": "message",
        "status": "completed",
        "content": [
            {"type": "output_text", "annotations": [], "logprobs": [], "text": "Checking."}
        ],
        "role": "assistant",
        "phase": "commentary",
    }
    assert done == [DONE_REASONING, message, DONE_CALL]
    assert [d["output_index"] for n, d in got if n == "response.output_item.done"] == [0, 1, 2]
    completed = got[-1][1]["response"]
    assert completed["id"] == "resp_T_001"
    assert (completed["status"], completed["model"], completed["output"]) == (
        "completed",
        MODEL,
        done,
    )
    assert completed["reasoning"] == {"effort": "xhigh", "summary": None}
    assert completed["usage"] == {
        "input_tokens": 900,
        "input_tokens_details": {"cached_tokens": 512, "cache_write_tokens": 0},
        "output_tokens": 40,
        "output_tokens_details": {"reasoning_tokens": 30},
        "total_tokens": 940,
    }
    created = got[0][1]["response"]
    assert (created["status"], created["output"], created["usage"]) == ("in_progress", [], None)


def test_deltas_items_and_arguments(serve):
    ops = [{"reasoning_item": REASONING, "chunks": 2}, {"tool_calls": [CALL], "pieces": 3},
           {"text": "abcdef", "chunks": 3}, COMPLETED]  # fmt: skip
    provider = serve(scenario("T", {"respond": {"stream": ops}}))
    got = events(post(provider, request()))
    added = [d["item"] for n, d in got if n == "response.output_item.added"]
    # The added reasoning item's encrypted_content is incomplete, as the API documents.
    assert added[0] == {"id": "rs_1", "type": "reasoning", "summary": [],
                        "encrypted_content": ENC[: len(ENC) // 2]}  # fmt: skip
    assert added[1] == {**DONE_CALL, "status": "in_progress", "arguments": ""}
    summary = [(d["summary_index"], d["delta"]) for n, d in got if n.endswith("summary_text.delta")]
    assert summary == [(0, "**Plan**\n\n"), (0, "Look it up."), (1, "G"), (1, "o.")]
    args = [d for n, d in got if n == "response.function_call_arguments.delta"]
    assert "".join(d["delta"] for d in args) == DONE_CALL["arguments"]
    assert {(d["item_id"], d["output_index"]) for d in args} == {("fc_1", 1)}
    args_done = next(d for n, d in got if n == "response.function_call_arguments.done")
    assert (args_done["name"], args_done["arguments"]) == ("read_file", DONE_CALL["arguments"])
    text = [d for n, d in got if n == "response.output_text.delta"]
    assert [(d["delta"], d["item_id"], d["content_index"], d["logprobs"]) for d in text] == [
        ("ab", "msg_T_001_2", 0, []),
        ("cd", "msg_T_001_2", 0, []),
        ("ef", "msg_T_001_2", 0, []),
    ]


def test_delta_events_carry_an_obfuscation_pad_unless_turned_off(serve):
    def exchange(n: int) -> dict[str, Any]:
        reasoning = {**REASONING, "id": f"rs_{n}"}
        call = {**CALL, "id": f"call_{n}"}
        ops = [{"reasoning_item": reasoning}, {"tool_calls": [call]}, {"text": "hi"}, COMPLETED]
        return {"respond": {"stream": ops}}

    provider = serve(scenario("T", exchange(1), exchange(2), exchange(3)))
    first, again = post(provider, request()), post(provider, request())
    padded = {n for n, d in events(first) if "obfuscation" in d}
    assert padded == {
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
        "response.output_text.delta",
    }
    # Deterministic: the same exchange gives the same bytes on another cursor.
    other = httpx.post(provider.base_url("T", "r1", "pydantic") + "/responses", json=request())
    assert other.content == first.content
    assert again.content != first.content  # exchange 2: other ids
    off = post(provider, request(stream_options={"include_obfuscation": False}))
    assert not any("obfuscation" in d for _, d in events(off))


def test_an_item_cut_short_sends_no_done_events(serve):
    ops = [{"reasoning_item": REASONING, "done": False}, {"stall": True}]
    provider = serve(scenario("T", {"respond": {"stream": ops}}))
    url = provider.base_url("T", "r1", "our") + "/responses"
    seen: list[str] = []
    with (
        httpx.Client(timeout=0.3) as client,
        client.stream("POST", url, json=request()) as response,
        pytest.raises(httpx.ReadTimeout),  # the stream stalls after the cut
    ):
        for line in response.iter_lines():
            if line.startswith("event: "):
                seen.append(line.removeprefix("event: "))
    assert seen == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        # cut inside the last summary part: no text.done, part.done or output_item.done
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
    ]


def test_a_message_cut_short_before_an_error_event(serve):
    ops = [{"text": "Hi ag", "done": False},
           {"error": {"message": "The server had an error"}, "delay_ms": 5}]  # fmt: skip
    provider = serve(scenario("T", {"respond": {"stream": ops}}))
    response = post(provider, request())
    assert response.status_code == 200
    got = events(response)
    assert [n for n, _ in got][-3:] == [
        "response.content_part.added",
        "response.output_text.delta",
        "error",
    ]
    assert got[-1][1] == {
        "type": "error",
        "sequence_number": 5,
        "code": "server_error",
        "message": "The server had an error",
        "param": None,
    }


def test_a_failed_response(serve):
    ops = [{"text": "partial"}, {"failed": {"code": "rate_limit_exceeded", "message": "slow down"}}]
    provider = serve(scenario("T", {"respond": {"stream": ops}}))
    name, data = events(post(provider, request()))[-1]
    assert name == "response.failed"
    assert data["response"]["status"] == "failed"
    assert data["response"]["error"] == {"code": "rate_limit_exceeded", "message": "slow down"}
    assert [item["type"] for item in data["response"]["output"]] == ["message"]


def test_a_response_that_runs_out_of_output_tokens(serve):
    """As the API ends a response whose max_output_tokens ran out, here inside a reasoning item
    (only the items done so far are in its output)."""
    usage = {"input_tokens": 50, "output_tokens": 16, "reasoning_tokens": 16}
    ops = [{"reasoning_item": REASONING}, {"text": "Hel", "done": False}, {"incomplete": usage}]
    provider = serve(scenario("T", {"respond": {"stream": ops}}))
    got = events(post(provider, request()))
    assert [name for name, _ in got][-2:] == ["response.output_text.delta", "response.incomplete"]
    response = got[-1][1]["response"]
    assert (response["status"], response["incomplete_details"]) == (
        "incomplete",
        {"reason": "max_output_tokens"},
    )
    assert response["usage"]["output_tokens_details"] == {"reasoning_tokens": 16}
    assert response["output"] == [DONE_REASONING]


def test_rate_limit_with_retry_after_then_ok(serve):
    limited = {"respond": {"status": 429, "headers": {"retry-after": "1"}}}
    provider = serve(scenario("T", limited, says("ok")))
    first = post(provider, request())
    assert (first.status_code, first.headers["retry-after"]) == (429, "1")
    error = {"message": "Too Many Requests", "type": "api_error", "param": None, "code": None}
    assert first.json() == {"error": error}  # OpenAI's error shape
    assert post(provider, request()).status_code == 200


def test_each_api_is_answered_only_on_its_own_endpoint(serve, tmp_path):
    chat_scenario = {**scenario("C", says("ok")), "model": {"kind": "openrouter", "model": "m"}}
    chat_scenario["exchanges"] = [{"respond": {"stream": [{"text": "ok"}, {"finish": "stop"}]}}] * 2
    provider = serve(scenario("T", says("ok"), says("ok")), chat_scenario)
    chat_body = {"model": MODEL, "stream": True, "messages": [USER]}
    wrong = post(provider, chat_body, endpoint="chat/completions")
    assert wrong.status_code == 404
    assert wrong.headers["x-should-retry"] == "false"
    assert wrong.json()["error"]["message"] == (
        "scenario T is scripted for POST <base_url>/responses, not /chat/completions"
    )
    assert post(provider, request()).status_code == 200  # exchange 2: the 404 used exchange 1
    wire = tmp_path / "wire" / "T" / "r1" / "our"
    metas = [json.loads((wire / f"00{n}.meta.json").read_text()) for n in (1, 2)]
    assert [(m["path"].rsplit("/v1", 1)[1], m["status"]) for m in metas] == [
        ("/chat/completions", 404),
        ("/responses", 200),
    ]
    assert post(provider, request(), sid="C").status_code == 404
    assert post(provider, chat_body, sid="C", endpoint="chat/completions").status_code == 200


def test_bad_responses_requests(serve):
    provider = serve(scenario("T", *[says("ok")] * 5))  # each request uses an exchange
    assert post(provider, {"model": MODEL, "stream": True}).status_code == 400
    assert post(provider, request(stream=False)).status_code == 400
    assert post(provider, {**request(), "input": []}).status_code == 400
    assert post(provider, {**request(), "input": ["hi"]}).status_code == 400
    assert post(provider, {**request(), "input": "hi"}).status_code == 200  # one user message


def test_well_formed_items_of_every_served_type_pass(serve):
    provider = serve(scenario("T", says("ok")))
    items = [
        {"role": "developer", "content": [{"type": "input_text", "text": "Be brief."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {**DONE_REASONING, "encrypted_content": None},
        {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
         "phase": "commentary", "content": [{"type": "output_text", "text": "Checking."}]},
        DONE_CALL,
        {"type": "function_call_output", "call_id": "call_1",
         "output": [{"type": "input_text", "text": "3 rows"}]},
    ]  # fmt: skip
    assert post(provider, request(*items)).status_code == 200


OUTPUT_OK = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}


@pytest.mark.parametrize(
    ("item", "message", "param", "code"),
    [
        (
            {"type": "function_call_output", "call_id": "call_1"},
            "Missing required parameter: 'input[1].output'.",
            "input[1].output",
            "missing_required_parameter",
        ),
        (
            {**OUTPUT_OK, "output": None},
            "Invalid type for 'input[1].output': expected one of a string or an array, but got"
            " null instead.",
            "input[1].output",
            "invalid_type",
        ),
        (
            {k: v for k, v in OUTPUT_OK.items() if k != "call_id"},
            "Missing required parameter: 'input[1].call_id'.",
            "input[1].call_id",
            "missing_required_parameter",
        ),
        (
            {**DONE_CALL, "arguments": {"path": "a.pipe"}},
            "Invalid type for 'input[1].arguments': expected a string, but got an object instead.",
            "input[1].arguments",
            "invalid_type",
        ),
        (
            {k: v for k, v in DONE_CALL.items() if k != "name"},
            "Missing required parameter: 'input[1].name'.",
            "input[1].name",
            "missing_required_parameter",
        ),
        (
            {"role": "user"},
            "Missing required parameter: 'input[1].content'.",
            "input[1].content",
            "missing_required_parameter",
        ),
        (
            {"role": "tool", "content": "3 rows"},
            "Invalid value: 'tool'. Supported values are: 'user', 'assistant', 'system', and"
            " 'developer'.",
            "input[1].role",
            "invalid_value",
        ),
        (
            {"role": "assistant", "content": [{"type": "input_text", "text": "hi"}]},
            "Invalid value: 'input_text'. Supported values are: 'output_text' and 'refusal'.",
            "input[1].content[0]",
            "invalid_value",
        ),
        (
            {"role": "user", "content": [{"type": "input_text"}]},
            "Missing required parameter: 'input[1].content[0].text'.",
            "input[1].content[0].text",
            "missing_required_parameter",
        ),
        (
            {k: v for k, v in DONE_REASONING.items() if k != "id"},
            "Missing required parameter: 'input[1].id'.",
            "input[1].id",
            "missing_required_parameter",
        ),
        (
            {k: v for k, v in DONE_REASONING.items() if k != "summary"},
            "Missing required parameter: 'input[1].summary'.",
            "input[1].summary",
            "missing_required_parameter",
        ),
        (
            {**DONE_REASONING, "summary": ["Go."]},
            "Invalid type for 'input[1].summary[0]': expected an object, but got a string instead.",
            "input[1].summary[0]",
            "invalid_type",
        ),
        (
            {**DONE_REASONING, "encrypted_content": 7},
            "Invalid type for 'input[1].encrypted_content': expected a string, but got an integer"
            " instead.",
            "input[1].encrypted_content",
            "invalid_type",
        ),
        (
            {"call_id": "call_1", "output": "ok"},
            "Missing required parameter: 'input[1].type'.",
            "input[1].type",
            "missing_required_parameter",
        ),
        (
            {"type": "web_search_call", "id": "ws_1"},
            "Invalid value: 'web_search_call'. Supported values are: 'message', 'function_call',"
            " 'function_call_output', and 'reasoning'.",
            "input[1].type",
            "invalid_value",
        ),
    ],
)
def test_malformed_input_items_are_400(serve, tmp_path, item, message, param, code):
    """As the API answers them, before the script is read: an exchange that expects nothing
    refuses them too."""
    provider = serve(scenario("T", says("ok")))
    response = post(provider, request(USER, item))
    assert response.status_code == 400
    assert response.headers["x-should-retry"] == "false"
    error = {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    assert response.json() == {"error": error}
    meta = json.loads((tmp_path / "wire" / "T" / "r1" / "our" / "001.meta.json").read_text())
    assert (meta["status"], meta["error"]) == (400, message)


def replaying(reasoning: dict[str, Any]) -> dict[str, Any]:
    output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
    return request(USER, reasoning, DONE_CALL, output)


@pytest.mark.parametrize(
    ("reasoning", "status", "message"),
    [
        (DONE_REASONING, 200, None),
        (
            {k: v for k, v in DONE_REASONING.items() if k != "encrypted_content"},
            404,
            "Item with id 'rs_1' not found. Items are not persisted when `store` is set to false."
            " Try again with `store` set to true, or remove this item from your input.",
        ),
        (
            {**DONE_REASONING, "encrypted_content": ENC[: len(ENC) // 2]},  # the added item's
            400,
            "The encrypted content for item rs_1 could not be verified.",
        ),
        (
            {**DONE_REASONING, "id": "rs_other"},
            400,
            "The encrypted content for item rs_other could not be verified.",
        ),
    ],
)
def test_reject_unencrypted_reasoning(serve, reasoning, status, message):
    strict = {"reject_unencrypted_reasoning": True}
    ops = [{"reasoning_item": REASONING}, {"tool_calls": [CALL]}, COMPLETED]
    provider = serve(scenario("T", {"respond": {"stream": ops}}, says("ok"), strict=strict))
    post(provider, request())
    response = post(provider, replaying(reasoning))
    assert response.status_code == status
    if message:
        assert response.json()["error"]["message"] == message


def test_a_cut_reasoning_item_never_verifies(serve):
    cut = [{"reasoning_item": REASONING, "done": False}, {"stall": True}]
    strict = {"reject_unencrypted_reasoning": True}
    provider = serve(scenario("T", {"respond": {"stream": cut}}, says("ok"), strict=strict))
    with pytest.raises(httpx.ReadTimeout):  # the stream stalls
        httpx.post(provider.base_url("T", "r1", "our") + "/responses", json=request(), timeout=0.2)
    # Even the full encrypted content: the item was never done, so it was never issued.
    response = post(provider, replaying(DONE_REASONING))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_encrypted_content"


def test_reasoning_that_no_earlier_response_sent_never_verifies(serve):
    """A later exchange's reasoning item was never sent to this cursor, so its scripted
    encrypted content cannot verify yet; once a response has sent it, it does."""
    strict = {"reject_unencrypted_reasoning": True}
    provider = serve(scenario("T", says("ok"), CALLS, says("ok"), strict=strict))
    early = post(provider, replaying(DONE_REASONING))  # rs_1 comes with exchange 2
    assert early.status_code == 400
    assert early.json()["error"] == {
        "message": "The encrypted content for item rs_1 could not be verified.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_encrypted_content",
    }
    assert post(provider, request()).status_code == 200  # exchange 2 sends rs_1
    assert post(provider, replaying(DONE_REASONING)).status_code == 200


def test_reject_params_names_the_parameter(serve):
    provider = serve(scenario("T", says("ok"), strict={"reject_params": ["temperature"]}))
    response = post(provider, request(temperature=0.0))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "temperature"
    assert response.json()["error"]["message"] == "Unsupported parameter: 'temperature'."


OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": "3 rows"}
TOOL_TURN = [USER, DONE_REASONING, DONE_CALL, OUTPUT]
CALLS = {"respond": {"stream": [{"reasoning_item": REASONING}, {"tool_calls": [CALL]}, COMPLETED]}}


def test_expect_checks_pass_on_a_matching_request(serve):
    expect = {
        "body_has": ["tools"],
        "body_equals": {"store": False, "reasoning.effort": "xhigh", "tools.-1.name": "read_file"},
        "body_contains": {"include": "reasoning.encrypted_content"},
        "system_contains": "You are",
        "input_len": 4,
        "input_at": [{"index": 1, "type": "reasoning"}, {"index": -2, "contains": "a.pipe"}],
        "last_type": "function_call_output",
        "last_content_contains": "rows",
        "tool_result_contains": {"call_1": "3 rows"},
        "reasoning_replayed": ["rs_1"],
    }
    provider = serve(scenario("T", CALLS, says("ok", expect=expect), says("ok", expect=expect)))
    post(provider, request())
    shape = {
        "store": False,
        "reasoning": {"effort": "xhigh", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "tools": [{"type": "function", "name": "read_file", "parameters": {}}],
    }
    # The system prompt may be `instructions` or lead `input`: the input checks skip it.
    assert post(provider, request(*TOOL_TURN, instructions="You are a test.", **shape)).is_success
    system = {"role": "developer", "content": [{"type": "input_text", "text": "You are X."}]}
    assert post(provider, request(system, *TOOL_TURN, **shape)).is_success


@pytest.mark.parametrize(
    ("expect", "failure"),
    [
        ({"input_len": 3}, "4 input items, expected 3"),
        ({"last_type": "message"}, "last item is 'function_call_output', expected 'message'"),
        ({"last_role": "user"}, "last role is None, expected 'user'"),
        ({"last_content_contains": "bye"}, "last item lacks 'bye': '3 rows'"),
        (
            {"tool_result_contains": {"call_1": "4 rows"}},
            "function_call_output call_1 lacks '4 rows': '3 rows'",
        ),
        (
            {"input_at": [{"index": 0, "type": "reasoning"}]},
            "input item 0 type is 'message', expected 'reasoning'",
        ),
        (
            {"input_at": [{"index": 0, "phase": "commentary"}]},
            "input item 0 phase is None, expected 'commentary'",
        ),
        ({"input_at": [{"index": 9}]}, "no input item at index 9"),
        ({"system_contains": "Rocket"}, "system prompt lacks 'Rocket': ''"),
        (
            {"body_contains": {"include": "reasoning.encrypted_content"}},
            "body include is nothing, expected a list containing 'reasoning.encrypted_content'",
        ),
        (
            {"body_equals": {"input.0.role": "system"}},
            "body input.0.role is 'user', expected 'system'",
        ),
        ({"body_equals": {"input.7.role": "user"}}, "body lacks 'input.7.role'"),
    ],
)
def test_each_expect_mismatch_is_named(serve, expect, failure):
    provider = serve(scenario("T", CALLS, says("never", expect=expect)))
    post(provider, request())
    response = post(provider, request(*TOOL_TURN))
    assert response.status_code == 500
    assert response.json()["error"]["message"] == f"expect failed (exchange 2): {failure}"


def test_reasoning_summaries_stream_only_when_asked_for(serve):
    """The API streams no summary unless the request asks for one (`reasoning.summary`): the
    reasoning item's summary is then empty, and it is replayed so."""
    expect = {"reasoning_replayed": ["rs_1"]}
    provider = serve(scenario("T", CALLS, says("ok", expect=expect)))
    unasked = {"reasoning": {"effort": "xhigh"}}
    got = events(post(provider, request(**unasked)))
    assert not [name for name, _ in got if "summary" in name]
    reasoning = next(data["item"] for name, data in got if name == "response.output_item.done")
    assert reasoning == {**DONE_REASONING, "summary": []}
    assert post(provider, request(USER, reasoning, DONE_CALL, OUTPUT, **unasked)).is_success


@pytest.mark.parametrize(
    ("replayed", "failure"),
    [
        ([USER, DONE_CALL, OUTPUT], "reasoning rs_1 is replayed 0 times, expected once"),
        ([USER, DONE_REASONING, DONE_REASONING, DONE_CALL, OUTPUT], "replayed 2 times"),
        (
            [USER, {**DONE_REASONING, "summary": [], "status": "completed"}, DONE_CALL, OUTPUT],
            "reasoning rs_1 is not replayed as sent (adds 'status', changes 'summary')",
        ),
        (
            [
                USER,
                {k: v for k, v in DONE_REASONING.items() if k != "encrypted_content"},
                DONE_CALL,
            ],
            "reasoning rs_1 is not replayed as sent (lacks 'encrypted_content')",
        ),
    ],
)
def test_reasoning_must_be_replayed_exactly_as_sent(serve, replayed, failure):
    expect = {"reasoning_replayed": ["rs_1"]}
    provider = serve(scenario("T", CALLS, says("never", expect=expect)))
    post(provider, request())
    response = post(provider, request(*replayed))
    assert response.status_code == 500
    assert failure in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (
            lambda s: s["exchanges"][0]["respond"]["stream"].pop(),
            "$.exchanges[0].respond.stream: must end with one of completed, incomplete, error,"
            " failed, stall",
        ),
        (
            lambda s: s["exchanges"][0]["respond"]["stream"].insert(0, {"stall": True}),
            "$.exchanges[0].respond.stream: must end with one of",
        ),
        (
            lambda s: s["exchanges"][0]["respond"]["stream"][0].update(done=False),
            "$.exchanges[0].respond.stream[0]: done: false must come right before the incomplete,"
            " error, failed or stall",
        ),
        (
            lambda s: s["exchanges"][0]["respond"]["stream"].insert(0, {"finish": "stop"}),
            "$.exchanges[0].respond.stream[0]: expected exactly one of: text, reasoning_item,"
            " tool_calls, completed, incomplete, error, failed, stall",
        ),
        (lambda s: s.update(style="openai"), "Additional properties are not allowed ('style'"),
        (
            lambda s: s.update(strict={"reject_unsigned_reasoning": True}),
            "Additional properties are not allowed ('reject_unsigned_reasoning'",
        ),
        (
            lambda s: s["exchanges"].append(
                says("x") | {"respond": {"stream": [{"reasoning_item": REASONING}, COMPLETED]}}
            ),
            "$.exchanges: reasoning ids must be unique: ['rs_1']",
        ),
        (
            lambda s: s["exchanges"][0].update(expect={"reasoning_replayed": ["rs_9"]}),
            "$.exchanges[0].expect: no earlier exchange sends reasoning ['rs_9']",
        ),
        (  # a request cannot replay the reasoning its own response sends
            lambda s: s["exchanges"][0].update(expect={"reasoning_replayed": ["rs_1"]}),
            "$.exchanges[0].expect: no earlier exchange sends reasoning ['rs_1']",
        ),
        (lambda s: s["exchanges"][0].update(expect={"messages_len": 1}), "'messages_len'"),
        (lambda s: s["model"].update(kind="openai_chat"), "$.model.kind: 'openai_chat' is not one"),
    ],
)
def test_invalid_responses_scenarios_have_clear_errors(tmp_path: Path, change, error: str):
    data = scenario("T", says("ok"))
    data["exchanges"] = [CALLS]  # a fresh copy is changed below
    data = json.loads(json.dumps(data))
    change(data)
    (tmp_path / "T.json").write_text(json.dumps(data))
    with pytest.raises(ScenarioError) as info:
        load_scenario(tmp_path / "T.json")
    assert error in str(info.value)


@pytest.mark.parametrize("sid", [f"R0{n}" for n in range(1, 6)])
def test_r_scenarios_speak_the_responses_api(sid):
    s = load_scenario(SCENARIOS_DIR / f"{sid}.json")
    assert (s.api, s.style) == ("responses", "responses")
    assert s.model == {
        "kind": "openai_responses",
        "model": "gpt-6-luna",
        "reasoning": {"effort": "xhigh", "summary": "auto"},
        "temperature": None,
    }
    assert s.strict["reject_unencrypted_reasoning"] is True
    assert "temperature" in s.strict["reject_params"]


async def test_the_openai_sdk_parses_every_event(serve):
    """The fake's stream is what the official SDK expects: every event is a typed event."""
    openai = pytest.importorskip("openai")
    ops = [{"reasoning_item": REASONING}, {"text": "hi", "phase": "final_answer"},
           {"tool_calls": [CALL]}, COMPLETED]  # fmt: skip
    failing = [{"text": "x", "done": False}, {"error": {"message": "boom"}}]
    cut = [{"reasoning_item": {**REASONING, "id": "rs_2"}}, {"incomplete": COMPLETED["completed"]}]
    streams = [{"respond": {"stream": stream}} for stream in (ops, failing, cut)]
    provider = serve(scenario("T", *streams))
    client = openai.AsyncOpenAI(base_url=provider.base_url("T", "r1", "our"), api_key="dummy")
    async with client:
        stream = await client.responses.create(model=MODEL, input="hi", stream=True)
        seen = [event async for event in stream]
        assert all(type(e).__name__.startswith("Response") for e in seen)
        final = seen[-1].response
        assert final.usage.input_tokens_details.cached_tokens == 0
        reasoning, message, call = final.output
        assert reasoning.encrypted_content == ENC
        assert (message.phase, message.content[0].text) == ("final_answer", "hi")
        assert (call.call_id, call.arguments) == ("call_1", DONE_CALL["arguments"])
        stream = await client.responses.create(model=MODEL, input="hi", stream=True)
        last = [event async for event in stream][-1]
        assert (type(last).__name__, last.code, last.message) == (
            "ResponseErrorEvent",
            "server_error",
            "boom",
        )
        stream = await client.responses.create(model=MODEL, input="hi", stream=True)
        last = [event async for event in stream][-1]
        assert type(last).__name__ == "ResponseIncompleteEvent"
        assert last.response.incomplete_details.reason == "max_output_tokens"

import asyncio
import contextlib
import json
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bakeoff.fakeprov.__main__ import main
from bakeoff.fakeprov.script import SCENARIOS_DIR, ScenarioError, load_scenario
from bakeoff.fakeprov.server import FakeProvider

MODEL = "anthropic/claude-sonnet-4.5"
USER = {"role": "user", "content": "hi"}
DESIGN_TOOLS = {
    "list_components",
    "describe_component",
    "validate_pipeline",
    "list_files",
    "read_file",
    "write_file",
    "edit_file",
}
SIG = {"type": "reasoning.text", "signature": "sig", "index": 0}


def scenario(sid: str, *exchanges: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {
        "id": sid,
        "title": "test",
        "system": "You are a test.",
        "model": {"kind": "openrouter", "model": MODEL},
        "rules": {"*": "allow"},
        "limits": {"max_steps": 4},
        "engine": {"delay_ms": 0},
        "driver": [{"user": "hi"}],
        "exchanges": list(exchanges),
        "expect": {"stops": ["end_turn"]},
        **top,
    }


def says(text: str, **exchange: Any) -> dict[str, Any]:
    return {"respond": {"stream": [{"text": text}, {"finish": "stop"}]}, **exchange}


def request(*messages: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"model": MODEL, "stream": True, "messages": list(messages or [USER]), **extra}


def frames(response: httpx.Response) -> list[Any]:
    """SSE data payloads (JSON-decoded, except [DONE]) and comments of a whole response."""
    out: list[Any] = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            out.append(data if data == "[DONE]" else json.loads(data))
        elif line.startswith(":"):
            out.append(line)
    return out


def text_of(chunks: list[Any]) -> str:
    return "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if isinstance(c, dict) and c["choices"]
    )


@pytest.fixture
def serve(tmp_path: Path) -> Iterator[Callable[..., FakeProvider]]:
    with contextlib.ExitStack() as stack:

        def start(*scenarios: dict[str, Any]) -> FakeProvider:
            for s in scenarios:
                (tmp_path / f"{s['id']}.json").write_text(json.dumps(s))
            return stack.enter_context(FakeProvider(tmp_path, tmp_path / "wire"))

        yield start


def chat(provider: FakeProvider, body: Any, sid: str = "T", run: str = "r1", impl: str = "our"):
    url = provider.base_url(sid, run, impl) + "/chat/completions"
    return httpx.post(url, json=body)


def meta(provider: FakeProvider, n: int, sid: str = "T", run: str = "r1", impl: str = "our"):
    return json.loads((provider.wire_dir / sid / run / impl / f"{n:03d}.meta.json").read_text())


def deltas(response: httpx.Response) -> list[dict[str, Any]]:
    return [c["choices"][0]["delta"] for c in frames(response) if isinstance(c, dict)]


def test_sse_frames_are_http_chunks(serve):
    stream = [{"comment": "OPENROUTER PROCESSING"}, {"text": "Hello world", "chunks": 2}]
    provider = serve(scenario("T", {"respond": {"stream": [*stream, {"finish": "stop"}]}}))
    raw_body = json.dumps(request()).encode()
    with socket.create_connection(("127.0.0.1", provider.port)) as sock:
        head = f"POST /s/T/r1/our/v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {len(raw_body)}\r\n\r\n"
        sock.sendall(head.encode() + raw_body)
        data = b""
        while not data.endswith(b"\r\n0\r\n\r\n"):
            data += sock.recv(65536)
    headers, _, body = data.partition(b"\r\n\r\n")
    assert headers.startswith(b"HTTP/1.1 200")
    assert b"Transfer-Encoding: chunked" in headers
    assert b"Content-Type: text/event-stream" in headers
    events = []
    while body:
        size, _, body = body.partition(b"\r\n")
        events.append(body[: int(size, 16)])
        body = body[int(size, 16) + 2 :]
    assert events[-1] == b""  # the terminating zero-length chunk
    assert events[0] == b": OPENROUTER PROCESSING\n\n"
    assert events[-2] == b"data: [DONE]\n\n"
    assert all(e.startswith(b"data: {") and e.endswith(b"}\n\n") for e in events[1:-2])
    first = json.loads(events[1].removeprefix(b"data: "))
    assert first == {
        "id": "gen-T-001",
        "object": "chat.completion.chunk",
        "created": first["created"],
        "model": MODEL,
        "provider": "FakeProvider",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": None,
                "native_finish_reason": None,
            }
        ],
    }


def test_usage_chunk_repeats_finish_reason_and_carries_cost(serve):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0012, "cached_tokens": 8}
    ops = [{"text": "Hi"}, {"finish": "stop"}, {"usage": {**usage, "reasoning_tokens": 2}}]
    provider = serve(scenario("T", {"respond": {"stream": ops}}))
    chunks = frames(chat(provider, request()))
    finish, last = chunks[-3], chunks[-2]
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["choices"][0]["native_finish_reason"] == "stop"
    assert last["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.0012,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": 8},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }


def test_openai_style_sends_usage_only_when_asked_and_without_cost(serve):
    ops = [
        {"text": "Hi"},
        {"finish": "stop"},
        {"usage": {"prompt_tokens": 3, "completion_tokens": 1}},
    ]
    provider = serve(
        scenario("T", {"respond": {"stream": ops}}, {"respond": {"stream": ops}}, style="openai")
    )
    plain = frames(chat(provider, request()))
    assert all("usage" not in c for c in plain if isinstance(c, dict))
    assert all("provider" not in c for c in plain if isinstance(c, dict))
    assert plain[0]["id"] == "chatcmpl-T-001"
    assert plain[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Hi"}
    assert plain[1]["choices"][0]["delta"] == {}
    counted = frames(chat(provider, request(stream_options={"include_usage": True})))
    assert counted[-2]["choices"] == []
    assert counted[-2]["usage"]["total_tokens"] == 4
    assert "cost" not in counted[-2]["usage"]


def test_keep_alive_reuses_one_connection(serve):
    provider = serve(scenario("T", says("one"), says("two"), says("three")))
    url = provider.base_url("T", "r1", "our") + "/chat/completions"
    with httpx.Client() as client:
        assert text_of(frames(client.post(url, json=request()))) == "one"
        assert text_of(frames(client.post(url, json=request()))) == "two"
    with httpx.Client() as client:
        client.post(url, json=request())
    assert meta(provider, 1)["conn_id"] == meta(provider, 2)["conn_id"]
    assert meta(provider, 3)["conn_id"] != meta(provider, 1)["conn_id"]


def test_reasoning_goes_to_the_requested_delta_field(serve):
    ops = [
        {"reasoning": "Let me think", "chunks": 2},
        {"reasoning": "hm", "field": "reasoning_content"},
    ]
    provider = serve(scenario("T", {"respond": {"stream": [*ops, {"text": "ok"}]}}, style="openai"))
    assert deltas(chat(provider, request())) == [
        {"role": "assistant", "reasoning": "Let me"},
        {"reasoning": " think"},
        {"reasoning_content": "hm"},
        {"content": "ok"},
    ]


DETAILS = [
    {"type": "reasoning.text", "text": "Plan: ", "format": "anthropic-claude-v1", "index": 0},
    {"type": "reasoning.text", "signature": "sig", "index": 0},  # metadata only
    {"type": "reasoning.encrypted", "data": "opaque", "index": 1, "future_field": [1]},
]


@pytest.mark.parametrize("style", ["openrouter", "openai"])
def test_reasoning_details_are_sent_verbatim(serve, style):
    provider = serve(
        scenario("T", {"respond": {"stream": [{"reasoning_details": DETAILS}]}}, style=style)
    )
    sent = [d for d in deltas(chat(provider, request())) if "reasoning_details" in d]
    assert [d["reasoning_details"] for d in sent] == [[detail] for detail in DETAILS]
    # OpenRouter mirrors the text of a reasoning.text fragment (only one with text) in `reasoning`
    mirrored = ["Plan: ", None, None] if style == "openrouter" else [None] * 3
    assert [d.get("reasoning") for d in sent] == mirrored


def test_delay_ms_pauses_before_each_chunk(serve):
    provider = serve(
        scenario("T", {"respond": {"stream": [{"text": "abc", "chunks": 3, "delay_ms": 50}]}})
    )
    started = time.monotonic()
    assert text_of(frames(chat(provider, request()))) == "abc"
    assert time.monotonic() - started >= 0.15


def test_rate_limit_with_retry_after_then_ok(serve):
    limited = {"respond": {"status": 429, "headers": {"retry-after": "1"}}}
    provider = serve(scenario("T", limited, says("ok")))
    first = chat(provider, request())
    assert first.status_code == 429
    assert first.headers["retry-after"] == "1"
    assert first.json()["error"]["code"] == 429
    second = chat(provider, request())
    assert second.status_code == 200
    assert text_of(frames(second)) == "ok"
    assert [meta(provider, n)["status"] for n in (1, 2)] == [429, 200]


def test_openai_style_error_body(serve):
    provider = serve(scenario("T", {"respond": {"status": 503}}, style="openai"))
    error = {"message": "Service Unavailable", "type": "api_error", "param": None, "code": None}
    assert chat(provider, request()).json() == {"error": error}


def test_sse_error_ends_the_stream_after_http_200(serve):
    error = {"message": "Provider disconnected"}
    provider = serve(scenario("T", {"respond": {"stream": [{"text": "Hi"}, {"sse_error": error}]}}))
    response = chat(provider, request())
    chunks = frames(response)
    assert response.status_code == 200
    assert "[DONE]" not in chunks
    # OpenRouter's documented mid-stream error: a string code at the top level
    assert chunks[-1]["error"] == {"code": "server_error", "message": "Provider disconnected"}
    assert chunks[-1]["choices"] == [
        {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
    ]


SLOW = [{"text": "x" * 10, "chunks": 10, "delay_ms": 20}]  # still writing when the client leaves


@pytest.mark.parametrize("stream", [[{"text": "Let me"}, {"stall": True}], SLOW])
def test_server_survives_a_client_that_gives_up(serve, capfd, stream):
    provider = serve(scenario("T", {"respond": {"stream": stream}}, says("done")))
    url = provider.base_url("T", "r1", "our") + "/chat/completions"
    with httpx.Client() as client, client.stream("POST", url, json=request()) as response:
        assert next(response.iter_lines()).startswith("data: ")
    # the client closed mid-stream; the server carries on with the next exchange
    assert text_of(frames(chat(provider, request()))) == "done"
    time.sleep(0.25)  # outlast SLOW, so its handler writes into the closed socket before stop()
    provider.stop()  # joins the handler of the abandoned stream
    assert capfd.readouterr().err == ""


def test_server_survives_a_killed_client_process(serve):
    stalled = {"respond": {"stream": [{"text": "Let me"}, {"stall": True}]}}
    provider = serve(scenario("T", stalled, says("done")))
    url = provider.base_url("T", "r1", "our") + "/chat/completions"
    code = (
        "import httpx, sys\n"
        f"with httpx.stream('POST', {url!r}, json={request()!r}) as r:\n"
        "    print(next(r.iter_lines()), flush=True)\n"
        "    sys.stdin.read()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert child.stdout is not None and child.stdout.readline().startswith("data: ")
    child.kill()
    child.wait()
    assert text_of(frames(chat(provider, request()))) == "done"


@pytest.mark.parametrize("op", [{"stall": True}, {"text": "late", "delay_ms": 60_000}])
def test_stop_ends_a_waiting_stream_promptly(serve, op):
    provider = serve(scenario("T", {"respond": {"stream": [op]}}))
    url = provider.base_url("T", "r1", "our") + "/chat/completions"
    with httpx.Client() as client, client.stream("POST", url, json=request()) as response:
        started = time.monotonic()
        provider.stop()
        assert time.monotonic() - started < 1
        with pytest.raises(httpx.RemoteProtocolError):
            response.read()


CALLS_F = {
    "respond": {"stream": [{"tool_calls": [{"id": "call_1", "name": "f", "arguments": {}}]}]}
}
CALL_1 = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
TOOL_TURN = [
    USER,
    {"role": "assistant", "content": None, "tool_calls": [CALL_1]},
    {"role": "tool", "tool_call_id": "call_1", "content": "2 rows"},
]


def test_expect_checks_pass_on_a_matching_request(serve):
    expect = {
        "body_has": ["tools"],
        "body_lacks": ["reasoning"],
        "model": MODEL,
        "messages_len": 4,
        "last_role": "tool",
        "last_content_contains": "rows",
        "tool_result_contains": {"call_1": "3 rows"},
    }
    provider = serve(scenario("T", CALLS_F, says("ok", expect=expect)))
    assert chat(provider, request()).status_code == 200
    messages = [
        {"role": "system", "content": "s"},
        *TOOL_TURN[:2],
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "3 rows"}]},
    ]
    assert chat(provider, request(*messages, tools=[])).status_code == 200


@pytest.mark.parametrize(
    ("expect", "failure"),
    [
        ({"body_has": ["tools"]}, "body lacks 'tools'"),
        ({"body_lacks": ["stream"]}, "body has 'stream'"),
        ({"model": "other/model"}, f"model is '{MODEL}', expected 'other/model'"),
        ({"messages_len": 2}, "3 messages, expected 2"),
        ({"last_role": "user"}, "last role is 'tool', expected 'user'"),
        ({"last_content_contains": "bye"}, "last message lacks 'bye': '2 rows'"),
        (
            {"tool_result_contains": {"call_1": "3 rows"}},
            "tool result call_1 lacks '3 rows': '2 rows'",
        ),
        (
            {"messages_at": [{"index": 0, "role": "system"}]},
            "message 0 role is 'user', expected 'system'",
        ),
        ({"messages_at": [{"index": -1, "contains": "zzz"}]}, "message -1 lacks 'zzz'"),
        ({"messages_at": [{"index": 9}]}, "no message at index 9"),
        (
            {"body_equals": {"stream_options.include_usage": True}},
            "body lacks 'stream_options.include_usage'",
        ),
        ({"body_equals": {"stream": False}}, "body stream is True, expected False"),
    ],
)
def test_each_expect_mismatch_is_named(serve, expect, failure):
    provider = serve(scenario("T", CALLS_F, says("never", expect=expect)))
    chat(provider, request())
    response = chat(provider, request(*TOOL_TURN))
    assert response.status_code == 500
    assert response.json()["error"]["message"] == f"expect failed (exchange 2): {failure}"


def test_expect_mismatch_is_a_recorded_500(serve):
    expect = {"last_role": "tool", "tool_result_contains": {"call_1": "x"}}
    provider = serve(scenario("T", CALLS_F, says("never", expect=expect)))
    chat(provider, request())
    response = chat(provider, request())
    assert response.status_code == 500
    assert response.headers["x-should-retry"] == "false"
    message = response.json()["error"]["message"]
    assert message.startswith("expect failed (exchange 2): last role is 'user'")
    assert "no tool result for call_1" in message
    assert meta(provider, 2)["status"] == 500
    assert meta(provider, 2)["error"] == message


def test_reject_params_answers_400(serve):
    strict = {"reject_params": ["reasoning_effort", "stream_options"]}
    provider = serve(scenario("T", says("ok"), says("ok"), strict=strict))
    response = chat(provider, request(stream_options={"include_usage": True}))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "stream_options"
    assert chat(provider, request()).status_code == 200


def test_reject_unsigned_reasoning_merges_fragments_by_index(serve):
    exchanges = [says("ok", strict={"reject_unsigned_reasoning": True}) for _ in range(3)]
    provider = serve(scenario("T", *exchanges))
    text = {"type": "reasoning.text", "text": "thinking", "index": 0}
    encrypted = {"type": "reasoning.encrypted", "data": "opaque", "index": 1}

    def replaying(details: list[dict[str, Any]]) -> dict[str, Any]:
        assistant = {"role": "assistant", "content": "a", "reasoning_details": details}
        return request(USER, assistant, USER)

    unsigned = chat(provider, replaying([text, encrypted]))
    assert unsigned.status_code == 400
    assert "index 0" in unsigned.json()["error"]["message"]
    assert chat(provider, replaying([text, SIG, encrypted])).status_code == 200
    assert chat(provider, replaying([{**text, "signature": "sig"}])).status_code == 200


def test_wire_recording_is_verbatim_and_headerless(serve):
    provider = serve(scenario("T", says("one"), says("two")))
    stale = provider.wire_dir / "T" / "r1" / "our" / "007.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an older server")
    raw = (
        '{ "model" : "m",\n "stream":true, "messages":[{"role":"user","content":"héllo \\u00e9"}]}'
    )
    url = provider.base_url("T", "r1", "our") + "/chat/completions"
    secret = "Bearer sk-test-not-a-real-key"
    for _ in range(2):
        httpx.post(url, content=raw.encode(), headers={"Authorization": secret})
    recorded = sorted(p.name for p in stale.parent.iterdir())
    assert recorded == ["001.json", "001.meta.json", "002.json", "002.meta.json"]
    assert (stale.parent / "001.json").read_bytes() == raw.encode()
    assert set(meta(provider, 2)) == {"conn_id", "path", "t_us", "status"}
    assert meta(provider, 2)["path"] == "/s/T/r1/our/v1/chat/completions"
    assert meta(provider, 2)["t_us"] >= meta(provider, 1)["t_us"]
    assert all(secret not in p.read_text() for p in stale.parent.iterdir())


def test_base_url_claims_the_recording_even_without_requests(serve):
    provider = serve(scenario("T", says("one"), says("two")))
    folder = provider.wire_dir / "T" / "r1" / "our"
    folder.mkdir(parents=True)
    (folder / "001.json").write_text("from an older server")
    provider.base_url("T", "r1", "our")
    assert list(folder.iterdir()) == []  # a run that sends nothing leaves nothing stale
    chat(provider, request())
    provider.base_url("T", "r1", "our")  # later calls keep this run's recording
    assert sorted(p.name for p in folder.iterdir()) == ["001.json", "001.meta.json"]


def test_a_failed_recording_answers_500(serve):
    provider = serve(scenario("T", says("one")))
    url = provider.base_url("T", "r1", "our") + "/chat/completions"
    shutil.rmtree(provider.wire_dir)  # e.g. deleted under a running server
    response = httpx.post(url, json=request())
    assert response.status_code == 500
    assert response.json()["error"]["message"].startswith("recording failed: ")


def test_a_body_cut_short_is_not_answered_recorded_or_counted(serve):
    provider = serve(scenario("T", says("one"), says("two")))
    path = httpx.URL(provider.base_url("T", "r1", "our")).path + "/chat/completions"
    body = json.dumps(request()).encode()
    with socket.create_connection(("127.0.0.1", provider.port)) as sock:
        head = f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n\r\n"
        sock.sendall(head.encode() + body[: len(body) // 2])
        sock.shutdown(socket.SHUT_WR)  # the client dies mid-send
        assert sock.recv(65536) == b""  # the server hangs up without an answer
    assert text_of(frames(chat(provider, request()))) == "one"
    folder = provider.wire_dir / "T" / "r1" / "our"
    assert sorted(p.name for p in folder.iterdir()) == ["001.json", "001.meta.json"]


async def test_many_simultaneous_connections_are_all_served(serve):
    provider = serve(scenario("T", says("ok")))
    urls = [provider.base_url("T", f"r{n}", "our") + "/chat/completions" for n in range(64)]
    started = time.monotonic()
    async with httpx.AsyncClient() as client:  # one new connection per request
        responses = await asyncio.gather(*(client.post(url, json=request()) for url in urls))
    # A full listen backlog drops connects: they reset, or the client resends its SYN after 1 s.
    assert time.monotonic() - started < 1
    assert [text_of(frames(r)) for r in responses] == ["ok"] * 64
    assert len({meta(provider, 1, run=f"r{n}")["conn_id"] for n in range(64)}) == 64


def test_cursors_are_independent_per_run_and_impl(serve):
    provider = serve(scenario("T", says("first"), says("second")))
    for run, impl in [("r1", "our"), ("r1", "pydantic"), ("r2", "our")]:
        assert text_of(frames(chat(provider, request(), run=run, impl=impl))) == "first"
    assert text_of(frames(chat(provider, request()))) == "second"
    exhausted = chat(provider, request())
    assert exhausted.status_code == 500
    assert exhausted.json()["error"]["message"] == "script exhausted"


def test_interleaved_tool_call_arguments(serve):
    calls = [
        {"id": "call_a", "name": "read_file", "arguments": {"path": "a.pipe"}},
        {"id": "call_b", "name": "list_files", "arguments": '{"path": "."}'},
    ]
    op = {"tool_calls": calls, "pieces": 3, "interleave": True}
    provider = serve(scenario("T", {"respond": {"stream": [op, {"finish": "tool_calls"}]}}))
    deltas = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in frames(chat(provider, request()))
        if isinstance(c, dict) and "tool_calls" in c["choices"][0]["delta"]
    ]
    assert [d["index"] for d in deltas] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert [d.get("id") for d in deltas[:2]] == ["call_a", "call_b"]
    assert deltas[0]["function"] == {"name": "read_file", "arguments": ""}
    arguments = [
        "".join(d["function"]["arguments"] for d in deltas if d["index"] == i) for i in (0, 1)
    ]
    assert arguments == ['{"path": "a.pipe"}', '{"path": "."}']


def test_bad_requests_and_models_endpoint(serve):
    provider = serve(scenario("T", says("ok")))
    url = provider.base_url("T", "r1", "our")
    assert httpx.post(url + "/chat/completions", content=b"{not json").status_code == 400
    assert chat(provider, {**request(), "stream": False}).status_code == 400
    assert chat(provider, {**request(), "messages": []}).status_code == 400
    assert chat(provider, {**request(), "messages": ["hi"]}).status_code == 400
    assert httpx.post(url + "/chat/completions", content=iter([b"{}"])).status_code == 411
    assert httpx.post(url + "/embeddings", json=request()).status_code == 404
    assert httpx.get(url + "/chat/completions").status_code == 404
    assert httpx.get(url + "/models").json()["data"][0]["id"] == MODEL
    assert httpx.get(provider.base_url("Nope", "r1", "our") + "/models").status_code == 404
    with pytest.raises(ValueError):
        provider.base_url("..", "r1", "our")


SCENARIO_FILES = sorted(SCENARIOS_DIR.glob("*.json"))


def test_every_designed_scenario_exists():
    expected = {f"S{n:02d}" for n in range(1, 16)} - {"S10"} | {"S10a", "S10b", "S12b"}
    expected |= {f"R{n:02d}" for n in range(1, 6)}  # the Responses API scenarios
    assert {p.stem for p in SCENARIO_FILES} == expected


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_scenario_file_is_valid_and_consistent(path: Path):
    s = load_scenario(path)
    assert s.expect["requests"] == len(s.exchanges)  # every scripted exchange gets used
    for exchange in s.exchanges:
        for op in exchange["respond"].get("stream", []):
            for call in op.get("tool_calls", []):
                assert call["id"].startswith(f"call_{s.id}_")
                assert call["name"] in DESIGN_TOOLS
                assert isinstance(call["arguments"], dict)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (
            lambda s: s["exchanges"][0]["respond"]["stream"].insert(0, {"txt": "x"}),
            "$.exchanges[0].respond.stream[0]: expected exactly one of: text, reasoning",
        ),
        (
            lambda s: s["exchanges"][0]["respond"]["stream"].insert(0, {"stall": True}),
            "$.exchanges[0].respond.stream[0]: sse_error and stall must be last",
        ),
        (lambda s: s["expect"]["stops"].append("end_turn"), "$.expect.stops: lists 2 stops for 1"),
        (
            lambda s: s["exchanges"].extend([CALLS_F, CALLS_F]),
            "$.exchanges: tool call ids must be unique: ['call_1']",
        ),
        (
            lambda s: s["driver"].append({"crash_after": "tool.end"}),
            "$.driver[1]: crash_after must be followed by a user step",
        ),
        (lambda s: s["driver"].insert(0, {"crash_after": "boom"}), "$.driver[0].crash_after"),
        (
            lambda s: s["driver"].insert(0, {"crash_after": "turn.end", "call_id": "call_1"}),
            "$.driver[0].call_id: turn.end events name no tool call",
        ),
        (
            lambda s: s["driver"].insert(0, {"crash_after": "item", "call_id": "call_x"}),
            "$.expect: unknown tool call ids: ['call_x']",
        ),
        (lambda s: s.update(id="Other"), "$.id: 'Other' must match the file name"),
        (
            lambda s: s["expect"].update(tool_runs={"call_x": 1}),
            "$.expect: unknown tool call ids: ['call_x']",
        ),
        (
            lambda s: s["expect"].update(tools_overlap=["call_x", "call_y"]),
            "$.expect: unknown tool call ids: ['call_x', 'call_y']",
        ),
        (lambda s: s["expect"].update(tools_overlap=["call_x"]), "$.expect.tools_overlap"),
        (
            lambda s: s["expect"].update(cancel_within_ms=200),
            "$.expect.cancel_within_ms: no driver step has cancel_after_ms",
        ),
    ],
)
def test_invalid_scenarios_have_clear_errors(tmp_path: Path, change, error: str):
    data = scenario("T", says("ok"))
    change(data)
    (tmp_path / "T.json").write_text(json.dumps(data))
    with pytest.raises(ScenarioError) as info:
        load_scenario(tmp_path / "T.json")
    assert str(info.value).startswith("T.json: ")
    assert error in str(info.value)


def test_check_command(tmp_path: Path, capsys):
    assert main(["--check"]) == 0
    assert capsys.readouterr().out.count("ok    ") == len(SCENARIO_FILES)
    (tmp_path / "T.json").write_text("{}")
    assert main(["--check", "--scenarios", str(tmp_path)]) == 1
    assert capsys.readouterr().out.startswith("FAIL  T.json: ")


def test_a_broken_scenario_is_reported_by_the_server(serve):
    provider = serve(scenario("T", {"respond": {"status": 200}}))
    response = chat(provider, request())
    assert response.status_code == 500
    assert "status 200 needs a stream" in response.json()["error"]["message"]


def test_min_gap_rejects_an_immediate_retry_and_accepts_a_patient_one(serve):
    """A retry that ignores retry-after arrives too soon and fails the scenario."""
    limited = {"respond": {"status": 429, "headers": {"retry-after": "1"}}}
    provider = serve(
        scenario("T", limited, says("ok", expect={"min_gap_ms": 200}), limited, says("ok"))
    )
    assert chat(provider, request()).status_code == 429
    early = chat(provider, request())
    assert early.status_code == 500
    assert "expected at least 200 ms" in early.json()["error"]["message"]

    provider2 = serve(scenario("U", limited, says("ok", expect={"min_gap_ms": 200})))
    assert chat(provider2, request(), sid="U").status_code == 429
    time.sleep(0.25)
    assert chat(provider2, request(), sid="U").status_code == 200


def test_restart_is_a_new_server_run(serve):
    """stop() + start() on the same instance: cursors restart and recordings are fresh."""
    provider = serve(scenario("T", says("first"), says("second")))
    assert text_of(frames(chat(provider, request()))) == "first"
    provider.stop()
    provider.start()
    assert text_of(frames(chat(provider, request()))) == "first"
    wire = provider.wire_dir / "T" / "r1" / "our"
    assert sorted(p.name for p in wire.glob("*.json")) == ["001.json", "001.meta.json"]

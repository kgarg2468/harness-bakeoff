"""Scenario scripts for the fake provider.

`load_scenario()` reads and validates a scenario file. `reply()` answers one request against
a scenario's exchange: it enforces the strict modes and the exchange's `expect`, then turns
the scripted `respond` into the SSE frames and control ops that the server plays back.
A scenario speaks one API, chosen by its model kind: chat completions (OpenRouter or OpenAI
style), or OpenAI's Responses API (`openai_responses`, section "Responses API" below).
Everything here is deterministic: the same script and request always give the same bytes.
The file format is documented in `fakeprov/README.md`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import string
from dataclasses import dataclass, field
from http.client import responses
from pathlib import Path
from typing import Any, Literal, NoReturn, get_args

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import best_match

from bakeoff.shared.contract import LOOP_EVENTS, SHARED_EVENTS, StopReason

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
CREATED = 1758758400  # the fixed `created` of every chunk
PROVIDER = "FakeProvider"
type Api = Literal["chat", "responses"]
# The path under `<base_url>` that each API answers on.
ENDPOINTS: dict[Api, str] = {"chat": "chat/completions", "responses": "responses"}


class ScenarioError(ValueError):
    """A scenario file is unreadable or breaks the schema. The message names the spot."""


@dataclass(frozen=True, slots=True)
class Scenario:
    """A validated scenario file (see fakeprov/README.md for each field)."""

    id: str
    title: str
    system: str
    model: dict[str, Any]
    rules: dict[str, Any]
    limits: dict[str, Any]
    engine: dict[str, Any]
    driver: list[dict[str, Any]]
    exchanges: list[dict[str, Any]]
    expect: dict[str, Any]
    # Wire style of every exchange that does not set its own: "openrouter" or "openai" (chat
    # completions), or "responses" (the Responses API, the only style of its scenarios).
    style: str
    strict: dict[str, Any]

    @property
    def api(self) -> Api:
        """The API the scenario is scripted for, which decides the endpoint it answers on."""
        return "responses" if self.style == "responses" else "chat"


@dataclass(frozen=True, slots=True)
class Sleep:
    """Stream control op: pause before the next frame."""

    seconds: float


@dataclass(frozen=True, slots=True)
class Stall:
    """Stream control op: send nothing more, but keep the socket open until the client leaves."""


type Op = bytes | Sleep | Stall


@dataclass(frozen=True, slots=True)
class Reply:
    """The answer to one request: a JSON `body`, or (status 200) a `stream` of SSE ops."""

    status: int
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    stream: list[Op] = field(default_factory=list)
    error: str | None = None  # why the fake server itself rejected the request


# --- schema ---------------------------------------------------------------------------------

_STR: dict[str, Any] = {"type": "string"}
_STRS: dict[str, Any] = {"type": "array", "items": _STR}
_INT0: dict[str, Any] = {"type": "integer", "minimum": 0}
_INT1: dict[str, Any] = {"type": "integer", "minimum": 1}
_BOOL: dict[str, Any] = {"type": "boolean"}


def _obj(properties: dict[str, Any], *required: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _kinds(kinds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """An object that is exactly one of `kinds`, told apart by which key it has."""
    return {
        "type": "object",
        "anyOf": [{"required": [kind]} for kind in kinds],
        "allOf": [{"if": {"required": [kind]}, "then": schema} for kind, schema in kinds.items()],
    }


def _op(kind: str, value: dict[str, Any], **options: dict[str, Any]) -> dict[str, Any]:
    return _obj({kind: value, "delay_ms": _INT0, **options}, kind)


_CALLS = {
    "type": "array",
    "minItems": 1,
    "items": _obj(
        {"id": _STR, "name": _STR, "arguments": {"type": ["string", "object"]}},
        "id",
        "name",
        "arguments",
    ),
}
_OPS = {  # chat completions
    "text": _op("text", _STR, chunks=_INT1),
    "reasoning": _op(
        "reasoning", _STR, chunks=_INT1, field={"enum": ["reasoning", "reasoning_content"]}
    ),
    "reasoning_details": _op(
        "reasoning_details", {"type": "array", "items": {"type": "object", "required": ["type"]}}
    ),
    "tool_calls": _op("tool_calls", _CALLS, pieces=_INT1, interleave=_BOOL),
    "comment": _op("comment", _STR),
    "finish": _op("finish", {"enum": ["stop", "length", "tool_calls", "content_filter"]}),
    "usage": _op(
        "usage",
        _obj(
            {
                "prompt_tokens": _INT0,
                "completion_tokens": _INT0,
                "cached_tokens": _INT0,
                "reasoning_tokens": _INT0,
                "cost": {"type": "number", "minimum": 0},
                "is_byok": _BOOL,
            },
            "prompt_tokens",
            "completion_tokens",
        ),
    ),
    "sse_error": _op(
        "sse_error", _obj({"code": {"type": ["integer", "string"]}, "message": _STR}, "message")
    ),
    "stall": _op("stall", {"const": True}),
}
_ERROR = _obj({"code": _STR, "message": _STR}, "message")
_RESPONSES_USAGE = {
    "input_tokens": _INT0,
    "output_tokens": _INT0,
    "cached_tokens": _INT0,
    "cache_write_tokens": _INT0,
    "reasoning_tokens": _INT0,
}
# `done: false` cuts the stream before the item is done: its done events are never sent.
_RESPONSES_OPS = {
    "text": _op(
        "text", _STR, chunks=_INT1, phase={"enum": ["commentary", "final_answer"]}, done=_BOOL
    ),
    "reasoning_item": _op(
        "reasoning_item",
        _obj(
            # minLength 2: the added event sends its first half, which must differ from it.
            # `text`: raw reasoning text parts (some models stream their reasoning itself).
            {
                "id": _STR,
                "encrypted_content": {"type": "string", "minLength": 2},
                "summary": _STRS,
                "text": _STRS,
            },
            "id",
            "encrypted_content",
        ),
        chunks=_INT1,
        done=_BOOL,
    ),
    # `status: incomplete`: max_output_tokens cut the last call, which is done as incomplete.
    "tool_calls": _op("tool_calls", _CALLS, pieces=_INT1, status={"const": "incomplete"}),
    "completed": _op("completed", _obj(_RESPONSES_USAGE, "input_tokens", "output_tokens")),
    # max_output_tokens ran out (say, while the model reasoned): the response ends with usage.
    "incomplete": _op(
        "incomplete",
        _obj(
            {**_RESPONSES_USAGE, "reason": {"enum": ["max_output_tokens", "content_filter"]}},
            "input_tokens",
            "output_tokens",
        ),
    ),
    "error": _op("error", _ERROR),
    "failed": _op("failed", _ERROR),
    "stall": _op("stall", {"const": True}),
}
# The ops that end a Responses stream: every stream has exactly one, as its last op.
_RESPONSES_ENDS = ("completed", "incomplete", "error", "failed", "stall")

_STEPS = {
    "user": _obj({"user": _STR, "cancel_after_ms": _INT0}, "user"),
    "approve": _obj(
        {
            "approve": _obj(
                {"allow": {"anyOf": [{"const": "all"}, _STRS]}, "deny": _STRS, "reason": _STR}
            ),
            "new_process": _BOOL,
        },
        "approve",
    ),
    "crash_after": _obj(
        {"crash_after": {"enum": [*LOOP_EVENTS, *SHARED_EVENTS]}, "call_id": _STR}, "crash_after"
    ),
    "resume": _obj({"resume": {"const": "crash"}}, "resume"),
    "revert": _obj({"revert": _INT1}, "revert"),
    "compact": _obj({"compact": _STR}, "compact"),
}

# Top-level keys every scenario has; each becomes a `Scenario` field of the same name.
_REQUIRED = tuple("id title system model rules limits engine driver exchanges expect".split())
# Event types a `crash_after` step can narrow down to one tool call (see fakeprov/README.md).
_CALL_EVENTS = ("tool_call.ready", "permission.asked", "tool.start", "tool.end", "item")
_DECISION = {"enum": ["allow", "ask", "deny"]}
_KINDS = {"enum": ["openrouter", "openai_compat", "openai_responses"]}
_STYLE = {"enum": ["openrouter", "openai"]}
_STRICT = {
    "chat": _obj({"reject_params": _STRS, "reject_unsigned_reasoning": _BOOL}),
    "responses": _obj({"reject_params": _STRS, "reject_unencrypted_reasoning": _BOOL}),
}
# Exchange `expect` keys that read the body as it is, in both APIs.
_BODY_EXPECT = {
    "body_has": _STRS,
    "body_lacks": _STRS,
    "model": _STR,
    # {"dotted.path": value}: the body's value at that path equals it exactly. A path segment
    # that is an integer indexes a list (negative from the end), e.g. "tools.0.name".
    "body_equals": {"type": "object"},
    # {"dotted.path": value}: the body's value at that path is a list that contains it.
    "body_contains": {"type": "object"},
    # The request must arrive at least this long after the cursor's previous request
    # (e.g. a retry that honours `retry-after`).
    "min_gap_ms": {"type": "number", "minimum": 0},
}
_EXCHANGE_EXPECT = {
    "chat": _obj(
        {
            **_BODY_EXPECT,
            "last_role": {"enum": ["system", "developer", "user", "assistant", "tool"]},
            "last_content_contains": _STR,
            "messages_len": _INT1,
            "tool_result_contains": {"type": "object", "additionalProperties": _STR},
            # Checks on specific messages; `index` may be negative (from the end).
            "messages_at": {
                "type": "array",
                "items": _obj({"index": {"type": "integer"}, "role": _STR, "contains": _STR}),
            },
        }
    ),
    # The input checks skip the system prompt (`instructions`, or leading system/developer
    # messages in `input`), so they hold wherever a loop puts it; `system_contains` checks it.
    "responses": _obj(
        {
            **_BODY_EXPECT,
            "input_len": _INT1,
            "last_type": {
                "enum": ["message", "function_call", "function_call_output", "reasoning"]
            },
            "last_role": {"enum": ["user", "assistant"]},
            "last_content_contains": _STR,
            "input_at": {
                "type": "array",
                "items": _obj(
                    {
                        "index": {"type": "integer"},
                        "type": _STR,
                        "role": _STR,
                        "contains": _STR,
                        "phase": _STR,
                    },
                    "index",
                ),
            },
            # {call_id: substring}: a function_call_output for that call contains it.
            "tool_result_contains": {"type": "object", "additionalProperties": _STR},
            # Reasoning ids whose done item must be in `input` exactly as it was sent.
            "reasoning_replayed": {"type": "array", "items": _STR, "minItems": 1},
            "system_contains": _STR,
        }
    ),
}
_FINAL_EXPECT = _obj(
    {
        "stops": {"type": "array", "items": {"enum": list(get_args(StopReason))}},
        "files": {"type": "object", "additionalProperties": {"type": ["string", "null"]}},
        "tool_runs": {"type": "object", "additionalProperties": _INT0},
        "commits": _INT0,
        "requests": _INT0,
        "text_contains": _STR,
        "cost_usd": {"type": "number", "minimum": 0},
        "usage": _obj({"input_tokens": _INT0, "output_tokens": _INT0, "cached_tokens": _INT0}),
        "cost_source": {"enum": ["provider", "estimate", "none"]},
        # Calls whose tools must all be running at one moment (parallel tools, S03).
        "tools_overlap": {"type": "array", "items": _STR, "minItems": 2, "uniqueItems": True},
        # Every turn the driver cancels (`cancel_after_ms`) ends this soon after the cancel.
        "cancel_within_ms": {"type": "number", "minimum": 0},
    },
    "stops",
)


def _schema(api: Api) -> dict[str, Any]:
    """The scenario file schema for one API. Only chat scenarios have wire styles."""
    ops = _OPS if api == "chat" else _RESPONSES_OPS
    styles = {"style": _STYLE} if api == "chat" else {}
    respond = _obj(
        {
            "status": {"type": "integer", "minimum": 200, "maximum": 599},
            "headers": {"type": "object", "additionalProperties": _STR},
            "body": {"type": "object"},
            "stream": {"type": "array", "items": _kinds(ops)},
        }
    )
    exchange = {
        "note": _STR,
        **styles,
        "strict": _STRICT[api],
        "expect": _EXCHANGE_EXPECT[api],
        "respond": respond,
    }
    model = {
        "kind": _KINDS,
        "model": _STR,
        "reasoning": {"type": ["object", "null"]},
        # null: send no temperature (reasoning models reject one); absent: ModelConfig's default
        "temperature": {"type": ["number", "null"]},
        "compat": {"type": "object"},
    }
    rules = {
        "type": "object",
        "additionalProperties": {
            "anyOf": [_DECISION, {"type": "object", "additionalProperties": _DECISION}]
        },
    }
    return _obj(
        {
            "id": _STR,
            "title": _STR,
            "system": _STR,
            "model": _obj(model, "kind", "model"),
            "rules": rules,
            "limits": _obj({"max_steps": _INT1}, "max_steps"),
            "engine": _obj({"delay_ms": _INT0}, "delay_ms"),
            **styles,
            "strict": _STRICT[api],
            "driver": {"type": "array", "minItems": 1, "items": _kinds(_STEPS)},
            "exchanges": {
                "type": "array",
                "minItems": 1,
                "items": _obj(exchange, "respond"),
            },
            "expect": _FINAL_EXPECT,
        },
        *_REQUIRED,
    )


_VALIDATORS: dict[Api, Draft202012Validator] = {
    api: Draft202012Validator(_schema(api)) for api in ("chat", "responses")
}


# --- loading --------------------------------------------------------------------------------


def load_scenario(path: Path) -> Scenario:
    """Read and validate one scenario file. Raises `ScenarioError` with a precise location."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScenarioError(f"{path.name}: {exc}") from exc
    model = data.get("model") if isinstance(data, dict) else None
    kind = model.get("kind") if isinstance(model, dict) else None
    api: Api = "responses" if kind == "openai_responses" else "chat"
    if (error := best_match(_VALIDATORS[api].iter_errors(data))) is not None:
        raise ScenarioError(f"{path.name}: {_describe(error)}")
    if data["id"] != path.stem:
        raise ScenarioError(f"{path.name}: $.id: {data['id']!r} must match the file name")
    _check_semantics(path.name, data, api)
    default_style = {"openai_compat": "openai", "openai_responses": "responses"}.get(kind)
    return Scenario(
        **{key: data[key] for key in _REQUIRED},
        style=data.get("style", default_style or "openrouter"),
        strict=data.get("strict", {}),
    )


def _describe(error: ValidationError) -> str:
    # A step or op with no known key fails the `anyOf` of `_kinds()`; name the choices.
    kinds = error.parent if error.parent is not None else error
    if kinds.validator == "anyOf" and all("required" in s for s in kinds.validator_value):
        names = ", ".join(s["required"][0] for s in kinds.validator_value)
        return f"{kinds.json_path}: expected exactly one of: {names}"
    return f"{error.json_path}: {error.message}"


def _check_semantics(name: str, data: dict[str, Any], api: Api) -> None:
    """Rules the JSON schema cannot express."""

    def fail(where: str, message: str) -> NoReturn:
        raise ScenarioError(f"{name}: {where}: {message}")

    call_ids: list[str] = []
    reasoning_ids: list[str] = []
    for n, exchange in enumerate(data["exchanges"]):
        # A request can replay only reasoning that an earlier response sent (never its own).
        replayed = exchange.get("expect", {}).get("reasoning_replayed", [])
        if unsent := sorted(set(replayed) - set(reasoning_ids)):
            fail(f"$.exchanges[{n}].expect", f"no earlier exchange sends reasoning {unsent}")
        respond = exchange["respond"]
        where = f"$.exchanges[{n}].respond"
        if (respond.get("status", 200) == 200) != ("stream" in respond):
            fail(where, "status 200 needs a stream; other statuses take none")
        ops = respond.get("stream", [])
        if api == "chat":
            for i, op in enumerate(ops[:-1]):
                if "sse_error" in op or "stall" in op:
                    fail(f"{where}.stream[{i}]", "sse_error and stall must be last")
        elif "stream" in respond:
            ends = [i for i, op in enumerate(ops) if any(k in op for k in _RESPONSES_ENDS)]
            if ends != [len(ops) - 1]:
                fail(f"{where}.stream", f"must end with one of {', '.join(_RESPONSES_ENDS)}")
            for i, op in enumerate(ops[:-1]):
                # An item cut short is the last one of a stream that fails, stalls or runs out
                # of output tokens.
                cut = "done: false" if op.get("done") is False else None
                cut = "status: incomplete" if op.get("status") == "incomplete" else cut
                if cut and (i != len(ops) - 2 or "completed" in ops[-1]):
                    fail(f"{where}.stream[{i}]", f"{cut} must come right before the"
                         " incomplete, error, failed or stall that ends the stream")  # fmt: skip
            reasoning_ids += [op["reasoning_item"]["id"] for op in ops if "reasoning_item" in op]
        call_ids += [call["id"] for op in ops for call in op.get("tool_calls", [])]
    for what, ids in (("tool call", call_ids), ("reasoning", reasoning_ids)):
        if duplicates := sorted({c for c in ids if ids.count(c) > 1}):
            fail("$.exchanges", f"{what} ids must be unique: {duplicates}")

    referenced = set(data["expect"].get("tool_runs", {}))
    referenced |= set(data["expect"].get("tools_overlap", []))
    for exchange in data["exchanges"]:
        referenced |= set(exchange.get("expect", {}).get("tool_result_contains", {}))
    steps = data["driver"]
    turns = 0  # steps that run a turn to its end in this process
    for i, step in enumerate(steps):
        crashed = i > 0 and "crash_after" in steps[i - 1]
        if "approve" in step or "resume" in step or ("user" in step and not crashed):
            turns += 1
        if "crash_after" in step and (i + 1 == len(steps) or "user" not in steps[i + 1]):
            fail(f"$.driver[{i}]", "crash_after must be followed by a user step")
        if "call_id" in step:
            referenced.add(step["call_id"])
            if step["crash_after"] not in _CALL_EVENTS:
                fail(f"$.driver[{i}].call_id", f"{step['crash_after']} events name no tool call")
        if approve := step.get("approve"):
            allowed = approve.get("allow", [])
            referenced |= set(approve.get("deny", []))
            referenced |= set() if allowed == "all" else set(allowed)
    if unknown := sorted(referenced - set(call_ids)):
        fail("$.expect", f"unknown tool call ids: {unknown}")
    if len(data["expect"]["stops"]) != turns:
        fail("$.expect.stops", f"lists {len(data['expect']['stops'])} stops for {turns} turns")
    if "cancel_within_ms" in data["expect"] and not any("cancel_after_ms" in s for s in steps):
        fail("$.expect.cancel_within_ms", "no driver step has cancel_after_ms")


# --- answering ------------------------------------------------------------------------------


def rejected(status: int, message: str, param: str | None = None, code: str | None = None) -> Reply:
    """A request the fake server refuses itself (bad request, strict mode, failed expect)."""
    kind = "invalid_request_error" if status < 500 else "fake_provider_error"
    body = {"error": {"message": message, "type": kind, "param": param, "code": code}}
    # The OpenAI SDK honours this header, so a harness failure is not retried into the script.
    return Reply(status, body, headers={"x-should-retry": "false"}, error=message)


def reply(
    scenario: Scenario, index: int, raw: bytes, gap_ms: float | None = None, api: Api = "chat"
) -> Reply:
    """Answer request number `index` (0-based) of one (scenario, run, impl) cursor, which was
    sent to the endpoint of `api`.

    `gap_ms` is the time since the cursor's previous request (None for the first one)."""
    if api != scenario.api:
        return rejected(
            404,
            f"scenario {scenario.id} is scripted for POST <base_url>/{ENDPOINTS[scenario.api]},"
            f" not /{ENDPOINTS[api]}",
        )
    try:
        body = json.loads(raw)
    except ValueError as exc:
        return rejected(400, f"request body is not JSON: {exc}")
    if problem := (_chat_body_problem if api == "chat" else _responses_body_problem)(body):
        return rejected(400, problem)
    if api == "responses" and (malformed := _malformed_item(_input_items(body))):
        return rejected(400, *malformed)
    if body.get("stream") is not True:
        return rejected(400, "the fake provider only serves stream=true")
    if index >= len(scenario.exchanges):
        return rejected(500, "script exhausted")
    exchange = scenario.exchanges[index]
    strict = scenario.strict | exchange.get("strict", {})
    expect = exchange.get("expect", {})
    if api == "chat":
        if problem := _strict_violation(strict, body):
            return rejected(400, *problem)
        failures = _expect_failures(expect, body, gap_ms)
    else:
        if refused := _responses_strict(strict, body, scenario, index):
            return refused
        failures = _body_failures(expect, body, gap_ms)
        failures += _input_failures(expect, body, scenario, index)
    if failures:
        return rejected(500, f"expect failed (exchange {index + 1}): " + "; ".join(failures))

    style = exchange.get("style", scenario.style)
    respond = exchange["respond"]
    status = respond.get("status", 200)
    if status != 200:
        body = respond.get("body") or _error_body(status, style)
        return Reply(status, body, headers=respond.get("headers", {}))
    options = body.get("stream_options")
    options = options if isinstance(options, dict) else {}
    stream: _Stream | _ResponsesStream
    if api == "responses":
        stream = _ResponsesStream(
            response_id=f"resp_{scenario.id}_{index + 1:03d}",
            model=scenario.model["model"],
            effort=(scenario.model.get("reasoning") or {}).get("effort"),
            obfuscate=options.get("include_obfuscation") is not False,
            summarize=_asks_summary(body),
        )
    else:
        prefix = "gen" if style == "openrouter" else "chatcmpl"
        stream = _Stream(
            chunk_id=f"{prefix}-{scenario.id}-{index + 1:03d}",
            model=scenario.model["model"],
            style=style,
            include_usage=options.get("include_usage") is True,
        )
    return Reply(200, headers=respond.get("headers", {}), stream=stream.render(respond["stream"]))


def _chat_body_problem(body: object) -> str | None:
    messages = body.get("messages") if isinstance(body, dict) else None
    if not messages or not isinstance(messages, list):
        return "request body needs a non-empty `messages` list"
    if not all(isinstance(m, dict) for m in messages):
        return "every message must be an object"
    return None


def _strict_violation(strict: dict[str, Any], body: dict[str, Any]) -> tuple[str, str] | None:
    for key in strict.get("reject_params", []):
        if key in body:
            return f"Unrecognized request argument supplied: {key}", key
    if strict.get("reject_unsigned_reasoning"):
        for n, message in enumerate(body["messages"]):
            if message.get("role") == "assistant" and (
                unsigned := _unsigned_reasoning(message.get("reasoning_details"))
            ):
                return f"messages.{n}: reasoning.text without a signature ({unsigned})", "messages"
    return None


def _unsigned_reasoning(details: object) -> list[str]:
    """`reasoning.text` details without a signature. Fragments of one detail share an index."""
    signed: dict[str, bool] = {}
    for n, detail in enumerate(details if isinstance(details, list) else []):
        if isinstance(detail, dict) and detail.get("type") == "reasoning.text":
            index = detail.get("index")
            key = f"position {n}" if index is None else f"index {index}"
            signed[key] = signed.get(key, False) or bool(detail.get("signature"))
    return [key for key, ok in signed.items() if not ok]


_MISSING = object()


def _body_failures(
    expect: dict[str, Any], body: dict[str, Any], gap_ms: float | None = None
) -> list[str]:
    """The `expect` checks that read the body as it is (both APIs)."""
    failures = [f"body lacks {key!r}" for key in expect.get("body_has", []) if key not in body]
    failures += [f"body has {key!r}" for key in expect.get("body_lacks", []) if key in body]
    if "model" in expect and body.get("model") != expect["model"]:
        failures.append(f"model is {body.get('model')!r}, expected {expect['model']!r}")
    for path, want in expect.get("body_equals", {}).items():
        got = _at(body, path)
        if got is _MISSING:
            failures.append(f"body lacks {path!r}")
        elif got != want:
            failures.append(f"body {path} is {got!r}, expected {want!r}")
    for path, want in expect.get("body_contains", {}).items():
        got = _at(body, path)
        if not isinstance(got, list) or want not in got:
            shown = "nothing" if got is _MISSING else repr(got)
            failures.append(f"body {path} is {shown}, expected a list containing {want!r}")
    if "min_gap_ms" in expect and (gap_ms is None or gap_ms < expect["min_gap_ms"]):
        got = "no previous request" if gap_ms is None else f"{gap_ms:.0f} ms"
        failures.append(f"arrived after {got}, expected at least {expect['min_gap_ms']} ms")
    return failures


def _at(body: Any, path: str) -> Any:
    """The value at a dotted path; an integer segment indexes a list. _MISSING if absent."""
    got = body
    for key in path.split("."):
        if isinstance(got, dict):
            got = got.get(key, _MISSING)
        elif (
            isinstance(got, list) and key.lstrip("-").isdigit() and -len(got) <= int(key) < len(got)
        ):
            got = got[int(key)]
        else:
            return _MISSING
    return got


def _expect_failures(
    expect: dict[str, Any], body: dict[str, Any], gap_ms: float | None = None
) -> list[str]:
    messages: list[dict[str, Any]] = body["messages"]
    last = messages[-1]
    failures = _body_failures(expect, body, gap_ms)
    if "messages_len" in expect and len(messages) != expect["messages_len"]:
        failures.append(f"{len(messages)} messages, expected {expect['messages_len']}")
    if "last_role" in expect and last.get("role") != expect["last_role"]:
        failures.append(f"last role is {last.get('role')!r}, expected {expect['last_role']!r}")
    if "last_content_contains" in expect:
        needle, text = expect["last_content_contains"], _text(last.get("content"))
        if needle not in text:
            failures.append(f"last message lacks {needle!r}: {text[:200]!r}")
    for call_id, needle in expect.get("tool_result_contains", {}).items():
        results = [
            _text(m.get("content"))
            for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id") == call_id
        ]
        if not results:
            failures.append(f"no tool result for {call_id}")
        elif not any(needle in result for result in results):
            failures.append(f"tool result {call_id} lacks {needle!r}: {results[0][:200]!r}")
    for check in expect.get("messages_at", []):
        i = check["index"]
        if not -len(messages) <= i < len(messages):
            failures.append(f"no message at index {i}")
            continue
        message = messages[i]
        if "role" in check and message.get("role") != check["role"]:
            failures.append(
                f"message {i} role is {message.get('role')!r}, expected {check['role']!r}"
            )
        if "contains" in check and check["contains"] not in _text(message.get("content")):
            failures.append(f"message {i} lacks {check['contains']!r}")
    return failures


def _text(content: object) -> str:
    """Message content as plain text (a string, or a list of `{"type": "text"}` parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p["text"] for p in content if isinstance(p, dict) and "text" in p)
    return ""


def _error_body(status: int, style: str) -> dict[str, Any]:
    message = responses.get(status, "Error")
    if style == "openrouter":
        return {
            "error": {"code": status, "message": message, "metadata": {"provider_name": PROVIDER}}
        }
    return {"error": {"message": message, "type": "api_error", "param": None, "code": None}}


def _split(text: str, pieces: int) -> list[str]:
    """`text` in `pieces` near-equal parts (fewer when the text is shorter than that)."""
    bounds = [len(text) * i // pieces for i in range(pieces + 1)]
    return [text[a:b] for a, b in itertools.pairwise(bounds) if a < b]


class _Stream:
    """Renders one scripted response into SSE frames in the OpenRouter or OpenAI style."""

    def __init__(self, *, chunk_id: str, model: str, style: str, include_usage: bool) -> None:
        self.chunk_id = chunk_id
        self.model = model
        self.openrouter = style == "openrouter"
        self.include_usage = include_usage
        self.first = True
        self.finish: str | None = None  # repeated by the OpenRouter usage chunk
        self.tool_index = 0

    def render(self, ops: list[dict[str, Any]]) -> list[Op]:
        out: list[Op] = []
        for op in ops:
            delay = op.get("delay_ms", 0) / 1000
            for item in self._op(op):
                if delay:
                    out.append(Sleep(delay))
                out.append(item)
            if "sse_error" in op or "stall" in op:
                return out
        return [*out, b"data: [DONE]\n\n"]

    def _op(self, op: dict[str, Any]) -> list[Op]:
        if "text" in op:
            return [self._delta({"content": p}) for p in _split(op["text"], op.get("chunks", 1))]
        if "reasoning" in op:
            key = op.get("field", "reasoning")
            return [self._delta({key: p}) for p in _split(op["reasoning"], op.get("chunks", 1))]
        if "reasoning_details" in op:
            return [self._delta(self._detail(d)) for d in op["reasoning_details"]]
        if "tool_calls" in op:
            return [self._delta({"tool_calls": [d]}) for d in self._tool_deltas(op)]
        if "comment" in op:
            return [f": {op['comment']}\n\n".encode()]
        if "finish" in op:
            self.finish = op["finish"]
            return [self._delta({}, self.finish)]
        if "usage" in op:
            usage = self._usage(op["usage"])
            if self.openrouter:
                return [self._delta({}, self.finish, usage=usage)]
            return [self._frame([], usage=usage)] if self.include_usage else []
        if "sse_error" in op:
            # OpenRouter's documented mid-stream error chunk, in both styles (OpenAI has none).
            error = {"code": "server_error", **op["sse_error"]}
            choice = {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
            return [self._frame([choice], error=error)]
        return [Stall()]  # the only kind left

    def _detail(self, detail: dict[str, Any]) -> dict[str, Any]:
        # OpenRouter mirrors the text of a reasoning.text fragment in `delta.reasoning`.
        if self.openrouter and detail["type"] == "reasoning.text" and detail.get("text"):
            return {"reasoning": detail["text"], "reasoning_details": [detail]}
        return {"reasoning_details": [detail]}

    def _tool_deltas(self, op: dict[str, Any]) -> list[dict[str, Any]]:
        per_call = []
        for call in op["tool_calls"]:
            args = call["arguments"]
            args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            index, self.tool_index = self.tool_index, self.tool_index + 1
            function = {"name": call["name"], "arguments": ""}
            head = {"index": index, "id": call["id"], "type": "function", "function": function}
            pieces = _split(args, op.get("pieces", 1))
            per_call.append(
                [head, *({"index": index, "function": {"arguments": p}} for p in pieces)]
            )
        if op.get("interleave"):
            rounds = itertools.zip_longest(*per_call)
            return [delta for deltas in rounds for delta in deltas if delta is not None]
        return [delta for deltas in per_call for delta in deltas]

    def _usage(self, spec: dict[str, Any]) -> dict[str, Any]:
        prompt, completion = spec["prompt_tokens"], spec["completion_tokens"]
        usage: dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        if self.openrouter:
            usage |= {"cost": spec.get("cost", 0), "is_byok": spec.get("is_byok", False)}
        return usage | {
            "prompt_tokens_details": {"cached_tokens": spec.get("cached_tokens", 0)},
            "completion_tokens_details": {"reasoning_tokens": spec.get("reasoning_tokens", 0)},
        }

    def _delta(self, delta: dict[str, Any], finish: str | None = None, **extra: Any) -> bytes:
        if self.openrouter:
            delta = {"role": "assistant", "content": "", **delta}
            choice = {"index": 0, "delta": delta, "finish_reason": finish}
            choice["native_finish_reason"] = finish
        else:
            if self.first:
                delta = {"role": "assistant", **delta}
            choice = {"index": 0, "delta": delta, "finish_reason": finish}
        self.first = False
        return self._frame([choice], **extra)

    def _frame(self, choices: list[dict[str, Any]], **extra: Any) -> bytes:
        chunk: dict[str, Any] = {
            "id": self.chunk_id,
            "object": "chat.completion.chunk",
            "created": CREATED,
            "model": self.model,
        }
        if self.openrouter:
            chunk["provider"] = PROVIDER
        chunk = {**chunk, "choices": choices, **extra}
        data = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        return b"data: " + data.encode() + b"\n\n"


# --- Responses API ----------------------------------------------------------------------------
#
# Scenarios with model kind "openai_responses" answer on `POST <base_url>/responses`, as OpenAI's
# Responses API does (API reference, "Streaming events"): named SSE events (`event: <type>` and
# `data: {"type": <type>, "sequence_number": n, ...}`), no `data: [DONE]`. The response object in
# `response.created` / `.completed` / `.incomplete` / `.failed` does not echo the request, so
# every loop gets the same bytes. Item ids are fixed: reasoning ids come from the script,
# function calls are `fc_<call id without "call_">`, messages `msg_<scenario>_<NNN>_<output index>`.

_OBFUSCATION = string.ascii_letters + string.digits
_SYSTEM_ROLES = ("system", "developer")
_STORE_FALSE = (
    "Items are not persisted when `store` is set to false. Try again with `store` set to true,"
    " or remove this item from your input."
)


def _responses_body_problem(body: object) -> str | None:
    items = body.get("input") if isinstance(body, dict) else None
    if isinstance(items, str) and items:
        return None
    if not items or not isinstance(items, list):
        return "request body needs `input`: a non-empty string or list of items"
    if not all(isinstance(item, dict) for item in items):
        return "every input item must be an object"
    return None


type _Problem = tuple[str, str, str]  # (message, param, code) of the API's 400

# The input item types this fake serves, with their required fields and those fields' JSON types.
# The API checks these before anything else, so a malformed item is 400 whatever the script says.
_ITEM_FIELDS: dict[str, dict[str, tuple[type, ...]]] = {
    "message": {"role": (str,), "content": (str, list)},
    "function_call": {"call_id": (str,), "name": (str,), "arguments": (str,)},
    "function_call_output": {"call_id": (str,), "output": (str, list)},
    "reasoning": {"id": (str,), "summary": (list,)},
}
_ROLES = ("user", "assistant", "system", "developer")
_INPUT_PARTS = ("input_text", "input_image", "input_file")
_OUTPUT_PARTS = ("output_text", "refusal")  # an assistant message's, replayed
_JSON_TYPES: dict[type, str] = {
    dict: "an object",
    list: "an array",
    str: "a string",
    bool: "a boolean",
    int: "an integer",
    float: "a number",
    type(None): "null",
}


def _malformed_item(items: list[dict[str, Any]]) -> _Problem | None:
    """The API's 400 for the first input item whose shape it refuses, naming the field as it
    does (`input[3].output`). Item types other than the four this fake serves are refused too."""
    for n, item in enumerate(items):
        at, kind = f"input[{n}]", _item_type(item)
        if kind is None:
            return _missing(f"{at}.type")
        if not isinstance(kind, str) or kind not in _ITEM_FIELDS:
            return _invalid_value(kind, tuple(_ITEM_FIELDS), f"{at}.type")
        for key, types in _ITEM_FIELDS[kind].items():
            if problem := _field_problem(item, key, types, at):
                return problem
        problem = None
        match kind:
            case "message" if item["role"] not in _ROLES:
                problem = _invalid_value(item["role"], _ROLES, f"{at}.role")
            case "message":
                # Replayed assistant text must be output_text: input_text there is refused.
                parts = _OUTPUT_PARTS if item["role"] == "assistant" else _INPUT_PARTS
                problem = _malformed_parts(item["content"], parts, f"{at}.content")
            case "function_call_output":
                problem = _malformed_parts(item["output"], _INPUT_PARTS, f"{at}.output")
            case "reasoning":
                encrypted = item.get("encrypted_content")
                if encrypted is not None and not isinstance(encrypted, str):
                    problem = _invalid_type(f"{at}.encrypted_content", (str,), encrypted)
                else:
                    problem = _malformed_parts(item["summary"], ("summary_text",), f"{at}.summary")
        if problem:
            return problem
    return None


def _malformed_parts(parts: str | list[Any], types: tuple[str, ...], at: str) -> _Problem | None:
    """The first content part the API refuses (a string is plain text, never refused)."""
    for i, part in enumerate(parts if isinstance(parts, list) else []):
        where = f"{at}[{i}]"
        if not isinstance(part, dict):
            return _invalid_type(where, (dict,), part)
        if problem := _field_problem(part, "type", (str,), where):
            return problem
        if part["type"] not in types:
            return _invalid_value(part["type"], types, where)
        if part["type"].endswith("_text") and (
            problem := _field_problem(part, "text", (str,), where)
        ):
            return problem
    return None


def _field_problem(
    obj: dict[str, Any], key: str, types: tuple[type, ...], at: str
) -> _Problem | None:
    """The API's 400 if field `key` of the object at `at` is missing or of the wrong JSON type."""
    if key not in obj:
        return _missing(f"{at}.{key}")
    if not isinstance(obj[key], types):
        return _invalid_type(f"{at}.{key}", types, obj[key])
    return None


def _missing(param: str) -> _Problem:
    return f"Missing required parameter: '{param}'.", param, "missing_required_parameter"


def _invalid_type(param: str, types: tuple[type, ...], got: object) -> _Problem:
    names = [_JSON_TYPES[t] for t in types]
    want = names[0] if len(names) == 1 else "one of " + " or ".join(names)
    message = (
        f"Invalid type for '{param}': expected {want}, but got {_JSON_TYPES[type(got)]} instead."
    )
    return message, param, "invalid_type"


def _invalid_value(got: object, supported: tuple[str, ...], param: str) -> _Problem:
    *rest, last = [f"'{value}'" for value in supported]
    listed = f"{', '.join(rest)}{',' if len(rest) > 1 else ''} and {last}" if rest else last
    return f"Invalid value: '{got}'. Supported values are: {listed}.", param, "invalid_value"


def _input_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The request's `input` as items (a string is one user message, as the API reads it)."""
    items = body["input"]
    return [{"role": "user", "content": items}] if isinstance(items, str) else items


def _item_type(item: dict[str, Any]) -> str | None:
    """An input item's type; a message may leave it out (`{"role": ..., "content": ...}`)."""
    return item.get("type") or ("message" if "role" in item else None)


def _item_text(item: dict[str, Any]) -> str:
    """The text an input item carries, for `contains` checks."""
    match _item_type(item):
        case "message":
            return _text(item.get("content"))
        case "function_call_output":
            return _text(item.get("output"))
        case "function_call":
            return str(item.get("arguments", ""))
        case "reasoning":
            parts = item.get("summary")
            return _text(parts if isinstance(parts, list) else None)
    return ""


def _split_system(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """(the system prompt, the other input items). The system prompt is `instructions` plus any
    system/developer messages that `input` starts with: loops may put it in either place."""
    items = _input_items(body)
    n = next(
        (i for i, item in enumerate(items) if item.get("role") not in _SYSTEM_ROLES), len(items)
    )
    system = [_text(body.get("instructions"))] + [_text(i.get("content")) for i in items[:n]]
    return "\n".join(t for t in system if t), items[n:]


def _asks_summary(body: dict[str, Any]) -> bool:
    """Whether a request asks for reasoning summaries (`reasoning.summary`): the API streams
    none, and its reasoning items have an empty `summary`, unless it does."""
    reasoning = body.get("reasoning")
    return isinstance(reasoning, dict) and bool(reasoning.get("summary"))


def _done_reasoning(spec: dict[str, Any], summarized: bool) -> dict[str, Any]:
    """The done reasoning item of a scripted `reasoning_item`: what `response.output_item.done`
    sends (with its summary if the request asked for one), and exactly what a request must
    replay (`reasoning_replayed`)."""
    texts = spec.get("summary", []) if summarized else []
    summary = [{"type": "summary_text", "text": text} for text in texts]
    content = [{"type": "reasoning_text", "text": text} for text in spec.get("text", [])]
    return {
        "id": spec["id"],
        "type": "reasoning",
        "summary": summary,
        **({"content": content} if content else {}),
        "encrypted_content": spec["encrypted_content"],
    }


def _scripted_reasoning(scenario: Scenario, before: int) -> dict[str, dict[str, Any]]:
    """Reasoning id -> the op of every reasoning item (done or cut short) scripted for the
    requests before request `before` (0-based) of a cursor: the only ones it can have got."""
    return {
        op["reasoning_item"]["id"]: op
        for exchange in scenario.exchanges[:before]
        for op in exchange["respond"].get("stream", [])
        if "reasoning_item" in op
    }


def _responses_strict(
    strict: dict[str, Any], body: dict[str, Any], scenario: Scenario, index: int
) -> Reply | None:
    for key in strict.get("reject_params", []):
        if key in body:
            return rejected(400, f"Unsupported parameter: '{key}'.", key, "unsupported_parameter")
    if not strict.get("reject_unencrypted_reasoning"):
        return None
    # As the API does with `store: false`: a reasoning item is known only by its encrypted
    # content, which must be the complete one a done event sent for that id. A later
    # exchange's item was never sent, so its scripted content does not verify either.
    scripted = _scripted_reasoning(scenario, index)
    for item in _input_items(body):
        if _item_type(item) != "reasoning":
            continue
        item_id, encrypted = item.get("id"), item.get("encrypted_content")
        if not encrypted:
            return rejected(404, f"Item with id '{item_id}' not found. {_STORE_FALSE}", "input")
        op = scripted.get(str(item_id))
        if (
            op is None
            or op.get("done") is False
            or op["reasoning_item"]["encrypted_content"] != encrypted
        ):
            message = f"The encrypted content for item {item_id} could not be verified."
            return rejected(400, message, None, "invalid_encrypted_content")
    return None


def _input_failures(
    expect: dict[str, Any], body: dict[str, Any], scenario: Scenario, index: int
) -> list[str]:
    """The Responses `expect` checks on `input` (after the system prompt) and the system prompt
    of request `index`."""
    system, items = _split_system(body)
    failures = []
    if "system_contains" in expect and expect["system_contains"] not in system:
        failures.append(f"system prompt lacks {expect['system_contains']!r}: {system[:200]!r}")
    if "input_len" in expect and len(items) != expect["input_len"]:
        failures.append(f"{len(items)} input items, expected {expect['input_len']}")
    last = items[-1] if items else {}
    if "last_type" in expect and _item_type(last) != expect["last_type"]:
        failures.append(f"last item is {_item_type(last)!r}, expected {expect['last_type']!r}")
    if "last_role" in expect and last.get("role") != expect["last_role"]:
        failures.append(f"last role is {last.get('role')!r}, expected {expect['last_role']!r}")
    if "last_content_contains" in expect:
        needle, text = expect["last_content_contains"], _item_text(last)
        if needle not in text:
            failures.append(f"last item lacks {needle!r}: {text[:200]!r}")
    for call_id, needle in expect.get("tool_result_contains", {}).items():
        outputs = [
            _item_text(item)
            for item in items
            if _item_type(item) == "function_call_output" and item.get("call_id") == call_id
        ]
        if not outputs:
            failures.append(f"no function_call_output for {call_id}")
        elif not any(needle in output for output in outputs):
            failures.append(
                f"function_call_output {call_id} lacks {needle!r}: {outputs[0][:200]!r}"
            )
    for check in expect.get("input_at", []):
        i = check["index"]
        if not -len(items) <= i < len(items):
            failures.append(f"no input item at index {i}")
            continue
        item = items[i]
        got = {"type": _item_type(item), "role": item.get("role"), "phase": item.get("phase")}
        for key in ("type", "role", "phase"):
            if key in check and got[key] != check[key]:
                failures.append(f"input item {i} {key} is {got[key]!r}, expected {check[key]!r}")
        if "contains" in check and check["contains"] not in _item_text(item):
            failures.append(f"input item {i} lacks {check['contains']!r}")
    scripted = _scripted_reasoning(scenario, index)  # the loader checks each id is in it
    for rid in expect.get("reasoning_replayed", []):
        # A thread's model config is fixed, so this request asks for summaries if the one that
        # got the item did.
        sent = _done_reasoning(scripted[rid]["reasoning_item"], _asks_summary(body))
        found = [
            item for item in items if _item_type(item) == "reasoning" and item.get("id") == rid
        ]
        if len(found) != 1:
            failures.append(f"reasoning {rid} is replayed {len(found)} times, expected once")
        elif found[0] != sent:
            failures.append(f"reasoning {rid} is not replayed as sent ({_diff(sent, found[0])})")
    return failures


def _diff(want: dict[str, Any], got: dict[str, Any]) -> str:
    """Which keys of a replayed item differ from what was sent."""
    parts = [f"lacks {k!r}" for k in want if k not in got]
    parts += [f"adds {k!r}" for k in got if k not in want]
    parts += [f"changes {k!r}" for k in want if k in got and got[k] != want[k]]
    return ", ".join(parts)


def _responses_usage(spec: dict[str, Any]) -> dict[str, Any]:
    tokens_in, tokens_out = spec["input_tokens"], spec["output_tokens"]
    return {
        "input_tokens": tokens_in,
        "input_tokens_details": {
            "cached_tokens": spec.get("cached_tokens", 0),
            "cache_write_tokens": spec.get("cache_write_tokens", 0),
        },
        "output_tokens": tokens_out,
        "output_tokens_details": {"reasoning_tokens": spec.get("reasoning_tokens", 0)},
        "total_tokens": tokens_in + tokens_out,
    }


class _ResponsesStream:
    """Renders one scripted response as the Responses API's named SSE events."""

    def __init__(
        self, *, response_id: str, model: str, effort: str | None, obfuscate: bool, summarize: bool
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.effort = effort
        self.summarize = summarize  # stream reasoning summaries (the request asked for them)
        # Delta events carry an `obfuscation` pad by default (`stream_options.include_obfuscation`).
        self.obfuscate = obfuscate
        self.seq = 0  # sequence_number of the next event
        self.started = 0  # output items started (the next output_index)
        self.output: list[dict[str, Any]] = []  # done items, for the response that ends it

    def render(self, ops: list[dict[str, Any]]) -> list[Op]:
        created = self._response("in_progress")
        out: list[Op] = [
            self._event("response.created", response=created),
            self._event("response.in_progress", response=created),
        ]
        for op in ops:
            delay = op.get("delay_ms", 0) / 1000
            for frame in self._op(op):
                if delay:
                    out.append(Sleep(delay))
                out.append(frame)
        return out  # the stream ends after its last event: no [DONE] in this API

    def _op(self, op: dict[str, Any]) -> list[Op]:
        if "text" in op:
            return self._message(op)
        if "reasoning_item" in op:
            return self._reasoning(op)
        if "tool_calls" in op:
            calls, pieces = op["tool_calls"], op.get("pieces", 1)
            cut = op.get("status") == "incomplete"  # the last call only
            return [
                frame
                for n, call in enumerate(calls)
                for frame in self._call(call, pieces, cut=cut and n == len(calls) - 1)
            ]
        if "completed" in op:
            usage = _responses_usage(op["completed"])
            return [self._event("response.completed", response=self._response("completed", usage))]
        if "incomplete" in op:
            spec = op["incomplete"]
            response = self._response("incomplete", _responses_usage(spec))
            response["incomplete_details"] = {"reason": spec.get("reason", "max_output_tokens")}
            return [self._event("response.incomplete", response=response)]
        if "error" in op:
            error = {"code": "server_error", **op["error"]}
            return [self._event("error", code=error["code"], message=error["message"], param=None)]
        if "failed" in op:
            error = {"code": "server_error", **op["failed"]}
            failed = self._response("failed", error=error)
            return [self._event("response.failed", response=failed)]
        return [Stall()]  # the only kind left

    def _message(self, op: dict[str, Any]) -> list[Op]:
        index = self._start()
        item_id = f"msg_{self.response_id.removeprefix('resp_')}_{index}"
        item: dict[str, Any] = {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "content": [],
            "role": "assistant",
        }
        if "phase" in op:
            item["phase"] = op["phase"]
        at = {"item_id": item_id, "output_index": index, "content_index": 0}
        empty = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
        frames: list[Op] = [
            self._event("response.output_item.added", output_index=index, item=item),
            self._event("response.content_part.added", **at, part=empty),
        ]
        for piece in _split(op["text"], op.get("chunks", 1)):
            frames.append(
                self._event("response.output_text.delta", **at, delta=piece, logprobs=[], pad=True)
            )
        if op.get("done") is False:
            return frames
        part = {**empty, "text": op["text"]}
        done = {**item, "status": "completed", "content": [part]}
        self.output.append(done)
        return [
            *frames,
            self._event("response.output_text.done", **at, text=op["text"], logprobs=[]),
            self._event("response.content_part.done", **at, part=part),
            self._event("response.output_item.done", output_index=index, item=done),
        ]

    def _reasoning(self, op: dict[str, Any]) -> list[Op]:
        spec = op["reasoning_item"]
        index = self._start()
        at = {"item_id": spec["id"], "output_index": index}
        encrypted = spec["encrypted_content"]
        # The API documents that the added item's encrypted_content may be incomplete: only the
        # done item may be replayed. Half of it makes a replay of the added item fail to verify.
        added = {
            "id": spec["id"],
            "type": "reasoning",
            "summary": [],
            "encrypted_content": encrypted[: len(encrypted) // 2],
        }
        frames: list[Op] = [
            self._event("response.output_item.added", output_index=index, item=added)
        ]
        # Raw reasoning text streams whether or not a summary is asked for: it is no summary.
        empty = {"type": "reasoning_text", "text": ""}
        for n, text in enumerate(spec.get("text", [])):
            part_at = {**at, "content_index": n}
            frames.append(self._event("response.content_part.added", **part_at, part=empty))
            for piece in _split(text, op.get("chunks", 1)):
                frames.append(
                    self._event("response.reasoning_text.delta", **part_at, delta=piece, pad=True)
                )
            frames += [
                self._event("response.reasoning_text.done", **part_at, text=text),
                self._event("response.content_part.done", **part_at, part={**empty, "text": text}),
            ]
        summary = spec.get("summary", []) if self.summarize else []
        for n, text in enumerate(summary):
            part_at = {**at, "summary_index": n}
            frames.append(
                self._event(
                    "response.reasoning_summary_part.added",
                    **part_at,
                    part={"type": "summary_text", "text": ""},
                )
            )
            for piece in _split(text, op.get("chunks", 1)):
                frames.append(
                    self._event(
                        "response.reasoning_summary_text.delta", **part_at, delta=piece, pad=True
                    )
                )
            if op.get("done") is False and n == len(summary) - 1:
                return frames  # cut inside the last summary part
            frames += [
                self._event("response.reasoning_summary_text.done", **part_at, text=text),
                self._event(
                    "response.reasoning_summary_part.done",
                    **part_at,
                    part={"type": "summary_text", "text": text},
                ),
            ]
        if op.get("done") is False:
            return frames
        done = _done_reasoning(spec, self.summarize)
        self.output.append(done)
        return [*frames, self._event("response.output_item.done", output_index=index, item=done)]

    def _call(self, call: dict[str, Any], pieces: int, *, cut: bool = False) -> list[Op]:
        index = self._start()
        args = call["arguments"]
        args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        item_id = "fc_" + call["id"].removeprefix("call_")
        item = {
            "id": item_id,
            "type": "function_call",
            "status": "in_progress",
            "arguments": "",
            "call_id": call["id"],
            "name": call["name"],
        }
        at = {"item_id": item_id, "output_index": index}
        frames: list[Op] = [
            self._event("response.output_item.added", output_index=index, item=item)
        ]
        for piece in _split(args, pieces):
            frames.append(
                self._event("response.function_call_arguments.delta", **at, delta=piece, pad=True)
            )
        done = {**item, "status": "incomplete" if cut else "completed", "arguments": args}
        self.output.append(done)
        if not cut:  # the arguments of a call cut short never finish
            frames.append(
                self._event(
                    "response.function_call_arguments.done", **at, name=call["name"], arguments=args
                )
            )
        return [*frames, self._event("response.output_item.done", output_index=index, item=done)]

    def _start(self) -> int:
        index, self.started = self.started, self.started + 1
        return index

    def _response(
        self,
        status: str,
        usage: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": CREATED,
            "status": status,
            "completed_at": CREATED + 1 if status == "completed" else None,
            "error": error,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": self.model,
            "output": list(self.output),
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": self.effort, "summary": None},
            "store": False,
            "temperature": 1,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
            "truncation": "disabled",
            "usage": usage,
            "user": None,
            "metadata": {},
        }

    def _event(self, kind: str, *, pad: bool = False, **fields: Any) -> bytes:
        event = {"type": kind, "sequence_number": self.seq, **fields}
        if pad and self.obfuscate:
            # Random-looking and deterministic: the same for every run and impl.
            digest = hashlib.sha256(f"{self.response_id}:{self.seq}".encode()).digest()
            event["obfuscation"] = "".join(
                _OBFUSCATION[b % len(_OBFUSCATION)] for b in digest[: 4 + digest[-1] % 12]
            )
        self.seq += 1
        data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        return f"event: {kind}\ndata: {data}\n\n".encode()

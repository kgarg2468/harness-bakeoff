"""Scenario scripts for the fake provider.

`load_scenario()` reads and validates a scenario file. `reply()` answers one request against
a scenario's exchange: it enforces the strict modes and the exchange's `expect`, then turns
the scripted `respond` into the SSE frames and control ops that the server plays back.
Everything here is deterministic: the same script and request always give the same bytes.
The file format is documented in `fakeprov/README.md`.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from http.client import responses
from pathlib import Path
from typing import Any, NoReturn, get_args

from jsonschema import Draft202012Validator, ValidationError
from jsonschema.exceptions import best_match

from bakeoff.shared.contract import LOOP_EVENTS, SHARED_EVENTS, StopReason

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
CREATED = 1758758400  # the fixed `created` of every chunk
PROVIDER = "FakeProvider"


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
    style: str  # wire style of every exchange that does not set its own
    strict: dict[str, Any]


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


_OPS = {
    "text": _op("text", _STR, chunks=_INT1),
    "reasoning": _op(
        "reasoning", _STR, chunks=_INT1, field={"enum": ["reasoning", "reasoning_content"]}
    ),
    "reasoning_details": _op(
        "reasoning_details", {"type": "array", "items": {"type": "object", "required": ["type"]}}
    ),
    "tool_calls": _op(
        "tool_calls",
        {
            "type": "array",
            "minItems": 1,
            "items": _obj(
                {"id": _STR, "name": _STR, "arguments": {"type": ["string", "object"]}},
                "id",
                "name",
                "arguments",
            ),
        },
        pieces=_INT1,
        interleave=_BOOL,
    ),
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
_STYLE = {"enum": ["openrouter", "openai"]}
_STRICT = _obj({"reject_params": _STRS, "reject_unsigned_reasoning": _BOOL})
_EXCHANGE_EXPECT = _obj(
    {
        "body_has": _STRS,
        "body_lacks": _STRS,
        "model": _STR,
        "last_role": {"enum": ["system", "developer", "user", "assistant", "tool"]},
        "last_content_contains": _STR,
        "messages_len": _INT1,
        "tool_result_contains": {"type": "object", "additionalProperties": _STR},
    }
)
_RESPOND = _obj(
    {
        "status": {"type": "integer", "minimum": 200, "maximum": 599},
        "headers": {"type": "object", "additionalProperties": _STR},
        "body": {"type": "object"},
        "stream": {"type": "array", "items": _kinds(_OPS)},
    }
)
_FINAL_EXPECT = _obj(
    {
        "stops": {"type": "array", "items": {"enum": list(get_args(StopReason))}},
        "files": {"type": "object", "additionalProperties": {"type": ["string", "null"]}},
        "tool_runs": {"type": "object", "additionalProperties": _INT0},
        "commits": _INT0,
        "requests": _INT0,
        "text_contains": _STR,
        "cost_usd": {"type": "number", "minimum": 0},
    },
    "stops",
)
_SCHEMA = _obj(
    {
        "id": _STR,
        "title": _STR,
        "system": _STR,
        "model": _obj(
            {
                "kind": {"enum": ["openrouter", "openai_compat"]},
                "model": _STR,
                "reasoning": {"type": ["object", "null"]},
                "compat": {"type": "object"},
            },
            "kind",
            "model",
        ),
        "rules": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [_DECISION, {"type": "object", "additionalProperties": _DECISION}]
            },
        },
        "limits": _obj({"max_steps": _INT1}, "max_steps"),
        "engine": _obj({"delay_ms": _INT0}, "delay_ms"),
        "style": _STYLE,
        "strict": _STRICT,
        "driver": {"type": "array", "minItems": 1, "items": _kinds(_STEPS)},
        "exchanges": {
            "type": "array",
            "minItems": 1,
            "items": _obj(
                {
                    "note": _STR,
                    "style": _STYLE,
                    "strict": _STRICT,
                    "expect": _EXCHANGE_EXPECT,
                    "respond": _RESPOND,
                },
                "respond",
            ),
        },
        "expect": _FINAL_EXPECT,
    },
    *_REQUIRED,
)
_VALIDATOR = Draft202012Validator(_SCHEMA)


# --- loading --------------------------------------------------------------------------------


def load_scenario(path: Path) -> Scenario:
    """Read and validate one scenario file. Raises `ScenarioError` with a precise location."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScenarioError(f"{path.name}: {exc}") from exc
    if (error := best_match(_VALIDATOR.iter_errors(data))) is not None:
        raise ScenarioError(f"{path.name}: {_describe(error)}")
    if data["id"] != path.stem:
        raise ScenarioError(f"{path.name}: $.id: {data['id']!r} must match the file name")
    _check_semantics(path.name, data)
    byok = data["model"]["kind"] == "openai_compat"
    return Scenario(
        **{key: data[key] for key in _REQUIRED},
        style=data.get("style", "openai" if byok else "openrouter"),
        strict=data.get("strict", {}),
    )


def _describe(error: ValidationError) -> str:
    # A step or op with no known key fails the `anyOf` of `_kinds()`; name the choices.
    kinds = error.parent if error.parent is not None else error
    if kinds.validator == "anyOf" and all("required" in s for s in kinds.validator_value):
        names = ", ".join(s["required"][0] for s in kinds.validator_value)
        return f"{kinds.json_path}: expected exactly one of: {names}"
    return f"{error.json_path}: {error.message}"


def _check_semantics(name: str, data: dict[str, Any]) -> None:
    """Rules the JSON schema cannot express."""

    def fail(where: str, message: str) -> NoReturn:
        raise ScenarioError(f"{name}: {where}: {message}")

    call_ids: list[str] = []
    for n, exchange in enumerate(data["exchanges"]):
        respond = exchange["respond"]
        if (respond.get("status", 200) == 200) != ("stream" in respond):
            fail(f"$.exchanges[{n}].respond", "status 200 needs a stream; other statuses take none")
        ops = respond.get("stream", [])
        for i, op in enumerate(ops[:-1]):
            if "sse_error" in op or "stall" in op:
                fail(f"$.exchanges[{n}].respond.stream[{i}]", "sse_error and stall must be last")
        call_ids += [call["id"] for op in ops for call in op.get("tool_calls", [])]
    if duplicates := sorted({c for c in call_ids if call_ids.count(c) > 1}):
        fail("$.exchanges", f"tool call ids must be unique: {duplicates}")

    referenced = set(data["expect"].get("tool_runs", {}))
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


# --- answering ------------------------------------------------------------------------------


def rejected(status: int, message: str, param: str | None = None) -> Reply:
    """A request the fake server refuses itself (bad request, strict mode, failed expect)."""
    kind = "invalid_request_error" if status < 500 else "fake_provider_error"
    body = {"error": {"message": message, "type": kind, "param": param, "code": None}}
    # The OpenAI SDK honours this header, so a harness failure is not retried into the script.
    return Reply(status, body, headers={"x-should-retry": "false"}, error=message)


def reply(scenario: Scenario, index: int, raw: bytes) -> Reply:
    """Answer request number `index` (0-based) of one (scenario, run, impl) cursor."""
    try:
        body = json.loads(raw)
    except ValueError as exc:
        return rejected(400, f"request body is not JSON: {exc}")
    messages = body.get("messages") if isinstance(body, dict) else None
    if not messages or not isinstance(messages, list):
        return rejected(400, "request body needs a non-empty `messages` list")
    if not all(isinstance(m, dict) for m in messages):
        return rejected(400, "every message must be an object")
    if body.get("stream") is not True:
        return rejected(400, "the fake provider only serves stream=true")
    if index >= len(scenario.exchanges):
        return rejected(500, "script exhausted")
    exchange = scenario.exchanges[index]
    strict = scenario.strict | exchange.get("strict", {})
    if problem := _strict_violation(strict, body):
        return rejected(400, *problem)
    if failures := _expect_failures(exchange.get("expect", {}), body):
        return rejected(500, f"expect failed (exchange {index + 1}): " + "; ".join(failures))

    style = exchange.get("style", scenario.style)
    respond = exchange["respond"]
    status = respond.get("status", 200)
    if status != 200:
        body = respond.get("body") or _error_body(status, style)
        return Reply(status, body, headers=respond.get("headers", {}))
    prefix = "gen" if style == "openrouter" else "chatcmpl"
    options = body.get("stream_options")
    stream = _Stream(
        chunk_id=f"{prefix}-{scenario.id}-{index + 1:03d}",
        model=scenario.model["model"],
        style=style,
        include_usage=isinstance(options, dict) and options.get("include_usage") is True,
    )
    return Reply(200, headers=respond.get("headers", {}), stream=stream.render(respond["stream"]))


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


def _expect_failures(expect: dict[str, Any], body: dict[str, Any]) -> list[str]:
    messages: list[dict[str, Any]] = body["messages"]
    last = messages[-1]
    failures = [f"body lacks {key!r}" for key in expect.get("body_has", []) if key not in body]
    failures += [f"body has {key!r}" for key in expect.get("body_lacks", []) if key in body]
    if "model" in expect and body.get("model") != expect["model"]:
        failures.append(f"model is {body.get('model')!r}, expected {expect['model']!r}")
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
            error = {"code": 502, **op["sse_error"]}
            return [self._delta({}, "error", error=error)]
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

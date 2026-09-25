"""Loop A: the harness loop on pydantic-ai, used the way its docs recommend (A_CHECKLIST.md).

A turn is one `Agent.iter()` run over the native history rebuilt from the items. The run
executes in its own task and hands events to `run_turn` through a queue: pydantic-ai's
`CancellationToken` cancels the task that drives the run, and that must never be the
consumer's task.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Iterable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from typing import Any

import httpx
import openai
import pydantic_ai
from pydantic import ValidationError
from pydantic_ai import (
    Agent,
    AgentRun,
    CancellationToken,
    CostNotFoundWarning,
    DeferredToolRequests,
    DeferredToolResults,
    ModelAPIError,
    ModelHTTPError,
    ModelMessage,
    ModelRequestContext,
    ModelResponse,
    ModelResponseStreamEvent,
    ModelRetry,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunCancelled,
    RunContext,
    SkipToolExecution,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    Tool,
    ToolCallPart,
    ToolDefinition,
    ToolDenied,
    ToolFailed,
    UsageLimitExceeded,
    UsageLimits,
)
from pydantic_ai.capabilities import Hooks, ProcessHistory
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from bakeoff.pydantic_version import mapping
from bakeoff.pydantic_version.model import build_model, run_settings
from bakeoff.shared.contract import (
    Event,
    Item,
    ModelConfig,
    Resume,
    ToolCall,
    ToolHost,
    ToolSpec,
    TurnInput,
)

try:  # the HTTP library of the OpenAI SDK in use: httpx (openai 2.x) or httpx2 (3.x)
    import httpx2

    _DROPPED: tuple[type[Exception], ...] = (httpx.TransportError, httpx2.TransportError)
except ImportError:
    _DROPPED = (httpx.TransportError,)

# The OpenAI SDK retries inside one call and numbers the attempts only in this request header.
_RETRY_HEADER = "x-stainless-retry-count"
# Backoff before retrying a failed stream: the OpenAI SDK's own schedule (0.5 s doubling, max 8 s).
_STREAM_RETRY_BASE_S = 0.5
_STEP_CAP_RESULT = "Not run: the turn reached its step limit."


class _StreamRetry(Exception):
    """A stream failed midway and may be retried; carries the failed run's usage."""

    def __init__(self, usage: RunUsage) -> None:
        self.usage = usage


@dataclass
class _Turn:
    """Per-turn state: the run's deps (tools and hooks reach it as `ctx.deps`) and the HTTP
    hooks' target."""

    turn_id: str
    tools: ToolHost
    max_steps: int
    out: asyncio.Queue[Event | None] = field(default_factory=asyncio.Queue)
    steps: int = 0
    attempt: int = 0  # of the current step
    # (status, monotonic time, reason) of the last failed attempt, until the next one starts.
    failure: tuple[int | None, float, str] | None = None
    emitted: dict[int, ModelMessage] = field(default_factory=dict)
    items: list[Item] = field(default_factory=list)  # emitted this turn
    responses: list[tuple[int, ModelResponse]] = field(default_factory=list)  # usage not sent yet

    def emit(self, type_: str, data: dict[str, Any]) -> None:
        self.out.put_nowait(Event(type_, data))

    def request_started(self, sdk_attempt: int) -> None:
        """Each HTTP attempt: a new step, or a retry (the SDK's, or ours after a failed stream)."""
        if sdk_attempt == 1 and self.failure is None:
            self.steps += 1
            self.attempt = 1
        else:
            self.attempt += 1
            status, failed_at, reason = self.failure or (None, time.monotonic(), "connection error")
            self.emit(
                "retry",
                {
                    "attempt": self.attempt,
                    "status": status,
                    "wait_ms": round((time.monotonic() - failed_at) * 1000),
                    "reason": reason,
                },
            )
        self.failure = None
        self.emit("request.start", {"step": self.steps, "attempt": self.attempt})

    def stream_event(self, event: ModelResponseStreamEvent) -> None:
        match event:
            case (
                PartStartEvent(part=TextPart(content=text))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=text))
            ) if text:
                self.emit("text.delta", {"text": text})
            case (
                PartStartEvent(part=ThinkingPart(content=text))
                | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=text))
            ) if text:
                self.emit("reasoning.delta", {"text": text})
            case PartEndEvent(part=ToolCallPart() as call):
                self.emit("tool_call.ready", _call_data(call))

    def flush(self, messages: Iterable[ModelMessage]) -> None:
        """Emit an item for each wire message not emitted yet, then the queued usage events."""
        for message in messages:
            if id(message) in self.emitted:
                continue
            self.emitted[id(message)] = message  # keeps the object alive, so ids stay unique
            interrupted = isinstance(message, ModelResponse) and message.state == "interrupted"
            for piece in mapping.split(message):
                for openai_message in mapping.to_openai(piece):
                    item = Item(
                        id=uuid.uuid4().hex,
                        turn_id=self.turn_id,
                        message=openai_message,
                        status="incomplete" if interrupted else "complete",
                        native=mapping.dump(piece),
                    )
                    self.items.append(item)
                    self.emit("item", {"item": item})
        for step, response in self.responses:
            self.emit("usage", _usage(response, step))
        self.responses.clear()


_TURN: ContextVar[_Turn] = ContextVar("pydantic_version_turn")


class PydanticLoop:
    """`contract.Loop` on pydantic-ai. The Agent is built once per tool set and the model (with
    its HTTP pool) once per endpoint; every thread shares them."""

    name = "pydantic"

    def __init__(self) -> None:
        # Nothing may reach stdout/stderr (rule 1): newer releases print a first-run banner, the
        # library warns each time it drops `temperature` for a reasoning model (it drops it
        # correctly either way), and it warns at the end of a run with a cost limit that no
        # price is known for (the usage events already say cost_source="none").
        pydantic_ai.BANNER_ENABLED = False
        warnings.filterwarnings("ignore", "Sampling parameters", UserWarning, "pydantic_ai")
        warnings.filterwarnings("ignore", category=CostNotFoundWarning)
        self._agents: dict[str, Agent[_Turn, str | DeferredToolRequests]] = {}
        self._models: dict[str, OpenAIChatModel] = {}

    async def run_turn(
        self, turn: TurnInput, tools: ToolHost, cancel: asyncio.Event
    ) -> AsyncIterator[Event]:
        state = _Turn(turn.turn_id, tools, turn.limits.max_steps)
        task = asyncio.create_task(self._drive(turn, state, cancel))
        try:
            while (event := await state.out.get()) is not None:
                yield event
                state.out.task_done()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def aclose(self) -> None:
        for model in self._models.values():
            await model.client.close()
        self._agents.clear()
        self._models.clear()

    async def _drive(self, turn: TurnInput, state: _Turn, cancel: asyncio.Event) -> None:
        """Run the turn in this task and queue its events, ending with turn.end and None."""
        _TURN.set(state)
        token = CancellationToken()
        watcher = asyncio.create_task(_cancel_when_set(cancel, token))
        history, usage = turn.history, None
        try:
            while True:
                try:
                    end = await self._run(turn, history, usage, state, token)
                    break
                except _StreamRetry as retry:
                    # pydantic-ai does not retry a stream that fails midway (A_CHECKLIST), so run
                    # the step again from what is saved. The failed attempt is not a new step.
                    cause = retry.__cause__
                    status = cause.status_code if isinstance(cause, ModelHTTPError) else None
                    state.failure = (status, time.monotonic(), _retry_reason(cause) or "")
                    history = [*turn.history, *state.items]
                    usage = replace(retry.usage, requests=retry.usage.requests - 1)
                    wait = min(8.0, _STREAM_RETRY_BASE_S * 2 ** (state.attempt - 1))
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(cancel.wait(), wait * random.uniform(0.75, 1.0))
                    if cancel.is_set():
                        end = {"stop": "cancelled"}
                        break
        except RunCancelled:
            end = {"stop": "cancelled"}
        except Exception as exc:  # every turn ends with turn.end (rule 7)
            retryable = _retry_reason(exc) is not None
            kind, message = type(exc).__name__, f"{type(exc).__name__}: {exc}"
            state.emit("error", {"kind": kind, "message": message, "retryable": retryable})
            end = {"stop": "error", "error": message}
        finally:
            watcher.cancel()
        state.emit("turn.end", {**end, "steps": state.steps})
        state.out.put_nowait(None)

    async def _run(
        self,
        turn: TurnInput,
        items: list[Item],
        usage: RunUsage | None,
        state: _Turn,
        token: CancellationToken,
    ) -> dict[str, Any]:
        """One agent run over the history rebuilt from `items`. Returns the turn.end data."""
        history = mapping.to_history(items)
        deferred = None
        if turn.resume is None:
            # A new message after a pause nobody answered: its calls get the result the library
            # would synthesize, saved as items (rule 4).
            closing = mapping.close_pending(history)
            state.flush(closing)
            history += closing
        elif pending := mapping.pending_calls(history):
            deferred, asks = _answers(turn.resume, pending, state.tools)
            if asks:
                return _pause(state, asks)
        limits = turn.limits
        cost_limit = None if limits.max_cost_usd is None else Decimal(str(limits.max_cost_usd))
        model = self._model(turn.model)
        run: AgentRun[_Turn, str | DeferredToolRequests] | None = None
        try:
            async with self._agent(state.tools.specs()).iter(
                message_history=history,  # ends with the user's request: the library resumes it
                deferred_tool_results=deferred,
                model=model,
                model_settings=run_settings(model, turn.model),
                instructions=turn.system,
                deps=state,
                usage_limits=UsageLimits(request_limit=limits.max_steps, cost_limit=cost_limit),
                usage=usage,
                retries=limits.max_steps,  # a tool's retry budget never outlasts the step cap
                cancellation_token=token,
            ) as run:
                try:
                    # Persist at every node boundary, so a crash loses at most the current step.
                    async for node in run:
                        if not Agent.is_model_request_node(node):
                            state.flush(run.new_messages())
                            continue
                        # Tool results ride on the request node until it is sent: persist them now,
                        # and send it only once the runner has handled (saved) them.
                        unsent = [] if node.is_resuming_without_prompt else [node.request]
                        state.flush([*run.new_messages(), *unsent])
                        await state.out.join()
                        async with node.stream(run.ctx) as stream:
                            try:
                                async for event in stream:
                                    state.stream_event(event)
                            except Exception as exc:
                                # No tool runs before a response is complete, so a retry is safe.
                                if _retry_reason(exc) and state.attempt <= turn.model.max_retries:
                                    raise _StreamRetry(run.usage) from exc
                                raise
                except UsageLimitExceeded:
                    state.flush(run.new_messages())
                    over_budget = cost_limit is not None and (run.usage.cost or 0) > cost_limit
                    return {"stop": "budget" if over_budget else "max_steps"}
                result = run.result
        except _StreamRetry:
            raise  # the partial response is dropped, not saved
        except Exception as exc:
            # Keep what completed and close the calls left open (rule 4). After a cancel the
            # history is complete only once the run has unwound, in the RunCancelled snapshot.
            ended = exc if isinstance(exc, RunCancelled) else run
            if ended is not None:
                state.flush([*ended.new_messages(), *mapping.close_pending(ended.all_messages())])
            raise
        if result is not None and isinstance(result.output, DeferredToolRequests):
            return _pause(state, result.output.approvals)
        return {"stop": "end_turn"}

    def _agent(self, specs: list[ToolSpec]) -> Agent[_Turn, str | DeferredToolRequests]:
        key = json.dumps([asdict(spec) for spec in specs], sort_keys=True)
        if (agent := self._agents.get(key)) is None:
            toolset = ApprovalRequiredToolset(
                FunctionToolset([_tool(spec) for spec in specs]),
                approval_required_func=_needs_approval,
            )
            agent = self._agents[key] = Agent(
                deps_type=_Turn,
                output_type=[str, DeferredToolRequests],
                toolsets=[toolset],
                capabilities=[
                    Hooks(after_model_request=_on_model_response, before_tool_execute=_step_cap),
                    ProcessHistory(mapping.replayable),
                ],
                name=self.name,
            )
        return agent

    def _model(self, cfg: ModelConfig) -> OpenAIChatModel:
        key = json.dumps(asdict(replace(cfg, session_id=None)), sort_keys=True)
        if (model := self._models.get(key)) is None:
            hooks = {"request": [_on_request], "response": [_on_response]}
            model = self._models[key] = build_model(cfg, hooks)
        return model


def _tool(spec: ToolSpec) -> Tool[_Turn]:
    async def call(ctx: RunContext[_Turn], **_: Any) -> str | ToolDenied:
        # The ToolHost sends tool.start straight to the runner; wait until the consumer has
        # handled everything queued before it (this call's tool_call.ready).
        await ctx.deps.out.join()
        result = await ctx.deps.tools.run(_running_call(ctx))
        # The library's three outcomes for a call that did not succeed (A_CHECKLIST).
        if result.ok:
            return result.content
        if result.error == "invalid_args":
            raise ModelRetry(result.content)
        if result.error == "denied":
            return ToolDenied(result.content)
        raise ToolFailed(result.content)

    tool = Tool.from_schema(
        call,
        name=spec.name,
        description=spec.description,
        json_schema=spec.parameters,
        takes_ctx=True,
    )
    # The default (strict=None) lets the OpenAI schema transformer add
    # `additionalProperties: false` to every object, which turns free-form objects such as
    # `validate_pipeline.pipeline` into "must be empty". Our schemas are not strict schemas.
    tool.strict = False
    return tool


def _needs_approval(ctx: RunContext[_Turn], tool_def: ToolDefinition, args: dict[str, Any]) -> bool:
    return ctx.deps.tools.check(_running_call(ctx)) == "ask"


def _step_cap(
    ctx: RunContext[_Turn], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: Any
) -> Any:
    """The step cap is checked before the next request, so the calls of the last allowed response
    would run although their results can never be sent. Skip them; each still gets a result."""
    if ctx.usage.requests >= ctx.deps.max_steps:
        raise SkipToolExecution(_STEP_CAP_RESULT)
    return args


def _running_call(ctx: RunContext[_Turn]) -> ToolCall:
    """The call `ctx` is executing, with the argument text exactly as the model streamed it."""
    response = next(m for m in reversed(ctx.messages) if isinstance(m, ModelResponse))
    return _to_call(next(c for c in response.tool_calls if c.tool_call_id == ctx.tool_call_id))


def _to_call(part: ToolCallPart) -> ToolCall:
    # args_as_json_str() returns the streamed text verbatim when it is a JSON object.
    return ToolCall(id=part.tool_call_id, name=part.tool_name, arguments=part.args_as_json_str())


def _call_data(part: ToolCallPart) -> dict[str, Any]:
    return {
        "call_id": part.tool_call_id,
        "name": part.tool_name,
        "arguments": part.args_as_json_str(),
    }


def _answers(
    resume: Resume, pending: list[ToolCallPart], tools: ToolHost
) -> tuple[DeferredToolResults, list[ToolCallPart]]:
    """Approvals for the open calls: the user's decision, else `check()` again (a crash resume).
    Calls that come back "ask" are returned to be asked again."""
    approvals: dict[str, bool | ToolDenied] = {}
    asks: list[ToolCallPart] = []
    for call in pending:
        decision = resume.decisions.get(call.tool_call_id)
        if decision == "deny":
            approvals[call.tool_call_id] = ToolDenied(
                f"Denied by user: {resume.reason or 'no reason given'}"
            )
        elif decision == "allow" or tools.check(_to_call(call)) != "ask":
            approvals[call.tool_call_id] = True  # run() still enforces "deny" rules
        else:
            asks.append(call)
    return DeferredToolResults(approvals=approvals), asks


def _pause(state: _Turn, calls: list[ToolCallPart]) -> dict[str, Any]:
    for call in calls:
        state.emit("permission.asked", _call_data(call))
    return {"stop": "paused", "pending": [call.tool_call_id for call in calls]}


def _retry_reason(exc: BaseException | None) -> str | None:
    """Why a provider error may pass on a new attempt, or None: a 429 or 5xx, a dropped
    connection, or an error event mid-stream. pydantic-ai wraps only some of these, so the
    others arrive as they were raised (library behaviors, A_CHECKLIST)."""
    if isinstance(exc, ModelHTTPError):
        status = exc.status_code
        return f"HTTP {status}" if status == 429 or status >= 500 else None
    if isinstance(exc, ModelAPIError):
        return exc.message
    if isinstance(exc, openai.APIError):  # an OpenAI-style error event mid-stream (BYOK)
        return f"stream error: {exc.message}"
    if isinstance(exc, _DROPPED):  # the connection dropped mid-stream (openai 2.x)
        return f"connection error: {type(exc).__name__}"
    if isinstance(exc, ValidationError) and exc.title == "_OpenRouterError":
        # OpenRouter's error chunk, whose string `code` the library's model rejects, or no error
        # body at all: a stream that dropped (openai 3.x).
        error = exc.errors()[0]
        return f"OpenRouter error chunk: {error['input']}" if error["loc"] else "connection error"
    return None


def _usage(response: ModelResponse, step: int) -> dict[str, Any]:
    usage = response.usage
    billed = (response.provider_details or {}).get("cost")
    # Without usage from the provider there is nothing to price: no estimate.
    estimate = float(usage.cost) if usage.cost is not None and usage.has_values() else None
    source = "provider" if billed is not None else "estimate" if estimate is not None else "none"
    return {
        "step": step,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_tokens": usage.cache_read_tokens,
        "reasoning_tokens": usage.details.get("reasoning_tokens", 0),
        "cost_usd": billed if billed is not None else estimate,
        "cost_source": source,
    }


def _on_model_response(
    ctx: RunContext[_Turn], /, *, request_context: ModelRequestContext, response: ModelResponse
) -> ModelResponse:
    """Every response the model returns, before the library adds it to history and checks the
    limits. OpenRouter's billed cost becomes the response cost, so `RunUsage` and `cost_limit`
    count what was charged, and the usage event is queued (the response that crosses the cost
    limit was billed too, although the library drops it)."""
    if (cost := (response.provider_details or {}).get("cost")) is not None:
        response.usage.cost = Decimal(str(cost))
    ctx.deps.responses.append((ctx.deps.steps, response))
    return response


async def _cancel_when_set(cancel: asyncio.Event, token: CancellationToken) -> None:
    await cancel.wait()
    token.cancel()


async def _on_request(request: Any) -> None:
    if (turn := _TURN.get(None)) is not None:
        turn.request_started(int(request.headers.get(_RETRY_HEADER, "0")) + 1)


async def _on_response(response: Any) -> None:
    if (turn := _TURN.get(None)) is not None and response.status_code >= 400:
        status = response.status_code
        turn.failure = (status, time.monotonic(), f"HTTP {status}")

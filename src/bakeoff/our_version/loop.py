"""The lean loop: one streamed request per step, eager and parallel tools, cheap cancel."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

import httpx

from bakeoff.shared.contract import Event, Item, ToolCall, ToolHost, ToolResult, ToolSpec, TurnInput

from .compat import REASONING_FIELDS, merge_detail, static_body, usage_fields
from .provider import (
    Stream,
    StreamedCall,
    Wire,
    as_provider_error,
    http_error,
    replayed,
    stream_error,
)
from .retry import backoff

CANCELLED = "Cancelled by user"
_MAX_THREADS = 64  # request caches kept for this many recently used threads
_DRAIN_S = 0.05  # once the answer is complete, wait this long for the body's end (keep-alive)
_EMPTY: dict[str, Any] = {}
_NO_CHOICE = (_EMPTY,)
_UNPRICED = "max_cost_usd is set but the endpoint reports no cost"


def _role(item: Item) -> Any:
    return item.message.get("role")


class _Cancelled(Exception):
    """The request was stopped by the turn's cancel event."""


class OurLoop:
    """Agent loop on raw httpx against OpenAI-compatible streaming chat completions.

    `http_client` replaces the pooled per-base-URL clients (tests pass a MockTransport client);
    `sleep` is awaited for retry backoff.
    """

    name = "our"

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http_client
        self._sleep = sleep
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._wires: dict[str, Wire] = {}

    def run_turn(
        self, turn: TurnInput, tools: ToolHost, cancel: asyncio.Event
    ) -> AsyncIterator[Event]:
        return _Turn(self, turn, tools, cancel).run()

    async def aclose(self) -> None:
        clients, self._clients = self._clients, {}
        for client in clients.values():
            await client.aclose()

    def _client(self, base_url: str) -> httpx.AsyncClient:
        if self._http is not None:
            return self._http
        client = self._clients.get(base_url)
        if client is None:
            limits = httpx.Limits(
                max_connections=32, max_keepalive_connections=8, keepalive_expiry=120
            )
            client = self._clients[base_url] = httpx.AsyncClient(trust_env=False, limits=limits)
        return client

    def _wire(self, turn: TurnInput, tools: list[ToolSpec]) -> Wire:
        key = (turn.model, turn.system, tools)
        wire = self._wires.pop(turn.thread_id, None)
        if wire is None or wire.key != key:
            session = turn.model.session_id or turn.thread_id
            wire = Wire(key, static_body(turn.model, turn.system, tools, session))
        self._wires[turn.thread_id] = wire  # re-inserted last: most recently used
        if len(self._wires) > _MAX_THREADS:
            del self._wires[next(iter(self._wires))]
        return wire


class _Turn:
    """State of one run_turn call. Rebuilt from history every turn (contract rule 2)."""

    def __init__(
        self, loop: OurLoop, turn: TurnInput, tools: ToolHost, cancel: asyncio.Event
    ) -> None:
        self.loop, self.turn, self.tools, self.cancel = loop, turn, tools, cancel
        self.model = turn.model
        specs = tools.specs()
        self.read_only = {s.name for s in specs if s.read_only}
        self.wire = loop._wire(turn, specs)
        self.url = self.model.base_url.rstrip("/") + "/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.model.api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = httpx.Timeout(self.model.timeout_s, connect=10.0)
        self.items = replayed(turn.history)
        resume = turn.resume
        self.user = resume.decisions if resume else {}
        self.reason = (resume.reason if resume else None) or "no reason given"
        # A resume continues the logical turn (all after the last user item), so the limits count
        # the steps and cost it already spent.
        history = turn.history
        start = next(
            (i + 1 for i in reversed(range(len(history))) if _role(history[i]) == "user"), 0
        )
        spent = [it.usage or {} for it in history[start:] if _role(it) == "assistant"]
        self.steps = len(spent)
        self.cost = sum(u.get("cost_usd") or 0.0 for u in spent)
        self.unpriced = any(u.get("cost_source") == "none" for u in spent)
        self.ended = False
        self.jobs: dict[str, asyncio.Task[ToolResult]] = {}  # tool runs by call id
        self.tasks: set[asyncio.Task[Any]] = set()  # everything the cancel watcher must stop
        self.response: httpx.Response | None = None

    async def run(self) -> AsyncIterator[Event]:
        """The whole turn. The SSE reader is inlined here on purpose: every async-generator level
        between it and the runner would cost each streamed chunk another suspend/resume."""
        watcher = asyncio.ensure_future(self._watch())
        history, limits = self.turn.history, self.turn.limits
        # Incomplete items (cancelled or truncated output) are never replayed: they are no answer.
        last = next((_role(it) for it in reversed(history) if it.status == "complete"), None)
        try:
            if pending := self._pending():  # approval or crash resume
                async for event in self._tools(pending):
                    yield event
            elif last == "assistant":
                yield self._end("end_turn")  # resumed after the final answer: nothing left to do
            while not self.ended:  # one step: a model request, then its tool calls
                if self.cancel.is_set():
                    yield self._end("cancelled")
                elif self.steps >= limits.max_steps:
                    yield self._end("max_steps")
                elif limits.max_cost_usd is not None and self.cost >= limits.max_cost_usd:
                    yield self._end("budget")
                elif limits.max_cost_usd is not None and self.unpriced:
                    # Budget policy: a response without a cost is never free; stop, don't guess.
                    for event in self._abort("budget_unenforceable", _UNPRICED, stop="budget"):
                        yield event
                if self.ended:
                    break
                self.steps += 1
                body = self.wire.body(self.items)
                for attempt in itertools.count(1):
                    yield Event("request.start", {"step": self.steps, "attempt": attempt})
                    stream, failure = Stream(), None
                    try:
                        lines = (await self._open(body)).aiter_lines()
                        async for line in lines:
                            if not line.startswith("data:"):
                                continue  # blank separators, ": keep-alive" comments, other fields
                            data = line[5:]
                            if data[-6:] == "[DONE]":
                                stream.done = True
                                break
                            chunk = json.loads(data)
                            if (error := chunk.get("error")) is not None:
                                raise stream_error(error)
                            choice = (chunk.get("choices") or _NO_CHOICE)[0]
                            if finish := choice.get("finish_reason"):
                                stream.finish = finish
                            delta = choice.get("delta") or _EMPTY
                            if text := delta.get("content"):
                                stream.text.append(text)
                                yield Event("text.delta", {"text": text})
                            for field in REASONING_FIELDS:
                                if thought := delta.get(field):
                                    yield Event("reasoning.delta", {"text": thought})
                                    break
                            for fragment in delta.get("reasoning_details") or ():
                                merge_detail(stream.details, fragment)
                            for tool_delta in delta.get("tool_calls") or ():
                                if done := stream.tool_delta(tool_delta, self.read_only):
                                    yield self._ready(done, stream)
                            if usage := chunk.get("usage"):
                                stream.usage = usage  # replaced, never added: counted once
                                if stream.finish:  # finish_reason and usage: nothing else is due
                                    stream.done = True
                                    break
                        if stream.done:  # read the body's end only to keep the connection
                            with contextlib.suppress(Exception):
                                async with asyncio.timeout(_DRAIN_S):
                                    async for _ in lines:
                                        pass
                        stream.check_end()
                    except Exception as exc:
                        failure = exc
                    finally:
                        if (resp := self.response) is not None:
                            self.response = None
                            await resp.aclose()
                    if failure is None:
                        break
                    events, wait = await self._failed(failure, stream, attempt)
                    for event in events:
                        yield event
                    if self.ended:
                        return
                    await self._settle(self.loop._sleep(wait))
                    if self.cancel.is_set():
                        yield self._end("cancelled")
                        return
                usage = {"step": self.steps, **usage_fields(stream.usage)}
                self.cost += usage["cost_usd"]
                self.unpriced |= usage["cost_source"] == "none"
                if stream.finish == "length":  # cut off at max_tokens: keep the text, run no call
                    await self._stop_jobs()
                    cut = stream.partial() or {"role": "assistant", "content": None}
                    yield self._item(cut, status="incomplete", usage=usage)
                    yield Event("usage", usage)
                    cut_at = f"Output truncated at max_tokens={self.model.max_tokens}"
                    for event in self._abort("output_truncated", cut_at):
                        yield event
                    return
                for call in stream.calls.values():
                    if not call.ready:
                        yield self._ready(call)
                yield self._item(stream.message(), usage=usage)
                yield Event("usage", usage)
                if calls := stream.tool_calls():
                    async for event in self._tools(calls):
                        yield event
                else:
                    yield self._end("end_turn")
        except Exception as exc:  # a bug must still end the turn (contract rule 7)
            await self._stop_jobs()
            for event in self._abort("internal", repr(exc)):
                yield event
        finally:
            watcher.cancel()
            for task in list(self.tasks):
                task.cancel()

    async def _watch(self) -> None:
        await self.cancel.wait()
        for task in list(self.tasks):
            task.cancel()
        if self.response is not None:
            with contextlib.suppress(Exception):
                await self.response.aclose()  # wakes the stream reader; no per-chunk polling

    def _pending(self) -> list[ToolCall]:
        """Calls of the last assistant message that have no result item yet."""
        for i in range(len(self.items) - 1, -1, -1):
            msg = self.items[i].message
            if msg.get("role") == "assistant":
                answered = {it.message.get("tool_call_id") for it in self.items[i + 1 :]}
                return [
                    ToolCall(tc["id"], tc["function"]["name"], tc["function"]["arguments"])
                    for tc in msg.get("tool_calls") or ()
                    if tc["id"] not in answered
                ]
        return []

    async def _open(self, body: bytes) -> httpx.Response:
        """POST the request as a task the cancel watcher can stop; return the 200 response."""
        client = self.loop._client(self.model.base_url)
        request = client.build_request(
            "POST", self.url, content=body, headers=self.headers, timeout=self.timeout
        )
        send = await self._settle(client.send(request, stream=True))
        if send.cancelled():
            raise _Cancelled
        resp = self.response = send.result()  # from here on, cancel closes it (no polling)
        if self.cancel.is_set():
            raise _Cancelled
        if resp.status_code != 200:
            raise await http_error(resp)
        return resp

    async def _failed(
        self, exc: Exception, stream: Stream, attempt: int
    ) -> tuple[list[Event], float]:
        """The events for a failed attempt (ending the turn, or a `retry`) and the retry wait."""
        if self.cancel.is_set():
            await self._stop_jobs()
            partial = stream.partial()
            cut = [] if partial is None else [self._item(partial, status="incomplete")]
            return [*cut, self._end("cancelled")], 0.0
        err = as_provider_error(exc)  # re-raises a bug
        # Once a tool has started, a retry could replay its call id: fail instead.
        if self.jobs or not err.retryable or attempt > self.model.max_retries:
            await self._stop_jobs()
            return self._abort(err.kind, err.message, retryable=err.retryable), 0.0
        wait = backoff(attempt - 1) if err.wait_s is None else err.wait_s
        retry = {
            "attempt": attempt,
            "status": err.status,
            "wait_ms": round(wait * 1000),
            "reason": err.message,
        }
        return [Event("retry", retry)], wait

    def _ready(self, streamed: StreamedCall, stream: Stream | None = None) -> Event:
        """A call's arguments are complete. Mid-stream (`stream` given), start what may start early."""
        call = streamed.finish()
        if stream is not None:
            self._start_early(stream)
        return Event(
            "tool_call.ready", {"call_id": call.id, "name": call.name, "arguments": call.arguments}
        )

    def _start_early(self, stream: Stream) -> None:
        """Start complete, allowed read-only calls while the model is still streaming, in call
        order: a call starts early only if every call before it did, so none can run ahead of
        an earlier write."""
        for streamed in stream.calls.values():
            if streamed.id in self.jobs:
                continue
            if not streamed.ready or streamed.name not in self.read_only:
                return
            call = streamed.call()
            if self.tools.check(call) != "allow":
                return
            self._start(call)

    async def _tools(self, calls: list[ToolCall]) -> AsyncIterator[Event]:
        """Run the calls and append one result per call, in call order.

        Consecutive read-only calls run concurrently. Any other call runs alone, after everything
        before it, and everything after waits for it, so no call sees older state than the calls
        before it left. "ask" calls pause the turn; the calls after the first of them run on resume.
        """
        asked = [
            c
            for c in calls
            if c.id not in self.jobs and c.id not in self.user and self.tools.check(c) == "ask"
        ]
        for call in asked:
            yield Event(
                "permission.asked",
                {"call_id": call.id, "name": call.name, "arguments": call.arguments},
            )
        now = calls[: calls.index(asked[0])] if asked else calls
        done = 0  # calls of `now` whose result is appended
        for i, call in enumerate(now):
            if call.id in self.jobs or self.user.get(call.id) == "deny":
                continue  # started early, or denied by the user
            if call.name in self.read_only:  # allowed, or denied by a rule (run() enforces that)
                self._start(call)
                continue
            for c in now[done:i]:  # a write waits for everything before it ...
                yield self._result(await self._outcome(c))
            self._start(call)
            yield self._result(await self._outcome(call))  # ... and everything after waits for it
            done = i + 1
        for call in now[done:]:
            yield self._result(await self._outcome(call))
        if self.cancel.is_set():  # no orphans: calls that did not run get a result too
            for call in calls[len(now) :]:
                yield self._result(ToolResult(call.id, False, CANCELLED))
            yield self._end("cancelled")
        elif asked:
            yield self._end("paused", pending=[call.id for call in asked])

    def _start(self, call: ToolCall) -> None:
        self.jobs[call.id] = self._spawn(self.tools.run(call))

    async def _outcome(self, call: ToolCall) -> ToolResult:
        job = self.jobs.pop(call.id, None)
        if job is None:
            return ToolResult(call.id, False, f"Denied by user: {self.reason}")
        if not job.done():
            await asyncio.wait([job])
        return ToolResult(call.id, False, CANCELLED) if job.cancelled() else job.result()

    async def _stop_jobs(self) -> None:
        """Cancel tool runs whose results will not be used; wait so their tool.end precedes ours."""
        for job in self.jobs.values():
            job.cancel()
        if self.jobs:
            await asyncio.wait(self.jobs.values())

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        if self.cancel.is_set():
            task.cancel()
        return task

    async def _settle(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        """Await `coro` as a task the cancel watcher can stop; return the finished task."""
        task = self._spawn(coro)
        await asyncio.wait([task])
        return task

    def _item(
        self,
        message: dict[str, Any],
        *,
        status: Literal["complete", "incomplete"] = "complete",
        usage: dict[str, Any] | None = None,
    ) -> Event:
        item = Item(uuid.uuid4().hex, self.turn.turn_id, message, status, usage=usage)
        if status == "complete":
            self.items.append(item)
        return Event("item", {"item": item})

    def _result(self, result: ToolResult) -> Event:
        return self._item(
            {"role": "tool", "tool_call_id": result.call_id, "content": result.content}
        )

    def _abort(
        self, kind: str, message: str, *, stop: str = "error", retryable: bool = False
    ) -> list[Event]:
        """An `error` event, then the turn's end."""
        error = {"kind": kind, "message": message, "retryable": retryable}
        end = self._end(stop, error=message) if stop == "error" else self._end(stop)
        return [Event("error", error), end]

    def _end(self, stop: str, **extra: Any) -> Event:
        self.ended = True
        return Event("turn.end", {"stop": stop, "steps": self.steps, **extra})

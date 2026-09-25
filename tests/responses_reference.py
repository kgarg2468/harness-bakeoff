"""A minimal Responses API client that follows the contract, for the tests only.

It proves that R01-R05 can pass: the scripts, fakeprov's checks and the invariants agree with
a correct client. It is not a loop under comparison: no eager tools, deltas or retry backoff,
it runs calls one at a time, and it reads each response whole (racing it against `cancel`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from bakeoff.shared.contract import Event, Item, ToolCall, ToolHost, ToolResult, TurnInput


class RetryableError(Exception):
    def __init__(self, reason: str, wait_s: float = 0.0) -> None:
        super().__init__(reason)
        self.wait_s = wait_s


class ReferenceLoop:
    # The thread's impl: a child process loads it by this name (see reference_worker.py).
    name = "reference"

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=30)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def run_turn(
        self, turn: TurnInput, tools: ToolHost, cancel: asyncio.Event
    ) -> AsyncIterator[Event]:
        history = list(turn.history)
        # Calls of the last response without a result: an approval (or crash) resume.
        answered = {i.message.get("tool_call_id") for i in history}
        last = next((i for i in reversed(history) if i.message.get("role") == "assistant"), None)
        open_calls = [c for c in _calls(last) if c.id not in answered] if last else []
        step = 0
        while True:
            if open_calls:
                paused = []
                for call in open_calls:
                    decision = (turn.resume.decisions if turn.resume else {}).get(call.id)
                    decision = decision or tools.check(call)
                    if decision == "ask":
                        paused.append(call)
                        continue
                    if decision == "deny" and turn.resume and call.id in turn.resume.decisions:
                        result = ToolResult(call.id, False, f"Denied by user: {turn.resume.reason}")
                    else:
                        result = await tools.run(call)
                    message = {"role": "tool", "tool_call_id": call.id, "content": result.content}
                    item = Item(f"{turn.turn_id}:{call.id}", turn.turn_id, message)
                    history.append(item)
                    yield Event("item", {"item": item})
                if paused:
                    for call in paused:
                        data = {"call_id": call.id, "name": call.name, "arguments": call.arguments}
                        yield Event("permission.asked", data)
                    pending = [c.id for c in paused]
                    yield Event("turn.end", {"stop": "paused", "steps": step, "pending": pending})
                    return
            if step == turn.limits.max_steps:
                yield Event("turn.end", {"stop": "max_steps", "steps": step})
                return
            step += 1
            yield Event("request.start", {"step": step, "attempt": 1})
            events: list[dict[str, Any]] | None = None
            for attempt in range(1, turn.model.max_retries + 2):
                reader = asyncio.create_task(self._request(turn, tools, history))
                waiter = asyncio.create_task(cancel.wait())
                await asyncio.wait({reader, waiter}, return_when=asyncio.FIRST_COMPLETED)
                waiter.cancel()
                if cancel.is_set():
                    reader.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await reader
                    yield Event("turn.end", {"stop": "cancelled", "steps": step})
                    return
                try:
                    events = reader.result()
                    break
                except RetryableError as exc:
                    wait_ms = round(exc.wait_s * 1000)
                    yield Event(
                        "retry", {"attempt": attempt, "status": None, "wait_ms": wait_ms,
                                  "reason": str(exc)}
                    )  # fmt: skip
                    await asyncio.sleep(exc.wait_s)
            if events is None:
                yield Event("turn.end", {"stop": "error", "steps": step, "error": "retries"})
                return
            output = [e["item"] for e in events if e["type"] == "response.output_item.done"]
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(
                    part["text"]
                    for item in output
                    if item["type"] == "message"
                    for part in item["content"]
                )
                or None,
            }
            calls = [item for item in output if item["type"] == "function_call"]
            if calls:
                message["tool_calls"] = [
                    {"id": c["call_id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ]  # fmt: skip
            item = Item(f"{turn.turn_id}:{step}", turn.turn_id, message, native=output)
            history.append(item)
            yield Event("item", {"item": item})
            usage = next(e for e in events if e["type"] == "response.completed")["response"][
                "usage"
            ]
            yield Event("usage", {
                "step": step,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cached_tokens": usage["input_tokens_details"]["cached_tokens"],
                "reasoning_tokens": usage["output_tokens_details"]["reasoning_tokens"],
                "cost_usd": 0.0,
                "cost_source": "none",
            })  # fmt: skip
            open_calls = _calls(item)
            if not open_calls:
                yield Event("turn.end", {"stop": "end_turn", "steps": step})
                return

    async def _request(
        self, turn: TurnInput, tools: ToolHost, history: list[Item]
    ) -> list[dict[str, Any]]:
        """One request: the whole stream's events. Raises RetryableError on 429/5xx or a
        stream that fails midway."""
        model = turn.model
        body: dict[str, Any] = {
            "model": model.model,
            "stream": True,
            "store": False,  # we keep the history, so reasoning goes back encrypted
            "include": ["reasoning.encrypted_content"],
            "instructions": turn.system,
            "input": [entry for item in history for entry in _input(item)],
            "tools": [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": False,
                }
                for t in tools.specs()
            ],
            "max_output_tokens": model.max_tokens,
        }
        if model.reasoning:
            body["reasoning"] = model.reasoning
        if model.temperature is not None:
            body["temperature"] = model.temperature
        async with self.client.stream(
            "POST", f"{model.base_url}/responses", json=body,
            headers={"Authorization": f"Bearer {model.api_key}"},
        ) as response:  # fmt: skip
            if response.status_code == 429 or response.status_code >= 500:
                await response.aread()
                wait = float(response.headers.get("retry-after", "0"))
                raise RetryableError(f"HTTP {response.status_code}", wait)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {(await response.aread())!r}")
            events = [
                json.loads(line.removeprefix("data: "))
                async for line in response.aiter_lines()
                if line.startswith("data: ")
            ]
        if events[-1]["type"] in ("error", "response.failed"):
            raise RetryableError(json.dumps(events[-1])[:200], 0.05)
        return events


def _input(item: Item) -> list[dict[str, Any]]:
    """The Responses input items of one history item: a response's own output items,
    verbatim (reasoning with its encrypted_content), or the runner's chat-shaped messages."""
    if item.native is not None:
        return list(item.native)
    message = item.message
    if message.get("role") == "tool":
        output = message["content"]
        return [
            {"type": "function_call_output", "call_id": message["tool_call_id"], "output": output}
        ]
    return [{"role": message["role"], "content": message["content"]}]


def _calls(item: Item | None) -> list[ToolCall]:
    calls = (item.message.get("tool_calls") or []) if item else []
    return [ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls]

"""The seam between the shared harness and a loop implementation.

Everything in `bakeoff.shared` is used by every loop. A loop implementation
(`our_version`, `pydantic_version`, ...) only has to implement `Loop` below, so
the line-count comparison covers exactly the code the decision is about.

Rules every Loop must follow (tests enforce them, see DESIGN.md):

1. A loop only *yields events*. It never touches the session log, git, the UI
   or stdout/stderr. The shared runner persists and publishes what it yields.
2. A loop keeps no state between turns (connection pools and caches are fine).
   It rebuilds everything from `TurnInput.history`, so "resume after approval"
   and "resume after a crash" are the same code path.
3. Every durable change is an `item` event. The next model request must be the
   previous request's messages plus new items only: history is append-only.
4. Every tool call gets exactly one tool-result item, including on deny and
   cancel. A tool never runs twice for the same call id.
5. Tools are called through `ToolHost`. Call `check()` first: "ask" means pause
   (emit `permission.asked` per call, then `turn.end` with stop="paused" and the
   pending ids); "deny" and "allow" both go to `run()`, which enforces deny.
6. When `cancel` is set, stop within 200 ms, leave no orphan tool calls, and
   end with `turn.end` stop="cancelled".
7. The last event of every turn is `turn.end`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Decision = Literal["allow", "ask", "deny"]
StopReason = Literal["end_turn", "paused", "max_steps", "budget", "cancelled", "error"]

# Event types. Loops emit the first group; the runner and ToolHost emit the rest.
LOOP_EVENTS = (
    "request.start",  # {step, attempt}
    "reasoning.delta",  # {text}
    "text.delta",  # {text}
    "tool_call.ready",  # {call_id, name, arguments}: a call's arguments are complete
    "item",  # {item: Item}: the only event that changes durable history
    "usage",  # {step, input_tokens, output_tokens, cached_tokens, reasoning_tokens, cost_usd, cost_source}
    "retry",  # {attempt, status, wait_ms, reason}
    "permission.asked",  # {call_id, name, arguments}
    "error",  # {kind, message, retryable}
    "turn.end",  # {stop: StopReason, steps, pending?: [call_id], error?}
)
SHARED_EVENTS = (
    "turn.start",  # {turn_id, resume?}: runner
    "tool.start",  # {call_id, name}: ToolHost
    "tool.end",  # {call_id, name, ok, ms}: ToolHost
    "commit",  # {sha, files}: runner, after a turn completes
)


@dataclass(slots=True, frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments object
    read_only: bool = False  # safe to start before the model finishes streaming


@dataclass(slots=True, frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text exactly as streamed; never re-serialize it


@dataclass(slots=True, frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    content: str  # exactly what the model will see


@dataclass(slots=True)
class Item:
    """One durable entry in a thread's history. Items are append-only."""

    id: str
    turn_id: str
    # An OpenAI chat-completions message ({"role": ..., ...}) as it goes on the wire.
    message: dict[str, Any]
    # "incomplete" = cut short by a cancel; how a loop replays it is its own policy.
    status: Literal["complete", "incomplete"] = "complete"
    # Optional loop-private payload, e.g. pydantic-ai's native ModelMessage JSON.
    native: Any = None
    usage: dict[str, Any] | None = None


@dataclass(slots=True)
class Event:
    type: str
    data: dict[str, Any]


@dataclass(slots=True, frozen=True)
class Limits:
    max_steps: int = 12  # model requests per turn
    max_cost_usd: float | None = None


@dataclass(slots=True, frozen=True)
class Resume:
    kind: Literal["approval", "crash"]
    decisions: dict[str, Literal["allow", "deny"]] = field(default_factory=dict)
    reason: str | None = None  # shown to the model when a call is denied


@dataclass(slots=True, frozen=True)
class ModelConfig:
    base_url: str  # e.g. https://openrouter.ai/api/v1 or the local fake server
    model: str  # e.g. "anthropic/claude-sonnet-5"
    api_key: str = "dummy"
    kind: Literal["openrouter", "openai_compat"] = "openrouter"
    max_tokens: int = 4096
    temperature: float | None = 0.0
    reasoning: dict[str, Any] | None = None  # OpenRouter `reasoning`, e.g. {"effort": "low"}
    session_id: str | None = None  # OpenRouter sticky routing, keeps prompt-cache hits
    # Per-endpoint quirks for OpenAI-compatible (BYOK) servers. Keys are documented in
    # DESIGN.md ("Endpoint compat flags"); unknown keys must be ignored.
    compat: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    timeout_s: float = 600.0


@dataclass(slots=True, frozen=True)
class TurnInput:
    thread_id: str
    turn_id: str
    system: str  # frozen for the whole thread
    history: list[Item]  # everything so far, including this turn's user item
    resume: Resume | None
    limits: Limits
    model: ModelConfig


class ToolHost(Protocol):
    """Shared tool registry + permission rules. Emits tool.start/tool.end itself."""

    def specs(self) -> list[ToolSpec]:
        """All tools, always in the same order (keeps the prompt prefix stable)."""
        ...

    def check(self, call: ToolCall) -> Decision: ...

    async def run(self, call: ToolCall) -> ToolResult:
        """Validate args, enforce deny, execute. Never raises."""
        ...


class Loop(Protocol):
    name: str  # "our", "pydantic", ...

    def run_turn(
        self, turn: TurnInput, tools: ToolHost, cancel: asyncio.Event
    ) -> AsyncIterator[Event]: ...

    async def aclose(self) -> None: ...

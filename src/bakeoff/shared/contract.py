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
5. Tools are called through `ToolHost`. For a new call, call `check()` first:
   "ask" means pause (emit `permission.asked` per call, then `turn.end` with
   stop="paused" and the pending ids); "deny" and "allow" both go to `run()`,
   which enforces deny. On an approval resume, the user's answer replaces
   `check()` for the pending calls: `Resume.decisions[id] == "allow"` goes
   straight to `run()` (`run()` only blocks "deny" rules), and "deny" becomes a
   `ToolResult(ok=False, content="Denied by user: <reason>", error="denied")`
   without running.
   A crash resume has no decisions, so pending calls go through `check()` again.
6. When `cancel` is set, stop within 200 ms, leave no orphan tool calls, and
   end with `turn.end` stop="cancelled".
7. The last event a loop emits in every turn is `turn.end`. After it, the runner
   may add exactly one `commit` event (completed turns only), so consumers treat
   `commit` as "turn fully done" and `turn.end` as "the loop is done".
8. If history contains a compaction item (`Item.compaction`), a request carries
   only the system prompt plus the last compaction item and everything after it.
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


ToolError = Literal["invalid_args", "denied", "failed"]


@dataclass(slots=True, frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    content: str  # exactly what the model will see
    # Why ok=False: the model sent bad arguments (it can retry), the call was denied (by rules
    # or the user), or the tool itself failed. None when ok=True. Loops may map these onto
    # their own idioms (e.g. pydantic-ai's ModelRetry for invalid_args).
    error: ToolError | None = None


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
    # True for a runner-written summary that replaces everything before it (rule 8).
    compaction: bool = False


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
    # "openrouter" and "openai_compat" speak chat completions (`<base_url>/chat/completions`).
    # "openai_responses" speaks OpenAI's Responses API (`<base_url>/responses`): `input` items
    # instead of messages, named SSE events. Its items still carry a chat-completions-shaped
    # `Item.message` (for the shared log, UI and checks); a loop may keep the exact Responses
    # items it must replay (e.g. reasoning items with `encrypted_content`) in `Item.native`.
    kind: Literal["openrouter", "openai_compat", "openai_responses"] = "openrouter"
    max_tokens: int = 4096
    temperature: float | None = 0.0
    # OpenRouter's (or the Responses API's) `reasoning` object, e.g. {"effort": "low"}
    reasoning: dict[str, Any] | None = None
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

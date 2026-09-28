"""The seam between the shared harness and a loop implementation.

Everything in `bakeoff.shared` is used by every loop. A loop implementation
(`our_version`, `pydantic_version`, ...) only has to implement `Loop` below, so
the line-count comparison covers exactly the code the decision is about.

Rules every Loop must follow (tests enforce them, see DESIGN.md):

1. A loop only *yields events*. It never touches the session log, the workspace,
   the UI or stdout/stderr. The shared runner persists and publishes what it yields.
2. A loop keeps no state between turns (connection pools and caches are fine).
   It rebuilds everything from `TurnInput.history`, so "resume after approval",
   "resume with tool results" and "resume after a crash" are the same code path.
3. Every durable change is an `item` event. The next model request must be the
   previous request's messages plus new items only: history is append-only.
4. Every tool call gets exactly one tool-result item, including on deny and
   cancel. A tool never runs twice for the same call id.
5. Tools are called through `ToolHost`. For a new call, await `check()` first:
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
   may add exactly one `turn.saved` event (completed turns only: the version of the
   thread's workspace that holds the turn's changes), so consumers treat `turn.saved`
   as "turn fully done" and `turn.end` as "the loop is done".
8. If history contains a compaction item (`Item.compaction`), a request carries
   only the system prompt plus the last compaction item and everything after it.
   A loop never writes summaries: when a response's input fills the context
   window to `CONTEXT_NEAR_LIMIT`, it emits `context.near_limit` (at most once
   per turn), and the runtime decides whether to compact between turns.
9. A tool may start work that finishes later: `run()` returns a result with
   `pending=True`, and the call gets no result item yet (rule 4 holds once the
   result comes). The loop handles the step's other calls under the usual
   ordering, then ends the turn with stop="waiting" and the pending ids (or
   "paused", if a call must be asked; the pending calls wait on). On every
   resume `Resume.waiting` lists the calls still waiting: the loop never runs
   them. `Resume.results` holds finished results (a "tool_result" resume, or
   the crash resume of one): the loop appends one result item for each, never
   running its call, then goes on with the next model request, or ends
   "waiting" again, without a request, if calls are still waiting. A cancel
   gives a waiting call of the step a result like any other (rule 6).

Turn ownership (the runtime's side of the seam):

- The runtime runs each turn in its own task, independent of any client
  connection. A client that disconnects neither stops nor pauses the turn; only
  `cancel` stops it, and a client may attach to its events again later.
- One conversation (thread) runs in one place at a time, under a lock that
  expires if its holder dies. Turns, reverts and compactions of a thread take it;
  a resume after a crash takes it once the dead holder's lock has expired.

In this repo a turn runs in the task that calls `Runner.turn`: a worker process
of its own for `bakeoff turn`, `approve`, `deliver` and `resume`, which no client
holds open. The lock is an OS file lock per thread (`runner._try_lock`), which
the kernel drops when the process that holds it dies.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Decision = Literal["allow", "ask", "deny"]
StopReason = Literal["end_turn", "paused", "waiting", "max_steps", "budget", "cancelled", "error"]

# The share of `ModelConfig.context_window` a response's input tokens must reach for the loop
# to emit `context.near_limit` (rule 8).
CONTEXT_NEAR_LIMIT = 0.8

# Event types. Loops emit the first group; the runner and ToolHost emit the rest.
LOOP_EVENTS = (
    "request.start",  # {step, attempt}
    "reasoning.delta",  # {text}
    "text.delta",  # {text}
    "tool_call.ready",  # {call_id, name, arguments}: a call's arguments are complete
    "item",  # {item: Item}: the only event that changes durable history
    "usage",  # {step, input_tokens, output_tokens, cached_tokens, reasoning_tokens, cost_usd, cost_source}
    "context.near_limit",  # {input_tokens, context_window}: after the step's usage (rule 8)
    "retry",  # {attempt, status, wait_ms, reason}
    "permission.asked",  # {call_id, name, arguments}
    "error",  # {kind, message, retryable}
    "turn.end",  # {stop: StopReason, steps, pending?: [call_id], error?}: pending if paused/waiting
)
SHARED_EVENTS = (
    "turn.start",  # {turn_id, resume?}: runner
    "tool.start",  # {call_id, name}: ToolHost
    "tool.progress",  # {call_id, name, message}: ToolHost, what a running tool reports
    "tool.end",  # {call_id, name, ok, ms, pending?}: ToolHost; pending: the work goes on (rule 9)
    "turn.saved",  # {version, files}: runner, after a turn completes (a git sha here)
)
# Events a runtime may deliver live without storing them: they show a turn as it happens, and
# nothing that resumes or judges a turn reads them. The stored events are the rest: messages
# (`item`), tool start and end, permission requests, usage, and the turn boundaries (turn.start,
# turn.end, turn.saved), plus request.start, tool_call.ready, context.near_limit, retry and
# error. A runtime that does not store an event gives it no `seq`, so the stored seqs still
# have no gaps (I3). The reference runner stores every event.
LIVE_ONLY_EVENTS = ("text.delta", "reasoning.delta", "tool.progress")


@dataclass(slots=True, frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments object
    read_only: bool = False  # safe to start before the model finishes streaming
    timeout_s: float | None = None  # the ToolHost fails a run that takes longer (None: no limit)


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
    # The tool started work that finishes later (rule 9): no result item now. The content says
    # what started; the result comes in `Resume.results` of a later resume.
    pending: bool = False


# What a user message says: text, or a list of content parts in the OpenAI chat format,
# {"type": "text", "text": ...} and {"type": "image_url", "image_url": {"url": ..., "detail"?}}.
# The runner stores it as given and caps its size (`runner.MAX_USER_BYTES`): large files go by
# reference (a URL), not inline. A loop converts the parts for its API.
UserContent = str | list[dict[str, Any]]


@dataclass(slots=True)
class Item:
    """One durable entry in a thread's history. Items are append-only."""

    id: str
    turn_id: str
    # An OpenAI chat-completions message ({"role": ..., ...}) as it goes on the wire. A user
    # message's content is `UserContent`.
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
    kind: Literal["approval", "crash", "tool_result"]
    decisions: dict[str, Literal["allow", "deny"]] = field(default_factory=dict)
    reason: str | None = None  # shown to the model when a call is denied
    # Finished results of waiting calls, by call id (rule 9): a "tool_result" resume's, or what
    # the crash resume of one delivers again.
    results: dict[str, ToolResult] = field(default_factory=dict)
    # Set by the runner on every resume: the calls whose results are still to come. Never run.
    waiting: tuple[str, ...] = ()


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
    # The model's context window in tokens, if known: `context.near_limit` needs it (rule 8).
    context_window: int | None = None


@dataclass(slots=True, frozen=True)
class TurnInput:
    thread_id: str
    turn_id: str
    # The thread's system prompt. It changes only between turns, and then history has the
    # runner's note of the change before the user's message (`Runner.turn`).
    system: str
    history: list[Item]  # everything so far, including this turn's user item
    resume: Resume | None
    limits: Limits
    model: ModelConfig


class ToolHost(Protocol):
    """Shared tool registry + permission rules. Emits tool.start, tool.progress and tool.end
    itself."""

    def specs(self) -> list[ToolSpec]:
        """All tools, always in the same order (keeps the prompt prefix stable)."""
        ...

    async def check(self, call: ToolCall) -> Decision:
        """allow / ask / deny. Async, so a runtime may read its rules from a database."""
        ...

    async def run(self, call: ToolCall) -> ToolResult:
        """Validate args, enforce deny, execute within the spec's `timeout_s`. Never raises. A
        pending result means the tool's work goes on (rule 9)."""
        ...


class Loop(Protocol):
    name: str  # "our", "pydantic", ...

    def run_turn(
        self, turn: TurnInput, tools: ToolHost, cancel: asyncio.Event
    ) -> AsyncIterator[Event]: ...

    async def aclose(self) -> None: ...

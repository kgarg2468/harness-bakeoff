"""Drives turns: TurnInput -> loop events -> persist -> publish -> save the workspace.

See DESIGN.md, "The seam". The runner is shared by every loop, so persistence, event
stamping, tool timing and saved versions are identical for all of them. A thread's workspace
is a `Workspace`; the default is `WorkCopy`, where a saved version is a git commit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import aclosing, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from bakeoff.shared import permissions
from bakeoff.shared.contract import (
    Event,
    Item,
    Limits,
    Loop,
    ModelConfig,
    Resume,
    ToolHost,
    ToolResult,
    TurnInput,
    UserContent,
)
from bakeoff.shared.sessionlog import EventRow, SessionLog, event_row, item_to_json
from bakeoff.shared.workcopy import WorkCopy, Workspace

Sink = Callable[[dict[str, Any]], None]
MakeTools = Callable[[Path, dict[str, Any], Callable[[Event], None]], ToolHost]
# Opens a thread's workspace at its directory; the second argument is the descriptor of the
# thread's lock, for a workspace whose processes must keep it (see `_try_lock`).
MakeWorkspace = Callable[[Path, int | None], Workspace]

_BATCH = 64  # non-item events buffered before a flush
_CANCEL_POLL_S = 0.05
_LOCK_POLL_S = 0.02
_STATUS = {"end_turn": "done", "max_steps": "done", "budget": "done", "cancelled": "cancelled"}
_DEFAULT_LIMITS = Limits()
_THREAD_ID = re.compile(r"[\w-]+")
# While the last workspace turn is paused, its pending calls must be answered first: a new user
# message or a summary between a call and its result breaks the history (see `Runner.turn`).
_PAUSED = {"paused": "is paused: resolve the pending approval first"}
# The same holds while calls wait for their results (contract rule 9): `_refuse_while` then also
# refuses while any call of the thread waits, whatever turn started it.
_OPEN = {**_PAUSED, "waiting": "is waiting for tool results: deliver them first"}
# A revert also waits while the last turn has changes that no saved version holds (see
# `Runner.revert`).
_REVERT_WAITS = {
    **_OPEN,
    "error": "failed to save its changes: run a turn first, its saved version includes them",
}
# The stops after which a turn is not saved: its calls are still open (answered by a resume).
_OPEN_STOPS = ("paused", "waiting")
logger = logging.getLogger(__name__)

# Starts the content of a compaction item; see contract rule 8.
SUMMARY_PREFIX = "[harness] Conversation summary:"
# Starts the content of the item that notes a change of the thread's settings (`Runner.turn`).
CONFIG_PREFIX = "[harness] Configuration changed:"
# How that note names a new system prompt: I1 lets the system messages change only at such a note.
NEW_SYSTEM_PROMPT = "a new system prompt"
# The most a user message's content may take as JSON (UTF-8). Inline data (an image as a data URL)
# counts; a large file goes by reference (a URL) instead.
MAX_USER_BYTES = 256 * 1024

try:
    import fcntl
except ImportError:  # Windows: no advisory locks; the log's running-turn check still applies
    fcntl = None  # type: ignore[assignment]


class ThreadBusy(RuntimeError):
    """Another turn, revert or compaction of this thread holds its lock (maybe in another
    process)."""


def _try_lock(lock_path: Path) -> int | None:
    """Take the thread's OS advisory lock without waiting and return its descriptor (None where
    the OS has no advisory locks). Raises ThreadBusy if another descriptor holds it.

    The kernel drops the lock when the last descriptor of it closes. Every git process of the
    thread inherits the descriptor (see `WorkCopy`), so a worker that dies (e.g. SIGKILL) keeps
    the lock until its last git process exits. A crash resume can take it after that, and two
    live processes (e.g. two concurrent crash resumes) can never both hold it.
    """
    if fcntl is None:
        return None
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException as exc:
        os.close(fd)
        if isinstance(exc, BlockingIOError):
            raise ThreadBusy(
                f"thread {lock_path.stem} is busy: a turn, revert or compaction holds its lock"
            ) from None
        raise
    return fd


def _unlock(fd: int | None) -> None:
    if fd is not None:
        os.close(fd)  # the lock stays while a git process still has its inherited copy


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[int | None]:
    """Hold the thread's lock (see `_try_lock`) for the block; yields its descriptor."""
    fd = _try_lock(lock_path)
    try:
        yield fd
    finally:
        _unlock(fd)


async def _wait_lock(lock_path: Path, wait_s: float) -> int | None:
    """`_try_lock`, retried for up to `wait_s` seconds."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return _try_lock(lock_path)
        except ThreadBusy:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(_LOCK_POLL_S)


def _check_loop_item(item: Any, turn_id: str) -> None:
    """A loop's items belong to its own turn, and only the runner writes compaction items
    (contract rule 8) and notes of changed settings: a loop cannot make history before an item
    of its own disappear, or change the system prompt behind one (I1)."""
    if getattr(item, "compaction", False):
        raise ValueError(f"item {item.id!r} is a compaction item: only the runner compacts")
    if getattr(item, "id", None) in (f"{turn_id}:user", f"{turn_id}:config"):
        raise ValueError(f"item {item.id!r} has the id of a runner item")
    if getattr(item, "turn_id", turn_id) != turn_id:
        raise ValueError(f"item {item.id!r} belongs to turn {item.turn_id!r}, not {turn_id!r}")


def _content_problem(content: Any) -> str | None:
    """What is wrong with a user message's content (`contract.UserContent`), or None."""
    if isinstance(content, list):
        if not content:
            return "it has no content parts"
        for n, part in enumerate(content):
            kind = part.get("type") if isinstance(part, dict) else None
            image = part.get("image_url") if kind == "image_url" else None
            if not (
                (kind == "text" and isinstance(part.get("text"), str))
                or (isinstance(image, dict) and isinstance(image.get("url"), str))
            ):
                return f"part {n} is not a text part or an image_url part with a url"
    elif not isinstance(content, str):
        return "the content must be text or a list of content parts"
    # A lone surrogate (half an emoji, say) is valid JSON text: it counts, it is not an error.
    size = len(json.dumps(content, ensure_ascii=False).encode(errors="surrogatepass"))
    if size > MAX_USER_BYTES:
        return (
            f"its content takes {size} bytes as JSON, more than {MAX_USER_BYTES}:"
            " pass large files by URL"
        )
    return None


def _complete_answer(item: Item) -> bool:
    return item.message.get("role") == "assistant" and item.status == "complete"


def _new_settings(
    thread: dict[str, Any], system: str | None, rules: dict[str, Any] | None
) -> tuple[dict[str, Any], str] | None:
    """The thread's settings (`{"system", "meta"}`) with a new system prompt and new rules, and
    the note that says what changed; None if nothing changes."""
    changes, settings = [], {"system": thread["system"], "meta": thread["meta"]}
    if system is not None and system != thread["system"]:
        changes.append(NEW_SYSTEM_PROMPT)
        settings["system"] = system
    if rules is not None and rules != thread["meta"]["rules"]:
        changes.append(f"new permission rules {json.dumps(rules)}")
        settings["meta"] = {**thread["meta"], "rules": rules}
    return (settings, f"{CONFIG_PREFIX} {', and '.join(changes)}.") if changes else None


@contextmanager
def _logged_failure(what: str) -> Iterator[None]:
    """Log a failure instead of raising it, while another error is on its way up."""
    try:
        yield
    except Exception:
        logger.exception("%s failed", what)


class _Publisher:
    """Stamps one turn's events, persists them and publishes them to the sink. It stores every
    event, the live-only ones too (`contract.LIVE_ONLY_EVENTS`), and numbers them all.

    `item` and `tool.start` events are persisted (with everything buffered before them) before
    they are published, a `tool.start` before its tool runs, so no crash can hide a run from I2.
    `turn.end` is persisted before it is published. Other events are published at once and
    persisted in batches. `finish` records the end of the turn in one transaction.
    """

    def __init__(
        self, log: SessionLog, sink: Sink | None, thread_id: str, turn_id: str, impl: str
    ) -> None:
        self._log = log
        self._sink = sink
        self._thread = thread_id
        self.turn_id = turn_id
        self._head = {"v": 1, "thread": thread_id, "turn": turn_id, "impl": impl}
        self._seq = log.next_seq(thread_id)
        self._t0 = time.perf_counter_ns()
        self._batch: list[EventRow] = []
        self._unsent: list[str] = []  # buffered events whose store failed: published once stored
        self.end: dict[str, Any] | None = None  # the data of the turn's turn.end, once stamped
        self.ended = False  # tool events go to the turn row from now on (see `publish`)

    def _t_us(self) -> int:
        return (time.perf_counter_ns() - self._t0) // 1000

    def _row(self, seq: int, type_: str, data: dict[str, Any]) -> EventRow:
        env = {**self._head, "seq": seq, "t_us": self._t_us(), "type": type_, "data": data}
        if type_ == "item":
            env["data"] = {**data, "item": item_to_json(data["item"])}
        # Serialized now, so a later change to `data` cannot alter the record, and a value
        # that is not JSON fails here, at the event that carries it.
        return event_row(env)

    def emit(
        self, type_: str, data: dict[str, Any], *, settings: dict[str, Any] | None = None
    ) -> None:
        """Stamp, persist and publish one event. `settings` (with an `item`) replaces the
        thread's system prompt and meta in the item's transaction."""
        row = self._row(self._seq, type_, data)
        if type_ == "item":
            self._log.append_item(
                self._thread, data["item"], [*self._batch, row], settings=settings
            )
            self._batch.clear()
        elif type_ == "tool.start" or (type_ == "tool.end" and data.get("pending")):
            # A pending tool.end is stored at once too: a crash resume must know the call waits.
            self._log.append_events([*self._batch, row])
            self._batch.clear()
        else:
            self._batch.append(row)
        self._seq += 1
        if type_ == "turn.end":
            self.end, self.ended = data, True
        if len(self._batch) >= _BATCH or type_ == "turn.end":
            try:
                self.flush()
            except Exception:
                self._unsent.append(row[-1])  # still buffered; published when a retry stores it
                raise
        self._to_sink(row[-1])

    def _to_sink(self, envelope_json: str) -> None:
        if self._sink is None:
            return
        try:
            self._sink(json.loads(envelope_json))  # a private copy, equal to the stored one
        except Exception:
            # A broken consumer (e.g. a closed stdout) must not fail the turn.
            logger.exception("event sink failed; it gets no more events this turn")
            self._sink = None

    def _send_unsent(self) -> None:
        unsent, self._unsent = self._unsent, []
        for envelope in unsent:
            self._to_sink(envelope)

    def publish(self, event: Event) -> None:
        """The ToolHost's `emit` callback. A tool event after the loop's `turn.end` cannot
        join the stream (`turn.saved` must follow `turn.end` directly), so it is stored on the
        turn row (`late`) at once: also after the row is complete, e.g. from a tool task that
        outlived its loop."""
        if not self.ended:
            self.emit(event.type, event.data)
            return
        late = {"t_us": self._t_us(), "type": event.type, "data": event.data}
        with _logged_failure(f"storing a tool event after the end of turn {self.turn_id}"):
            self._log.append_late(self.turn_id, late)

    def flush(self) -> None:
        if self._batch:
            self._log.append_events(self._batch)
            self._batch.clear()
        self._send_unsent()

    def finish(
        self,
        status: str,
        *,
        stop: str | None = None,
        pending: list[str] | None = None,
        item: Item | None = None,
        saved: tuple[str, list[str]] | None = None,
    ) -> None:
        """Record the end of the turn in one transaction: its status (and saved version), what
        is still buffered, and the runner's last events (`item`, then `turn.saved`). They are
        published only after that, so a consumer that sees `turn.saved` finds the turn complete
        in the log (rule 7), and nothing is published that the log does not have."""
        rows = []
        if item is not None:
            rows.append(self._row(self._seq, "item", {"item": item}))
        if saved is not None:
            version, files = saved
            data = {"version": version, "files": files}
            rows.append(self._row(self._seq + len(rows), "turn.saved", data))
        self._log.set_turn_status(
            self.turn_id,
            status,
            stop=stop,
            pending=pending,
            commit_sha=None if saved is None else saved[0],
            events=[*self._batch, *rows],
            item=item,
        )
        self._batch.clear()
        self._seq += len(rows)
        self.ended = True
        self._send_unsent()
        for row in rows:
            self._to_sink(row[-1])

    def close(self) -> None:
        """Store what is still buffered (e.g. after a crash); the turn row stays as it is."""
        self.ended = True
        self.flush()


class NdjsonMirror:
    """A sink that appends each event envelope to a file as one JSON line."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def __call__(self, envelope: dict[str, Any]) -> None:
        self._file.write(json.dumps(envelope) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class Runner:
    """Runs turns of any `Loop` against the session log and one workspace per thread
    (`make_workspace`, by default a git `WorkCopy`).

    Each turn, revert and compaction holds the thread's OS lock (see `_try_lock`) while it runs.
    A resume (approval, tool results or crash) waits up to `lock_wait_s` seconds for it: it
    follows the end of the turn before it, whose worker, or that worker's last git process, may
    still hold it. Anything else raises ThreadBusy at once.
    """

    def __init__(
        self,
        log: SessionLog,
        wc_root: Path,
        make_tools: MakeTools,
        sink: Sink | None = None,
        lock_wait_s: float = 5.0,
        make_workspace: MakeWorkspace = WorkCopy,
    ) -> None:
        self.log = log
        self.wc_root = wc_root
        self.make_tools = make_tools
        self.sink = sink
        self.lock_wait_s = lock_wait_s
        self.make_workspace = make_workspace

    def workdir(self, thread_id: str) -> Path:
        """The directory of the thread's workspace."""
        return (self.wc_root / thread_id).absolute()

    def new_thread(
        self,
        *,
        impl: str,
        system: str,
        rules: dict[str, Any],
        model: ModelConfig,
        thread_id: str | None = None,
    ) -> str:
        """Create a thread (its meta keeps rules and the model config minus the api key) and
        its workspace."""
        thread_id = thread_id or uuid.uuid4().hex[:12]
        if not _THREAD_ID.fullmatch(thread_id):
            raise ValueError(f"thread id must match {_THREAD_ID.pattern}: {thread_id!r}")
        model_meta = asdict(model)
        del model_meta["api_key"]
        # The workspace first: once the thread row exists, every turn can rely on it.
        self.make_workspace(self.workdir(thread_id), None).init_sync()
        self.log.create_thread(
            thread_id, impl=impl, system=system, meta={"rules": rules, "model": model_meta}
        )
        return thread_id

    async def turn(
        self,
        loop: Loop,
        thread_id: str,
        *,
        model: ModelConfig,
        user_text: UserContent | None = None,
        resume: Resume | None = None,
        limits: Limits = _DEFAULT_LIMITS,
        cancel: asyncio.Event | None = None,
        watch_cancel: bool = False,
        system: str | None = None,
        rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one turn: a new user message, or a resume (approval, tool results or crash).

        `user_text` is the message's content, text or content parts (`contract.UserContent`),
        stored as given. One whose JSON is larger than `MAX_USER_BYTES`, or with a part that is
        not text or an image URL, is refused (ValueError) before anything is recorded.

        While the last turn is paused, a new user message is refused: the pending calls need an
        approval resume first (it may deny them, with a reason), since a user message between a
        call and its result breaks the history. The same goes while calls wait for results
        (contract rule 9): a turn that ends "waiting" is recorded like a paused one, not saved,
        and `Resume(kind="tool_result", results={call_id: ToolResult})` delivers them, in any
        process and as they come. Each must be a waiting call's (ValueError otherwise), and a
        delivery waits for a pending approval. On every resume the runner sets `Resume.waiting`
        to the calls still waiting, which the loop never runs, and a crash resume of a
        tool_result turn delivers again the results that turn had not saved.

        `system` and `rules` are the thread's system prompt and permission rules from this turn
        on (rules that are not valid raise ValueError at once). If they differ from the thread's
        (read once the lock is held), the runner stores them and appends a runner item
        before the user's message, a user message that starts with `CONFIG_PREFIX` and names
        what changed, so the history explains why the request prefix changed (I1 lets it reset
        there). Only a new user message changes them: on a resume they must be None or unchanged,
        since a note between a call and its result breaks the history.

        Returns `{"turn_id", "stop", "pending", "version"}` (`version`: the saved version, None
        if the turn was not saved). A loop exception ends the turn with stop="error" instead of
        raising. `asyncio.CancelledError` propagates and leaves the turn "running", like a crash;
        resume it with `Resume(kind="crash")`. Any other failure (the workspace, or the session
        log) is raised after the turn is recorded without a saved version: as "error" (the next
        turn's saved version includes its changes), or still "paused" or "waiting" if it ended
        so. If even that cannot be written, the turn stays "running" for a crash resume.

        A resume waits up to `lock_wait_s` for the thread's lock; a new user message raises
        ThreadBusy at once while another turn, revert or compaction holds it.
        """
        if (user_text is None) == (resume is None):
            raise ValueError("pass exactly one of user_text and resume")
        if user_text is not None and (problem := _content_problem(user_text)):
            raise ValueError(f"cannot send this user message: {problem}")
        if rules is not None:
            permissions.validate_rules(rules)  # before they are stored: else no turn could run
        if loop.name != (impl := self._thread(thread_id)["impl"]):
            raise ValueError(f"thread {thread_id} belongs to {impl!r}, not {loop.name!r}")
        wait_s = self.lock_wait_s if resume is not None else 0.0
        lock_fd = await _wait_lock(self._lock_path(thread_id), wait_s)
        try:
            return await self._turn(
                loop,
                thread_id,
                model,
                user_text,
                resume,
                limits,
                cancel,
                watch_cancel,
                lock_fd,
                system,
                rules,
            )
        finally:
            _unlock(lock_fd)

    async def _turn(
        self,
        loop: Loop,
        thread_id: str,
        model: ModelConfig,
        user_text: UserContent | None,
        resume: Resume | None,
        limits: Limits,
        cancel: asyncio.Event | None,
        watch_cancel: bool,
        lock_fd: int | None,
        system: str | None,
        rules: dict[str, Any] | None,
    ) -> dict[str, Any]:
        kind = resume.kind if resume else "user"
        # Read with the lock held: a resume may have waited for a turn that changed the settings.
        thread = self._thread(thread_id)
        if resume is not None and _new_settings(thread, system, rules):
            raise ValueError("the system prompt and the rules change with a user message only")
        self._retire_dead(thread_id, lock_fd)
        if resume is None:
            self._refuse_while(thread_id, "start a user turn", _OPEN)
        else:
            resume = self._resume(thread_id, resume)
        row = self.log.start_turn(thread_id, kind)  # raises if a turn is running (unless crash)
        turn_id = row["id"]
        ws = self.make_workspace(self.workdir(thread_id), lock_fd)
        pub = _Publisher(self.log, self.sink, thread_id, turn_id, thread["impl"])
        cancel = cancel or asyncio.Event()
        watcher = (
            asyncio.create_task(self._watch_cancel(thread_id, row["started_us"], cancel))
            if watch_cancel
            else None
        )
        try:
            await self._reconcile(ws, thread_id, turn_id)
            start: dict[str, Any] = {"turn_id": turn_id}
            if resume is not None:
                start["resume"] = asdict(resume)
            pub.emit("turn.start", start)
            if changed := _new_settings(thread, system, rules):
                settings, note = changed
                item = Item(f"{turn_id}:config", turn_id, {"role": "user", "content": note})
                pub.emit("item", {"item": item}, settings=settings)
                thread = {**thread, **settings}
            if user_text is not None:
                message = {"role": "user", "content": user_text}
                pub.emit("item", {"item": Item(f"{turn_id}:user", turn_id, message)})
            turn_input = TurnInput(
                thread_id=thread_id,
                turn_id=turn_id,
                system=thread["system"],
                history=self.log.items(thread_id),
                resume=resume,
                limits=limits,
                model=model,
            )
            end = await self._drive(loop, turn_input, cancel, pub, thread["meta"]["rules"])
            stop = end.get("stop", "error")
            if stop in _OPEN_STOPS:
                pending = list(end.get("pending") or [])
                pub.finish(stop, stop=stop, pending=pending)
                return {"turn_id": turn_id, "stop": stop, "pending": pending, "version": None}
            saved = await ws.save(f"turn {row['idx'] + 1}: {stop}")
            # The row's status and version with `turn.saved`, which is published after it.
            pub.finish(_STATUS.get(stop, "error"), stop=stop, saved=saved)
            return {"turn_id": turn_id, "stop": stop, "pending": [], "version": saved[0]}
        except Exception:  # unlike a crash (CancelledError), a failure ends the turn
            self._record_failure(pub)
            raise
        except BaseException:  # a crash: the turn stays "running" for a crash resume
            with _logged_failure(f"storing the events of turn {turn_id}"):
                pub.close()
            raise
        finally:
            if watcher is not None:
                watcher.cancel()

    async def _drive(
        self,
        loop: Loop,
        turn_input: TurnInput,
        cancel: asyncio.Event,
        pub: _Publisher,
        rules: dict[str, Any],
    ) -> dict[str, Any]:
        """Consume the loop's events up to `turn.end` and return its data.

        If the loop fails or stops early, publish a synthesized `turn.end` with stop="error".
        """
        steps: set[Any] = set()
        end: dict[str, Any] | None = None
        error = "loop ended without turn.end"
        try:
            tools = self.make_tools(self.workdir(turn_input.thread_id), rules, pub.publish)
            async with aclosing(loop.run_turn(turn_input, tools, cancel)) as events:
                async for event in events:
                    if event.type == "request.start":
                        steps.add(event.data.get("step"))
                    elif event.type == "item":
                        _check_loop_item(event.data.get("item"), turn_input.turn_id)
                    pub.emit(event.type, event.data)
                    if event.type == "turn.end":
                        end = event.data
                        break
        except Exception as exc:  # CancelledError is not an Exception: it propagates (crash)
            if end is None and pub.end is not None:
                raise  # the loop's turn.end is stamped, but the log could not store it
            if end is None:  # else it failed while closing after its turn.end: nothing to add
                error = f"{type(exc).__name__}: {exc}"
                pub.emit("error", {"kind": "loop", "message": error, "retryable": False})
        if end is None:
            end = {"stop": "error", "steps": len(steps), "error": error}
            pub.emit("turn.end", end)
        return end

    async def revert(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        """Undo a saved turn's changes with a new version, recorded as a new "revert" turn.

        Appends a runner item telling the model what was reverted. Returns the same summary
        shape as `turn()`. The note, the version and `turn.saved` are recorded in one
        transaction; if the workspace fails (e.g. a git conflict) or the log cannot record them,
        the workspace is reset and no turn is recorded. If its process dies, the next call on
        the thread undoes it (see `_retire_dead`). It refuses while the last turn is paused or
        failed to save: that turn's changes are in no version, so the revert would take them
        into its own, or lose them if the workspace aborts it.
        """
        thread = self._thread(thread_id)
        target = next((t for t in self.log.turns(thread_id) if t["id"] == turn_id), None)
        if target is None or not target["commit_sha"]:
            raise ValueError(f"turn {turn_id} of thread {thread_id} has no saved version to revert")
        # Held until the revert is fully recorded: while its row says "running", a crash resume
        # that got the lock would take the workspace's revert for a version no turn recorded.
        with _exclusive(self._lock_path(thread_id)) as lock_fd:
            self._retire_dead(thread_id, lock_fd)
            self._refuse_while(thread_id, "revert", _REVERT_WAITS)
            row = self.log.start_turn(thread_id, "revert")
            ws = self.make_workspace(self.workdir(thread_id), lock_fd)
            try:
                await self._reconcile(ws, thread_id, row["id"])  # e.g. after a failed revert
                head = await ws.head()
                saved = await ws.revert(target["commit_sha"])
            except Exception:
                self._drop_turn(row["id"])  # it recorded nothing
                raise
            files_text = ", ".join(saved[1]) or "none"
            note = f"[harness] Reverted turn {target['idx'] + 1}; files: {files_text}"
            message = {"role": "user", "content": note}
            try:
                pub = _Publisher(self.log, self.sink, thread_id, row["id"], thread["impl"])
                item = Item(f"{row['id']}:revert", row["id"], message)
                pub.finish("done", item=item, saved=saved)
            except Exception:  # nothing is recorded: the thread goes back to where it was
                with _logged_failure(f"undoing the workspace's revert for turn {row['id']}"):
                    await ws.recover(head, keep=False)
                self._drop_turn(row["id"])
                raise
        return {"turn_id": row["id"], "stop": None, "pending": [], "version": saved[0]}

    def compact(self, thread_id: str, summary: str) -> Item:
        """Append a compaction item (contract rule 8) as its own "compact" turn.

        The item and the finished row are recorded in one transaction, or nothing is.
        Compaction changes no files, so the turn saves no version. It is refused while the last
        turn is paused: the summary would come between the pending calls and their results.
        """
        thread = self._thread(thread_id)
        with _exclusive(self._lock_path(thread_id)) as lock_fd:
            self._retire_dead(thread_id, lock_fd)
            self._refuse_while(thread_id, "compact", _OPEN)
            row = self.log.start_turn(thread_id, "compact")
            item = Item(
                id=f"{row['id']}:compact",
                turn_id=row["id"],
                message={"role": "user", "content": f"{SUMMARY_PREFIX} {summary}"},
                compaction=True,
            )
            try:
                pub = _Publisher(self.log, self.sink, thread_id, row["id"], thread["impl"])
                pub.finish("done", item=item)
            except Exception:
                self._drop_turn(row["id"])
                raise
        return item

    def _lock_path(self, thread_id: str) -> Path:
        # Next to the workspace, not inside it, so it is never saved.
        return self.workdir(thread_id).parent / f"{thread_id}.lock"

    def _thread(self, thread_id: str) -> dict[str, Any]:
        thread = self.log.get_thread(thread_id)
        if thread is None:
            raise KeyError(f"unknown thread {thread_id}")
        return thread

    def _record_failure(self, pub: _Publisher) -> None:
        """Record the end of a turn that failed before its saved version was recorded, with its
        buffered events (e.g. a turn.end whose flush failed).

        A paused (or waiting) turn stays so with its pending calls: the next turn must answer
        them.
        Any other turn becomes "error" without a version, so it does not block the thread; the
        next turn repairs the workspace and its version includes the changes. Best effort: if
        the log fails again (e.g. it is still locked), that is only logged, so the caller raises
        the first error, and the turn stays "running" for a crash resume to take over.
        """
        end = pub.end or {}
        stop = end.get("stop", "error")
        with _logged_failure(f"recording the end of turn {pub.turn_id}"):
            if stop in _OPEN_STOPS:
                pub.finish(stop, stop=stop, pending=list(end.get("pending") or []))
            else:
                pub.finish("error", stop=stop)

    def _drop_turn(self, turn_id: str) -> None:
        """Delete a revert or compaction turn that recorded nothing, while its error is on its
        way up. If the log fails, mark it "error" instead; if that fails too, it stays "running"
        for a crash resume. Both failures are only logged."""
        try:
            self.log.discard_turn(turn_id)
        except Exception:
            logger.exception("discarding turn %s failed", turn_id)
            with _logged_failure(f"recording turn {turn_id} as an error"):
                self.log.set_turn_status(turn_id, "error")

    def _retire_dead(self, thread_id: str, lock_fd: int | None) -> None:
        """Retire the thread's last turn if it is a revert or compaction that died.

        Called with the thread's lock held. Every revert and compaction holds it while it runs,
        so a "running" one is dead: its process died, or its failure could not be recorded. It
        recorded nothing (each records its end in one transaction). A compaction is deleted; a
        revert is marked "error" and the next `_reconcile` undoes what the workspace did for it.
        A dead loop turn is left for a crash resume. Without OS locks (`lock_fd` None) a running
        turn may be alive, so nothing is retired.
        """
        last = self.log.last_turn(thread_id)
        if lock_fd is None or last is None or last["status"] != "running":
            return
        if last["kind"] == "compact":
            self.log.discard_turn(last["id"])
        elif last["kind"] == "revert":
            self.log.set_turn_status(last["id"], "error")

    def _refuse_while(self, thread_id: str, what: str, waits: dict[str, str]) -> None:
        """Raise if the thread's last workspace turn has no saved version and a status in
        `waits`: it left something that must be resolved before `what`. With "waiting" in
        `waits`, also while any call of the thread waits for its result (say, one whose turn
        then failed)."""
        if "waiting" in waits and (waiting := self._waiting(thread_id)):
            raise RuntimeError(f"cannot {what}: calls {waiting} wait for their results")
        turns = self._ws_turns(thread_id)
        last = turns[-1] if turns else None
        # A revert without a version recorded nothing, and `_reconcile` undoes its changes.
        if last is None or last["kind"] == "revert" or last["commit_sha"]:
            return
        if last["status"] in waits:
            raise RuntimeError(f"cannot {what}: turn {last['id']} {waits[last['status']]}")

    def _waiting(self, thread_id: str) -> list[str]:
        """The calls that wait for their results (contract rule 9): calls of the last complete
        assistant item that no result item after it answers, whose run said `pending` (a stored
        `tool.end`) in that item's turn or a later one. A pending run whose call never reached
        history (a read-only call started early, before its response failed or was cut off)
        waits for nothing, and neither does an older call with the same id."""
        items = self.log.items(thread_id)
        last = next((n for n in reversed(range(len(items))) if _complete_answer(items[n])), None)
        if last is None:
            return []
        order = {t["id"]: t["idx"] for t in self.log.turns(thread_id)}
        since = order.get(items[last].turn_id, 0)
        ends = self.log.events(thread_id, types=("tool.end",))
        pending = {
            e["data"].get("call_id")
            for e in ends
            if e["data"].get("pending") and order.get(e["turn"], -1) >= since
        }
        answered = {item.message.get("tool_call_id") for item in items[last + 1 :]}
        calls = [call.get("id") for call in items[last].message.get("tool_calls") or ()]
        return [c for c in calls if c in pending and c not in answered]

    def _resume(self, thread_id: str, resume: Resume) -> Resume:
        """`resume` as the loop gets it: with the calls still waiting in `waiting` and, for the
        crash resume of a tool_result turn, the results that turn delivered but did not save.
        Raises ValueError for a delivery of a call that does not wait."""
        waiting = self._waiting(thread_id)
        results = dict(resume.results)
        if resume.decisions and resume.kind != "approval":
            raise ValueError("only an approval resume carries decisions")
        if asked := [c for c in resume.decisions if c in waiting]:
            raise ValueError(f"calls {asked} wait for their results: nobody asked about them")
        if results and resume.kind == "approval":
            raise ValueError("results come with a tool_result resume")
        if resume.kind == "tool_result":
            self._refuse_while(thread_id, "deliver tool results", _PAUSED)
            wrong = [
                c for c, r in results.items() if c not in waiting or r.call_id != c or r.pending
            ]
            if wrong or not results:
                raise ValueError(f"cannot deliver results for {wrong}: waiting for {waiting}")
        elif resume.kind == "crash":
            last = self.log.last_turn(thread_id)
            died = last["id"] if last is not None and last["status"] == "running" else None
            delivered = self._delivered(thread_id, died) if died else {}
            results = {c: r for c, r in delivered.items() if c in waiting}
        return replace(
            resume, results=results, waiting=tuple(c for c in waiting if c not in results)
        )

    def _delivered(self, thread_id: str, turn_id: str) -> dict[str, ToolResult]:
        """The results the `turn.start` of turn `turn_id` delivered, if it was a resume."""
        start = next(
            (e for e in self.log.events(thread_id, types=("turn.start",)) if e["turn"] == turn_id),
            None,
        )
        resume = (start or {}).get("data", {}).get("resume") or {}
        return {c: ToolResult(**r) for c, r in (resume.get("results") or {}).items()}

    def _ws_turns(self, thread_id: str) -> list[dict[str, Any]]:
        """The thread's turns that use the workspace (all but compactions), in order."""
        return [t for t in self.log.turns(thread_id) if t["kind"] != "compact"]

    async def _reconcile(self, ws: Workspace, thread_id: str, turn_id: str) -> None:
        """Before turn `turn_id` uses the workspace, repair what the workspace turn before it
        left, if that turn died ("running") or failed ("error" without a version): the workspace
        may hold a version that no turn recorded (with git: a commit, a stale lock, or a revert
        in progress).

        A revert records its note, version and `turn.saved` at once, so one without a version
        recorded nothing: undo all that the workspace did for it (it started from a clean tree
        at the last recorded version) and mark it "error", so the model never sees changes it
        was not told about. For any other turn, go back to the last recorded version and keep
        the files, so this turn's version includes the changes.
        """
        turns = [t for t in self._ws_turns(thread_id) if t["id"] != turn_id]
        last = turns[-1] if turns else None
        if last is None or last["status"] not in ("running", "error") or last["commit_sha"]:
            return
        versions = [t["commit_sha"] for t in turns if t["commit_sha"]]
        if last["kind"] != "revert":
            await ws.recover(versions[-1] if versions else None)
            return
        await ws.recover(versions[-1] if versions else None, keep=False)
        if last["status"] == "running":
            self.log.set_turn_status(last["id"], "error")

    async def _watch_cancel(self, thread_id: str, since_us: int, cancel: asyncio.Event) -> None:
        # Polls because the request may come from another process (`bakeoff cancel`).
        while not cancel.is_set():
            if self.log.cancel_requested(thread_id, since_us):
                cancel.set()
            else:
                await asyncio.sleep(_CANCEL_POLL_S)

"""Drives turns: TurnInput -> loop events -> persist -> publish -> git commit.

See DESIGN.md, "The seam". The runner is shared by every loop, so persistence, event
stamping, tool timing and commits are identical for all of them.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bakeoff.shared.contract import (
    Event,
    Item,
    Limits,
    Loop,
    ModelConfig,
    Resume,
    ToolHost,
    TurnInput,
)
from bakeoff.shared.sessionlog import EventRow, SessionLog, event_row, item_to_json
from bakeoff.shared.workcopy import WorkCopy

Sink = Callable[[dict[str, Any]], None]
MakeTools = Callable[[Path, dict[str, Any], Callable[[Event], None]], ToolHost]

_BATCH = 64  # non-item events buffered before a flush
_CANCEL_POLL_S = 0.05
_STATUS = {"end_turn": "done", "max_steps": "done", "budget": "done", "cancelled": "cancelled"}
_DEFAULT_LIMITS = Limits()
_THREAD_ID = re.compile(r"[\w-]+")
logger = logging.getLogger(__name__)

# Starts the content of a compaction item; see contract rule 8.
SUMMARY_PREFIX = "[harness] Conversation summary:"

try:
    import fcntl
except ImportError:  # Windows: no advisory locks; the log's running-turn check still applies
    fcntl = None  # type: ignore[assignment]


class ThreadBusy(RuntimeError):
    """Another live process is running a turn (or a revert) on this thread."""


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an OS advisory lock on `lock_path` for the duration of a turn.

    The kernel drops the lock when its process dies, so a crash resume can always take it,
    while two live processes (e.g. two concurrent crash resumes) can never both hold it.
    """
    if fcntl is None:
        yield
        return
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ThreadBusy(f"thread is busy in another process: {lock_path.stem}") from None
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


class _Publisher:
    """Stamps one turn's events, persists them and publishes them to the sink.

    `item` events are persisted (with everything buffered before them) before they are
    published; other events are published at once and persisted in batches.
    """

    def __init__(
        self, log: SessionLog, sink: Sink | None, thread_id: str, turn_id: str, impl: str
    ) -> None:
        self._log = log
        self._sink = sink
        self._thread = thread_id
        self._head = {"v": 1, "thread": thread_id, "turn": turn_id, "impl": impl}
        self._seq = log.next_seq(thread_id)
        self._t0 = time.perf_counter_ns()
        self._batch: list[EventRow] = []
        self._open = True  # until the loop's turn.end
        # Tool events after the loop's turn.end: `{"t_us", "type", "data"}`.
        self.late: list[dict[str, Any]] = []

    def _t_us(self) -> int:
        return (time.perf_counter_ns() - self._t0) // 1000

    def emit(self, type_: str, data: dict[str, Any]) -> None:
        env = {**self._head, "seq": self._seq, "t_us": self._t_us(), "type": type_, "data": data}
        item: Item | None = data["item"] if type_ == "item" else None
        if item is not None:
            env["data"] = {**data, "item": item_to_json(item)}
        # Serialized now, so a later change to `data` cannot alter the record, and a value
        # that is not JSON fails here, at the event that carries it.
        row = event_row(env)
        if item is not None:
            self._log.append_item(self._thread, item, [*self._batch, row])
            self._batch.clear()
        else:
            self._batch.append(row)
        self._seq += 1
        if type_ == "turn.end":
            self._open = False
        if len(self._batch) >= _BATCH or type_ == "turn.end":
            self.flush()
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

    def publish(self, event: Event) -> None:
        """The ToolHost's `emit` callback. A tool event after the loop's `turn.end` cannot
        join the stream (`commit` must follow `turn.end` directly), so it goes to `late`."""
        if self._open:
            self.emit(event.type, event.data)
        else:
            self.late.append({"t_us": self._t_us(), "type": event.type, "data": event.data})

    def flush(self) -> None:
        if self._batch:
            self._log.append_events(self._batch)
            self._batch.clear()

    def close(self) -> None:
        self.flush()
        self._open = False


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
    """Runs turns of any `Loop` against the session log and one git working copy per thread."""

    def __init__(
        self,
        log: SessionLog,
        wc_root: Path,
        make_tools: MakeTools,
        sink: Sink | None = None,
    ) -> None:
        self.log = log
        self.wc_root = wc_root
        self.make_tools = make_tools
        self.sink = sink

    def workdir(self, thread_id: str) -> Path:
        """The thread's git working copy."""
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
        its git working copy."""
        thread_id = thread_id or uuid.uuid4().hex[:12]
        if not _THREAD_ID.fullmatch(thread_id):
            raise ValueError(f"thread id must match {_THREAD_ID.pattern}: {thread_id!r}")
        model_meta = asdict(model)
        del model_meta["api_key"]
        # The repository first: once the thread row exists, every turn can rely on it.
        WorkCopy(self.workdir(thread_id)).init_sync()
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
        user_text: str | None = None,
        resume: Resume | None = None,
        limits: Limits = _DEFAULT_LIMITS,
        cancel: asyncio.Event | None = None,
        watch_cancel: bool = False,
    ) -> dict[str, Any]:
        """Run one turn: a new user message, or a resume (approval or crash).

        Returns `{"turn_id", "stop", "pending", "commit"}`. A loop exception ends the turn with
        stop="error" instead of raising. `asyncio.CancelledError` propagates and leaves the turn
        "running", like a crash; resume it with `Resume(kind="crash")`. If the commit fails,
        the turn is recorded as "error" without a commit and the git error is raised.
        """
        if (user_text is None) == (resume is None):
            raise ValueError("pass exactly one of user_text and resume")
        thread = self._thread(thread_id)
        if loop.name != thread["impl"]:
            raise ValueError(f"thread {thread_id} belongs to {thread['impl']!r}, not {loop.name!r}")
        with _exclusive(self._lock_path(thread_id)):
            return await self._turn(
                loop, thread, thread_id, model, user_text, resume, limits, cancel, watch_cancel
            )

    async def _turn(
        self,
        loop: Loop,
        thread: dict[str, Any],
        thread_id: str,
        model: ModelConfig,
        user_text: str | None,
        resume: Resume | None,
        limits: Limits,
        cancel: asyncio.Event | None,
        watch_cancel: bool,
    ) -> dict[str, Any]:
        kind = resume.kind if resume else "user"
        row = self.log.start_turn(thread_id, kind)  # raises if a turn is running (unless crash)
        turn_id = row["id"]
        wc = WorkCopy(self.workdir(thread_id))
        pub = _Publisher(self.log, self.sink, thread_id, turn_id, thread["impl"])
        cancel = cancel or asyncio.Event()
        watcher = (
            asyncio.create_task(self._watch_cancel(thread_id, row["started_us"], cancel))
            if watch_cancel
            else None
        )
        try:
            if kind == "crash":
                await wc.recover(self._last_commit(thread_id))
            start: dict[str, Any] = {"turn_id": turn_id}
            if resume is not None:
                start["resume"] = asdict(resume)
            pub.emit("turn.start", start)
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
            if stop == "paused":
                pending = list(end.get("pending") or [])
                self.log.set_turn_status(turn_id, "paused", stop=stop, pending=pending)
                if pub.late:  # no commit event to carry them
                    logger.warning("turn %s: tool events after turn.end: %s", turn_id, pub.late)
                return {"turn_id": turn_id, "stop": stop, "pending": pending, "commit": None}
            try:
                sha, files = await wc.commit(f"turn {row['idx'] + 1}: {stop}")
            except Exception:
                # Record it, or the turn stays "running" and blocks the thread.
                self.log.set_turn_status(turn_id, "error", stop=stop)
                raise
            self.log.set_turn_status(turn_id, _STATUS.get(stop, "error"), stop=stop, commit_sha=sha)
            commit: dict[str, Any] = {"sha": sha, "files": files}
            if pub.late:  # evidence for the checks (I2) without breaking rule 7
                commit["late"] = pub.late
            # Last, so a consumer that sees `commit` finds the turn row complete (rule 7).
            pub.emit("commit", commit)
            return {"turn_id": turn_id, "stop": stop, "pending": [], "commit": sha}
        finally:
            if watcher is not None:
                watcher.cancel()
            pub.close()

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
                    pub.emit(event.type, event.data)
                    if event.type == "turn.end":
                        end = event.data
                        break
        except Exception as exc:  # CancelledError is not an Exception: it propagates (crash)
            if end is None:  # else it failed while closing after its turn.end: nothing to add
                error = f"{type(exc).__name__}: {exc}"
                pub.emit("error", {"kind": "loop", "message": error, "retryable": False})
        if end is None:
            end = {"stop": "error", "steps": len(steps), "error": error}
            pub.emit("turn.end", end)
        return end

    async def revert(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        """Undo a committed turn's changes with a new commit, recorded as a new "revert" turn.

        Appends a runner item telling the model what was reverted. Returns the same summary
        shape as `turn()`. If git fails (e.g. a conflict), no turn is recorded.
        """
        thread = self._thread(thread_id)
        target = next((t for t in self.log.turns(thread_id) if t["id"] == turn_id), None)
        if target is None or not target["commit_sha"]:
            raise ValueError(f"turn {turn_id} of thread {thread_id} has no commit to revert")
        with _exclusive(self._lock_path(thread_id)):
            # The running turn row keeps other turns out while git works.
            row = self.log.start_turn(thread_id, "revert")
            try:
                sha, files = await WorkCopy(self.workdir(thread_id)).revert(target["commit_sha"])
            except Exception:
                self.log.discard_turn(row["id"])  # it recorded nothing yet
                raise
        pub = _Publisher(self.log, self.sink, thread_id, row["id"], thread["impl"])
        note = f"[harness] Reverted turn {target['idx'] + 1}; files: {', '.join(files) or 'none'}"
        message = {"role": "user", "content": note}
        pub.emit("item", {"item": Item(f"{row['id']}:revert", row["id"], message)})
        self.log.set_turn_status(row["id"], "done", commit_sha=sha)
        pub.emit("commit", {"sha": sha, "files": files})
        pub.close()
        return {"turn_id": row["id"], "stop": None, "pending": [], "commit": sha}

    def compact(self, thread_id: str, summary: str) -> Item:
        """Append a compaction item (contract rule 8) as its own "compact" turn.

        Compaction changes no files, so the turn has no commit.
        """
        thread = self._thread(thread_id)
        row = self.log.start_turn(thread_id, "compact")
        item = Item(
            id=f"{row['id']}:compact",
            turn_id=row["id"],
            message={"role": "user", "content": f"{SUMMARY_PREFIX} {summary}"},
            compaction=True,
        )
        pub = _Publisher(self.log, self.sink, thread_id, row["id"], thread["impl"])
        pub.emit("item", {"item": item})
        pub.close()
        self.log.set_turn_status(row["id"], "done")
        return item

    def _lock_path(self, thread_id: str) -> Path:
        # Next to the working copy, not inside it, so it is never committed.
        return self.workdir(thread_id).parent / f"{thread_id}.lock"

    def _thread(self, thread_id: str) -> dict[str, Any]:
        thread = self.log.get_thread(thread_id)
        if thread is None:
            raise KeyError(f"unknown thread {thread_id}")
        return thread

    def _last_commit(self, thread_id: str) -> str | None:
        shas = [t["commit_sha"] for t in self.log.turns(thread_id) if t["commit_sha"]]
        return shas[-1] if shas else None

    async def _watch_cancel(self, thread_id: str, since_us: int, cancel: asyncio.Event) -> None:
        # Polls because the request may come from another process (`bakeoff cancel`).
        while not cancel.is_set():
            if self.log.cancel_requested(thread_id, since_us):
                cancel.set()
            else:
                await asyncio.sleep(_CANCEL_POLL_S)

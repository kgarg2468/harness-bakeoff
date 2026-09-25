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
_LOCK_POLL_S = 0.02
_STATUS = {"end_turn": "done", "max_steps": "done", "budget": "done", "cancelled": "cancelled"}
_DEFAULT_LIMITS = Limits()
_THREAD_ID = re.compile(r"[\w-]+")
# A revert waits while the last turn has changes that no commit holds (see `Runner.revert`).
_REVERT_WAITS = {
    "paused": "is paused: resolve the pending approval first",
    "error": "failed to commit its changes: run a turn first, its commit includes them",
}
logger = logging.getLogger(__name__)

# Starts the content of a compaction item; see contract rule 8.
SUMMARY_PREFIX = "[harness] Conversation summary:"

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


@contextmanager
def _logged_failure(what: str) -> Iterator[None]:
    """Log a failure instead of raising it, while another error is on its way up."""
    try:
        yield
    except Exception:
        logger.exception("%s failed", what)


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
        self.ended = False  # the loop's turn.end is stamped (or the publisher is closed)
        # Tool events after the loop's turn.end, `{"t_us", "type", "data"}`; kept on the turn row.
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
            self.ended = True
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
        join the stream (`commit` must follow `turn.end` directly), so it goes to `late`, which
        the runner stores on the turn row."""
        if not self.ended:
            self.emit(event.type, event.data)
        else:
            self.late.append({"t_us": self._t_us(), "type": event.type, "data": event.data})

    def flush(self) -> None:
        if self._batch:
            self._log.append_events(self._batch)
            self._batch.clear()

    def close(self) -> None:
        """Store what is still buffered. Nothing is stored after this, even if it fails."""
        self.ended = True
        batch, self._batch = self._batch, []
        if batch:
            self._log.append_events(batch)


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
    """Runs turns of any `Loop` against the session log and one git working copy per thread.

    Each turn and revert holds the thread's OS lock (see `_try_lock`) while it runs.
    A resume (approval or crash) waits up to `lock_wait_s` seconds for it: it follows the end of
    the turn before it, whose worker, or that worker's last git process, may still hold it.
    Anything else raises ThreadBusy at once.
    """

    def __init__(
        self,
        log: SessionLog,
        wc_root: Path,
        make_tools: MakeTools,
        sink: Sink | None = None,
        lock_wait_s: float = 5.0,
    ) -> None:
        self.log = log
        self.wc_root = wc_root
        self.make_tools = make_tools
        self.sink = sink
        self.lock_wait_s = lock_wait_s

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
        "running", like a crash; resume it with `Resume(kind="crash")`. Any other failure (git,
        or the session log) records the turn as "error" without a commit and is raised; the
        next turn's commit then includes its changes.
        """
        if (user_text is None) == (resume is None):
            raise ValueError("pass exactly one of user_text and resume")
        thread = self._thread(thread_id)
        if loop.name != thread["impl"]:
            raise ValueError(f"thread {thread_id} belongs to {thread['impl']!r}, not {loop.name!r}")
        wait_s = self.lock_wait_s if resume is not None else 0.0
        lock_fd = await _wait_lock(self._lock_path(thread_id), wait_s)
        try:
            return await self._turn(
                loop,
                thread,
                thread_id,
                model,
                user_text,
                resume,
                limits,
                cancel,
                watch_cancel,
                lock_fd,
            )
        finally:
            _unlock(lock_fd)

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
        lock_fd: int | None,
    ) -> dict[str, Any]:
        kind = resume.kind if resume else "user"
        row = self.log.start_turn(thread_id, kind)  # raises if a turn is running (unless crash)
        turn_id = row["id"]
        wc = WorkCopy(self.workdir(thread_id), lock_fd)
        pub = _Publisher(self.log, self.sink, thread_id, turn_id, thread["impl"])
        cancel = cancel or asyncio.Event()
        watcher = (
            asyncio.create_task(self._watch_cancel(thread_id, row["started_us"], cancel))
            if watch_cancel
            else None
        )
        stop = "error"  # until the loop's turn.end says otherwise
        try:
            await self._reconcile(wc, thread_id, turn_id)
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
                self.log.set_turn_status(
                    turn_id, "paused", stop=stop, pending=pending, late=pub.late or None
                )
                return {"turn_id": turn_id, "stop": stop, "pending": pending, "commit": None}
            sha, files = await wc.commit(f"turn {row['idx'] + 1}: {stop}")
            status = _STATUS.get(stop, "error")
            self.log.set_turn_status(
                turn_id, status, stop=stop, commit_sha=sha, late=pub.late or None
            )
            # Last, so a consumer that sees `commit` finds the turn row complete (rule 7).
            pub.emit("commit", {"sha": sha, "files": files})
            return {"turn_id": turn_id, "stop": stop, "pending": [], "commit": sha}
        except Exception:  # unlike a crash (CancelledError), a failure ends the turn
            self._record_failure(turn_id, stop, pub)
            raise
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
            if end is None and pub.ended:
                raise  # the loop's turn.end is stamped, but the log could not store it
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
        shape as `turn()`. If git fails (e.g. a conflict), no turn is recorded. It refuses while
        the last turn is paused or failed to commit: that turn's changes are not committed, so
        the revert would take them into its own commit, or lose them if git aborts it.
        """
        thread = self._thread(thread_id)
        target = next((t for t in self.log.turns(thread_id) if t["id"] == turn_id), None)
        if target is None or not target["commit_sha"]:
            raise ValueError(f"turn {turn_id} of thread {thread_id} has no commit to revert")
        # Held until the revert is fully recorded: while its row says "running", a crash resume
        # that got the lock would take git's revert commit for one that no turn recorded.
        with _exclusive(self._lock_path(thread_id)) as lock_fd:
            last = self._git_turns(thread_id)[-1]  # there is one: the target
            if last["status"] in _REVERT_WAITS and not last["commit_sha"]:
                raise RuntimeError(
                    f"cannot revert: turn {last['id']} {_REVERT_WAITS[last['status']]}"
                )
            row = self.log.start_turn(thread_id, "revert")
            try:
                wc = WorkCopy(self.workdir(thread_id), lock_fd)
                sha, files = await wc.revert(target["commit_sha"])
            except Exception:
                self.log.discard_turn(row["id"])  # it recorded nothing yet
                raise
            pub = _Publisher(self.log, self.sink, thread_id, row["id"], thread["impl"])
            files_text = ", ".join(files) or "none"
            note = f"[harness] Reverted turn {target['idx'] + 1}; files: {files_text}"
            message = {"role": "user", "content": note}
            try:
                pub.emit("item", {"item": Item(f"{row['id']}:revert", row["id"], message)})
                self.log.set_turn_status(row["id"], "done", commit_sha=sha)
                pub.emit("commit", {"sha": sha, "files": files})
                pub.close()
            except Exception:
                self._record_failure(row["id"], None, pub)
                raise
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

    def _record_failure(self, turn_id: str, stop: str | None, pub: _Publisher) -> None:
        """End a failed turn as "error" without a commit, or it would stay "running" and block
        the thread (the next turn repairs git). Best effort: if the log fails again (e.g. it is
        still locked), that is only logged, so the caller raises the first error; a crash
        resume can still take the turn over."""
        with _logged_failure(f"recording turn {turn_id} as an error"):
            self.log.set_turn_status(turn_id, "error", stop=stop, late=pub.late or None)
        with _logged_failure(f"storing the events of turn {turn_id}"):
            pub.close()  # e.g. a turn.end whose flush failed

    def _git_turns(self, thread_id: str) -> list[dict[str, Any]]:
        """The thread's turns that use the working copy (all but compactions), in order."""
        return [t for t in self.log.turns(thread_id) if t["kind"] != "compact"]

    async def _reconcile(self, wc: WorkCopy, thread_id: str, turn_id: str) -> None:
        """Before turn `turn_id` uses git: if the turn before it died ("running") or failed to
        commit ("error" without a sha), git may hold a commit that no turn recorded or a stale
        lock. Move HEAD back to the last recorded commit; the changes after it stay staged, so
        this turn's commit includes them."""
        turns = [t for t in self._git_turns(thread_id) if t["id"] != turn_id]
        if turns and turns[-1]["status"] in ("running", "error") and not turns[-1]["commit_sha"]:
            shas = [t["commit_sha"] for t in turns if t["commit_sha"]]
            await wc.recover(shas[-1] if shas else None)

    async def _watch_cancel(self, thread_id: str, since_us: int, cancel: asyncio.Event) -> None:
        # Polls because the request may come from another process (`bakeoff cancel`).
        while not cancel.is_set():
            if self.log.cancel_requested(thread_id, since_us):
                cancel.set()
            else:
                await asyncio.sleep(_CANCEL_POLL_S)

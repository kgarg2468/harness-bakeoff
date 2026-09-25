"""Scenario driver: runs a fakeprov scenario against one loop and judges it.

It uses the real `Runner`, `SessionLog`, `WorkCopy` and `ToolHost` (on `MockEngine`) against the
fake provider. Steps that need another process (`approve` with `new_process`, and the user turn
after `crash_after`) run `python -m bakeoff.cli ...` as real OS processes; a crash is a SIGKILL
the child sends itself from its runner's sink. Pass or fail comes only from the wire recordings,
the session log and the working copy (DESIGN.md, "Scenarios").

Output layout (the report reads it):

    out/runs/<run_id>/summary.json                     the matrix, git sha, loop versions
    out/runs/<run_id>/<scenario>/<impl>/log.sqlite     the SessionLog
                                        wire/          fakeprov's recordings of the requests
                                        events.ndjson  every published event, all processes
                                        wc/            the git working copy root
                                        result.json    see `run_scenario`
    out/runs/latest -> <run_id>
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from bakeoff import loops
from bakeoff.fakeprov.script import SCENARIOS_DIR, Scenario, load_scenario
from bakeoff.fakeprov.server import FakeProvider
from bakeoff.shared.contract import Event, Limits, Loop, ModelConfig, Resume, ToolHost
from bakeoff.shared.engine.mock import MockEngine
from bakeoff.shared.invariants import (
    Check,
    check_commits,
    check_prefix,
    check_seq,
    check_tool_results,
    load_wire,
)
from bakeoff.shared.runner import NdjsonMirror, Runner, Sink
from bakeoff.shared.sessionlog import SessionLog
from bakeoff.shared.toolhost import build_toolhost
from bakeoff.shared.workcopy import GIT_CONFIG, git_env

RESULT_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
LOOP_TURNS = ("user", "approval", "crash")  # turn kinds that run a loop (not revert/compact)
MODEL_TIMEOUT_S = 30.0  # a scenario request that hangs longer than this is a failure anyway
CHILD_TIMEOUT_S = 60.0
STRAY_WAIT_S = 2.0  # how long tasks a loop left behind get to finish once cancelled
_ONE_LINE = 160


class DriverError(RuntimeError):
    """A driver step could not be carried out (the scenario fails with this as its error)."""


# --- output capture (I5) ---------------------------------------------------------------------


@dataclass(slots=True)
class Captured:
    """What was written to stdout/stderr while `capture_output` was active."""

    stdout: str = ""
    stderr: str = ""


@contextmanager
def capture_output() -> Iterator[Captured]:
    """Capture everything this process writes to stdout/stderr: through `sys.stdout`/`sys.stderr`
    and straight to file descriptors 1 and 2 (C code, subprocesses, other threads)."""
    captured = Captured()
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        if stream is not None:
            stream.flush()
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        saved = (os.dup(1), os.dup(2))
        streams = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        try:
            yield captured
        finally:
            text_out, text_err = sys.stdout.getvalue(), sys.stderr.getvalue()  # type: ignore[attr-defined]
            sys.stdout, sys.stderr = streams
            for stream in (sys.__stdout__, sys.__stderr__):  # buffered writes to the real fds
                if stream is not None:
                    stream.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)
            out.seek(0)
            err.seek(0)
            captured.stdout = text_out + out.read().decode(errors="replace")
            captured.stderr = text_err + err.read().decode(errors="replace")


def check_quiet(stdout: str, stderr: str) -> Check:
    """I5: the loop wrote nothing to stdout/stderr."""
    info = {"stdout": stdout[:2000], "stderr": stderr[:2000]}
    if not stdout and not stderr:
        return Check("I5", True, "nothing written to stdout/stderr", info)
    written = [f"{len(t)} chars to {n}" for n, t in (("stdout", stdout), ("stderr", stderr)) if t]
    first = (stdout or stderr).strip().splitlines() or [""]
    return Check("I5", False, f"{', '.join(written)}: {first[0][:120]!r}", info)


# --- sessions: one directory with log.sqlite, wc/ and events.ndjson -------------------------


class Workspace:
    """A run directory: the session log, the working copies and the event mirror.

    `sinks` get every published event after the mirror has it (so a crash in a sink never loses
    the event from events.ndjson). Paths default to the layout above, relative to `db`'s folder.
    """

    def __init__(
        self,
        db: Path,
        *,
        wc: Path | None = None,
        events: Path | None = None,
        engine_delay_ms: int = 0,
        sinks: Sequence[Sink] = (),
    ) -> None:
        self.root = db.parent
        self.log = SessionLog(db)
        self.mirror = NdjsonMirror(events or self.root / "events.ndjson")
        self.engine_delay_ms = engine_delay_ms
        self.sinks = list(sinks)
        self.runner = Runner(self.log, wc or self.root / "wc", self._tools, sink=self._publish)

    def _tools(
        self, workdir: Path, rules: dict[str, Any], emit: Callable[[Event], None]
    ) -> ToolHost:
        return build_toolhost(workdir, rules, emit, MockEngine(delay_ms=self.engine_delay_ms))

    def _publish(self, envelope: dict[str, Any]) -> None:
        self.mirror(envelope)
        for sink in self.sinks:
            sink(envelope)

    def thread(self, thread_id: str) -> dict[str, Any]:
        """The thread row; raises DriverError for an unknown thread."""
        thread = self.log.get_thread(thread_id)
        if thread is None:
            raise DriverError(f"unknown thread {thread_id} in {self.log_path}")
        return thread

    @property
    def log_path(self) -> Path:
        return self.root / "log.sqlite"

    def close(self) -> None:
        self.mirror.close()
        self.log.close()


def crash_sink(event_type: str, call_id: str | None = None) -> Sink:
    """A sink that SIGKILLs this process at the first published event of `event_type` (about
    `call_id`, if given; see fakeprov/README.md, "Crash points"). The runner publishes an `item`,
    `tool.start` or `turn.end` only after storing it, so the crash comes right after it is
    durable and before the loop can run again."""

    def sink(envelope: dict[str, Any]) -> None:
        if envelope["type"] != event_type:
            return
        data = envelope["data"]
        if call_id is not None:
            about = (
                data["item"]["message"].get("tool_call_id")
                if event_type == "item"
                else data.get("call_id")
            )
            if about != call_id:
                return
        os.kill(os.getpid(), signal.SIGKILL)

    return sink


def model_from_meta(meta: dict[str, Any], api_key: str) -> ModelConfig:
    """The thread's ModelConfig (its meta keeps everything but the key)."""
    return ModelConfig(**meta, api_key=api_key)


def pending_calls(log: SessionLog, thread_id: str) -> list[str]:
    """The call ids the thread's paused turn waits for; raises if its last turn is not paused."""
    last = log.last_turn(thread_id)
    if last is None or last["status"] != "paused":
        state = "no turns" if last is None else f"last turn {last['id']} is {last['status']}"
        raise DriverError(f"thread {thread_id} has no paused turn ({state})")
    return list(last["pending"] or [])


def decide(
    pending: Sequence[str], allow: Sequence[str] | Literal["all"], deny: Sequence[str]
) -> dict[str, Literal["allow", "deny"]]:
    """The approval decisions for the pending calls. Every pending call needs an answer, and
    every answered call must be pending (a typo must not silently deny or allow nothing)."""
    decisions: dict[str, Literal["allow", "deny"]] = {}
    for call_id in pending:
        if call_id in deny:
            decisions[call_id] = "deny"
        elif allow == "all" or call_id in allow:
            decisions[call_id] = "allow"
    named = [*deny, *([] if allow == "all" else allow)]
    if unknown := [c for c in named if c not in pending]:
        raise DriverError(f"not pending: {unknown} (pending: {list(pending)})")
    if unanswered := [c for c in pending if c not in decisions]:
        raise DriverError(f"no decision for pending calls {unanswered}")
    return decisions


async def worker_turn(
    db: Path,
    thread_id: str,
    *,
    api_key: str,
    user_text: str | None = None,
    resume: Resume | None = None,
    max_steps: int = Limits().max_steps,
    engine_delay_ms: int = 0,
    wc: Path | None = None,
    events: Path | None = None,
    sinks: Sequence[Sink] = (),
) -> dict[str, Any]:
    """Run one turn of an existing thread in this process (the `bakeoff turn/approve/resume`
    workers). The loop comes from the registry by the thread's impl; the model from its meta.
    A `bakeoff cancel` from another process stops it. Returns the runner's summary plus the pid.
    """
    ws = Workspace(db, wc=wc, events=events, engine_delay_ms=engine_delay_ms, sinks=sinks)
    try:
        thread = ws.thread(thread_id)
        loop = loops.load(thread["impl"])()
        try:
            summary = await ws.runner.turn(
                loop,
                thread_id,
                model=model_from_meta(thread["meta"]["model"], api_key),
                user_text=user_text,
                resume=resume,
                limits=Limits(max_steps=max_steps),
                watch_cancel=True,
            )
        finally:
            await loop.aclose()
    finally:
        ws.close()
    return {"pid": os.getpid(), "thread": thread_id, **summary}


# --- running one scenario ----------------------------------------------------------------------


def scenario_ids(scenarios_dir: Path = SCENARIOS_DIR) -> list[str]:
    """Every scenario id, in order (S01 ... S15)."""
    return sorted(p.stem for p in scenarios_dir.glob("*.json"))


def new_run_id() -> str:
    """A sortable, unique run id, e.g. 20260925-141502-3f9a."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


@dataclass(slots=True)
class CancelMark:
    """When the driver cancelled one turn (a `cancel_after_ms` step), for `cancel_within_ms`.

    `after_start_us` is measured on the driver's clock from the moment the turn's `turn.start`
    was published, which is when the runner stamped it; the log's `t_us` gives the rest."""

    turn: str | None = None  # None: the turn never started
    started: float = 0.0  # perf_counter() at turn.start
    after_start_us: int | None = None  # None: the turn ended before the cancel fired


class _ScenarioRun:
    """The state of one scenario run while its driver steps execute."""

    loop: Loop  # set by `drive`

    def __init__(
        self, sc: Scenario, impl: str, run_id: str, directory: Path, provider: FakeProvider
    ) -> None:
        self.sc, self.impl, self.run_id, self.dir = sc, impl, run_id, directory
        self.thread_id = f"{sc.id}-{impl}"
        self.limits = Limits(max_steps=sc.limits["max_steps"])
        # A fresh fakeprov cursor for every run: one provider may serve many runs (the test
        # suite shares one), even with the same run id under another `out`. Each starts at the
        # first exchange. Its recordings move to `wire/` afterwards.
        self.cursor = f"{run_id}.{uuid.uuid4().hex[:6]}"
        self.recorded = provider.wire_dir / sc.id / self.cursor / impl
        self.model = ModelConfig(
            base_url=provider.base_url(sc.id, self.cursor, impl),  # claims the recording folder
            model=sc.model["model"],
            kind=sc.model["kind"],
            reasoning=sc.model.get("reasoning"),
            compat=sc.model.get("compat") or {},
            timeout_s=MODEL_TIMEOUT_S,
        )
        self.ws = Workspace(
            directory / "log.sqlite", engine_delay_ms=sc.engine["delay_ms"], sinks=[self._on_event]
        )
        # I5: what this process wrote while the loop existed, then each child's extra output.
        self.stdout: list[str] = []
        self.stderr: list[str] = []
        self.processes: list[dict[str, Any]] = []
        self.cancels: list[CancelMark] = []
        self._on_start: Callable[[str], None] | None = None  # arms a step's cancel timer

    def _on_event(self, envelope: dict[str, Any]) -> None:
        if envelope["type"] == "turn.start" and self._on_start is not None:
            self._on_start(envelope["turn"])
            self._on_start = None

    async def drive(self, loop: Loop) -> None:
        """Create the thread and perform every driver step, in order."""
        self.loop = loop
        self.ws.runner.new_thread(
            impl=loop.name,
            system=self.sc.system,
            rules=self.sc.rules,
            model=self.model,
            thread_id=self.thread_id,
        )
        crash: dict[str, Any] | None = None
        for n, step in enumerate(self.sc.driver, 1):
            try:
                if "crash_after" in step:
                    crash = step
                elif "user" in step and crash is not None:
                    args = ["--user", step["user"], "--crash-after", crash["crash_after"]]
                    if "call_id" in crash:
                        args += ["--call-id", crash["call_id"]]
                    await self._child(n, "turn", args, killed=True)
                    crash = None
                elif "user" in step:
                    await self._turn(user_text=step["user"], cancel_ms=step.get("cancel_after_ms"))
                elif "approve" in step:
                    await self._approve(n, step)
                elif "resume" in step:
                    await self._turn(resume=Resume(kind="crash"))
                elif "revert" in step:
                    turn = self.ws.log.turns(self.thread_id)[step["revert"] - 1]
                    await self.ws.runner.revert(self.thread_id, turn["id"])
                elif "compact" in step:
                    self.ws.runner.compact(self.thread_id, step["compact"])
            except Exception as exc:
                raise DriverError(f"step {n} {json.dumps(step)[:80]}: {exc}") from exc

    async def _turn(
        self,
        *,
        user_text: str | None = None,
        resume: Resume | None = None,
        cancel_ms: int | None = None,
    ) -> None:
        cancel = asyncio.Event()
        timers: list[asyncio.TimerHandle] = []
        if cancel_ms is not None:
            aio = asyncio.get_running_loop()
            mark = CancelMark()
            self.cancels.append(mark)

            def fire() -> None:
                mark.after_start_us = round((time.perf_counter() - mark.started) * 1e6)
                cancel.set()

            def arm(turn_id: str) -> None:
                mark.turn, mark.started = turn_id, time.perf_counter()
                timers.append(aio.call_later(cancel_ms / 1000, fire))

            self._on_start = arm
        try:
            await self.ws.runner.turn(
                self.loop,
                self.thread_id,
                model=self.model,
                user_text=user_text,
                resume=resume,
                limits=self.limits,
                cancel=cancel,
            )
        finally:
            self._on_start = None
            for timer in timers:
                timer.cancel()

    async def _approve(self, n: int, step: dict[str, Any]) -> None:
        answer = step["approve"]
        pending = pending_calls(self.ws.log, self.thread_id)
        decisions = decide(pending, answer.get("allow", []), answer.get("deny", []))
        reason = answer.get("reason")
        if not step.get("new_process"):
            await self._turn(resume=Resume(kind="approval", decisions=decisions, reason=reason))
            return
        args = [f"--{d}={c}" for c, d in decisions.items()]
        if reason is not None:
            args.append(f"--reason={reason}")
        await self._child(n, "approve", args, killed=False)

    async def _child(self, n: int, command: str, args: list[str], *, killed: bool) -> None:
        """Run `bakeoff <command>` for this thread as a separate OS process.

        The worker prints one JSON summary line and nothing else; anything more on stdout or
        stderr is the loop's output (I5). A `killed` child must die of SIGKILL at its crash point.
        """
        argv = [
            sys.executable,
            "-m",
            "bakeoff.cli",
            command,
            self.thread_id,
            f"--db={self.ws.log_path}",
            f"--max-steps={self.limits.max_steps}",
            f"--engine-delay-ms={self.sc.engine['delay_ms']}",
            *args,
        ]
        # Unbuffered, so what the child writes before a SIGKILL still reaches the pipe.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), CHILD_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise DriverError(f"{command} process {proc.pid} timed out") from None
        out, err = raw_out.decode(errors="replace"), raw_err.decode(errors="replace")
        record: dict[str, Any] = {
            "step": n,
            "command": command,
            "pid": proc.pid,
            "exit": proc.returncode,
        }
        self.processes.append(record)
        if killed:
            self.stdout.append(out)
            self.stderr.append(err)
            if proc.returncode != -signal.SIGKILL:
                raise DriverError(
                    f"{command} process {proc.pid} was to die of SIGKILL at its crash point,"
                    f" but exited with {proc.returncode}: {err.strip()[-300:]}"
                )
            return
        *noise, last = out.splitlines() or [""]
        self.stdout.append("\n".join(noise))
        self.stderr.append(err)
        if proc.returncode != 0:
            raise DriverError(
                f"{command} process {proc.pid} exited with {proc.returncode}: {err.strip()[-300:]}"
            )
        record["summary"] = json.loads(last)


async def run_scenario(
    scenario: str | Path,
    impl: str,
    *,
    out: Path,
    run_id: str,
    provider: FakeProvider,
    loop_factory: Callable[[], Loop] | None = None,
) -> dict[str, Any]:
    """Run one scenario (an id in fakeprov/scenarios, or a path) for one loop, judge it, and
    write `out/runs/<run_id>/<scenario>/<impl>/result.json`. Returns the result:

    `{"v", "run_id", "scenario", "title", "impl", "stops", "expect": {key: {"ok", "detail"}},
    "invariants": {"I1"|"I2"|"I3"|"I5"|"I7": {"ok", "detail", "info"}}, "requests", "tool_runs",
    "usage", "duration_ms", "passed", "error", "thread", "processes"}`.

    `provider` serves the scenario (its folder must hold it); `loop_factory` replaces the
    registry's loop class (child processes still use the registry, by the loop's name).
    Raises DriverError if that directory exists: a run id is never reused.
    """
    path = (
        Path(scenario)
        if str(scenario).endswith(".json")
        else provider.scenarios_dir / f"{scenario}.json"
    )
    sc = load_scenario(path)
    directory = out / "runs" / run_id / sc.id / impl
    try:
        directory.mkdir(parents=True)  # never over an earlier run: that would erase its results
    except FileExistsError:
        raise DriverError(f"{directory} already exists: pick another --run-id") from None
    started = time.perf_counter()
    run = _ScenarioRun(sc, impl, run_id, directory, provider)
    captured = Captured()
    try:
        # I5 covers the loop's whole life, so output from its background tasks and threads
        # counts too, also while the driver waits for a child process. The driver itself
        # prints nothing here, and children write to their own pipes.
        with capture_output() as captured:
            factory = loop_factory or (lambda: loops.load(impl)())  # a load error is the error
            error = await _drive_and_close(run, factory)
    finally:
        run.stdout.insert(0, captured.stdout)
        run.stderr.insert(0, captured.stderr)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    wire = directory / "wire"
    if run.recorded.exists():
        shutil.move(run.recorded, wire)
    wire.mkdir(exist_ok=True)
    try:
        result = judge(run, wire)
    finally:
        run.ws.close()
    result |= {"duration_ms": duration_ms, "error": error}
    result["passed"] = error is None and result["passed"]
    write_json(directory / "result.json", result)
    return result


async def _drive_and_close(run: _ScenarioRun, factory: Callable[[], Loop]) -> str | None:
    """Create the loop, run the scenario's steps, close the loop. Returns the run's error, if
    any: a failure of the loop or of a driver step is the scenario's result, not a crash of the
    whole matrix."""
    errors: list[str] = []
    loop: Loop | None = None
    before = asyncio.all_tasks()
    try:
        loop = factory()
        await run.drive(loop)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    except asyncio.CancelledError:
        if _cancelled_from_outside():
            raise
        errors.append("CancelledError: a turn raised CancelledError instead of ending")
    finally:
        if loop is not None:
            try:
                await loop.aclose()
            except Exception as exc:
                errors.append(f"aclose: {type(exc).__name__}: {exc}")
            except asyncio.CancelledError:
                if _cancelled_from_outside():
                    raise
                errors.append("aclose: CancelledError")
        if strays := await _stop_strays(before):
            errors.append(f"tasks still running after aclose: {strays}")
    return "; ".join(errors) or None


async def _stop_strays(before: set[asyncio.Task[Any]]) -> list[str]:
    """Cancel the tasks the loop left running after `aclose` (e.g. a tool it shielded from a
    cancel), give them a moment to finish, and name them. Their last events and output then
    land in this scenario's log and I5 capture, not in the next scenario's, which may be
    another loop's."""
    current = asyncio.current_task()
    strays = [t for t in asyncio.all_tasks() - before if t is not current and not t.done()]
    for task in strays:
        task.cancel()
    if strays:
        await asyncio.wait(strays, timeout=STRAY_WAIT_S)
    return [getattr(t.get_coro(), "__qualname__", t.get_name()) for t in strays]


def _cancelled_from_outside() -> bool:
    """Whether someone cancelled this task (Ctrl-C, a timeout): that CancelledError must
    propagate. One that a loop lets escape by itself only fails the scenario."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


# --- judging -----------------------------------------------------------------------------------


@dataclass(slots=True)
class Observed:
    """What a scenario run left behind, read from the log, the wire and the working copy."""

    stops: list[str]
    tool_runs: dict[str, int]
    requests: int
    commits: int | None
    last_text: str
    usage: list[dict[str, Any]]  # the data of every usage event
    workdir: Path
    bodies: list[bytes] = field(default_factory=list)
    # call id -> one (turn, tool.start t_us, tool.end t_us or None) per run, from the events
    tool_spans: dict[str, list[ToolSpan]] = field(default_factory=dict)
    # per cancelled turn: (turn id, ms from the cancel to turn.end); ms is 0.0 if the turn
    # ended before its cancel fired, None if it never started or never ended
    cancels: list[tuple[str | None, float | None]] = field(default_factory=list)


ToolSpan = tuple[str, int, int | None]


def observe(
    log: SessionLog,
    thread_id: str,
    wire: Path,
    workdir: Path,
    cancels: Sequence[CancelMark] = (),
) -> Observed:
    """Collect the facts the final `expect` and the invariants are judged on."""
    turns, events = log.turns(thread_id), log.events(thread_id)
    starts = [e["data"].get("call_id") for e in events if e["type"] == "tool.start"]
    starts += [
        x["data"].get("call_id")
        for t in turns
        for x in t.get("late") or []
        if x["type"] == "tool.start"
    ]
    loop_turns = [t for t in turns if t["kind"] in LOOP_TURNS]
    last_text = ""
    if loop_turns:
        last_id = loop_turns[-1]["id"]
        texts = [
            _text(i.message.get("content"))
            for i in log.items(thread_id)
            if i.turn_id == last_id and i.message.get("role") == "assistant"
        ]
        last_text = next((t for t in reversed(texts) if t), "")
    bodies = [body for body, _ in load_wire(wire)]
    return Observed(
        stops=[t["stop"] for t in loop_turns if t["status"] != "running"],
        tool_runs=dict(Counter(starts)),
        requests=len(bodies),
        commits=_commit_count(workdir),
        last_text=last_text,
        usage=[e["data"] for e in events if e["type"] == "usage"],
        workdir=workdir,
        bodies=bodies,
        tool_spans=tool_spans(events),
        cancels=[cancel_latency(events, mark) for mark in cancels],
    )


def tool_spans(events: Sequence[dict[str, Any]]) -> dict[str, list[ToolSpan]]:
    """Each call's runs as (turn, start t_us, end t_us): a `tool.end` closes the open run of
    its call in its turn. A run whose `tool.end` never joined its turn's events has no end."""
    spans: dict[str, list[ToolSpan]] = {}
    for e in events:
        call_id = e["data"].get("call_id")
        if e["type"] == "tool.start":
            spans.setdefault(call_id, []).append((e["turn"], e["t_us"], None))
        elif e["type"] == "tool.end":
            runs = spans.get(call_id, [])
            for i in reversed(range(len(runs))):
                if runs[i][0] == e["turn"] and runs[i][2] is None:
                    runs[i] = (runs[i][0], runs[i][1], e["t_us"])
                    break
    return spans


def cancel_latency(
    events: Sequence[dict[str, Any]], mark: CancelMark
) -> tuple[str | None, float | None]:
    """(turn, ms from the driver's cancel to the loop's turn.end) for one cancelled turn."""
    if mark.turn is None:
        return None, None
    start = next((e["t_us"] for e in events if e["turn"] == mark.turn), None)
    end = next(
        (e["t_us"] for e in events if e["turn"] == mark.turn and e["type"] == "turn.end"), None
    )
    if start is None or end is None:
        return mark.turn, None
    if mark.after_start_us is None:  # the turn ended first; the timer never fired
        return mark.turn, 0.0
    return mark.turn, max(0.0, (end - start - mark.after_start_us) / 1000)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _commit_count(workdir: Path) -> int | None:
    """Commits after the working copy's empty initial one (None if there is no repository)."""
    if not (workdir / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", *GIT_CONFIG, "rev-list", "--count", "HEAD"],
        cwd=workdir,
        env=git_env(workdir),
        capture_output=True,
        text=True,
        check=False,
    )
    return int(proc.stdout) - 1 if proc.returncode == 0 else None


def usage_totals(usage: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Token and cost totals over usage events; cost_source is the common source, "mixed" if
    they differ, "none" if there were none."""
    totals: dict[str, Any] = {
        key: sum(u.get(key) or 0 for u in usage)
        for key in ("input_tokens", "output_tokens", "cached_tokens")
    }
    totals["cost_usd"] = round(sum(u.get("cost_usd") or 0.0 for u in usage), 12)
    sources = {u.get("cost_source") for u in usage}
    totals["cost_source"] = sources.pop() if len(sources) == 1 else "mixed" if sources else "none"
    return totals


def evaluate_expect(expect: dict[str, Any], obs: Observed) -> dict[str, dict[str, Any]]:
    """Judge every key of a scenario's final `expect` (fakeprov/README.md, "Final expect")."""
    return {
        key: dict(zip(("ok", "detail"), _expect(key, want, obs), strict=True))
        for key, want in expect.items()
    }


def _expect(key: str, want: Any, obs: Observed) -> tuple[bool, str]:
    def equal(got: Any) -> tuple[bool, str]:
        return (got == want, f"{got}" if got == want else f"got {got}, expected {want}")

    match key:
        case "stops":
            return equal(obs.stops)
        case "requests":
            return equal(obs.requests)
        case "commits":
            return equal(obs.commits)
        case "files":
            problems = [
                p
                for path, needle in want.items()
                if (p := _file_problem(obs.workdir, path, needle))
            ]
            return (not problems, "; ".join(problems) or f"{len(want)} files as expected")
        case "tool_runs":
            wrong = {
                c: obs.tool_runs.get(c, 0) for c, n in want.items() if obs.tool_runs.get(c, 0) != n
            }
            if wrong:
                return False, "; ".join(f"{c} ran {wrong[c]}x, expected {want[c]}x" for c in wrong)
            return True, f"{len(want)} calls ran as expected"
        case "text_contains":
            ok = want in obs.last_text
            return ok, f"{'found' if ok else 'missing'} {want!r} in {obs.last_text[:120]!r}"
        case "cost_usd":
            got = usage_totals(obs.usage)["cost_usd"]
            ok = abs(got - want) <= 1e-9
            return ok, f"{got}" if ok else f"got {got}, expected {want}"
        case "usage":
            totals = usage_totals(obs.usage)
            return equal({k: totals[k] for k in want})
        case "cost_source":
            sources = sorted({u.get("cost_source") for u in obs.usage}, key=str)
            if not sources:
                return False, "no usage events"
            return equal(sources[0] if len(sources) == 1 else sources)
        case "tools_overlap":
            return _overlap(want, obs.tool_spans)
        case "cancel_within_ms":
            if not obs.cancels:
                return False, "no turn was cancelled"
            times = [
                f"{turn or 'a turn that never started'}: "
                + ("no turn.end" if ms is None else f"{ms:.1f} ms")
                for turn, ms in obs.cancels
            ]
            ok = all(ms is not None and ms <= want for _, ms in obs.cancels)
            return ok, f"turn.end after the cancel: {', '.join(times)} (limit {want} ms)"
    return False, f"unknown expect key {key!r}"


def _overlap(call_ids: Sequence[str], spans: dict[str, list[ToolSpan]]) -> tuple[bool, str]:
    """Whether the tools of `call_ids` ran at one moment: each once, in one turn, and the last
    `tool.start` before the first `tool.end`."""
    runs = {c: spans.get(c, []) for c in call_ids}
    if wrong := [f"{c} ran {len(r)}x" for c, r in runs.items() if len(r) != 1]:
        return False, "; ".join(wrong) + ", expected once each"
    closed: dict[str, tuple[int, int]] = {}
    for call_id, [(_, start, end)] in runs.items():
        if end is not None:
            closed[call_id] = (start, end)
    if running := [c for c in runs if c not in closed]:
        return False, f"no tool.end in the turn for {running}"
    if len({turn for [(turn, _, _)] in runs.values()}) > 1:
        return False, "the calls ran in different turns"
    t0 = min(start for start, _ in closed.values())
    shown = ", ".join(
        f"{c} {(start - t0) / 1000:.0f}-{(end - t0) / 1000:.0f} ms"
        for c, (start, end) in closed.items()
    )
    last_start = max(start for start, _ in closed.values())
    first_end = min(end for _, end in closed.values())
    if last_start < first_end:
        together = (first_end - last_start) / 1000
        return True, f"all {len(closed)} running together for {together:.0f} ms: {shown}"
    return False, f"not all running at once: {shown}"


def _file_problem(workdir: Path, path: str, needle: str | None) -> str | None:
    full = workdir / path
    if needle is None:
        return f"{path} exists, expected absent" if full.exists() else None
    if not full.is_file():
        return f"{path} is missing"
    if needle not in full.read_text(encoding="utf-8", errors="replace"):
        return f"{path} lacks {needle!r}"
    return None


def check_invariants(
    log: SessionLog, thread_id: str, obs: Observed, stdout: str, stderr: str, *, wire: bool = True
) -> dict[str, Check]:
    """I1 (only with wire recordings), I2, I3, I5 and I7 for one thread."""
    items, events, turns = log.items(thread_id), log.events(thread_id), log.turns(thread_id)
    checks: dict[str, Callable[[], Check]] = {
        "I1": lambda: check_prefix(obs.bodies, items, turns),
        "I2": lambda: check_tool_results(items, events, turns),
        "I3": lambda: check_seq(events, items),
        "I5": lambda: check_quiet(stdout, stderr),
        "I7": lambda: check_commits(log, thread_id, obs.workdir),
    }
    if not wire:
        del checks["I1"]
    results = {}
    for name, check in checks.items():
        try:
            results[name] = check()
        except Exception as exc:  # a broken log must fail the check, not the driver
            results[name] = Check(name, False, f"check failed: {type(exc).__name__}: {exc}")
    return results


def judge(run: _ScenarioRun, wire: Path) -> dict[str, Any]:
    """The result of a finished scenario run (without duration and error)."""
    workdir = run.ws.runner.workdir(run.thread_id)
    obs = observe(run.ws.log, run.thread_id, wire, workdir, run.cancels)
    expect = evaluate_expect(run.sc.expect, obs)
    checks = check_invariants(
        run.ws.log, run.thread_id, obs, "".join(run.stdout), "".join(run.stderr)
    )
    invariants = {
        name: {"ok": c.ok, "detail": c.detail, "info": c.info} for name, c in checks.items()
    }
    return {
        "v": RESULT_VERSION,
        "run_id": run.run_id,
        "scenario": run.sc.id,
        "title": run.sc.title,
        "impl": run.impl,
        "stops": obs.stops,
        "expect": expect,
        "invariants": invariants,
        "requests": obs.requests,
        "tool_runs": obs.tool_runs,
        "usage": usage_totals(obs.usage),
        "passed": all(e["ok"] for e in expect.values()) and all(c.ok for c in checks.values()),
        "thread": run.thread_id,
        "processes": run.processes,
    }


def reason(result: dict[str, Any]) -> str:
    """One line on why a result failed ("ok" if it passed)."""
    if result["passed"]:
        return "ok"
    problems = [f"error: {result['error']}"] if result.get("error") else []
    problems += [f"{k}: {v['detail']}" for k, v in result.get("expect", {}).items() if not v["ok"]]
    problems += [
        f"{k}: {v['detail']}" for k, v in result.get("invariants", {}).items() if not v["ok"]
    ]
    first = problems[0].splitlines()[0] if problems else "failed"
    if len(first) > _ONE_LINE:
        first = first[: _ONE_LINE - 3] + "..."
    return first + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")


# --- the matrix ------------------------------------------------------------------------------------


def failing(result: dict[str, Any]) -> set[str]:
    """The final `expect` keys and invariants a result fails."""
    return {
        key
        for group in ("expect", "invariants")
        for key, check in result.get(group, {}).items()
        if not check["ok"]
    }


def status(result: dict[str, Any]) -> str:
    """pass; xfail: it fails exactly as documented (`loops.KnownFailure`); XPASS: a documented
    failure passed; FAIL: anything else, including a documented cell that fails differently."""
    known = loops.known_failure(result["impl"], result["scenario"])
    if result["passed"]:
        return "XPASS" if known else "pass"
    if known and result["error"] is None and failing(result) == known.checks:
        return "xfail"
    return "FAIL"


async def run_matrix(
    scenarios: Sequence[str],
    impls: Sequence[str],
    *,
    out: Path,
    run_id: str | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    scenarios_dir: Path = SCENARIOS_DIR,
) -> dict[str, Any]:
    """Run every scenario for every loop against one fake provider, then write summary.json
    and point `out/runs/latest` at the run. Returns the summary. Raises DriverError before
    anything runs if `out/runs/<run_id>` exists: a run id is never reused."""
    run_id = run_id or new_run_id()
    if (out / "runs" / run_id).exists():  # its summary.json and cells would be replaced
        raise DriverError(f"{out / 'runs' / run_id} already exists: pick another --run-id")
    staging = out / "runs" / run_id / ".wire"  # fakeprov records here; each run moves its part
    results = []
    with FakeProvider(scenarios_dir, wire_dir=staging) as provider:
        for sid in scenarios:
            for impl in impls:
                result = await run_scenario(sid, impl, out=out, run_id=run_id, provider=provider)
                results.append(result)
                if on_result is not None:
                    on_result(result)
    shutil.rmtree(staging, ignore_errors=True)
    summary = summarize(run_id, results, impls)
    write_json(out / "runs" / run_id / "summary.json", summary)
    point_latest(out / "runs", run_id)
    return summary


def summarize(
    run_id: str, results: Sequence[dict[str, Any]], impls: Sequence[str]
) -> dict[str, Any]:
    """summary.json: scenario x impl -> passed + one-line reason, plus git sha and loop versions."""
    matrix: dict[str, dict[str, Any]] = {}
    for r in results:
        known = loops.known_failure(r["impl"], r["scenario"])
        matrix.setdefault(r["scenario"], {})[r["impl"]] = {
            "passed": r["passed"],
            "status": status(r),
            "reason": reason(r),
            "expected_failure": None if known is None else known.why,
            "expected_checks": None if known is None else sorted(known.checks),
            "duration_ms": r["duration_ms"],
        }
    sha, dirty = git_state()
    return {
        "v": RESULT_VERSION,
        "run_id": run_id,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "loops": {
            impl: {
                "target": loops.REGISTRY[impl].target if impl in loops.REGISTRY else None,
                "versions": loops.versions(impl) if impl in loops.REGISTRY else {},
            }
            for impl in impls
        },
        "scenarios": {r["scenario"]: r["title"] for r in results},
        "matrix": matrix,
    }


def format_matrix(summary: dict[str, Any]) -> str:
    """The compact text matrix `bakeoff scenario` prints: one row per scenario, then why each
    cell that did not pass failed."""
    impls = list(summary["loops"])
    width = max(len(i) for i in impls) + 7
    lines = [("scenario  " + "".join(i.ljust(width) for i in impls)).rstrip()]
    notes = []
    for sid, row in summary["matrix"].items():
        cells = []
        for impl in impls:
            cell = row.get(impl)
            cells.append(("-" if cell is None else cell["status"]).ljust(width))
            if cell is not None and cell["status"] != "pass":
                why = cell["reason"]
                if cell["status"] == "XPASS":
                    why = f"passed, but documented to fail: {cell['expected_failure']}"
                elif cell["status"] == "FAIL" and cell["expected_checks"]:
                    why += f" [documented to fail only {', '.join(cell['expected_checks'])}]"
                notes.append(f"  {sid}/{impl} {cell['status']}: {why}")
        lines.append(f"{sid:<10}" + "".join(cells).rstrip())
    totals = []
    for impl in impls:
        cells = [row[impl] for row in summary["matrix"].values() if impl in row]
        counts = Counter(c["status"] for c in cells)
        totals.append(
            f"{impl} {counts['pass']}/{len(cells)} pass"
            + "".join(f", {n} {s}" for s, n in sorted(counts.items()) if s != "pass")
        )
    return "\n".join([*lines, *notes, "; ".join(totals)])


def unexpected(summary: dict[str, Any]) -> list[str]:
    """The cells that did not do what was expected: FAIL, and XPASS (a documented failure is
    gone, so its entry in `loops.REGISTRY` must go too)."""
    return [
        f"{sid}/{impl}"
        for sid, row in summary["matrix"].items()
        for impl, cell in row.items()
        if cell["status"] in ("FAIL", "XPASS")
    ]


# --- files -----------------------------------------------------------------------------------------


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (a reader never sees half a file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def point_latest(parent: Path, run_id: str) -> None:
    """Make `parent/latest` a relative symlink to `run_id`, replacing it atomically."""
    link, tmp = parent / "latest", parent / f".latest.{os.getpid()}"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(run_id)
    os.replace(tmp, link)


def git_state() -> tuple[str | None, bool | None]:
    """This checkout's HEAD sha and whether it has uncommitted changes (None outside git)."""

    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--untracked-files=no")
    return sha, None if changes is None else bool(changes)

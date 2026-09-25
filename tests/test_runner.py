import asyncio
import json
import subprocess
from typing import ClassVar

import pytest

from bakeoff.shared.contract import Event, Item, ModelConfig, Resume, ToolCall, ToolResult
from bakeoff.shared.invariants import check_commits, check_seq, check_tool_results
from bakeoff.shared.runner import NdjsonMirror, Runner
from bakeoff.shared.sessionlog import SessionLog
from bakeoff.shared.workcopy import GIT_CONFIG, WorkCopy, git_env

MODEL = ModelConfig(base_url="http://127.0.0.1:9/v1", model="fake/model", api_key="sk-secret")
RULES = {"*": "allow", "write_file": {"*.pipe": "allow", "*": "ask"}}
# The events of a turn driven by `writes()`.
WRITE_TURN = [
    "turn.start",
    "item",  # the user message
    "request.start",
    "item",
    "tool.start",
    "tool.end",
    "item",
    "turn.end",
    "commit",
]


class FakeLoop:
    """A scripted Loop: `script(turn, tools, cancel)` is an async generator of events."""

    name = "our"

    def __init__(self, script):
        self.script = script
        self.inputs = []

    def run_turn(self, turn, tools, cancel):
        self.inputs.append(turn)
        return self.script(turn, tools, cancel)

    async def aclose(self):
        pass


class StubTools:
    """Writes `{"path", "content"}` into the working copy; emits tool.start/end like ToolHost."""

    made: ClassVar[list["StubTools"]] = []

    def __init__(self, workdir, rules, emit):
        self.workdir, self.rules, self.emit = workdir, rules, emit
        StubTools.made.append(self)

    def specs(self):
        return []

    def check(self, call):
        return "allow"

    async def run(self, call):
        self.emit(Event("tool.start", {"call_id": call.id, "name": call.name}))
        args = json.loads(call.arguments)
        (self.workdir / args["path"]).write_text(args["content"])
        self.emit(Event("tool.end", {"call_id": call.id, "name": call.name, "ok": True, "ms": 0}))
        return ToolResult(call.id, True, "written")


def assistant(*calls):
    tool_calls = [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in calls
    ]
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def item(turn, suffix, message):
    return Event("item", {"item": Item(f"{turn.turn_id}:{suffix}", turn.turn_id, message)})


def tool_item(turn, result):
    message = {"role": "tool", "tool_call_id": result.call_id, "content": result.content}
    return item(turn, f"tool-{result.call_id}", message)


def write_call(turn, path):
    return ToolCall(f"{turn.turn_id}:c", "write_file", json.dumps({"path": path, "content": "{}"}))


def writes(path):
    async def script(turn, tools, cancel):
        yield Event("request.start", {"step": 1, "attempt": 1})
        call = write_call(turn, path)
        yield item(turn, "a", assistant(call))
        yield tool_item(turn, await tools.run(call))
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    return script


def only(*events):
    async def script(turn, tools, cancel):
        for event in events:
            yield event

    return script


def git(root, *args):
    return subprocess.run(
        ["git", *GIT_CONFIG, *args], cwd=root, env=git_env(root), capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def log(tmp_path):
    log = SessionLog(tmp_path / "log.sqlite")
    yield log
    log.close()


@pytest.fixture
def published():
    return []


@pytest.fixture
def runner(log, tmp_path, published):
    StubTools.made.clear()
    return Runner(log, tmp_path / "wc", StubTools, sink=published.append)


@pytest.fixture
def tid(runner):
    return runner.new_thread(impl="our", system="You build pipelines.", rules=RULES, model=MODEL)


def types(log, tid):
    return [e["type"] for e in log.events(tid)]


def test_new_thread_keeps_config_but_not_the_api_key(runner, log, tmp_path):
    tid = runner.new_thread(impl="our", system="sys", rules=RULES, model=MODEL, thread_id="t-1")
    assert tid == "t-1"
    meta = log.get_thread(tid)["meta"]
    assert meta["rules"] == RULES
    assert meta["model"]["base_url"] == MODEL.base_url
    assert "api_key" not in meta["model"]
    assert b"sk-secret" not in (tmp_path / "log.sqlite").read_bytes()
    with pytest.raises(ValueError, match="thread id"):
        runner.new_thread(impl="our", system="s", rules={}, model=MODEL, thread_id="../x")


def test_new_thread_creates_the_working_copy(runner, log, tid):
    assert git(runner.workdir(tid), "log", "--format=%s") == "init"
    runner.compact(tid, "nothing yet")  # a thread without any loop turn
    assert check_commits(log, tid, runner.workdir(tid)).ok


async def test_user_turn(runner, log, tid, published):
    loop = FakeLoop(writes("a.pipe"))
    summary = await runner.turn(loop, tid, model=MODEL, user_text="build it")

    turn_id = f"{tid}.0"
    head = git(runner.workdir(tid), "rev-parse", "HEAD")
    assert summary == {"turn_id": turn_id, "stop": "end_turn", "pending": [], "commit": head}
    assert types(log, tid) == WRITE_TURN
    events = log.events(tid)
    assert published == events
    assert [e["seq"] for e in events] == list(range(1, 10))
    assert all(e["v"] == 1 and e["impl"] == "our" and e["turn"] == turn_id for e in events)
    assert [e["t_us"] for e in events] == sorted(e["t_us"] for e in events)
    assert events[0]["data"] == {"turn_id": turn_id}
    assert events[1]["data"]["item"]["message"] == {"role": "user", "content": "build it"}
    assert events[-1]["data"] == {"sha": head, "files": ["a.pipe"]}

    (turn_input,) = loop.inputs
    assert turn_input.system == "You build pipelines."
    assert turn_input.resume is None
    assert turn_input.model == MODEL
    user = turn_input.history[-1]
    assert (user.id, user.native, user.message["content"]) == (f"{turn_id}:user", None, "build it")
    assert StubTools.made[0].workdir == runner.workdir(tid)
    assert StubTools.made[0].rules == RULES

    turn = log.last_turn(tid)
    assert (turn["kind"], turn["status"], turn["stop"], turn["commit_sha"]) == (
        "user",
        "done",
        "end_turn",
        head,
    )
    assert git(runner.workdir(tid), "log", "-1", "--format=%s") == "turn 1: end_turn"
    assert [i.message["role"] for i in log.items(tid)] == ["user", "assistant", "tool"]


async def test_next_turn_sees_history_and_seq_continues(runner, log, tid):
    await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    loop = FakeLoop(writes("b.pipe"))
    await runner.turn(loop, tid, model=MODEL, user_text="two")
    history = loop.inputs[0].history
    assert [i.message["role"] for i in history] == ["user", "assistant", "tool", "user"]
    assert history[-1].message["content"] == "two"
    events = log.events(tid)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    for check in (
        check_seq(events, log.items(tid)),
        check_tool_results(log.items(tid), events, log.turns(tid)),
        check_commits(log, tid, runner.workdir(tid)),
    ):
        assert check.ok, check.detail


async def test_pause_then_approve_in_another_process(runner, log, tid, tmp_path):
    async def ask(turn, tools, cancel):
        call = write_call(turn, "a.txt")
        yield item(turn, "a", assistant(call))
        yield Event("permission.asked", {"call_id": call.id, "name": call.name, "arguments": "{}"})
        yield Event("turn.end", {"stop": "paused", "steps": 1, "pending": [call.id]})

    summary = await runner.turn(FakeLoop(ask), tid, model=MODEL, user_text="write a.txt")
    call_id = f"{tid}.0:c"
    assert summary == {
        "turn_id": f"{tid}.0",
        "stop": "paused",
        "pending": [call_id],
        "commit": None,
    }
    paused = log.last_turn(tid)
    assert (paused["status"], paused["pending"], paused["commit_sha"]) == (
        "paused",
        [call_id],
        None,
    )
    assert types(log, tid)[-1] == "turn.end"

    async def approve(turn, tools, cancel):
        assert turn.resume.decisions == {call_id: "allow"}
        (spec,) = turn.history[-1].message["tool_calls"]
        call = ToolCall(spec["id"], spec["function"]["name"], spec["function"]["arguments"])
        yield tool_item(turn, await tools.run(call))
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    # A new process: its own log connection and runner.
    log2 = SessionLog(tmp_path / "log.sqlite")
    runner2 = Runner(log2, tmp_path / "wc", StubTools)
    loop = FakeLoop(approve)
    resume = Resume(kind="approval", decisions={call_id: "allow"})
    summary = await runner2.turn(loop, tid, model=MODEL, resume=resume)
    log2.close()

    assert summary["stop"] == "end_turn"
    assert [i.message["role"] for i in loop.inputs[0].history] == ["user", "assistant"]
    events = log.events(tid)
    start = [e for e in events if e["type"] == "turn.start"][1]
    assert start["data"] == {
        "turn_id": f"{tid}.1",
        "resume": {"kind": "approval", "decisions": {call_id: "allow"}, "reason": None},
    }
    assert events[-1]["data"]["files"] == ["a.txt"]
    assert [t["kind"] for t in log.turns(tid)] == ["user", "approval"]
    assert check_commits(log, tid, runner.workdir(tid)).ok
    assert check_seq(events, log.items(tid)).ok


async def test_turn_arguments_are_validated(runner, tid):
    loop = FakeLoop(writes("a.pipe"))
    with pytest.raises(ValueError, match="exactly one"):
        await runner.turn(loop, tid, model=MODEL)
    with pytest.raises(ValueError, match="exactly one"):
        await runner.turn(loop, tid, model=MODEL, user_text="x", resume=Resume(kind="crash"))
    loop.name = "pydantic"
    with pytest.raises(ValueError, match="belongs to 'our'"):
        await runner.turn(loop, tid, model=MODEL, user_text="x")
    with pytest.raises(KeyError):
        await runner.turn(loop, "missing", model=MODEL, user_text="x")


async def test_crash_leaves_turn_running_and_crash_resume_continues(runner, log, tid):
    blocked = asyncio.Event()

    async def hang(turn, tools, cancel):
        call = write_call(turn, "a.pipe")
        yield item(turn, "a", assistant(call))
        yield tool_item(turn, await tools.run(call))
        yield Event("text.delta", {"text": "almost"})
        blocked.set()
        await asyncio.Event().wait()
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    task = asyncio.create_task(runner.turn(FakeLoop(hang), tid, model=MODEL, user_text="go"))
    await blocked.wait()
    with pytest.raises(RuntimeError, match=r"running turn|busy"):
        await runner.turn(FakeLoop(writes("b.pipe")), tid, model=MODEL, user_text="again")
    with pytest.raises(RuntimeError, match="running turn"):
        runner.compact(tid, "s")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    crashed = log.last_turn(tid)
    assert (crashed["status"], crashed["commit_sha"]) == ("running", None)
    assert types(log, tid)[-1] == "text.delta"  # buffered events were flushed

    async def finish(turn, tools, cancel):
        assert turn.resume.kind == "crash"
        assert [i.message["role"] for i in turn.history] == ["user", "assistant", "tool"]
        yield item(turn, "final", {"role": "assistant", "content": "done"})
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    summary = await runner.turn(FakeLoop(finish), tid, model=MODEL, resume=Resume(kind="crash"))
    assert summary["stop"] == "end_turn"
    assert [(t["kind"], t["status"]) for t in log.turns(tid)] == [
        ("user", "running"),
        ("crash", "done"),
    ]
    assert log.events(tid)[-1]["data"]["files"] == ["a.pipe"]
    for check in (
        check_seq(log.events(tid), log.items(tid)),
        check_tool_results(log.items(tid), log.events(tid), log.turns(tid)),
        check_commits(log, tid, runner.workdir(tid)),
    ):
        assert check.ok, check.detail


async def test_concurrent_turns_on_one_thread_are_rejected(runner, log, tid):
    await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    results = await asyncio.gather(
        runner.turn(FakeLoop(writes("b.pipe")), tid, model=MODEL, user_text="two"),
        runner.turn(FakeLoop(writes("c.pipe")), tid, model=MODEL, user_text="three"),
        return_exceptions=True,
    )
    assert results[0]["stop"] == "end_turn"
    assert isinstance(results[1], RuntimeError)
    assert [t["status"] for t in log.turns(tid)] == ["done", "done"]


async def test_a_turn_cannot_start_while_a_revert_runs_git(runner, log, tid):
    await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    reverted, rejected = await asyncio.gather(
        runner.revert(tid, f"{tid}.0"),
        runner.turn(FakeLoop(writes("b.pipe")), tid, model=MODEL, user_text="two"),
        return_exceptions=True,
    )
    assert reverted["commit"]
    assert isinstance(rejected, RuntimeError)
    assert "running turn" in str(rejected) or "busy" in str(rejected)
    assert [(t["kind"], t["status"]) for t in log.turns(tid)] == [
        ("user", "done"),
        ("revert", "done"),
    ]
    assert check_seq(log.events(tid), log.items(tid)).ok
    assert check_commits(log, tid, runner.workdir(tid)).ok


async def test_loop_exception_ends_turn_with_error(runner, log, tid):
    async def boom(turn, tools, cancel):
        yield Event("request.start", {"step": 1, "attempt": 1})
        yield Event("request.start", {"step": 1, "attempt": 2})
        raise ValueError("boom")

    summary = await runner.turn(FakeLoop(boom), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "error"
    assert summary["commit"]
    events = log.events(tid)
    assert [e["type"] for e in events][-3:] == ["error", "turn.end", "commit"]
    assert events[-3]["data"] == {"kind": "loop", "message": "ValueError: boom", "retryable": False}
    assert events[-2]["data"] == {"stop": "error", "steps": 1, "error": "ValueError: boom"}
    assert log.last_turn(tid)["status"] == "error"


async def test_tool_host_construction_failure_ends_turn_with_error(log, tmp_path):
    def broken_tools(workdir, rules, emit):
        raise ValueError("bad rules")

    runner = Runner(log, tmp_path / "wc", broken_tools)
    tid = runner.new_thread(impl="our", system="s", rules={}, model=MODEL)
    summary = await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "error"
    assert log.events(tid)[-2]["data"]["error"] == "ValueError: bad rules"


def usage(cost):
    return Event("usage", {"step": 1, "cost_usd": cost})


@pytest.mark.parametrize(
    "bad",
    [
        lambda turn: usage(object()),
        lambda turn: Event("item", {"item": Item("x", turn.turn_id, {}, native=object())}),
    ],
)
async def test_event_data_that_is_not_json_ends_the_turn_with_error(runner, log, tid, bad):
    async def script(turn, tools, cancel):
        yield bad(turn)
        yield item(turn, "a", {"role": "assistant", "content": "never recorded"})
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    summary = await runner.turn(FakeLoop(script), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "error"
    assert types(log, tid) == ["turn.start", "item", "error", "turn.end", "commit"]
    assert "not JSON serializable" in log.events(tid)[2]["data"]["message"]
    assert [i.message["role"] for i in log.items(tid)] == ["user"]
    assert log.last_turn(tid)["status"] == "error"
    again = await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="again")
    assert again["stop"] == "end_turn"
    assert check_seq(log.events(tid), log.items(tid)).ok


async def test_events_are_recorded_as_emitted(runner, log, tid, published):
    async def mutates(turn, tools, cancel):
        event = usage(0.01)
        yield event
        event.data["cost_usd"] = 999.0
        message = {"role": "assistant", "content": "v1"}
        yield item(turn, "a", message)
        message["content"] = "v2"
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    await runner.turn(FakeLoop(mutates), tid, model=MODEL, user_text="go")
    events = log.events(tid)
    assert published == events
    assert events[2]["data"]["cost_usd"] == 0.01
    assert events[3]["data"]["item"]["message"]["content"] == "v1"
    assert log.items(tid)[-1].message["content"] == "v1"


@pytest.mark.parametrize(
    "fails",
    [lambda seen: len(seen) >= 4, lambda seen: seen[-1] == "turn.end"],
    ids=["from-the-4th-event-on", "once-at-turn-end"],
)
async def test_a_failing_sink_does_not_affect_the_turn(log, tmp_path, caplog, fails):
    seen = []

    def sink(envelope):
        seen.append(envelope["type"])
        if fails(seen):
            raise BrokenPipeError("stdout is closed")

    runner = Runner(log, tmp_path / "wc", StubTools, sink=sink)
    tid = runner.new_thread(impl="our", system="s", rules={}, model=MODEL)
    summary = await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "end_turn"
    assert log.last_turn(tid)["status"] == "done"
    assert types(log, tid) == WRITE_TURN
    assert seen == WRITE_TURN[: len(seen)]  # nothing after the failure
    assert [r.getMessage() for r in caplog.records] == [
        "event sink failed; it gets no more events this turn"
    ]


async def test_commit_failure_ends_the_turn_without_blocking_the_thread(runner, log, tid):
    lock = runner.workdir(tid) / ".git" / "index.lock"
    lock.write_text("")  # left behind by a git process that was killed
    with pytest.raises(RuntimeError, match="git add failed"):
        await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    turn = log.last_turn(tid)
    assert (turn["status"], turn["stop"], turn["commit_sha"]) == ("error", "end_turn", None)
    assert types(log, tid)[-1] == "turn.end"
    # The next turn cleans up after git (the stale lock) and commits both turns' changes.
    summary = await runner.turn(FakeLoop(writes("b.pipe")), tid, model=MODEL, user_text="two")
    assert summary["stop"] == "end_turn"
    assert log.events(tid)[-1]["data"]["files"] == ["a.pipe", "b.pipe"]
    check = check_commits(log, tid, runner.workdir(tid))
    assert check.ok, check.detail


async def test_a_commit_git_made_but_no_turn_recorded_is_undone_by_the_next_turn(
    runner, log, tid, monkeypatch
):
    real = WorkCopy._head_change

    async def fails_once(self):  # git committed, then reading the new commit failed
        monkeypatch.setattr(WorkCopy, "_head_change", real)
        raise RuntimeError("git diff-tree failed")

    monkeypatch.setattr(WorkCopy, "_head_change", fails_once)
    with pytest.raises(RuntimeError, match="diff-tree"):
        await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    wd = runner.workdir(tid)
    assert git(wd, "rev-list", "--count", "HEAD") == "2"  # init + a commit no turn recorded
    assert (log.last_turn(tid)["status"], log.last_turn(tid)["commit_sha"]) == ("error", None)

    await runner.turn(FakeLoop(writes("b.pipe")), tid, model=MODEL, user_text="two")
    assert git(wd, "rev-list", "--count", "HEAD") == "2"  # init + turn 2, built on init
    assert log.events(tid)[-1]["data"]["files"] == ["a.pipe", "b.pipe"]
    check = check_commits(log, tid, wd)
    assert check.ok, check.detail


async def test_crash_resume_cleans_up_after_git(runner, log, tid):
    await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    # The worker died while committing turn 1: git's commit landed without being recorded,
    # and a git process that was killed left its index lock.
    log.start_turn(tid, "user")
    wd = runner.workdir(tid)
    (wd / "b.pipe").write_text("{}")
    git(wd, "add", "-A")
    git(wd, "commit", "-q", "-m", "turn 1: end_turn")
    (wd / ".git" / "index.lock").write_text("")

    finish = FakeLoop(only(Event("turn.end", {"stop": "end_turn", "steps": 0})))
    summary = await runner.turn(finish, tid, model=MODEL, resume=Resume(kind="crash"))
    assert summary["stop"] == "end_turn"
    assert log.events(tid)[-1]["data"]["files"] == ["b.pipe"]
    check = check_commits(log, tid, wd)
    assert check.ok, check.detail


async def test_loop_without_turn_end(runner, log, tid):
    summary = await runner.turn(FakeLoop(only()), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "error"
    end = log.events(tid)[-2]
    assert end["type"] == "turn.end"
    assert end["data"] == {"stop": "error", "steps": 0, "error": "loop ended without turn.end"}
    assert log.last_turn(tid)["status"] == "error"


async def test_commit_directly_follows_turn_end(runner, log, tid, published):
    async def sloppy(turn, tools, cancel):
        try:
            yield Event("turn.end", {"stop": "max_steps", "steps": 12})
            yield Event("text.delta", {"text": "never consumed"})
        finally:  # runs when the runner closes the loop, after turn.end
            tools.emit(Event("tool.end", {"call_id": "c9", "name": "x", "ok": True, "ms": 1}))
            raise RuntimeError("cleanup failed")

    summary = await runner.turn(FakeLoop(sloppy), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "max_steps"
    assert types(log, tid) == ["turn.start", "item", "turn.end", "commit"]
    assert published == log.events(tid)
    assert log.last_turn(tid)["status"] == "done"
    assert published[-1]["data"] == {"sha": log.last_turn(tid)["commit_sha"], "files": []}
    (late,) = log.last_turn(tid)["late"]  # kept as evidence, outside the stream
    assert (late["type"], late["data"]["call_id"]) == ("tool.end", "c9")
    assert late["t_us"] >= published[-2]["t_us"]
    StubTools.made[0].emit(Event("tool.start", {"call_id": "c9", "name": "x"}))  # after commit
    assert len(log.events(tid)) == len(published) == 4


async def test_a_tool_run_after_turn_end_fails_i2(runner, log, tid):
    async def reruns(turn, tools, cancel):
        call = write_call(turn, "a.pipe")
        yield item(turn, "a", assistant(call))
        try:
            yield tool_item(turn, await tools.run(call))
            yield Event("turn.end", {"stop": "end_turn", "steps": 1})
        finally:  # runs when the runner closes the loop, after turn.end
            await tools.run(call)

    await runner.turn(FakeLoop(reruns), tid, model=MODEL, user_text="go")
    events = log.events(tid)
    assert [e["type"] for e in events][-2:] == ["turn.end", "commit"]
    assert [x["type"] for x in log.last_turn(tid)["late"]] == ["tool.start", "tool.end"]
    check = check_tool_results(log.items(tid), events, log.turns(tid))
    call_id = f"{tid}.0:c"
    assert (check.ok, check.info["reran"], check.info["late_runs"]) == (False, [call_id], [call_id])


async def test_a_tool_run_after_a_paused_turn_end_is_recorded_and_fails_i2(runner, log, tid):
    async def pauses(turn, tools, cancel):
        call = write_call(turn, "a.txt")
        yield item(turn, "a", assistant(call))
        try:
            yield Event("turn.end", {"stop": "paused", "steps": 1, "pending": [call.id]})
        finally:  # runs when the runner closes the loop: the pending call runs anyway
            await tools.run(call)

    summary = await runner.turn(FakeLoop(pauses), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "paused"
    assert types(log, tid)[-1] == "turn.end"  # nothing after turn.end in the stream (rule 7)
    call_id = f"{tid}.0:c"
    reader = SessionLog(runner.wc_root.parent / "log.sqlite")  # durable, e.g. for `bakeoff report`
    paused = reader.last_turn(tid)
    assert (paused["status"], paused["pending"]) == ("paused", [call_id])
    assert [(x["type"], x["data"]["call_id"]) for x in paused["late"]] == [
        ("tool.start", call_id),
        ("tool.end", call_id),
    ]
    check = check_tool_results(reader.items(tid), reader.events(tid), reader.turns(tid))
    reader.close()
    assert (check.ok, check.info["late_runs"]) == (False, [call_id])


async def wait_for_cancel(turn, tools, cancel):
    await asyncio.wait_for(cancel.wait(), 2)
    yield Event("turn.end", {"stop": "cancelled", "steps": 0})


async def test_cancel_event(runner, log, tid):
    cancel = asyncio.Event()
    task = asyncio.create_task(
        runner.turn(FakeLoop(wait_for_cancel), tid, model=MODEL, user_text="go", cancel=cancel)
    )
    await asyncio.sleep(0.01)
    cancel.set()
    assert (await task)["stop"] == "cancelled"
    turn = log.last_turn(tid)
    assert (turn["status"], turn["stop"]) == ("cancelled", "cancelled")
    assert turn["commit_sha"]


async def test_watch_cancel_polls_the_log(runner, log, tid, tmp_path):
    log.request_cancel(tid)  # stale: from before the turn started
    task = asyncio.create_task(
        runner.turn(FakeLoop(wait_for_cancel), tid, model=MODEL, user_text="go", watch_cancel=True)
    )
    await asyncio.sleep(0.12)
    assert not task.done()
    other = SessionLog(tmp_path / "log.sqlite")  # e.g. `bakeoff cancel` in another process
    other.request_cancel(tid)
    other.close()
    summary = await asyncio.wait_for(task, 1)
    assert summary["stop"] == "cancelled"
    assert log.last_turn(tid)["status"] == "cancelled"


async def test_batching_and_publish_order(log, tmp_path):
    reader = SessionLog(tmp_path / "log.sqlite")
    stored_counts = {}
    unstored = []

    def sink(envelope):
        if envelope["type"] == "item":  # published only after it is stored
            stored = envelope["data"]["item"]["id"] in [i.id for i in reader.items(tid)]
            if not stored or reader.events(tid)[-1] != envelope:
                unstored.append(envelope)
        if envelope["type"] == "commit":  # the turn row is already complete
            turn = reader.last_turn(tid)
            if (turn["status"], turn["commit_sha"]) != ("done", envelope["data"]["sha"]):
                unstored.append(envelope)

    async def deltas(turn, tools, cancel):
        for n in range(1, 131):
            yield Event("text.delta", {"text": str(n)})
            if n in (63, 64, 130):
                stored_counts[n] = len(reader.events(tid))
        yield item(turn, "a", {"role": "assistant", "content": "done"})
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    runner = Runner(log, tmp_path / "wc", StubTools, sink=sink)
    tid = runner.new_thread(impl="our", system="s", rules={}, model=MODEL)
    summary = await runner.turn(FakeLoop(deltas), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "end_turn"
    assert unstored == []
    # turn.start + user item are stored with the item; then one flush per 64 events.
    assert stored_counts == {63: 2, 64: 66, 130: 130}
    assert len(reader.events(tid)) == 2 + 130 + 3
    reader.close()


async def test_revert(runner, log, tid, published):
    await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    await runner.turn(FakeLoop(writes("b.pipe")), tid, model=MODEL, user_text="two")
    published.clear()

    summary = await runner.revert(tid, f"{tid}.0")

    wd = runner.workdir(tid)
    head = git(wd, "rev-parse", "HEAD")
    assert summary == {"turn_id": f"{tid}.2", "stop": None, "pending": [], "commit": head}
    assert not (wd / "a.pipe").exists()
    assert (wd / "b.pipe").exists()
    note = log.items(tid)[-1]
    assert note.message == {"role": "user", "content": "[harness] Reverted turn 1; files: a.pipe"}
    assert note.native is None
    assert [e["type"] for e in published] == ["item", "commit"]
    assert published[1]["data"] == {"sha": head, "files": ["a.pipe"]}
    turn = log.last_turn(tid)
    assert (turn["kind"], turn["status"], turn["commit_sha"]) == ("revert", "done", head)
    assert check_commits(log, tid, wd).ok
    assert check_seq(log.events(tid), log.items(tid)).ok

    # The model sees the note in the next turn.
    loop = FakeLoop(writes("c.pipe"))
    await runner.turn(loop, tid, model=MODEL, user_text="three")
    assert loop.inputs[0].history[-2].message["content"].startswith("[harness] Reverted turn 1")


async def test_revert_errors_record_no_turn(runner, log, tid):
    await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    (runner.workdir(tid) / "a.pipe").write_text("changed")
    no_op = FakeLoop(only(Event("turn.end", {"stop": "end_turn", "steps": 0})))
    await runner.turn(no_op, tid, model=MODEL, user_text="two")  # commits the change to a.pipe
    with pytest.raises(RuntimeError, match="git revert failed"):
        await runner.revert(tid, f"{tid}.0")
    with pytest.raises(ValueError, match="no commit"):
        await runner.revert(tid, f"{tid}.9")
    assert len(log.turns(tid)) == 2
    assert check_commits(log, tid, runner.workdir(tid)).ok


async def test_compact(runner, log, tid, published):
    await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="one")
    published.clear()

    summary_item = runner.compact(tid, "wrote a.pipe")

    assert summary_item.compaction
    assert summary_item.message == {
        "role": "user",
        "content": "[harness] Conversation summary: wrote a.pipe",
    }
    assert log.items(tid)[-1] == summary_item
    assert [e["type"] for e in published] == ["item"]
    assert published[0]["data"]["item"]["compaction"] is True
    turn = log.last_turn(tid)
    assert (turn["kind"], turn["status"], turn["commit_sha"]) == ("compact", "done", None)

    loop = FakeLoop(writes("b.pipe"))
    await runner.turn(loop, tid, model=MODEL, user_text="two")
    assert loop.inputs[0].history[-2] == summary_item
    assert check_commits(log, tid, runner.workdir(tid)).ok
    assert check_seq(log.events(tid), log.items(tid)).ok


async def test_ndjson_mirror(log, tmp_path):
    mirror = NdjsonMirror(tmp_path / "mirror" / "events.ndjson")
    runner = Runner(log, tmp_path / "wc", StubTools, sink=mirror)
    tid = runner.new_thread(impl="our", system="s", rules={}, model=MODEL)
    await runner.turn(FakeLoop(writes("a.pipe")), tid, model=MODEL, user_text="go")
    mirror.close()
    lines = (tmp_path / "mirror" / "events.ndjson").read_text().splitlines()
    assert [json.loads(line) for line in lines] == log.events(tid)


async def test_concurrent_crash_resumes_cannot_both_run(runner, log, tid):
    """Two crash resumes of one thread: exactly one runs, the other gets ThreadBusy."""
    from bakeoff.shared.runner import ThreadBusy

    started = asyncio.Event()

    async def hang(turn, tools, cancel):
        started.set()
        await asyncio.Event().wait()
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    crashed = asyncio.create_task(runner.turn(FakeLoop(hang), tid, model=MODEL, user_text="go"))
    await started.wait()
    crashed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await crashed

    release = asyncio.Event()

    async def slow_finish(turn, tools, cancel):
        await release.wait()
        yield item(turn, "final", {"role": "assistant", "content": "done"})
        yield Event("turn.end", {"stop": "end_turn", "steps": 1})

    first = asyncio.create_task(
        runner.turn(FakeLoop(slow_finish), tid, model=MODEL, resume=Resume(kind="crash"))
    )
    await asyncio.sleep(0.05)
    with pytest.raises(ThreadBusy):
        await runner.turn(FakeLoop(slow_finish), tid, model=MODEL, resume=Resume(kind="crash"))
    release.set()
    assert (await first)["stop"] == "end_turn"
    assert [(t["kind"], t["status"]) for t in log.turns(tid)] == [
        ("user", "running"),
        ("crash", "done"),
    ]

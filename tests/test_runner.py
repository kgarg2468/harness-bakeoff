import asyncio
import json
import subprocess
from typing import ClassVar

import pytest

from bakeoff.shared.contract import Event, Item, ModelConfig, Resume, ToolCall, ToolResult
from bakeoff.shared.invariants import check_commits, check_seq, check_tool_results
from bakeoff.shared.runner import NdjsonMirror, Runner
from bakeoff.shared.sessionlog import SessionLog
from bakeoff.shared.workcopy import GIT_CONFIG, git_env

MODEL = ModelConfig(base_url="http://127.0.0.1:9/v1", model="fake/model", api_key="sk-secret")
RULES = {"*": "allow", "write_file": {"*.pipe": "allow", "*": "ask"}}


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
    assert runner.workdir(tid).is_dir()
    with pytest.raises(ValueError, match="thread id"):
        runner.new_thread(impl="our", system="s", rules={}, model=MODEL, thread_id="../x")


async def test_user_turn(runner, log, tid, published):
    loop = FakeLoop(writes("a.pipe"))
    summary = await runner.turn(loop, tid, model=MODEL, user_text="build it")

    turn_id = f"{tid}.0"
    head = git(runner.workdir(tid), "rev-parse", "HEAD")
    assert summary == {"turn_id": turn_id, "stop": "end_turn", "pending": [], "commit": head}
    assert types(log, tid) == [
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
    assert git(runner.workdir(tid), "log", "-1", "--format=%s") == "turn 0: end_turn"
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
        check_tool_results(log.items(tid), events),
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
    with pytest.raises(RuntimeError, match="running turn"):
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
        check_tool_results(log.items(tid), log.events(tid)),
        check_commits(log, tid, runner.workdir(tid)),
    ):
        assert check.ok, check.detail


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


async def test_loop_without_turn_end(runner, log, tid):
    summary = await runner.turn(FakeLoop(only()), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "error"
    end = log.events(tid)[-2]
    assert end["type"] == "turn.end"
    assert end["data"] == {"stop": "error", "steps": 0, "error": "loop ended without turn.end"}
    assert log.last_turn(tid)["status"] == "error"


async def test_nothing_after_turn_end_is_consumed(runner, log, tid):
    closed = []

    async def chatty(turn, tools, cancel):
        try:
            yield Event("turn.end", {"stop": "max_steps", "steps": 12})
            yield Event("text.delta", {"text": "late"})
        finally:
            closed.append(True)

    summary = await runner.turn(FakeLoop(chatty), tid, model=MODEL, user_text="go")
    assert summary["stop"] == "max_steps"
    assert closed == [True]
    assert types(log, tid)[-2:] == ["turn.end", "commit"]
    assert log.last_turn(tid)["status"] == "done"


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


async def test_batching_and_persist_before_publish(log, tmp_path):
    reader = SessionLog(tmp_path / "log.sqlite")
    stored_counts = {}

    def sink(envelope):
        if envelope["type"] == "item":  # published only after it is stored
            assert envelope["data"]["item"]["id"] in [i.id for i in reader.items(tid)]
            assert reader.events(tid)[-1] == envelope

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
    assert summary["stop"] == "end_turn"  # a failed assertion in the sink would end it with error
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
    assert note.message == {"role": "user", "content": "[harness] Reverted turn 0; files: a.pipe"}
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
    assert loop.inputs[0].history[-2].message["content"].startswith("[harness] Reverted turn 0")


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

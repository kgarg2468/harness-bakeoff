import asyncio
import json

import pytest

from bakeoff.shared.contract import ToolCall
from bakeoff.shared.engine.mock import MockEngine
from bakeoff.shared.toolhost import MAX_OUTPUT, ToolHostImpl, build_toolhost

ORDER = [
    "list_components",
    "describe_component",
    "validate_pipeline",
    "list_files",
    "read_file",
    "write_file",
    "edit_file",
]
READ_ONLY = {
    "list_components",
    "describe_component",
    "validate_pipeline",
    "list_files",
    "read_file",
}


def call(name, args="{}", call_id="c1"):
    return ToolCall(
        id=call_id, name=name, arguments=args if isinstance(args, str) else json.dumps(args)
    )


@pytest.fixture
def events():
    return []


@pytest.fixture
def make(tmp_path, events):
    def make(rules=None, engine=None):
        return build_toolhost(
            tmp_path, {"*": "allow"} if rules is None else rules, events.append, engine
        )

    return make


def test_specs_order_and_read_only(make):
    host = make()
    assert isinstance(host, ToolHostImpl)
    specs = host.specs()
    assert [s.name for s in specs] == ORDER
    assert {s.name for s in specs if s.read_only} == READ_ONLY
    assert host.specs() == specs


async def test_run_emits_start_and_end_and_counts(make, events, tmp_path):
    host = make()
    result = await host.run(call("write_file", {"path": "a.pipe", "content": "{}"}))
    assert (result.call_id, result.ok, result.content) == ("c1", True, "Wrote 2 bytes to a.pipe")
    assert [(e.type, e.data["call_id"], e.data["name"]) for e in events] == [
        ("tool.start", "c1", "write_file"),
        ("tool.end", "c1", "write_file"),
    ]
    assert events[0].data == {"call_id": "c1", "name": "write_file"}
    assert set(events[1].data) == {"call_id", "name", "ok", "ms"}
    assert events[1].data["ok"] is True and events[1].data["ms"] >= 0
    assert host.run_counts == {"c1": 1}
    await host.run(call("read_file", {"path": "a.pipe"}, "c2"))
    assert host.run_counts == {"c1": 1, "c2": 1}
    assert (tmp_path / "a.pipe").read_text() == "{}"


async def test_default_engine_is_the_mock(make):
    result = await make().run(call("describe_component", {"name": "chat"}))
    assert result.ok and json.loads(result.content)["lanes"] == {"_source": ["questions"]}


def test_check_uses_rules_and_paths(make):
    host = make({"*": "allow", "write_file": {"*.pipe": "allow", "*": "ask"}, "edit_file": "deny"})
    assert host.check(call("write_file", {"path": "dir/../a.pipe", "content": ""})) == "allow"
    assert host.check(call("write_file", {"path": "a.txt", "content": ""})) == "ask"
    edit = {"path": "a.pipe", "old_string": "a", "new_string": "b"}
    assert host.check(call("edit_file", edit)) == "deny"
    assert host.check(call("list_components")) == "allow"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("read_file", {"path": "../outside"}),
        ("read_file", {"path": "/etc/passwd"}),
        ("list_files", {"path": ".."}),
        ("validate_pipeline", {"path": "../p.pipe"}),
        ("write_file", {"path": ".git/config", "content": "x"}),
        ("write_file", {"path": "sub/.git/HEAD", "content": "x"}),
        ("write_file", {"path": ".GIT/config", "content": "x"}),
        ("edit_file", {"path": "sub/.git", "old_string": "a", "new_string": "b"}),
    ],
)
async def test_path_escape_is_always_denied(make, events, name, args):
    host = make({"*": "allow"})
    assert host.check(call(name, args)) == "deny"
    result = await host.run(call(name, args))
    assert not result.ok and result.content.startswith("Denied: ")
    assert host.run_counts == {}
    assert [e.type for e in events] == ["tool.start", "tool.end"]


async def test_deny_is_enforced_by_run(make, tmp_path):
    host = make({"*": "allow", "write_file": "deny"})
    result = await host.run(call("write_file", {"path": "a.pipe", "content": "{}"}))
    assert (result.ok, result.content) == (False, "Denied by permission rules: write_file a.pipe")
    assert not (tmp_path / "a.pipe").exists()
    assert host.run_counts == {}


async def test_ask_is_the_loops_job(make, tmp_path):
    host = make({"*": "ask"})
    args = {"path": "a.pipe", "content": "{}"}
    assert host.check(call("write_file", args)) == "ask"
    assert (await host.run(call("write_file", args))).ok  # run() only enforces deny
    assert (tmp_path / "a.pipe").exists()


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("read_file", '{"path": ', "Invalid arguments for read_file: not valid JSON"),
        ("read_file", "[1]", "Invalid arguments for read_file: expected a JSON object"),
        ("read_file", "{}", "Invalid arguments for read_file: 'path' is a required property"),
        (
            "read_file",
            '{"path": 1}',
            "Invalid arguments for read_file: path: 1 is not of type 'string'",
        ),
        (
            "describe_component",
            "",
            "Invalid arguments for describe_component: 'name' is a required",
        ),
        (
            "validate_pipeline",
            '{"pipeline": [1]}',
            "Invalid arguments for validate_pipeline: pipeline:",
        ),
        ("validate_pipeline", "{}", "Invalid arguments for validate_pipeline: pass exactly one"),
        (
            "validate_pipeline",
            '{"pipeline": {}, "path": "a.pipe"}',
            "Invalid arguments for validate_pipeline: pass exactly one of 'pipeline' or 'path'",
        ),
        ("delete_everything", "{}", "Unknown tool: delete_everything"),
    ],
)
async def test_bad_calls_are_reported_not_raised(make, name, arguments, message):
    host = make({"*": "ask"})
    bad = ToolCall(id="c1", name=name, arguments=arguments)
    # run() rejects these without executing, so the loop need not ask first.
    assert host.check(bad) == "allow"
    result = await host.run(bad)
    assert not result.ok
    assert result.content.startswith(message)
    assert host.run_counts == {}


async def test_long_schema_errors_are_shortened(make):
    huge = json.dumps({"pipeline": ["x" * 10_000]})
    result = await make().run(call("validate_pipeline", huge))
    assert not result.ok and len(result.content) <= 300


class BrokenEngine:
    async def get_services(self):
        raise RuntimeError("engine exploded")


async def test_tool_exceptions_become_short_results(make, events):
    result = await make(engine=BrokenEngine()).run(call("list_components"))
    assert (result.ok, result.content) == (
        False,
        "list_components failed: RuntimeError: engine exploded",
    )
    assert events[-1].data["ok"] is False


async def test_output_is_capped(make, tmp_path):
    (tmp_path / "big.txt").write_text("x" * (MAX_OUTPUT + 1234))
    result = await make().run(call("read_file", {"path": "big.txt"}))
    assert result.content == "x" * MAX_OUTPUT + "\n... [truncated 1234 chars]"


async def test_cancel_propagates(tmp_path):
    events, started = [], asyncio.Event()

    def emit(event):
        events.append(event)
        started.set()

    host = build_toolhost(tmp_path, {"*": "allow"}, emit, MockEngine(delay_ms=5_000))
    task = asyncio.create_task(host.run(call("describe_component", {"name": "chat"})))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [e.type for e in events] == ["tool.start", "tool.end"]
    assert events[-1].data["ok"] is False
    assert host.run_counts == {"c1": 1}


async def test_parallel_runs_overlap(make, events):
    host = make(engine=MockEngine(delay_ms=100))
    calls = [
        call("describe_component", {"name": n}, f"c{n}")
        for n in ("chat", "llm_openai", "tool_pipe")
    ]
    results = await asyncio.gather(*(host.run(c) for c in calls))
    assert [r.call_id for r in results] == [c.id for c in calls]
    types = [e.type for e in events]
    assert types[:3] == ["tool.start"] * 3  # all started before any finished


def test_invalid_rules_fail_fast(tmp_path):
    with pytest.raises(ValueError):
        build_toolhost(tmp_path, {"*": "maybe"}, lambda e: None)

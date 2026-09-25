"""The scenario driver: expect evaluation, result.json and summary.json, cross-process steps."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bakeoff import loops
from bakeoff.fakeprov.script import SCENARIOS_DIR
from bakeoff.fakeprov.server import FakeProvider
from bakeoff.our_version import OurLoop
from bakeoff.shared import scenario
from bakeoff.shared.contract import Event, ModelConfig, ToolResult
from bakeoff.shared.scenario import (
    CancelMark,
    DriverError,
    Observed,
    cancel_latency,
    capture_output,
    crash_sink,
    decide,
    evaluate_expect,
    format_matrix,
    reason,
    run_matrix,
    run_scenario,
    tool_spans,
    unexpected,
    usage_totals,
)
from bakeoff.shared.sessionlog import SessionLog

# The result.json fields of the shared output format, plus the driver's thread and processes.
RESULT_KEYS = set(
    "v run_id scenario title impl stops expect invariants requests tool_runs usage duration_ms"
    " passed error thread processes".split()
)


def copy_scenario(tmp_path: Path, sid: str, new_id: str, **changes: Any) -> Path:
    """A copy of scenario `sid` named `new_id` in tmp_path/scenarios, with top-level changes."""
    data = json.loads((SCENARIOS_DIR / f"{sid}.json").read_text())
    data |= {"id": new_id, **changes}
    folder = tmp_path / "scenarios"
    folder.mkdir(exist_ok=True)
    path = folder / f"{new_id}.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def provider(tmp_path: Path):
    with FakeProvider(tmp_path / "scenarios", tmp_path / "wire") as served:
        yield served


@pytest.fixture
def real_provider(tmp_path: Path):
    with FakeProvider(wire_dir=tmp_path / "wire") as served:
        yield served


def observed(tmp_path: Path, **fields: Any) -> Observed:
    base: dict[str, Any] = {
        "stops": ["end_turn"],
        "tool_runs": {"call_1": 1},
        "requests": 2,
        "commits": 1,
        "last_text": "Hello, team!",
        "usage": [],
        "workdir": tmp_path,
    }
    return Observed(**(base | fields))


def usage_event(tokens_in: int, tokens_out: int, cached: int, cost: float) -> dict[str, Any]:
    return {
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cached_tokens": cached,
        "cost_usd": cost,
        "cost_source": "provider",
    }


# --- expect evaluation ------------------------------------------------------------------------


def test_every_expect_key_passes_on_matching_facts(tmp_path: Path) -> None:
    (tmp_path / "a.pipe").write_text('{"components": []}')
    usage = [usage_event(10, 2, 5, 0.001), usage_event(20, 3, 0, 0.002)]
    expect = {
        "stops": ["end_turn"],
        "files": {"a.pipe": "components", "gone.md": None},
        "tool_runs": {"call_1": 1, "call_2": 0},
        "commits": 1,
        "requests": 2,
        "text_contains": "team",
        "cost_usd": 0.003,
        "usage": {"input_tokens": 30, "output_tokens": 5, "cached_tokens": 5},
        "cost_source": "provider",
    }
    results = evaluate_expect(expect, observed(tmp_path, usage=usage))
    assert set(results) == set(expect)
    assert all(r["ok"] for r in results.values()), results


def test_each_expect_key_explains_a_mismatch(tmp_path: Path) -> None:
    (tmp_path / "a.pipe").write_text("{}")
    (tmp_path / "gone.md").write_text("still here")
    usage = [{"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.5, "cost_source": "estimate"}]
    expect = {
        "stops": ["paused", "end_turn"],
        "files": {"a.pipe": "components", "gone.md": None, "missing.txt": "x"},
        "tool_runs": {"call_1": 0},
        "commits": 2,
        "requests": 3,
        "text_contains": "bye",
        "cost_usd": 0.25,
        "usage": {"input_tokens": 2},
        "cost_source": "none",
    }
    results = evaluate_expect(expect, observed(tmp_path, usage=usage))
    details = {key: r["detail"] for key, r in results.items() if not r["ok"]}
    assert set(details) == set(expect)
    assert details["stops"] == "got ['end_turn'], expected ['paused', 'end_turn']"
    assert details["files"] == (
        "a.pipe lacks 'components'; gone.md exists, expected absent; missing.txt is missing"
    )
    assert details["tool_runs"] == "call_1 ran 1x, expected 0x"
    assert details["requests"] == "got 2, expected 3"
    assert details["text_contains"] == "missing 'bye' in 'Hello, team!'"
    assert details["cost_usd"] == "got 0.5, expected 0.25"
    assert details["cost_source"] == "got estimate, expected none"


def test_cost_source_needs_usage_events_and_one_source(tmp_path: Path) -> None:
    assert evaluate_expect({"cost_source": "none"}, observed(tmp_path))["cost_source"] == {
        "ok": False,
        "detail": "no usage events",
    }
    mixed = [{"cost_source": "provider"}, {"cost_source": "none"}]
    result = evaluate_expect({"cost_source": "none"}, observed(tmp_path, usage=mixed))
    assert not result["cost_source"]["ok"]
    assert usage_totals(mixed)["cost_source"] == "mixed"
    assert usage_totals([])["cost_source"] == "none"


def test_cost_is_compared_with_a_tolerance(tmp_path: Path) -> None:
    usage = [
        {"cost_usd": 0.1, "cost_source": "provider"},
        {"cost_usd": 0.2, "cost_source": "provider"},
    ]
    assert evaluate_expect({"cost_usd": 0.3}, observed(tmp_path, usage=usage))["cost_usd"]["ok"]


def tool_event(turn: str, type_: str, call_id: str, t_ms: float) -> dict[str, Any]:
    return {"turn": turn, "type": type_, "t_us": int(t_ms * 1000), "data": {"call_id": call_id}}


def test_tools_overlap_needs_every_run_at_one_moment(tmp_path: Path) -> None:
    parallel = [
        tool_event("t.0", "tool.start", "a", 0),
        tool_event("t.0", "tool.start", "b", 1),
        tool_event("t.0", "tool.end", "a", 300),
        tool_event("t.0", "tool.end", "b", 301),
    ]
    want = {"tools_overlap": ["a", "b"]}
    check = evaluate_expect(want, observed(tmp_path, tool_spans=tool_spans(parallel)))
    assert check["tools_overlap"] == {
        "ok": True,
        "detail": "all 2 running together for 299 ms: a 0-300 ms, b 1-301 ms",
    }
    serial = [
        tool_event("t.0", "tool.start", "a", 0),
        tool_event("t.0", "tool.end", "a", 300),
        tool_event("t.0", "tool.start", "b", 300),
        tool_event("t.0", "tool.end", "b", 600),
    ]
    check = evaluate_expect(want, observed(tmp_path, tool_spans=tool_spans(serial)))
    assert check["tools_overlap"] == {
        "ok": False,
        "detail": "not all running at once: a 0-300 ms, b 300-600 ms",
    }
    unfinished = serial[:3]
    check = evaluate_expect(want, observed(tmp_path, tool_spans=tool_spans(unfinished)))
    assert check["tools_overlap"]["detail"] == "no tool.end in the turn for ['b']"
    twice = [*parallel, tool_event("t.1", "tool.start", "a", 5)]
    check = evaluate_expect(want, observed(tmp_path, tool_spans=tool_spans(twice)))
    assert check["tools_overlap"]["detail"] == "a ran 2x, expected once each"


def test_cancel_latency_is_measured_from_the_drivers_cancel(tmp_path: Path) -> None:
    events = [
        {"turn": "t.0", "type": "turn.start", "t_us": 100, "data": {}},
        {"turn": "t.0", "type": "turn.end", "t_us": 400_100 + 3_500, "data": {"stop": "cancelled"}},
    ]
    fired = CancelMark(turn="t.0", after_start_us=400_000)
    assert cancel_latency(events, fired) == ("t.0", 3.5)
    assert cancel_latency(events, CancelMark(turn="t.0")) == ("t.0", 0.0)  # ended first
    assert cancel_latency(events[:1], fired) == ("t.0", None)  # no turn.end
    assert cancel_latency(events, CancelMark()) == (None, None)  # never started
    fast = observed(tmp_path, cancels=[("t.0", 3.5), ("t.1", 0.0)])
    check = evaluate_expect({"cancel_within_ms": 200}, fast)["cancel_within_ms"]
    assert check == {
        "ok": True,
        "detail": "turn.end after the cancel: t.0: 3.5 ms, t.1: 0.0 ms (limit 200 ms)",
    }
    slow = observed(tmp_path, cancels=[("t.0", 1605.0), ("t.1", None)])
    check = evaluate_expect({"cancel_within_ms": 200}, slow)["cancel_within_ms"]
    assert not check["ok"] and "t.0: 1605.0 ms, t.1: no turn.end" in check["detail"]
    assert not evaluate_expect({"cancel_within_ms": 200}, observed(tmp_path))["cancel_within_ms"][
        "ok"
    ]


def test_decide_answers_every_pending_call_exactly() -> None:
    assert decide(["a", "b"], "all", ["b"]) == {"a": "allow", "b": "deny"}
    assert decide(["a"], [], ["a"]) == {"a": "deny"}
    with pytest.raises(DriverError, match="no decision"):
        decide(["a", "b"], ["a"], [])
    with pytest.raises(DriverError, match="not pending"):
        decide(["a"], ["a", "typo"], [])


def test_capture_output_sees_python_and_file_descriptor_writes(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with capture_output() as captured:
        print("via print")
        sys.stderr.write("via sys.stderr\n")
        os.write(1, b"via fd 1\n")
        os.write(2, b"via fd 2\n")
    assert captured.stdout == "via print\nvia fd 1\n"
    assert captured.stderr == "via sys.stderr\nvia fd 2\n"
    print("after")
    assert capfd.readouterr() == ("after\n", "")


def test_crash_sink_matches_the_event_and_call(monkeypatch: pytest.MonkeyPatch) -> None:
    kills: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append(sig))
    sink = crash_sink("item", "call_1")
    tool_item = {
        "type": "item",
        "data": {"item": {"message": {"role": "tool", "tool_call_id": "call_1"}}},
    }
    other = {
        "type": "item",
        "data": {"item": {"message": {"role": "tool", "tool_call_id": "call_2"}}},
    }
    sink({"type": "tool.end", "data": {"call_id": "call_1"}})
    sink(other)
    assert kills == []
    sink(tool_item)
    assert kills == [signal.SIGKILL]
    crash_sink("tool.start")({"type": "tool.start", "data": {"call_id": "x"}})
    assert kills == [signal.SIGKILL, signal.SIGKILL]


# --- running scenarios -------------------------------------------------------------------------


async def test_result_json_shape_and_layout(real_provider: FakeProvider, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = await run_scenario("S01", "our", out=out, run_id="r1", provider=real_provider)
    directory = out / "runs" / "r1" / "S01" / "our"
    assert set(result) == RESULT_KEYS
    assert json.loads((directory / "result.json").read_text()) == result
    assert result["passed"] and result["error"] is None
    assert (result["v"], result["scenario"], result["impl"], result["run_id"]) == (
        1,
        "S01",
        "our",
        "r1",
    )
    assert result["stops"] == ["end_turn"]
    assert set(result["expect"]) == {"stops", "requests", "commits", "text_contains", "cost_usd"}
    assert set(result["invariants"]) == {"I1", "I2", "I3", "I5", "I7"}
    for check in [*result["expect"].values(), *result["invariants"].values()]:
        assert check["ok"] is True and isinstance(check["detail"], str)
    assert result["usage"] == {
        "input_tokens": result["usage"]["input_tokens"],
        "output_tokens": result["usage"]["output_tokens"],
        "cached_tokens": result["usage"]["cached_tokens"],
        "cost_usd": 0.001446,
        "cost_source": "provider",
    }
    assert result["requests"] == 1 and result["tool_runs"] == {}
    assert isinstance(result["duration_ms"], float)
    for name in ("log.sqlite", "events.ndjson", "wire/001.json", "wire/001.meta.json"):
        assert (directory / name).is_file(), name
    assert (directory / "wc" / "S01-our" / ".git").is_dir()
    types = [
        json.loads(line)["type"] for line in (directory / "events.ndjson").read_text().splitlines()
    ]
    assert types[0] == "turn.start" and types[-2:] == ["turn.end", "commit"]


async def test_a_run_id_is_never_reused(real_provider: FakeProvider, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = await run_scenario("S01", "our", out=out, run_id="r1", provider=real_provider)
    saved = (out / "runs" / "r1" / "S01" / "our" / "result.json").read_bytes()
    with pytest.raises(DriverError, match="already exists: pick another --run-id"):
        await run_scenario("S01", "our", out=out, run_id="r1", provider=real_provider)
    assert (out / "runs" / "r1" / "S01" / "our" / "result.json").read_bytes() == saved
    # Every run gets a fresh fakeprov cursor, so the same run id elsewhere starts over.
    again = await run_scenario(
        "S01", "our", out=tmp_path / "other", run_id="r1", provider=real_provider
    )
    assert first["passed"] and again["passed"], reason(again)
    assert again["requests"] == 1


async def test_failed_expectations_are_reported_per_key(
    provider: FakeProvider, tmp_path: Path
) -> None:
    expect = {"stops": ["end_turn"], "requests": 2, "text_contains": "Goodbye"}
    copy_scenario(tmp_path, "S01", "X01", expect=expect)
    result = await run_scenario("X01", "our", out=tmp_path / "out", run_id="r1", provider=provider)
    assert not result["passed"] and result["error"] is None
    assert [k for k, v in result["expect"].items() if not v["ok"]] == ["requests", "text_contains"]
    assert reason(result) == "requests: got 1, expected 2 (+1 more)"


class NoisyLoop(OurLoop):
    """Breaks I5: prints during its turn."""

    def run_turn(self, turn, tools, cancel):  # type: ignore[no-untyped-def]
        print("debug: turn started")
        return super().run_turn(turn, tools, cancel)


async def test_a_loop_that_prints_fails_i5(real_provider: FakeProvider, tmp_path: Path) -> None:
    result = await run_scenario(
        "S01",
        "our",
        out=tmp_path / "out",
        run_id="r1",
        provider=real_provider,
        loop_factory=NoisyLoop,
    )
    i5 = result["invariants"]["I5"]
    assert not result["passed"] and not i5["ok"]
    assert i5["info"]["stdout"] == "debug: turn started\n"
    assert reason(result).startswith("I5: 20 chars to stdout: 'debug: turn started'")


# Loops that break the contract in ways only timing or the tool events show. Each wraps OurLoop
# and changes one thing, so the scenario must fail for exactly that reason.


class _Tools:
    """A ToolHost that passes everything through; subclasses change `run`."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def specs(self):  # type: ignore[no-untyped-def]
        return self.inner.specs()

    def check(self, call):  # type: ignore[no-untyped-def]
        return self.inner.check(call)

    async def run(self, call):  # type: ignore[no-untyped-def]
        return await self.inner.run(call)


class _OneAtATime(_Tools):
    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.lock = asyncio.Lock()

    async def run(self, call):  # type: ignore[no-untyped-def]
        async with self.lock:
            return await self.inner.run(call)


class _Invents(_Tools):
    async def run(self, call):  # type: ignore[no-untyped-def]
        return ToolResult(call.id, True, "")


class _Shields(_Tools):
    async def run(self, call):  # type: ignore[no-untyped-def]
        return await asyncio.shield(self.inner.run(call))


class _FinishesFirst(_Tools):
    async def run(self, call):  # type: ignore[no-untyped-def]
        task = asyncio.ensure_future(self.inner.run(call))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task  # the cancel waits for the tool
            raise


def wrapping(tools_class: type[_Tools]) -> type[OurLoop]:
    class Wrapped(OurLoop):
        def run_turn(self, turn, tools, cancel):  # type: ignore[no-untyped-def]
            return super().run_turn(turn, tools_class(tools), cancel)

    return Wrapped


class LeaksCancelledError(OurLoop):
    """Raises CancelledError instead of ending a cancelled turn."""

    async def run_turn(self, turn, tools, cancel):  # type: ignore[no-untyped-def]
        async for event in super().run_turn(turn, tools, cancel):
            if event.type == "turn.end" and event.data.get("stop") == "cancelled":
                raise asyncio.CancelledError
            yield event


class PrintsAfterItsTurn(OurLoop):
    """Prints from a task 50 ms after each turn: in S05, while the approve child runs."""

    async def run_turn(self, turn, tools, cancel):  # type: ignore[no-untyped-def]
        try:
            async for event in super().run_turn(turn, tools, cancel):
                yield event
        finally:
            self.later = asyncio.get_running_loop().create_task(self._print_later())

    async def _print_later(self) -> None:
        await asyncio.sleep(0.05)
        print("late output from a loop task")


async def scenario_with(
    provider: FakeProvider, tmp_path: Path, sid: str, loop: type[OurLoop]
) -> dict[str, Any]:
    return await run_scenario(
        sid, "our", out=tmp_path / "out", run_id="r1", provider=provider, loop_factory=loop
    )


def failed_checks(result: dict[str, Any]) -> dict[str, str]:
    return {
        key: check["detail"]
        for group in ("expect", "invariants")
        for key, check in result[group].items()
        if not check["ok"]
    }


async def test_tools_run_one_at_a_time_fail_s03(
    real_provider: FakeProvider, tmp_path: Path
) -> None:
    result = await scenario_with(real_provider, tmp_path, "S03", wrapping(_OneAtATime))
    assert list(failed_checks(result)) == ["tools_overlap"] and result["error"] is None
    assert failed_checks(result)["tools_overlap"].startswith("not all running at once")


async def test_invented_tool_results_fail_i2(real_provider: FakeProvider, tmp_path: Path) -> None:
    # S12 checks no tool_runs; the result that no run produced still fails.
    result = await scenario_with(real_provider, tmp_path, "S12", wrapping(_Invents))
    assert failed_checks(result) == {"I2": "unrun_results: ['call_S12_1']"}


async def test_a_tool_left_running_after_a_cancel_fails(
    real_provider: FakeProvider, tmp_path: Path
) -> None:
    result = await scenario_with(real_provider, tmp_path, "S07", wrapping(_Shields))
    assert failed_checks(result) == {"I2": "unfinished: ['call_S07_1']"}
    # The driver cancelled the stray tool before judging, so its tool.end is on record (late)
    # and nothing reaches the next scenario.
    assert result["error"] == "tasks still running after aclose: ['ToolHostImpl.run']"
    assert result["invariants"]["I5"]["ok"]


async def test_a_cancel_that_waits_for_the_tool_is_too_slow(
    provider: FakeProvider, tmp_path: Path
) -> None:
    copy_scenario(tmp_path, "S07", "X07", engine={"delay_ms": 800})  # a shorter slow tool
    result = await scenario_with(provider, tmp_path, "X07", wrapping(_FinishesFirst))
    assert list(failed_checks(result)) == ["cancel_within_ms"] and result["error"] is None
    detail = failed_checks(result)["cancel_within_ms"]
    assert "X07-our.0: " in detail and detail.endswith("(limit 200 ms)")
    (slow,) = [ms for turn, ms in result_cancels(detail) if turn == "X07-our.1"]
    assert slow > 200


def result_cancels(detail: str) -> list[tuple[str, float]]:
    """(turn, ms) pairs from a cancel_within_ms detail."""
    pairs = detail.removeprefix("turn.end after the cancel: ").split(" (limit")[0].split(", ")
    return [(turn, float(ms.removesuffix(" ms"))) for turn, ms in (p.split(": ") for p in pairs)]


async def test_a_loop_that_cannot_load_is_the_error(
    real_provider: FakeProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = replace(loops.REGISTRY["our"], target="bakeoff.our_version:NoSuchLoop")
    monkeypatch.setitem(loops.REGISTRY, "our", entry)
    result = await run_scenario(
        "S01", "our", out=tmp_path / "out", run_id="r1", provider=real_provider
    )
    assert not result["passed"]
    assert (
        result["error"] == "LoopUnavailable: our: bakeoff.our_version:NoSuchLoop is not built yet"
    )


async def test_a_loop_that_leaks_cancelled_error_fails_only_its_scenario(
    real_provider: FakeProvider, tmp_path: Path
) -> None:
    result = await scenario_with(real_provider, tmp_path, "S07", LeaksCancelledError)
    assert not result["passed"]
    assert result["error"].startswith("CancelledError: a turn raised CancelledError")
    assert (tmp_path / "out" / "runs" / "r1" / "S07" / "our" / "result.json").is_file()


async def test_output_while_a_child_runs_fails_i5(
    real_provider: FakeProvider, tmp_path: Path
) -> None:
    result = await scenario_with(real_provider, tmp_path, "S05", PrintsAfterItsTurn)
    assert failed_checks(result) == {"I5": "29 chars to stdout: 'late output from a loop task'"}
    assert result["error"] is None and result["processes"][0]["command"] == "approve"


async def test_approve_in_a_new_process(real_provider: FakeProvider, tmp_path: Path) -> None:
    result = await run_scenario(
        "S05", "our", out=tmp_path / "out", run_id="r1", provider=real_provider
    )
    assert result["passed"], reason(result)
    (proc,) = result["processes"]
    assert proc["command"] == "approve" and proc["exit"] == 0
    assert proc["pid"] != os.getpid() and proc["summary"]["pid"] == proc["pid"]
    assert proc["summary"]["stop"] == "end_turn"
    assert result["stops"] == ["paused", "end_turn"]
    assert result["tool_runs"] == {"call_S05_1": 1, "call_S05_2": 1}


async def test_crash_in_a_child_and_resume(real_provider: FakeProvider, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = await run_scenario("S08", "our", out=out, run_id="r1", provider=real_provider)
    assert result["passed"], reason(result)
    (proc,) = result["processes"]
    assert (proc["command"], proc["exit"]) == ("turn", -signal.SIGKILL)
    assert "summary" not in proc and proc["pid"] != os.getpid()
    lines = (out / "runs" / "r1" / "S08" / "our" / "events.ndjson").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    crashed = [e for e in events if e["turn"] == "S08-our.0"]
    # The child died right after the call's result item was stored: turn 0 never ended, and the
    # next event is the resumed turn's start.
    last = crashed[-1]
    assert last["type"] == "item"
    assert last["data"]["item"]["message"]["tool_call_id"] == "call_S08_1"
    assert "turn.end" not in {e["type"] for e in crashed}
    assert events[len(crashed)]["type"] == "turn.start"
    assert events[len(crashed)]["data"]["resume"]["kind"] == "crash"
    assert result["tool_runs"] == {"call_S08_1": 1}


async def test_a_crash_point_that_never_comes_fails_the_run(
    provider: FakeProvider, tmp_path: Path
) -> None:
    driver = [
        {"crash_after": "permission.asked"},
        {"user": "Write hello.txt containing 'hello from S08', then confirm."},
        {"resume": "crash"},
    ]
    copy_scenario(tmp_path, "S08", "X08", driver=driver)
    result = await run_scenario("X08", "our", out=tmp_path / "out", run_id="r1", provider=provider)
    assert not result["passed"]
    assert "was to die of SIGKILL at its crash point, but exited with 0" in result["error"]
    assert result["processes"][0]["exit"] == 0


async def test_a_driver_step_that_cannot_run_is_the_error(
    provider: FakeProvider, tmp_path: Path
) -> None:
    driver = [
        {"user": "Validate a simple chat pipeline and save a README.md that describes it."},
        {"approve": {"allow": ["call_S05_1"]}, "new_process": True},
    ]
    copy_scenario(tmp_path, "S05", "X05", driver=driver)
    result = await run_scenario("X05", "our", out=tmp_path / "out", run_id="r1", provider=provider)
    assert not result["passed"]
    assert result["error"].startswith("DriverError: step 2")
    assert "not pending: ['call_S05_1']" in result["error"]
    assert result["stops"] == ["paused"]


async def test_run_matrix_writes_the_summary_and_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_scenario(tmp_path, "S01", "X01")
    wrong = {"stops": ["end_turn"], "text_contains": "Goodbye"}
    for sid in ("X02", "X03", "X05"):
        copy_scenario(tmp_path, "S01", sid, expect=wrong)
    copy_scenario(tmp_path, "S01", "X04", expect=wrong | {"requests": 2})
    goodbye = loops.KnownFailure(frozenset({"text_contains"}), "says hello, not goodbye")
    known = {"X03": goodbye, "X04": goodbye, "X01": goodbye} | {
        "X05": loops.KnownFailure(frozenset({"I5"}), "prints")
    }
    monkeypatch.setitem(loops.REGISTRY, "our", replace(loops.REGISTRY["our"], known_failures=known))
    out, seen = tmp_path / "out", []
    ids = ["X01", "X02", "X03", "X04", "X05"]
    summary = await run_matrix(
        ids,
        ["our"],
        out=out,
        run_id="m1",
        on_result=lambda r: seen.append(r["scenario"]),
        scenarios_dir=tmp_path / "scenarios",
    )
    assert seen == ids
    runs = out / "runs"
    assert json.loads((runs / "m1" / "summary.json").read_text()) == summary
    assert (runs / "latest").is_symlink() and os.readlink(runs / "latest") == "m1"
    assert not (runs / "m1" / ".wire").exists()
    assert (runs / "m1" / "X01" / "our" / "wire" / "001.json").is_file()
    assert summary["loops"]["our"]["target"] == "bakeoff.our_version:OurLoop"
    assert summary["loops"]["our"]["versions"]["httpx"]
    assert summary["git_sha"] is None or len(summary["git_sha"]) == 40
    cells = {sid: summary["matrix"][sid]["our"] for sid in ids}
    assert cells["X02"] == {
        "passed": False,
        "status": "FAIL",
        "reason": cells["X02"]["reason"],
        "expected_failure": None,
        "expected_checks": None,
        "expected_error": None,
        "duration_ms": cells["X02"]["duration_ms"],
    }
    assert cells["X02"]["reason"].startswith("text_contains: missing 'Goodbye' in 'Hello")
    assert (cells["X03"]["status"], cells["X03"]["expected_failure"]) == (
        "xfail",
        "says hello, not goodbye",
    )
    assert cells["X03"]["expected_checks"] == ["text_contains"]
    # Documented, but it fails one more check (X04) or another one (X05): not an xfail.
    assert cells["X04"]["status"] == cells["X05"]["status"] == "FAIL"
    assert cells["X01"]["status"] == "XPASS"  # a documented failure that passes
    assert unexpected(summary) == ["X01/our", "X02/our", "X04/our", "X05/our"]
    lines = format_matrix(summary).splitlines()
    assert lines[:6] == [
        "scenario  our",
        "X01       XPASS",
        "X02       FAIL",
        "X03       xfail",
        "X04       FAIL",
        "X05       FAIL",
    ]
    assert lines[6] == "  X01/our XPASS: passed, but documented to fail: says hello, not goodbye"
    assert lines[7].startswith("  X02/our FAIL: text_contains: missing 'Goodbye'")
    assert lines[8].startswith("  X03/our xfail: text_contains: missing 'Goodbye'")
    assert lines[9].startswith("  X04/our FAIL: text_contains: missing 'Goodbye'")
    assert lines[9].endswith(" (+1 more) [documented to fail only text_contains]")
    assert lines[10].endswith(" [documented to fail only I5]")
    assert lines[-1] == "our 0/5 pass, 3 FAIL, 1 XPASS, 1 xfail"

    await run_matrix(["X01"], ["our"], out=out, run_id="m2", scenarios_dir=tmp_path / "scenarios")
    assert os.readlink(runs / "latest") == "m2"


# Loops whose aclose leaves a task that outlives its cancel: `LingerLoop`'s task ends at a
# second cancel, `DeafLoop`'s never does. A module, because run_matrix loads loops by name.
STUBBORN_LOOPS = """
import asyncio

from bakeoff.our_version import OurLoop

TASKS = []


async def linger() -> None:
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await asyncio.sleep(3600)


async def deaf() -> None:
    while True:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass


class LingerLoop(OurLoop):
    async def aclose(self) -> None:
        await super().aclose()
        TASKS.append(asyncio.get_running_loop().create_task(linger()))
        await asyncio.sleep(0)  # it starts, so a cancel lands in its try


class DeafLoop(OurLoop):
    async def aclose(self) -> None:
        await super().aclose()
        TASKS.append(asyncio.get_running_loop().create_task(deaf()))
        await asyncio.sleep(0)  # it starts, so a cancel lands in its try
"""


async def test_a_task_that_outlives_its_cancel_stops_the_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stubborn_loops.py").write_text(STUBBORN_LOOPS)
    monkeypatch.syspath_prepend(str(tmp_path))
    entry = replace(loops.REGISTRY["our"], target="stubborn_loops:LingerLoop")
    monkeypatch.setitem(loops.REGISTRY, "our", entry)
    monkeypatch.setattr(scenario, "STRAY_WAIT_S", 0.2)
    for sid in ("X01", "X02"):
        copy_scenario(tmp_path, "S01", sid)
    out, seen = tmp_path / "out", []
    summary = await run_matrix(
        ["X01", "X02"],
        ["our"],
        out=out,
        run_id="m1",
        on_result=lambda r: seen.append(r["scenario"]),
        scenarios_dir=tmp_path / "scenarios",
    )
    assert seen == ["X01"]  # X02 would have shared the event loop with the task
    result = json.loads((out / "runs" / "m1" / "X01" / "our" / "result.json").read_text())
    assert result["error"] == (
        "tasks still running after aclose: ['linger'];"
        " still running 0.2 s after being cancelled: ['linger']"
    )
    assert summary["matrix"]["X01"]["our"]["status"] == "FAIL"
    assert summary["stopped"] == (
        "X01/our left tasks running after they were cancelled: ['linger'];"
        " 1 of 2 runs did not start"
    )
    assert json.loads((out / "runs" / "m1" / "summary.json").read_text()) == summary
    assert format_matrix(summary).splitlines()[-1] == f"stopped: {summary['stopped']}"


def test_the_scenario_command_ends_even_if_a_task_ignores_every_cancel(tmp_path: Path) -> None:
    (tmp_path / "stubborn_loops.py").write_text(STUBBORN_LOOPS)
    out = tmp_path / "out"
    script = f"""
import sys
from dataclasses import replace
from bakeoff import loops
from bakeoff.cli import main
from bakeoff.shared import scenario
scenario.STRAY_WAIT_S = 0.2
loops.REGISTRY["our"] = replace(loops.REGISTRY["our"], target="stubborn_loops:DeafLoop")
sys.exit(main(["scenario", "S01", "S13", "--impl", "our", "--out", {str(out)!r}, "--run-id", "k1"]))
"""
    path = os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")]))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1, proc.stderr
    assert "bakeoff scenario: stopped: S01/our left tasks running after they were cancelled:" in (
        proc.stderr
    )
    summary = json.loads((out / "runs" / "k1" / "summary.json").read_text())
    assert list(summary["matrix"]) == ["S01"] and summary["stopped"].startswith("S01/our left")


def test_a_documented_failure_may_name_the_driver_error_it_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stops = frozenset({"stops"})
    known = {
        "X01": loops.KnownFailure(stops, "never pauses", "has no paused turn"),
        "X02": loops.KnownFailure(stops, "wrong stop"),
    }
    monkeypatch.setitem(loops.REGISTRY, "our", replace(loops.REGISTRY["our"], known_failures=known))

    def result(sid: str, error: str | None, *failed: str) -> dict[str, Any]:
        expect = {key: {"ok": key not in failed, "detail": ""} for key in ("stops", "requests")}
        return {"impl": "our", "scenario": sid, "passed": False, "error": error,
                "expect": expect, "invariants": {}}  # fmt: skip

    no_pause = "DriverError: step 2 {...}: thread X01-our has no paused turn (last turn is error)"
    assert scenario.status(result("X01", no_pause, "stops")) == "xfail"
    assert scenario.status(result("X01", None, "stops")) == "FAIL"  # the error did not come
    assert scenario.status(result("X01", "DriverError: other", "stops")) == "FAIL"
    assert scenario.status(result("X01", no_pause, "stops", "requests")) == "FAIL"
    assert scenario.status(result("X02", None, "stops")) == "xfail"
    assert scenario.status(result("X02", no_pause, "stops")) == "FAIL"  # an undocumented error


async def test_the_scenario_model_reaches_the_loop_unchanged(
    real_provider: FakeProvider, tmp_path: Path
) -> None:
    seen: list[ModelConfig] = []

    class Records(OurLoop):
        async def run_turn(self, turn, tools, cancel):  # type: ignore[override]
            seen.append(turn.model)
            yield Event("turn.end", {"stop": "end_turn", "steps": 0})

    out = tmp_path / "out"
    await run_scenario(
        "R01", "our", out=out, run_id="r1", provider=real_provider, loop_factory=Records
    )
    [model] = seen
    assert (model.kind, model.model, model.reasoning) == (
        "openai_responses",
        "gpt-6-luna",
        {"effort": "xhigh"},
    )
    assert model.temperature is None  # the scenario's null: reasoning models reject one
    assert model.base_url.endswith("/v1")
    # A child process (bakeoff approve/turn) rebuilds the same config from the thread's meta.
    log = SessionLog(out / "runs" / "r1" / "R01" / "our" / "log.sqlite")
    try:
        meta = log.get_thread("R01-our")["meta"]["model"]
    finally:
        log.close()
    assert scenario.model_from_meta(meta, "dummy") == replace(model, api_key="dummy")
    # Without a `temperature` in the scenario, ModelConfig's default stays.
    seen.clear()
    await run_scenario(
        "S01", "our", out=out, run_id="r1", provider=real_provider, loop_factory=Records
    )
    assert (seen[0].kind, seen[0].temperature) == ("openrouter", 0.0)


def test_unexpected_lists_failures_and_passes_of_documented_failures() -> None:
    cell = {"passed": False, "reason": "x", "expected_failure": None, "duration_ms": 1.0}
    summary = {
        "matrix": {
            "S01": {"our": cell | {"status": "FAIL"}},
            "S02": {"our": cell | {"status": "pass", "passed": True}},
            "S03": {"our": cell | {"status": "XPASS", "passed": True}},
            "S11": {"our": cell | {"status": "xfail"}},
        }
    }
    assert unexpected(summary) == ["S01/our", "S03/our"]


# --- the loop registry -------------------------------------------------------------------------


def test_load_tells_missing_loops_from_broken_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules = {
        "fake_loop_class_absent": "class Other: pass\n",
        "fake_loop_extra_absent": "import no_such_dependency_xyz\n",  # an extra not installed
        "fake_loop_renamed_submodule": "import httpx.no_such_submodule\n",  # a library change
        "fake_loop_bad_first_party": "import bakeoff.no_such_module\n",
        "fake_loop_raises": "raise RuntimeError('boom')\n",
    }
    for name, source in modules.items():
        (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    expected = {
        "fake_loop_not_there": True,  # the loop's own module is not built yet
        "fake_loop_class_absent": True,
        "fake_loop_extra_absent": True,
        "fake_loop_renamed_submodule": False,
        "fake_loop_bad_first_party": False,
        "fake_loop_raises": False,
    }
    for module, missing in expected.items():
        entry = loops.LoopEntry("probe", f"{module}:ProbeLoop", ())
        monkeypatch.setitem(loops.REGISTRY, "probe", entry)
        with pytest.raises(loops.LoopUnavailable) as info:
            loops.load("probe")
        assert info.value.missing is missing, (module, info.value.reason)
        assert loops.available(["probe"]) == {}


def test_scenario_ids_are_every_file_in_order() -> None:
    ids = scenario.scenario_ids()
    assert ids[0] == "R01" and ids[-1] == "S15" and {"R05", "S01", "S12b"} <= set(ids)
    assert ids == sorted(p.stem for p in SCENARIOS_DIR.glob("*.json"))


async def test_cancel_command_reaches_a_worker_turn(
    real_provider: FakeProvider, tmp_path: Path
) -> None:
    """`bakeoff cancel` from another process stops the turn a `bakeoff turn` worker runs."""
    db = tmp_path / "log.sqlite"
    ws = scenario.Workspace(db)
    model = ModelConfig(base_url=real_provider.base_url("S07", "r1", "our"), model="m")
    ws.runner.new_thread(impl="our", system="s", rules={"*": "allow"}, model=model, thread_id="t1")
    ws.close()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    bakeoff = (sys.executable, "-m", "bakeoff.cli")
    pipes = {"stdout": asyncio.subprocess.PIPE, "stderr": asyncio.subprocess.PIPE, "env": env}
    # S07's first answer streams "Let me", then stalls until the client leaves.
    worker = await asyncio.create_subprocess_exec(
        *bakeoff, "turn", "t1", f"--db={db}", "--user=hi", **pipes
    )
    events = tmp_path / "events.ndjson"
    for _ in range(200):
        if events.exists() and '"text.delta"' in events.read_text():
            break
        await asyncio.sleep(0.05)
    cancel = await asyncio.create_subprocess_exec(*bakeoff, "cancel", "t1", f"--db={db}", **pipes)
    assert json.loads((await cancel.communicate())[0]) == {"thread": "t1", "cancel": "requested"}
    out, err = await asyncio.wait_for(worker.communicate(), 10)
    assert err == b""
    summary = json.loads(out)
    assert (summary["stop"], summary["pid"]) == ("cancelled", worker.pid)


async def test_two_matrices_with_one_run_id_cannot_both_start(tmp_path):
    """Greptile #4104014805: the run directory is reserved atomically, so of two matrices
    started at once on the same run id exactly one runs; the other is refused."""
    from bakeoff.shared.scenario import DriverError, run_matrix

    first = run_matrix(["S01"], ["our"], out=tmp_path, run_id="race")
    second = run_matrix(["S01"], ["our"], out=tmp_path, run_id="race")
    outcomes = await asyncio.gather(first, second, return_exceptions=True)
    assert sum(isinstance(o, DriverError) for o in outcomes) == 1
    assert any(isinstance(o, dict) for o in outcomes)
    assert (tmp_path / "runs" / "race" / "S01" / "our" / "result.json").is_file()


async def test_a_provider_that_cannot_start_releases_the_run_id(tmp_path, monkeypatch):
    """Greptile #4104055671: nothing ran, so the reservation is released and the same run id
    can be retried."""
    from bakeoff.fakeprov.server import FakeProvider
    from bakeoff.shared.scenario import run_matrix

    real_start = FakeProvider.start
    calls = []

    def failing_once(self):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("address already in use (simulated)")
        return real_start(self)

    monkeypatch.setattr(FakeProvider, "start", failing_once)
    with pytest.raises(OSError, match="simulated"):
        await run_matrix(["S01"], ["our"], out=tmp_path, run_id="retry")
    assert not (tmp_path / "runs" / "retry").exists()
    summary = await run_matrix(["S01"], ["our"], out=tmp_path, run_id="retry")
    assert summary and (tmp_path / "runs" / "retry" / "S01" / "our" / "result.json").is_file()

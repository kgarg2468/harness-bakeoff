"""The report page: built from small fixture folders written by each test."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from bakeoff.report import build, data
from bakeoff.shared.contract import Item
from bakeoff.shared.sessionlog import SessionLog, event_row, item_to_json

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
SECTIONS = ["scorecard", "matrix", "replay", "wire", "live", "wins"]
EVIL = "</script><script>alert(1)</script>"
_VOID = {"meta", "input", "br", "img", "hr", "link", "col", "source", "wbr", "area", "base"}


class Page(HTMLParser):
    """Checks that every tag is closed in order, and collects sections and the embedded data."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.problems: list[str] = []
        self.sections: list[str] = []
        self._in_data = False
        self.data_text = ""
        self.feed(text)
        self.close()
        if self.stack:
            self.problems.append(f"unclosed: {self.stack}")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "section" and a.get("data-section"):
            self.sections.append(str(a["data-section"]))
        self._in_data = tag == "script" and a.get("id") == "report-data"
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        pass  # <rect/> and friends: open and closed at once

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"</{tag}> at {self.getpos()} but open: {self.stack[-3:]}")
            return
        self.stack.pop()
        self._in_data = False

    def handle_data(self, text: str) -> None:
        if self._in_data:
            self.data_text += text

    @property
    def data(self) -> dict[str, Any]:
        return json.loads(self.data_text)


def env(thread: str, turn: str, seq: int, t_us: int, type_: str, **data_: Any) -> dict[str, Any]:
    return {"v": 1, "thread": thread, "turn": turn, "impl": "x", "seq": seq, "t_us": t_us,
            "type": type_, "data": data_}  # fmt: skip


def item(turn: str, id_: str, message: dict[str, Any]) -> dict[str, Any]:
    """The data of an `item` event, as the runner stores it."""
    return {"item": item_to_json(Item(id_, turn, message))}


def our_events(thread: str = "S01-our") -> list[dict[str, Any]]:
    """One turn: a read_file call started while the stream runs, its result, the answer."""
    t = f"{thread}.0"
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
    }
    return [
        env(thread, t, 1, 0, "turn.start", turn_id=t),
        env(thread, t, 2, 50, "item", **item(t, "u1", {"role": "user", "content": "Read a.txt"})),
        env(thread, t, 3, 100, "request.start", step=1, attempt=1),
        env(thread, t, 4, 900, "tool_call.ready", call_id="c1", name="read_file", arguments='{"path": "a.txt"}'),
        env(thread, t, 5, 1000, "tool.start", call_id="c1", name="read_file", read_only=True),
        env(thread, t, 6, 1500, "item", **item(t, "a1", {"role": "assistant", "content": None, "tool_calls": [call]})),
        env(thread, t, 7, 2100, "tool.end", call_id="c1", name="read_file", ok=True, ms=1.1),
        env(thread, t, 8, 2200, "item", **item(t, "r1", {"role": "tool", "tool_call_id": "c1", "content": EVIL})),
        env(thread, t, 9, 2300, "request.start", step=2, attempt=1),
        env(thread, t, 10, 2800, "text.delta", text="All "),
        env(thread, t, 11, 2900, "usage", step=2, input_tokens=900, output_tokens=40, cached_tokens=0, cost_usd=0.0033, cost_source="provider"),
        env(thread, t, 12, 3000, "item", **item(t, "a2", {"role": "assistant", "content": "All done."})),
        env(thread, t, 13, 3100, "turn.end", stop="end_turn", steps=2),
        env(thread, t, 14, 5000, "commit", sha="abc1234def", files=["a.txt"]),
    ]  # fmt: skip


def write_pydantic_log(path: Path, thread: str = "S01-pydantic") -> None:
    """A session log with a turn that dies mid-turn (no turn.end) and its crash resume."""
    log = SessionLog(path)
    log.create_thread(thread, impl="pydantic", system="sys", meta={})
    t0 = log.start_turn(thread, "user")["id"]
    user = Item("u1", t0, {"role": "user", "content": "Read a.txt"})
    log.append_item(
        thread,
        user,
        [
            event_row(env(thread, t0, 1, 0, "turn.start", turn_id=t0)),
            event_row(env(thread, t0, 2, 10, "item", **item(t0, "u1", user.message))),
        ],
    )
    log.append_events(
        [
            event_row(env(thread, t0, 3, 200, "request.start", step=1, attempt=1)),
            event_row(env(thread, t0, 4, 700, "tool.start", call_id="c1", name="read_file")),
        ]
    )
    t1 = log.start_turn(thread, "crash")["id"]  # the worker died: turn 0 stays "running"
    resume = {"kind": "crash", "decisions": {}, "reason": None}
    log.append_events(
        [
            event_row(env(thread, t1, 5, 0, "turn.start", turn_id=t1, resume=resume)),
            event_row(env(thread, t1, 6, 300, "request.start", step=1, attempt=1)),
            event_row(
                env(
                    thread,
                    t1,
                    7,
                    400,
                    "retry",
                    attempt=1,
                    status=429,
                    wait_ms=1000,
                    reason="rate limited",
                )
            ),
            event_row(env(thread, t1, 8, 1400, "request.start", step=1, attempt=2)),
            event_row(env(thread, t1, 9, 1600, "text.delta", text="Done")),
        ]
    )
    answer = Item("a1", t1, {"role": "assistant", "content": "Done."})
    log.append_item(
        thread,
        answer,
        [event_row(env(thread, t1, 10, 1700, "item", **item(t1, "a1", answer.message)))],
    )
    log.set_turn_status(
        t1,
        "done",
        stop="end_turn",
        commit_sha="fff0000",
        events=[
            event_row(env(thread, t1, 11, 1800, "turn.end", stop="end_turn", steps=1)),
            event_row(env(thread, t1, 12, 2500, "commit", sha="fff0000", files=[])),
        ],
    )
    log.close()  # fmt: skip


def result(scenario: str, impl: str, passed: bool, **extra: Any) -> dict[str, Any]:
    expect = {"stops": {"ok": True, "detail": "stops ['end_turn']"}}
    if not passed:
        expect["tool_runs"] = {"ok": False, "detail": "c9 ran 1x, want 0"}
    invariants = {
        n: {"ok": True, "detail": f"{n} holds", "info": {}} for n in ("I1", "I2", "I3", "I5", "I7")
    }
    invariants["I1"]["info"] = {"requests": 2, "byte_prefix": True}
    return {"v": 1, "run_id": "r1", "scenario": scenario, "title": f"{scenario} title", "impl": impl,
            "stops": ["end_turn"], "expect": expect, "invariants": invariants, "requests": 2,
            "tool_runs": {"c1": 1}, "usage": {"input_tokens": 900, "output_tokens": 40, "cached_tokens": 0,
            "cost_usd": 0.0033, "cost_source": "provider"}, "duration_ms": 42.5, "passed": passed,
            "error": None, **extra}  # fmt: skip


def body(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    tools = [{"type": "function", "function": {"name": "read_file", "description": "x" * 50}}]
    return {"model": "m", "stream": True, **extra, "tools": tools, "messages": messages}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def out(tmp_path: Path) -> Path:
    """out/ with a run of two scenarios (S01 both loops, S02 only ours), metrics and a live run."""
    out = tmp_path / "out"
    run = out / "runs" / "r1"
    s01 = run / "S01"
    write_json(s01 / "our" / "result.json", result("S01", "our", True))
    (s01 / "our" / "events.ndjson").write_text("\n".join(map(json.dumps, our_events())) + "\n")
    system = {"role": "system", "content": "sys"}
    user = {"role": "user", "content": "Read a.txt"}
    for i, extra in enumerate(({}, {"session_id": "s"})):
        wire = s01 / "our" / "wire"
        write_json(wire / f"00{i + 1}.json", body([system, user][: i + 1], **extra))
        write_json(wire / f"00{i + 1}.meta.json", {"conn_id": 1, "status": 200, "t_us": 10 + i})
    write_json(s01 / "pydantic" / "result.json", result("S01", "pydantic", False))
    write_pydantic_log(s01 / "pydantic" / "log.sqlite")
    # The fake server's own layout under wire/: <scenario>/<run>/<impl>/NNN.json
    nested = s01 / "pydantic" / "wire" / "S01" / "r1" / "pydantic"
    write_json(nested / "001.json", body([system, user], tool_choice="auto"))
    write_json(run / "S02" / "our" / "result.json", result("S02", "our", False))
    write_json(run / "summary.json", {
        "v": 1, "run_id": "r1", "git": {"sha": "0123456789abcdef", "dirty": False},
        "loops": {"our": {"class": "bakeoff.our_version:OurLoop"}},
        "matrix": {"S02": {"our": {"passed": False, "reason": "known limit", "status": "xfail"}}},
    })  # fmt: skip
    (out / "runs" / "latest").symlink_to("r1")
    write_json(out / "metrics.json", {
        "schema": 1, "git": {"sha": "fedcba9876543210"},
        "loc": {"loops": {
            "our_version": {"total": {"files": 5, "code": 698, "comment": 18, "docstring": 90, "blank": 111},
                            "ported": {"Pi": {"code": 332}}, "imports_outside": [], "other_files": []},
            "pydantic_version": {"total": {"files": 3, "code": 512, "comment": 9, "docstring": 40, "blank": 60}},
        }, "shared": {"total": {"files": 13, "code": 1100, "comment": 47, "docstring": 211, "blank": 251}}},
        "bench": {"config": {"chunks": 500}, "loops": {
            "our_version": {"wall_ms": {"baseline": {"p50": 12.47}, "overhead": {"p50": 3.06, "p95": 3.8}},
                            "overhead_us_per_frame": {"wall": {"p50": 6.01}}},
            "pydantic_version": {"error": "bench exploded"}}},
        "deps": {
            "our_version": {"distributions": 9, "site_packages_mb": 4.2, "import_ms": 35.5,
                            "installed": {"httpx": "0.28.1"}, "third_party_code": {"loop_turn": {"total": 12000}}},
            "pydantic_version@2.31.1": {"distributions": 38, "site_packages_mb": 61.3, "import_ms": 850.0,
                                        "installed": {"pydantic-ai-slim": "2.31.1"}},
            "pydantic_version@latest": {"distributions": 40, "site_packages_mb": 70.1, "import_ms": None,
                                        "import_error": "ImportError: nope"},
        },
    })  # fmt: skip
    for impl, seconds in (("our", 2.5), ("pydantic", 3.25)):
        write_json(out / "live" / "L1" / impl / "result.json", {
            "v": 1, "run_id": "L1", "impl": impl, "model": "gpt-test", "prompt": "What is RocketRide?",
            "final_text": f"answer from {impl}", "stops": ["end_turn"], "requests": 3, "tool_runs": {"c": 1},
            "usage": {"input_tokens": 1234, "output_tokens": 56, "cached_tokens": 0, "cost_usd": None,
                      "cost_source": "none"}, "duration_ms": seconds * 1000, "error": None,
        })  # fmt: skip
    return out


def make(out: Path, **kwargs: Any) -> str:
    args = {"runs": out / "runs" / "latest", "live": out / "live", "metrics": out / "metrics.json"}
    return build.build(**{**args, **kwargs, "now": NOW, "discovered": []})


def test_full_page_is_well_formed_with_every_section(out: Path) -> None:
    html = make(out)
    page = Page(html)
    assert page.problems == []
    assert page.sections == SECTIONS
    assert "<link" not in html and "src=" not in html  # self-contained: no external assets
    assert "2026-09-25 12:00 UTC" in html
    # scorecard numbers
    for text in ("698", "332 ported", "512", "3.06 ms", "9</b> packages", "35.5 ms", "850.0 ms"):
        assert text in html
    assert "bench exploded" in html and "pydantic-ai ≥ 2.32" in html
    # matrix: pass, fail with its reason, and the summary's xfail
    assert 'id="cell-S01-our"' in html and 'class="cell fail"' in html
    assert "tool_runs: c9 ran 1x, want 0" in html
    assert 'class="cell xfail"' in html and "known limit" in html
    # live runs: prompt, answers, tokens, latency, steps
    assert "What is RocketRide?" in html and "answer from pydantic" in html
    assert "1,234" in html and "3.25 s" in html
    # where each wins: a measured win for A (fewer own lines), linked to its evidence
    assert "Fewer lines of its own code to maintain" in html and 'href="#loc"' in html
    assert "Answered faster: B 2.50 s vs A 3.25 s (live run L1, one sample" in html
    assert "Sent fewer input tokens" not in html  # a tie is nobody's win


def test_replay_and_wire_data(out: Path) -> None:
    scenarios = {s["id"]: s for s in Page(make(out)).data["scenarios"]}
    ours = scenarios["S01"]["runs"]["our"]["replay"]
    kinds = [c["k"] for c in ours["cards"]]
    assert kinds == ["user", "assistant", "result", "assistant", "end", "commit"]
    (tool,) = ours["lanes"]["tools"]
    assert tool["eager"] is True and tool["t0"] == 1.0 and tool["t1"] == 2.1  # ms
    assert [m["end"] for m in ours["lanes"]["model"]] == ["ok", "ok"]
    # From the session log: the killed turn gets a crash card, the resume retried once.
    theirs = scenarios["S01"]["runs"]["pydantic"]["replay"]
    assert [t["kind"] for t in theirs["turns"]] == ["user", "crash"]
    kinds = [c["k"] for c in theirs["cards"]]
    assert kinds == ["user", "crash", "resume", "retry", "assistant", "end", "commit"]
    assert theirs["stats"]["retries"] == 1
    assert [m["end"] for m in theirs["lanes"]["model"]] == ["killed", "retry", "ok"]
    # Wire: the tool list is stored once for all three bodies, found in both layouts.
    pool = Page(make(out)).data["pool"]
    our_wire = scenarios["S01"]["runs"]["our"]["wire"]
    their_wire = scenarios["S01"]["runs"]["pydantic"]["wire"]
    assert [w["n"] for w in our_wire] == [1, 2] and len(their_wire) == 1
    assert our_wire[1]["meta"] == {"conn_id": 1, "status": 200, "t_us": 11}
    refs = {w["body"]["tools"]["$ref"] for w in [*our_wire, *their_wire]}
    assert len(refs) == 1
    assert list(our_wire[1]["body"]) == ["model", "stream", "session_id", "tools", "messages"]
    messages = [pool[i] for i in our_wire[1]["body"]["messages"]["$refs"]]
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Read a.txt"},
    ]


def test_recorded_text_cannot_break_out_of_the_page(out: Path) -> None:
    html = make(out)
    assert EVIL not in html
    assert Page(html).problems == []
    tool_result = next(
        c
        for c in Page(html).data["scenarios"][0]["runs"]["our"]["replay"]["cards"]
        if c["k"] == "result"
    )
    assert tool_result["text"] == EVIL  # intact in the data, escaped in the page


def test_every_section_degrades_without_data(tmp_path: Path) -> None:
    html = build.build(runs=tmp_path / "nope", live=None, metrics=None, now=NOW, discovered=[])
    page = Page(html)
    assert page.problems == []
    assert page.sections == SECTIONS
    assert page.data["scenarios"] == []
    for text in ("No loops found", "No scenario runs found", "Nothing to replay yet",
                 "No wire recordings yet", "No live runs yet", "Not measured yet"):  # fmt: skip
        assert text in html


def test_partial_data_degrades_per_section(out: Path, tmp_path: Path) -> None:
    (out / "metrics.json").write_text("{not json")
    html = make(out, live=tmp_path / "no-live")
    assert Page(html).problems == []
    assert "metrics.json: unreadable" in html  # listed in the data notes
    assert "Not measured yet." in html and "No live runs yet." in html
    assert 'class="cell pass"' in html  # the runs still show


def test_output_is_deterministic(out: Path) -> None:
    first, second = make(out), make(out)
    assert first == second
    later = build.build(
        runs=out / "runs" / "latest", live=out / "live", metrics=out / "metrics.json",
        now=datetime(2027, 1, 1, tzinfo=UTC), discovered=[],
    )  # fmt: skip
    changed = [
        (a, b) for a, b in zip(first.splitlines(), later.splitlines(), strict=True) if a != b
    ]
    assert changed and all("2026-09-25 12:00 UTC" in a for a, _ in changed)


def test_over_the_size_budget_strings_are_cut(out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    big = "y" * 2_500  # under data.WIRE_LIMIT, so only the budget can cut it
    wire = out / "runs" / "r1" / "S01" / "our" / "wire"
    write_json(wire / "003.json", body([{"role": "user", "content": big}]))
    full = make(out)
    assert big in full
    monkeypatch.setattr(build, "BUDGET_BYTES", len(full.encode()) - 500)
    cut = make(out)
    assert big not in cut and "more characters not shown" in cut
    assert len(cut.encode()) <= build.BUDGET_BYTES
    assert Page(cut).problems == []


def test_long_wire_strings_are_clipped_with_a_pointer_to_the_file(out: Path) -> None:
    wire = out / "runs" / "r1" / "S01" / "our" / "wire"
    write_json(wire / "003.json", body([{"role": "user", "content": "z" * (data.WIRE_LIMIT + 10)}]))
    pool = Page(make(out)).data["pool"]
    clipped = next(
        v["content"] for v in pool if isinstance(v, dict) and "zzz" in str(v.get("content"))
    )
    assert clipped.endswith(
        "[10 more characters not shown; the full text is in S01/our/wire/003.json]"
    )


def test_session_log_is_read_without_touching_it(out: Path) -> None:
    log = out / "runs" / "r1" / "S01" / "pydantic" / "log.sqlite"
    before = {p.name: p.read_bytes() for p in log.parent.glob("log.sqlite*")}
    make(out)
    assert {p.name: p.read_bytes() for p in log.parent.glob("log.sqlite*")} == before


def test_parallel_tools_are_stacked_on_separate_rows() -> None:
    t = "T.0"
    events = [
        env("T", t, 1, 0, "turn.start", turn_id=t),
        env("T", t, 2, 100, "tool.start", call_id="a", name="x"),
        env("T", t, 3, 200, "tool.start", call_id="b", name="x"),
        env("T", t, 4, 300, "tool.end", call_id="a", name="x", ok=True),
        env("T", t, 5, 350, "tool.start", call_id="c", name="x"),
        env("T", t, 6, 400, "tool.end", call_id="b", name="x", ok=False),
        env("T", t, 7, 500, "tool.end", call_id="c", name="x", ok=True),
        env("T", t, 8, 600, "turn.end", stop="end_turn", steps=1),
    ]
    replay = data.build_replay(events, [])
    rows = {s["call"]: (s["row"], s["ok"]) for s in replay["lanes"]["tools"]}
    assert rows == {"a": (0, True), "b": (1, False), "c": (0, True)}
    assert replay["turns"][0]["rows"] == 2


def test_registry_skips_loops_that_are_missing_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "REGISTRY", (
        data.LoopInfo("our", "bakeoff.our_version:OurLoop", "B", "our loop", "teal"),
        data.LoopInfo("ghost", "bakeoff.not_merged_yet:GhostLoop", "G", "ghost", "grey"),
        data.LoopInfo("empty", "bakeoff.report:NoSuchLoop", "E", "empty", "grey"),
    ))  # fmt: skip
    assert data.discover() == ["our"]
    assert data.loop_info("mystery").letter == "?"  # a loop the registry does not know


def test_main_writes_the_page(out: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert build.main(["--out-dir", str(out)]) == 0
    page = out / "report.html"
    assert Page(page.read_text()).sections == SECTIONS
    assert f"wrote {page}" in capsys.readouterr().out

"""The report page: built from small fixture folders written by each test."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from bakeoff.report import build, data, render
from bakeoff.shared.contract import Item, ModelConfig
from bakeoff.shared.sessionlog import SessionLog, event_row, item_to_json

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
SECTIONS = ["scorecard", "matrix", "replay", "wire", "live", "wins"]
EVIL = "</script><script>alert(1)</script>"
_VOID = {"meta", "input", "br", "img", "hr", "link", "col", "source", "wbr", "area", "base"}


class Page(HTMLParser):
    """Checks that every tag is closed in order, and collects sections, the glossary terms used
    in <main> and the embedded data."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.problems: list[str] = []
        self.sections: list[str] = []
        self.terms: set[str] = set()
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
        if a.get("data-term") and "main" in self.stack:
            self.terms.add(str(a["data-term"]))
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
        }, "shared": {"total": {"files": 13, "code": 1100, "comment": 47, "docstring": 211, "blank": 251},
                      "ported": {"OpenCode": {"code": 322}, "RocketRide": {"code": 300}}}},
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
            "base_url": "https://api.openai.com/v1",
            "final_text": f"answer from {impl}", "stops": ["end_turn"], "requests": 3, "tool_runs": {"c": 1},
            "usage": {"input_tokens": 1234, "output_tokens": 56, "cached_tokens": 0, "cost_usd": None,
                      "cost_source": "none"}, "duration_ms": seconds * 1000, "passed": True,
            "error": None,
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
    assert "BYOK" in html and "api.openai.com" in html and "/v1" not in html
    # the shared layer's ported lines are counted too
    assert "<th>shared (once)</th>" in html and "622</td>" in html
    # where each wins: a measured win for A (fewer own lines), linked to its evidence
    assert "Fewer lines of its own code to maintain" in html and 'href="#loc"' in html
    # counts only over the scenarios both loops ran (S02 has no A run)
    assert "Passes more scenarios: B 1 vs A 0, of the 1 both loops ran." in html
    # one live run is not evidence of speed: shown as too close to call
    assert "Answered faster" not in html and "Sent fewer input tokens" not in html
    assert "one live run, too few to call" in html
    assert "as our own <span" in html and "written without an agent framework" in html


def test_every_glossary_term_has_a_tooltip_on_the_page(out: Path) -> None:
    used = Page(make(out)).terms
    assert {label for label, _ in render.GLOSSARY.values()} - used == set()


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


def test_input_problems_show_even_without_a_scenario_run(tmp_path: Path) -> None:
    """An unreadable metrics.json or live result is a data note even when no run folder exists."""
    (tmp_path / "metrics.json").write_text("{not json")
    (tmp_path / "live" / "L1" / "our").mkdir(parents=True)
    (tmp_path / "live" / "L1" / "our" / "result.json").write_text("{broken")
    html = build.build(
        runs=tmp_path / "runs", live=tmp_path / "live", metrics=tmp_path / "metrics.json",
        now=NOW, discovered=[],
    )  # fmt: skip
    assert Page(html).problems == []
    assert "2 data notes" in html
    assert "live/L1/our/result.json: unreadable" in html and "metrics.json: unreadable" in html


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


def test_identical_long_wire_strings_stay_identical_when_clipped(out: Path) -> None:
    long = {"role": "tool", "tool_call_id": "c1", "content": "z" * (data.WIRE_LIMIT + 10)}
    run = out / "runs" / "r1" / "S01"
    write_json(run / "our" / "wire" / "003.json", body([long]))
    write_json(run / "pydantic" / "wire" / "S01" / "r1" / "pydantic" / "002.json", body([long]))
    page = Page(make(out)).data
    clipped = [v for v in page["pool"] if isinstance(v, dict) and "zzz" in str(v.get("content"))]
    # One pool entry for both loops, and a note that names no file (it would differ per loop).
    assert len(clipped) == 1
    assert clipped[0]["content"].endswith("\n…[10 more characters not shown]")
    (s01,) = [s for s in page["scenarios"] if s["id"] == "S01"]
    ours, theirs = s01["runs"]["our"]["wire"][2], s01["runs"]["pydantic"]["wire"][1]
    assert ours["body"] == theirs["body"]
    assert ours["clipped"] == theirs["clipped"] == {"strings": 1, "chars": 10}
    assert ours["file"] == "S01/our/wire/003.json"
    assert theirs["file"] == "S01/pydantic/wire/S01/r1/pydantic/002.json"


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


def test_each_call_is_one_tool_run_whatever_the_order_of_its_events() -> None:
    """After turn.end, c1's tool.end was stored before its tool.start (late events): one row, from
    its first to its last event, finished and marked late. c2 started twice and ended once: one
    row that never finished and says it started twice."""
    t = "T.0"
    assistant = {"role": "assistant", "content": None,
                 "tool_calls": [call("c1", "read_file"), call("c2", "read_file")]}  # fmt: skip
    events = [
        env("T", t, 1, 0, "turn.start", turn_id=t),
        env("T", t, 2, 100, "request.start", step=1, attempt=1),
        env("T", t, 3, 200, "item", **item(t, "a1", assistant)),
        env("T", t, 4, 300, "tool.start", call_id="c2", name="read_file", read_only=True),
        env("T", t, 5, 400, "tool.end", call_id="c2", name="read_file", ok=True),
        env("T", t, 6, 450, "tool.start", call_id="c2", name="read_file", read_only=True),
        env("T", t, 7, 540, "turn.end", stop="cancelled", steps=1),
    ]
    late = [
        {"t_us": 610, "type": "tool.end", "data": {"call_id": "c1", "name": "read_file", "ok": True}},
        {"t_us": 560, "type": "tool.start", "data": {"call_id": "c1", "name": "read_file"}},
    ]  # fmt: skip
    turns = [{"id": t, "thread": "T", "idx": 0, "kind": "user", "status": "cancelled",
              "stop": "cancelled", "late": late}]  # fmt: skip
    replay = data.build_replay(events, turns)
    runs = {s["call"]: s for s in replay["lanes"]["tools"]}
    assert len(replay["lanes"]["tools"]) == 2 and replay["turns"][0]["rows"] == 2
    c1, c2 = runs["c1"], runs["c2"]
    assert (c1["t0"], c1["t1"], c1["ok"], c1["late"]) == (0.56, 0.61, True, True)
    assert "starts" not in c1
    assert (c2["t0"], c2["t1"], c2["ok"], c2["late"], c2["starts"]) == (0.3, 0.61, None, False, 2)


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


def call(id_: str, name: str) -> dict[str, Any]:
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_eager_means_read_only_and_started_while_its_message_streams() -> None:
    """Only c1 is eager. c4 starts after the stream ended (A's order), c7 is a write, and c2 was
    saved in turn 0 and runs in the approval turn while a new request is open."""
    t0, t1 = "T.0", "T.1"
    assistant = {"role": "assistant", "content": None,
                 "tool_calls": [call("c1", "read_file"), call("c4", "list_files"),
                                call("c7", "write_file"), call("c2", "read_file")]}  # fmt: skip
    resume = {"kind": "approval", "decisions": {"c2": "allow"}, "reason": None}
    events = [
        env("T", t0, 1, 0, "turn.start", turn_id=t0),
        env("T", t0, 2, 100, "request.start", step=1, attempt=1),
        env("T", t0, 3, 200, "tool.start", call_id="c1", name="read_file", read_only=True),
        env("T", t0, 4, 250, "tool.start", call_id="c7", name="write_file", read_only=False),
        env("T", t0, 5, 300, "item", **item(t0, "a1", assistant)),
        env("T", t0, 6, 400, "tool.start", call_id="c4", name="list_files", read_only=True),
        env("T", t0, 7, 450, "permission.asked", call_id="c2", name="read_file", arguments="{}"),
        env("T", t0, 8, 500, "turn.end", stop="paused", steps=1, pending=["c2"]),
        env("T", t1, 9, 0, "turn.start", turn_id=t1, resume=resume),
        env("T", t1, 10, 50, "request.start", step=2, attempt=1),
        env("T", t1, 11, 60, "tool.start", call_id="c2", name="read_file", read_only=True),
        env("T", t1, 12, 90, "turn.end", stop="end_turn", steps=2),
    ]  # fmt: skip
    replay = data.build_replay(events, [])
    eager = {seg["call"]: seg["eager"] for seg in replay["lanes"]["tools"]}
    assert eager == {"c1": True, "c7": False, "c4": False, "c2": False}
    assert replay["stats"]["eager"] == 1


def test_tool_result_cards_say_how_the_result_came_about() -> None:
    t0, t1 = "T.0", "T.1"
    names = ("a", "b", "c", "d", "e", "f")
    assistant = {"role": "assistant", "content": None,
                 "tool_calls": [call(n, "read_file") for n in names]}  # fmt: skip

    def result_(turn: str, seq: int, id_: str, text: str) -> dict[str, Any]:
        message = {"role": "tool", "tool_call_id": id_, "content": text}
        return env("T", turn, seq, seq, "item", **item(turn, f"r{id_}", message))

    resume = {"kind": "approval", "decisions": {"c": "deny"}, "reason": "no"}
    events = [
        env("T", t0, 1, 0, "turn.start", turn_id=t0),
        env("T", t0, 2, 1, "item", **item(t0, "a1", assistant)),
        env("T", t0, 3, 2, "tool.start", call_id="a", name="read_file", read_only=True),
        env("T", t0, 4, 3, "tool.end", call_id="a", name="read_file", ok=True),
        env("T", t0, 5, 4, "tool.start", call_id="b", name="read_file", read_only=True),
        env("T", t0, 6, 5, "tool.end", call_id="b", name="read_file", ok=False),
        env("T", t0, 7, 6, "tool.start", call_id="f", name="read_file", read_only=True),
        env("T", t0, 8, 7, "tool.end", call_id="f", name="read_file", ok=False),
        env("T", t0, 9, 8, "tool.start", call_id="e", name="read_file", read_only=True),
        result_(t0, 10, "a", "hello"),
        result_(t0, 11, "b", "read_file failed: no such file"),
        result_(t0, 12, "f", "Denied by permission rules: read_file .env"),
        result_(t0, 13, "d", "Not run: the turn reached its step limit."),
        env("T", t0, 14, 14, "turn.end", stop="paused", steps=1, pending=["c"]),
        env("T", t1, 15, 0, "turn.start", turn_id=t1, resume=resume),
        result_(t1, 16, "c", "Denied by user: no"),
        result_(t1, 17, "e", "Interrupted: the worker died while it ran."),
    ]  # fmt: skip
    cards = data.build_replay(events, [])["cards"]
    states = {c["call"]: c["state"] for c in cards if c["k"] == "result"}
    assert states == {"a": "ok", "b": "failed", "f": "denied", "d": "not run", "c": "denied",
                      "e": "unfinished"}  # fmt: skip


def test_scenario_counts_compare_only_what_both_loops_ran(tmp_path: Path) -> None:
    """B passed S01 and S02, A ran only S01 and passed it: that is a tie, not a win for B."""
    run = tmp_path / "runs" / "r1"
    for scenario, impls in (("S01", ("our", "pydantic")), ("S02", ("our",))):
        for impl in impls:
            write_json(run / scenario / impl / "result.json", result(scenario, impl, True))
    html = build.build(runs=run, live=None, metrics=None, now=NOW, discovered=[])
    assert "Passes more scenarios" not in html
    assert "Keeps a" not in html  # the byte prefix too: 1 each over the shared scenario
    assert "Passes as many scenarios: B 1 vs A 1, of the 1 both loops ran." in html
    assert "too close to call" in html


def test_timings_need_a_clear_margin_and_live_runs_more_than_one_sample(out: Path) -> None:
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["bench"]["loops"] = {
        "our_version": {"wall_ms": {"baseline": {"p50": 12.0}, "overhead": {"p50": 3.10, "p95": 5.3}}},
        "pydantic_version": {"wall_ms": {"baseline": {"p50": 12.0}, "overhead": {"p50": 3.09, "p95": 9.9}}},
    }  # fmt: skip
    metrics["deps"]["our_version"]["import_ms"] = 800.0  # within 10% of A's 850
    write_json(out / "metrics.json", metrics)
    live = json.loads((out / "live" / "L1" / "our" / "result.json").read_text())
    for impl in ("our", "pydantic"):  # same step count in every run: per-step values compare
        path = out / "live" / "L1" / impl / "result.json"
        write_json(path, {**json.loads(path.read_text()), "steps": 2})
    for impl, seconds in (("our", 2.6), ("pydantic", 3.3)):  # a second sample
        write_json(out / "live" / "L2" / impl / "result.json",
                   {**live, "run_id": "L2", "impl": impl, "steps": 2, "duration_ms": seconds * 1000})  # fmt: skip
    html = make(out)
    assert "Adds less time on top of the model" not in html
    assert "the median and the slow tail do not both differ by 10%" in html
    assert "Starts faster in a fresh process" not in html and "850.0 ms, within 10%" in html
    # Two live runs, same steps, B about 25% faster in both: a per-step win, stated as a median,
    # with the model's step counts as context.
    assert "Less time per step: median B" in html and "over 2 live runs" in html
    assert "steps per answer: median B" in html and "chosen by the model" in html
    text = re.sub(r"<[^>]+>", "", html)  # the label is a glossary term (a tooltip span)
    assert "Input tokens per step: median B" in text and "within 10%" in text  # equal tokens


def test_a_live_run_a_loop_did_not_finish_is_not_a_sample(out: Path) -> None:
    """A loop that failed at once (stop error, 0 tokens, 40 ms) must not look fast and cheap:
    that run is left out of the medians."""
    live = json.loads((out / "live" / "L1" / "our" / "result.json").read_text())
    for impl, seconds in (("our", 2.6), ("pydantic", 3.3)):  # a second, finished sample
        write_json(out / "live" / "L2" / impl / "result.json",
                   {**live, "run_id": "L2", "impl": impl, "duration_ms": seconds * 1000})  # fmt: skip
    write_json(out / "live" / "L3" / "our" / "result.json", {**live, "run_id": "L3", "impl": "our"})
    write_json(out / "live" / "L3" / "pydantic" / "result.json",
               {**live, "run_id": "L3", "impl": "pydantic", "stops": ["error"], "error": None,
                "duration_ms": 40.0, "usage": {"input_tokens": 0}})  # fmt: skip
    html = make(out)
    assert "over 2 live runs" in html and "over 3 live runs" not in html


def test_a_live_run_that_did_not_pass_is_not_a_sample(out: Path) -> None:
    """A run can end with end_turn and still fail (an invariant broke): it is not an answer."""
    live = json.loads((out / "live" / "L1" / "our" / "result.json").read_text())
    for impl, seconds in (("our", 2.6), ("pydantic", 3.3)):  # a second, passed sample
        write_json(out / "live" / "L2" / impl / "result.json",
                   {**live, "run_id": "L2", "impl": impl, "duration_ms": seconds * 1000})  # fmt: skip
    write_json(out / "live" / "L3" / "our" / "result.json",
               {**live, "run_id": "L3", "impl": "our", "passed": False, "duration_ms": 40.0,
                "invariants": {"I5": {"ok": False, "detail": "wrote to stderr"}}})  # fmt: skip
    write_json(out / "live" / "L3" / "pydantic" / "result.json",
               {**live, "run_id": "L3", "impl": "pydantic"})  # fmt: skip
    html = make(out)
    assert "over 2 live runs" in html and "over 3 live runs" not in html


def write_live(
    live: Path, run_id: str, impl: str, seconds: float, *, prompt: str = "What is RocketRide?",
    model: str = "gpt-test", reasoning: str = "low", system: str = "sys", max_tokens: int = 4096,
    **extra: Any,
) -> None:  # fmt: skip
    """One loop's live run: result.json (plus `extra`), and the session log's thread row with the
    system prompt and the model config (minus the key), as `bakeoff live` saves them."""
    folder = live / run_id / impl
    base_url = "https://api.openai.com/v1"
    write_json(folder / "result.json", {
        "v": 1, "run_id": run_id, "impl": impl, "model": model, "base_url": base_url,
        "prompt": prompt, "final_text": "ok", "stops": ["end_turn"], "steps": 1, "requests": 1,
        "usage": {"input_tokens": 100}, "duration_ms": seconds * 1000, "passed": True,
        "error": None, "thread": f"live-{impl}", **extra,
    })  # fmt: skip
    config = asdict(
        ModelConfig(base_url, model, kind="openai_compat", reasoning={"effort": reasoning},
                    max_tokens=max_tokens)
    )  # fmt: skip
    del config["api_key"]
    log = SessionLog(folder / "log.sqlite")
    log.create_thread(f"live-{impl}", impl=impl, system=system, meta={"rules": {}, "model": config})
    log.close()


def test_live_medians_pool_only_runs_of_one_prompt(out: Path, tmp_path: Path) -> None:
    """Two prompts are two groups, each with its own medians and saying what it is: B is faster
    on the first prompt, A on the second, and pooled they would hide both."""
    live = tmp_path / "live"
    for run_id, prompt, ours, theirs in (
        ("L1", "What is RocketRide?", 2.5, 3.3), ("L2", "What is RocketRide?", 2.6, 3.4),
        ("L3", "Build a chat pipeline.", 9.0, 3.0), ("L4", "Build a chat pipeline.", 9.2, 3.1),
    ):  # fmt: skip
        write_live(live, run_id, "our", ours, prompt=prompt)
        write_live(live, run_id, "pydantic", theirs, prompt=prompt)
    text = re.sub(r"<[^>]+>", "", make(out, live=live))
    assert "over 4 live runs" not in text
    setup = "gpt-test, reasoning low, api.openai.com"
    assert (
        "Less time per step: median B 2.55 s vs A 3.35 s over 2 live runs of one prompt and "
        f"setup (setup 2: “What is RocketRide?”; {setup};"
    ) in text
    assert (
        "Less time per step: median B 9.10 s vs A 3.05 s over 2 live runs of one prompt and "
        f"setup (setup 1: “Build a chat pipeline.”; {setup};"
    ) in text


def test_live_medians_pool_only_runs_of_one_model_setup(out: Path, tmp_path: Path) -> None:
    """The model settings come from the session log (result.json has no reasoning effort): other
    settings are another group, and a run whose loops used different models is in none."""
    live = tmp_path / "live"
    for run_id, reasoning in (("L1", "low"), ("L2", "low"), ("L3", "high"), ("L4", "high")):
        write_live(live, run_id, "our", 2.5, reasoning=reasoning)
        write_live(live, run_id, "pydantic", 3.3, reasoning=reasoning)
    write_live(live, "L5", "our", 2.5)
    write_live(live, "L5", "pydantic", 3.3, model="gpt-other")
    html = make(out, live=live)
    text = re.sub(r"<[^>]+>", "", html)
    assert "over 5 live runs" not in text and "over 4 live runs" not in text
    assert text.count("over 2 live runs of one prompt and setup") == 4  # 2 groups x 2 claims
    assert "gpt-test, reasoning low, api.openai.com" in text
    assert "gpt-test, reasoning high, api.openai.com" in text
    # The live section says which run is not compared, and why.
    assert "the loops ran different prompts or model settings: not in the medians" in text


def test_live_groups_never_read_the_same(out: Path, tmp_path: Path) -> None:
    """Two groups whose prompts share the quoted start, or whose settings differ outside the
    model, reasoning and endpoint, still read differently: each has a number (in section 5 and
    in its claims), names the settings that differ, and quotes up to where the prompts part."""
    stem = "Build a RocketRide pipeline that answers questions from a chat, with a vector store "
    live = tmp_path / "live"
    for run_id, prompt, max_tokens, ours in (
        ("L1", stem + "and a reranker.", 4096, 1.0), ("L2", stem + "and a reranker.", 4096, 1.1),
        ("L3", stem + "and no reranker.", 4096, 2.0), ("L4", stem + "and no reranker.", 4096, 2.1),
        ("L5", stem + "and a reranker.", 512, 3.0), ("L6", stem + "and a reranker.", 512, 3.1),
    ):  # fmt: skip
        write_live(live, run_id, "our", ours, prompt=prompt, max_tokens=max_tokens)
        write_live(live, run_id, "pydantic", 5.0, prompt=prompt, max_tokens=max_tokens)
    text = re.sub(r"<[^>]+>", "", make(out, live=live))
    setup = "gpt-test, reasoning low, api.openai.com, max_tokens"
    quote = (
        "“Build a RocketRide pipeline that answers questions from a chat, with a vector store and "
    )
    # Newest first among groups of one size: L5/L6, then L3/L4, then L1/L2.
    for n, seconds, ending, max_tokens in (
        (1, "3.05 s", "a reranker.”", 512), (2, "2.05 s", "no reranker.”", 4096),
        (3, "1.05 s", "a reranker.”", 4096),
    ):  # fmt: skip
        assert (
            f"median B {seconds} vs A 5.00 s over 2 live runs of one prompt and setup "
            f"(setup {n}: {quote}{ending}; {setup} {max_tokens};"
        ) in text
    for run_id, n, max_tokens in (("L6", 1, 512), ("L4", 2, 4096), ("L1", 3, 4096)):
        assert f"live run {run_id} · setup {n}: {setup} {max_tokens}prompt" in text


def test_live_section_says_which_runs_the_medians_leave_out(out: Path, tmp_path: Path) -> None:
    """A grouped run the medians skip (a loop did not pass, or did not run it) says so, with the
    same test the medians use; each loop's column says whether it passed and what failed."""
    live = tmp_path / "live"
    for run_id in ("L1", "L2", "L3"):
        write_live(live, run_id, "pydantic", 3.0)
    write_live(live, "L1", "our", 2.0)
    write_live(live, "L2", "our", 2.0)
    write_live(live, "L3", "our", 0.1, passed=False,
               invariants={"I5": {"ok": False, "detail": "wrote to stderr"}})  # fmt: skip
    write_live(live, "L4", "our", 0.1)  # only one loop ran it
    text = re.sub(r"<[^>]+>", "", make(out, live=live))
    assert "over 2 live runs of one prompt and setup (setup 1:" in text
    setup = "setup 1: gpt-test, reasoning low, api.openai.com"
    assert f"live run L4 · {setup} · not in the medians: A has no result" in text
    assert f"live run L3 · {setup} · not in the medians: B did not pass (I5 failed)" in text
    for run_id in ("L1", "L2"):
        assert f"live run {run_id} · {setup}prompt" in text
    assert "passedno" in text and "I5 failed: wrote to stderr" in text


def test_live_medians_pool_only_runs_of_one_system_prompt(out: Path, tmp_path: Path) -> None:
    """The system prompt changes with the harness code, not with the command line: runs of one
    user prompt and model config made before and after such a change are two groups."""
    live = tmp_path / "live"
    for run_id, system, ours, theirs in (
        ("L1", "sys v1", 0.5, 3.0), ("L2", "sys v1", 0.6, 3.1),
        ("L3", "sys v2", 3.0, 2.5), ("L4", "sys v2", 3.1, 2.6),
    ):  # fmt: skip
        write_live(live, run_id, "our", ours, system=system)
        write_live(live, run_id, "pydantic", theirs, system=system)
    text = re.sub(r"<[^>]+>", "", make(out, live=live))
    assert "over 4 live runs" not in text
    assert "Less time per step: median B 550.0 ms vs A 3.05 s over 2 live runs" in text
    assert "Less time per step: median B 3.05 s vs A 2.55 s over 2 live runs" in text


def test_metrics_errors_are_shown_not_hidden_or_fatal(out: Path) -> None:
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["bench"]["loops"] = {
        "our_version": {"error": "bench child timed out"},
        "pydantic_version": {"error": "No module named 'openai'"},
    }
    metrics["deps"]["pydantic_version@2.31.1"] = {
        "distributions": 38, "site_packages_mb": 61.3, "import_ms": None,
        "import_error": "ModuleNotFoundError: No module named 'bakeoff.pydantic_version.mapping'",
        "framework_import_ms": 897.8, "turn_error": None,
        "third_party_code": {"loop_turn": None, "framework_import": {"total": 50000}},
    }  # fmt: skip
    write_json(out / "metrics.json", metrics)
    html = make(out)
    assert Page(html).problems == []
    assert "bench child timed out" in html and "No module named &#x27;openai&#x27;" in html
    # A loop that fails to import shows the error, not the library-only time.
    assert "bakeoff.pydantic_version.mapping" in html
    assert "897.8 ms" not in html and "50,000" not in html


def test_live_runs_show_steps_apart_from_retried_requests(out: Path) -> None:
    path = out / "live" / "L1" / "our" / "result.json"
    write_json(path, {**json.loads(path.read_text()), "requests": 4, "steps": 3,
                      "latency": {"ttft_ms": 812.5, "total_ms": 2400.0}})  # fmt: skip
    html = make(out)
    assert ">steps</span></dt><dd>3</dd>" in html
    assert "<dt>requests (with retries)</dt><dd>4</dd>" in html
    assert "<dt>first token</dt><dd>812.5 ms</dd>" in html


def test_header_reads_the_driver_summary(out: Path) -> None:
    write_json(out / "runs" / "r1" / "summary.json", {
        "v": 1, "run_id": "r1", "git_sha": "0123456789abcdef", "git_dirty": True,
        "loops": {"our": {"target": "bakeoff.our_version:OurLoop", "versions": {"httpx": "0.28.1"}},
                  "pydantic": {"target": "bakeoff.pydantic_version:PydanticLoop",
                               "versions": {"pydantic-ai-slim": "2.31.1", "openai": "2.8.1"}}},
        "matrix": {"S01": {"our": {"passed": True, "status": "pass", "reason": "ok"}}},
    })  # fmt: skip
    html = make(out)
    assert "git <code>0123456789</code> (with uncommitted changes)" in html
    assert "bakeoff.our_version:OurLoop, httpx 0.28.1" in html
    assert "pydantic-ai-slim 2.31.1, openai 2.8.1" in html and "{&#x27;" not in html


def test_bakeoff_report_command_builds_the_page(out: Path) -> None:
    from bakeoff.cli import main

    target = out / "page.html"
    assert main(["report", "--out-dir", str(out), "-o", str(target)]) == 0
    assert target.read_text().startswith("<!doctype html>")

"""Read the run outputs into JSON-ready dicts for the report.

Inputs (all optional; a missing or unreadable piece becomes `None` or an empty list plus a note
in `problems`, never an exception):

- a scenario run: `out/runs/<run_id>/<scenario>/<impl>/{result.json, log.sqlite, events.ndjson,
  wire/}` and `out/runs/<run_id>/summary.json`;
- live runs: `out/live/<run_id>/<impl>/result.json`;
- `out/metrics.json` from the metrics package.

The session log is the source of truth for the replay (FAIRNESS.md rule 2); `events.ndjson` is
the fallback when there is no log. Every list is sorted, so the same inputs give the same output.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, NamedTuple

# Long texts are cut here; the page shows a shorter preview with an expand control.
CARD_LIMIT = 6_000  # one transcript card (a tool result, an answer)
WIRE_LIMIT = 3_000  # one string inside a recorded request body
LIVE_LIMIT = 20_000  # a live run's final answer
MAX_LIVE_RUNS = 8
_INFO_LIMIT = 2_000  # an invariant's `info`, as JSON
_LOOP_KINDS = ("user", "approval", "crash")  # turns that run a loop (not revert/compact)


@dataclass(frozen=True, slots=True)
class LoopInfo:
    """How the report names and colours one loop implementation."""

    impl: str  # the name in run folders and events ("our", "pydantic", ...)
    target: str  # "module:Class", imported only by `discover()`
    letter: str  # FAIRNESS.md naming: A = pydantic-ai, B = ours
    label: str
    color: str  # a CSS variable of the page (see style.css)

    @property
    def package(self) -> str:
        """The package under src/bakeoff, as the metrics name it."""
        return self.target.split(":")[0].rsplit(".", 1)[-1]


REGISTRY: tuple[LoopInfo, ...] = (
    LoopInfo("our", "bakeoff.our_version:OurLoop", "B", "our loop", "teal"),
    LoopInfo("pydantic", "bakeoff.pydantic_version:PydanticLoop", "A", "pydantic-ai", "violet"),
    LoopInfo("hybrid", "bakeoff.hybrid_version:HybridLoop", "A'", "hybrid", "rose"),
)
_BY_IMPL = {info.impl: info for info in REGISTRY}
_BY_PACKAGE = {info.package: info for info in REGISTRY}


def loop_info(impl: str) -> LoopInfo:
    """The registry entry of `impl`, or a neutral one for a loop the registry does not know."""
    return _BY_IMPL.get(impl) or LoopInfo(impl, f"?:{impl}", "?", impl, "muted")


def impl_of_package(package: str) -> str:
    """ "our_version" -> "our" (metrics name loops by package)."""
    info = _BY_PACKAGE.get(package.split("@")[0])
    return info.impl if info else package


def discover() -> list[str]:
    """The registry loops whose class imports in this checkout, in registry order.

    Imports lazily, one loop at a time, and skips a loop whose package is missing (not merged
    yet), still empty, or fails to import: the report still shows whatever data it has."""
    found = []
    for info in REGISTRY:
        module_name, cls = info.target.split(":")
        try:
            if importlib.util.find_spec(module_name) is None:
                continue
            if hasattr(importlib.import_module(module_name), cls):
                found.append(info.impl)
        except Exception:  # a broken optional loop must not break the report
            continue
    return found


def read_json(path: Path) -> tuple[Any, str | None]:
    """`(data, None)`, `(None, None)` if the file is absent, or `(None, problem)`."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        return None, f"{path.name}: unreadable ({type(exc).__name__}: {exc})"


def clip(text: str, limit: int) -> str:
    """`text` cut to `limit` characters, saying how much is missing.

    The note depends only on the text, so the same text clips the same way wherever it was
    recorded (the wire diff compares clipped strings)."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…[{len(text) - limit:,} more characters not shown]"


def natural_key(name: str) -> list[Any]:
    """Sort key that puts S2 before S10 and S10a before S10b."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


class Pool:
    """Stores each distinct JSON value once, so the page does not repeat the tool list and the
    conversation prefix in every recorded request."""

    def __init__(self) -> None:
        self.values: list[Any] = []
        self._index: dict[str, int] = {}

    def add(self, value: Any) -> int:
        # Key order is part of the identity: re-ordered keys are a real wire difference.
        key = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if key not in self._index:
            self._index[key] = len(self.values)
            self.values.append(value)
        return self._index[key]


# --- scenario runs ------------------------------------------------------------------------


def load_run(runs_dir: Path | None, scenario_titles: dict[str, str] | None = None) -> dict | None:
    """One scenario run (`out/runs/<run_id>`, or `out/runs` whose `latest` link is followed).

    Returns None if there is no such folder. Otherwise `{"run_id", "path", "summary",
    "scenarios": [{"id", "title", "runs": {impl: ...}}], "impls", "pool", "problems"}`."""
    if runs_dir is None or not runs_dir.is_dir():
        return None
    path = runs_dir.resolve()
    if not (path / "summary.json").exists() and (path / "latest").is_dir():
        path = (path / "latest").resolve()
    summary, problem = read_json(path / "summary.json")
    problems = [problem] if problem else []
    summary = summary if isinstance(summary, dict) else {}
    run_id = str(summary.get("run_id") or path.name)
    titles = scenario_titles if scenario_titles is not None else fakeprov_titles()
    pool = Pool()
    scenarios = []
    for sdir in sorted(_subdirs(path), key=lambda p: natural_key(p.name)):
        runs = {}
        for idir in sorted(_subdirs(sdir), key=lambda p: p.name):
            if any(
                (idir / name).exists() for name in ("result.json", "log.sqlite", "events.ndjson")
            ):
                runs[idir.name] = load_scenario_run(idir, sdir.name, run_id, pool)
        if runs:
            title = next(
                (r["result"]["title"] for r in runs.values() if (r["result"] or {}).get("title")),
                titles.get(sdir.name, ""),
            )
            scenarios.append({"id": sdir.name, "title": str(title), "runs": runs})
    impls = sorted({i for s in scenarios for i in s["runs"]}, key=impl_order)
    return {
        "run_id": run_id,
        "path": str(path),
        "summary": summary,
        "scenarios": scenarios,
        "impls": impls,
        "pool": pool.values,
        "problems": problems,
    }


def impl_order(impl: str) -> tuple[int, str]:
    """Registry order first (our, pydantic, hybrid), then unknown loops by name."""
    names = [info.impl for info in REGISTRY]
    return (names.index(impl), "") if impl in names else (len(names), impl)


def _subdirs(path: Path) -> list[Path]:
    # Skip hidden and private folders (a driver's scratch space) and links like `latest`.
    return [
        p
        for p in path.iterdir()
        if p.is_dir() and not p.is_symlink() and not p.name.startswith((".", "_"))
    ]


def fakeprov_titles() -> dict[str, str]:
    """Scenario titles from the fake provider's scenario files, if they ship with this package."""
    try:
        folder = resources.files("bakeoff.fakeprov") / "scenarios"
        files = [f for f in folder.iterdir() if f.name.endswith(".json")]
    except (ModuleNotFoundError, OSError):
        return {}
    titles = {}
    for f in files:
        try:
            titles[f.name.removesuffix(".json")] = str(json.loads(f.read_text())["title"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return titles


def load_scenario_run(idir: Path, scenario: str, run_id: str, pool: Pool) -> dict[str, Any]:
    """Result, replay timeline and wire recordings of one (scenario, impl) folder."""
    result, problem = read_json(idir / "result.json")
    problems = [problem] if problem else []
    if result is not None and not isinstance(result, dict):
        result, problems = None, [*problems, "result.json: not an object"]
    events, turns, source = read_events(idir)
    if source is None and not problems:
        problems.append("no log.sqlite or events.ndjson: no replay")
    wire_dir = find_wire_dir(idir / "wire", scenario, run_id, idir.name)
    return {
        "result": slim_result(result),
        "replay": build_replay(events, turns) if events else None,
        "replay_source": source,
        "wire": wire_requests(wire_dir, pool, idir) if wire_dir else [],
        "problems": problems,
    }


def slim_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """result.json with oversized invariant `info` replaced by a note."""
    if result is None:
        return None
    out = dict(result)
    invariants = {}
    for name, check in sorted((result.get("invariants") or {}).items()):
        check = dict(check) if isinstance(check, dict) else {"ok": None, "detail": str(check)}
        info = check.get("info")
        if len(json.dumps(info, default=str)) > _INFO_LIMIT:
            check["info"] = {"note": "too large to show; see result.json"}
        invariants[name] = check
    out["invariants"] = invariants
    return out


def read_events(idir: Path) -> tuple[list[dict], list[dict], str | None]:
    """`(events, turns, source)`: from the session log if readable, else from events.ndjson
    (which has no turn rows)."""
    log = idir / "log.sqlite"
    if log.exists():
        try:
            events, turns = read_log(log)
            return events, turns, "log.sqlite"
        except sqlite3.Error:
            pass
    ndjson = idir / "events.ndjson"
    if ndjson.exists():
        events = []
        for line in ndjson.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # a line cut short by a killed process
            if isinstance(event, dict) and "type" in event:
                events.append(event)
        events.sort(key=lambda e: (str(e.get("thread")), e.get("seq", 0)))
        return events, [], "events.ndjson"
    return [], [], None


@contextmanager
def _private_log(path: Path) -> Iterator[sqlite3.Connection]:
    """A connection to a private copy of a session log (with its write-ahead log): opening the
    original, even read-only, can write to it, and the report must never change a run's
    evidence."""
    with tempfile.TemporaryDirectory(prefix="bakeoff-report-") as tmp:
        copy = Path(tmp) / "log.sqlite"
        for suffix in ("", "-wal"):
            src = path.with_name(path.name + suffix)
            if src.exists():
                shutil.copyfile(src, copy.with_name(copy.name + suffix))
        db = sqlite3.connect(copy)
        try:
            db.row_factory = sqlite3.Row
            yield db
        finally:
            db.close()


def read_log(path: Path) -> tuple[list[dict], list[dict]]:
    """All events and turn rows of a session log, threads in creation order."""
    with _private_log(path) as db:
        threads = [r["id"] for r in db.execute("SELECT id FROM threads ORDER BY created_us")]
        order = {t: i for i, t in enumerate(threads)}
        events = [json.loads(r["json"]) for r in db.execute("SELECT json FROM events")]
        turns = [dict(r) for r in db.execute("SELECT * FROM turns")]
    for turn in turns:
        for key in ("pending", "late"):
            turn[key] = json.loads(turn[key]) if turn.get(key) else None
    events.sort(key=lambda e: (order.get(e.get("thread"), len(order)), e.get("seq", 0)))
    turns.sort(key=lambda t: (order.get(t["thread"], len(order)), t["idx"]))
    return events, turns


# --- replay -------------------------------------------------------------------------------


def _ms(t_us: Any) -> float:
    return round((t_us or 0) / 1000, 2)


def _args(raw: Any) -> str:
    """Tool arguments for display: pretty JSON when they parse, else the raw text."""
    if isinstance(raw, str):
        try:
            return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except ValueError:
            return raw
    return json.dumps(raw, indent=2, ensure_ascii=False)


def _text(content: Any) -> str:
    """A message's content as text (list content is joined text parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return json.dumps(content, ensure_ascii=False)


def _reasoning(message: dict[str, Any]) -> str:
    """The reasoning a message carries, as text (encrypted blocks are named, not shown)."""
    parts = []
    for detail in message.get("reasoning_details") or []:
        if not isinstance(detail, dict):
            continue
        if detail.get("text") or detail.get("summary"):
            parts.append(str(detail.get("text") or detail.get("summary")))
        elif detail.get("data"):
            parts.append(f"[{detail.get('type', 'reasoning')}: encrypted, not readable]")
    for key in ("reasoning", "reasoning_content"):
        if not parts and isinstance(message.get(key), str) and message[key]:
            parts.append(message[key])
    return "\n".join(parts)


@dataclass(slots=True)
class _Calls:
    """What the whole replay knows about each tool call, gathered before the turns are walked:
    a result card or a later turn may refer to a call from an earlier turn."""

    names: dict[str, str]  # call id -> tool name
    ok: dict[str, bool]  # call id -> its tool.end's ok (only calls that ran to the end)
    started: set[str]  # calls with a tool.start
    denied: set[str]  # calls the user denied on an approval resume
    saved: set[str]  # calls whose assistant item is saved so far (grows during the walk)

    @classmethod
    def scan(cls, events: list[dict[str, Any]]) -> _Calls:
        calls = cls({}, {}, set(), set(), set())
        for e in events:
            data = e.get("data") or {}
            call = data.get("call_id")
            if e.get("type") in ("tool_call.ready", "tool.start", "permission.asked"):
                calls.names.setdefault(call, data.get("name"))
            if e.get("type") == "tool.start":
                calls.started.add(call)
            elif e.get("type") == "tool.end":
                calls.ok[call] = bool(data.get("ok"))
            elif e.get("type") == "turn.start":
                decisions = (data.get("resume") or {}).get("decisions") or {}
                calls.denied |= {c for c, d in decisions.items() if d == "deny"}
        return calls

    def state(self, call: str | None, text: str) -> str:
        """How a tool result came about: ok, failed, denied, unfinished (started, never ended)
        or not run (the loop wrote the result without running the tool, e.g. at a step limit).
        """
        if call in self.ok:
            if self.ok[call]:
                return "ok"
            # A permission rule's deny goes through ToolHost.run() and fails there.
            return "denied" if text.startswith("Denied") else "failed"
        # A user's deny never runs the tool (contract rule 5).
        if call in self.denied or text.startswith("Denied by user"):
            return "denied"
        return "unfinished" if call in self.started else "not run"


def _item_card(item: dict[str, Any], calls: _Calls) -> dict:
    message = item.get("message") or {}
    role = message.get("role")
    text = clip(_text(message.get("content")), CARD_LIMIT)
    if role == "assistant":
        tool_calls = [
            {
                "id": c.get("id"),
                "name": (c.get("function") or {}).get("name"),
                "args": clip(_args((c.get("function") or {}).get("arguments")), CARD_LIMIT),
            }
            for c in message.get("tool_calls") or []
            if isinstance(c, dict)
        ]
        card = {"k": "assistant", "text": text, "calls": tool_calls}
        if reasoning := _reasoning(message):
            card["reasoning"] = clip(reasoning, CARD_LIMIT)
        if item.get("status") == "incomplete":
            card["incomplete"] = True
        return card
    if role == "tool":
        call = message.get("tool_call_id")
        return {
            "k": "result",
            "call": call,
            "name": calls.names.get(call),
            "state": calls.state(call, _text(message.get("content"))),
            "text": text,
        }
    if item.get("compaction"):
        return {"k": "summary", "text": text}
    if role == "user" and text.startswith("[harness]"):
        return {"k": "note", "text": text}
    return {"k": "user" if role == "user" else str(role), "text": text}


def _turn_order(events: list[dict], turns: list[dict]) -> list[str]:
    order = [t["id"] for t in turns]
    seen = set(order)
    for e in events:
        if (tid := e.get("turn")) not in seen:
            seen.add(tid)
            order.append(tid)
    return order


def _guess_kind(events: list[dict]) -> str:
    """A turn's kind when there is no turn row (events.ndjson only)."""
    for e in events:
        if e.get("type") == "turn.start":
            return ((e.get("data") or {}).get("resume") or {}).get("kind") or "user"
        if e.get("type") == "item":
            item = (e.get("data") or {}).get("item") or {}
            if item.get("compaction"):
                return "compact"
            if _text((item.get("message") or {}).get("content")).startswith("[harness] Reverted"):
                return "revert"
    return "user"


def _stack(segments: list[dict]) -> int:
    """Give overlapping tool runs separate rows (parallel tools); returns the row count."""
    ends: list[float] = []
    for seg in sorted(segments, key=lambda s: (s["t0"], s["t1"])):
        row = next((i for i, end in enumerate(ends) if end <= seg["t0"]), len(ends))
        ends[row : row + 1] = [seg["t1"]]
        seg["row"] = row
    return len(ends)


def build_replay(events: list[dict[str, Any]], turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Lanes and transcript cards of one scenario run.

    Times are ms since the start of their turn (`turn` = index into `turns`). The page lays the
    turns end to end on one axis and cuts the idle time between them (a user deciding, a new
    process starting), with the same rule for every loop.

    Lanes: `model` (one segment per request: waiting for the first token, then streaming),
    `tools` (tool.start to tool.end, stacked when they overlap), `perm` (approval asked and
    answered) and `git` (commits, reverts, compaction)."""
    rows = {t["id"]: t for t in turns}
    by_turn: dict[str, list[dict]] = {}
    for e in events:
        by_turn.setdefault(e.get("turn"), []).append(e)
    calls = _Calls.scan(events)
    replay: dict[str, Any] = {
        "turns": [],
        "lanes": {"model": [], "tools": [], "perm": [], "git": []},
        "cards": [],
        "stats": dict.fromkeys(("requests", "retries", "tool_runs", "eager", "errors"), 0),
    }
    for ti, tid in enumerate(_turn_order(events, turns)):
        evs = by_turn.get(tid, [])
        row = rows.get(tid, {})
        late = [x for x in row.get("late") or [] if isinstance(x, dict)]
        kind = row.get("kind") or _guess_kind(evs)
        span = max([_ms(e.get("t_us")) for e in evs] + [_ms(x.get("t_us")) for x in late] + [0.0])
        turn = _TurnWalk(ti, replay, calls).run(evs, late)
        if kind in _LOOP_KINDS and evs and not turn["ended"]:
            # The worker died mid-turn (a crash scenario kills it on purpose).
            replay["cards"].append({"turn": ti, "t": span, "k": "crash"})
        replay["turns"].append(
            {
                "id": tid,
                "kind": kind,
                "status": row.get("status"),
                "stop": row.get("stop") or turn["stop"],
                "ms": span,
                "rows": turn["rows"],
            }
        )
    return replay


class _ToolEvent(NamedTuple):
    """A tool.start or tool.end of one call, kept until the turn is walked (see `_runs`)."""

    t: float
    start: bool  # tool.start; else tool.end
    late: bool  # from the turn row's `late` list: after the loop's turn.end
    data: dict[str, Any]
    eager: bool = False  # a start while the model still streamed its call's message


class _TurnWalk:
    """Turns one turn's events into lane segments and cards (see `build_replay`). Each event
    type has an `on_<type>` handler; times are ms since the turn started."""

    def __init__(self, ti: int, replay: dict[str, Any], calls: _Calls) -> None:
        self.ti, self.calls = ti, calls
        self.lanes: dict[str, list[dict]] = replay["lanes"]
        self.cards: list[dict] = replay["cards"]
        self.stats: dict[str, int] = replay["stats"]
        self.req: dict | None = None  # the model request being streamed
        self.by_step: dict[Any, dict] = {}  # step -> its latest request (usage arrives late)
        self.calls_seen: dict[str, list[_ToolEvent]] = {}  # call id -> its tool events
        self.tools: list[dict] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0}
        self.ended, self.stop = False, None

    def run(self, events: list[dict], late: list[dict]) -> dict[str, Any]:
        """Walk the turn's events, then its `late` tool events; returns how it ended."""
        stream = [(e, False) for e in events] + [(x, True) for x in late]
        for event, is_late in stream:
            handler = getattr(self, "on_" + str(event.get("type")).replace(".", "_"), None)
            if handler is not None:
                handler(_ms(event.get("t_us")), event.get("data") or {}, is_late)
        last = max([_ms(e.get("t_us")) for e, _ in stream] + [0.0])
        self.close(last, "open" if self.ended else "killed")
        for call, tool_events in self.calls_seen.items():
            self.tools += self._runs(call, tool_events, last)
        rows = _stack(self.tools)
        self.lanes["tools"].extend(sorted(self.tools, key=lambda s: (s["t0"], s["row"])))
        return {"ended": self.ended, "stop": self.stop, "rows": rows}

    def close(self, t: float, outcome: str) -> None:
        """End the open model request segment, if any."""
        if self.req is not None:
            self.req["t1"], self.req["end"] = t, outcome
            self.lanes["model"].append(self.req)
            self.req = None

    def card(self, t: float, body: dict[str, Any]) -> None:
        self.cards.append({"turn": self.ti, "t": t, **body})

    def on_request_start(self, t: float, data: dict, is_late: bool) -> None:
        self.close(t, "open")
        self.stats["requests"] += 1
        step, attempt = data.get("step"), data.get("attempt")
        self.req = {
            "turn": self.ti,
            "t0": t,
            "t1": t,
            "first": None,
            "step": step,
            "attempt": attempt,
        }
        self.by_step[step] = self.req

    def on_text_delta(self, t: float, data: dict, is_late: bool) -> None:
        if self.req is not None and self.req["first"] is None:
            self.req["first"] = t  # the first token: waiting ends, streaming starts

    on_reasoning_delta = on_tool_call_ready = on_text_delta

    def on_usage(self, t: float, data: dict, is_late: bool) -> None:
        if (seg := self.by_step.get(data.get("step")) or self.req) is not None:
            seg["usage"] = {k: data.get(k) for k in (*self.usage, "cost_source")}
        for key in self.usage:
            self.usage[key] += data.get(key) or 0

    def on_retry(self, t: float, data: dict, is_late: bool) -> None:
        self.close(t, "retry")
        self.stats["retries"] += 1
        fields = {k: data.get(k) for k in ("attempt", "status", "wait_ms")}
        self.card(t, {"k": "retry", **fields, "reason": clip(str(data.get("reason") or ""), 400)})

    def on_error(self, t: float, data: dict, is_late: bool) -> None:
        self.close(t, "error")
        self.stats["errors"] += 1
        text = clip(str(data.get("message")), 2000)
        self.card(t, {"k": "error", "kind": data.get("kind"), "text": text})

    def on_item(self, t: float, data: dict, is_late: bool) -> None:
        item = data.get("item") or {}
        message = item.get("message") or {}
        if message.get("role") == "assistant":
            self.close(t, "cut" if item.get("status") == "incomplete" else "ok")
            self.calls.saved.update(c.get("id") for c in message.get("tool_calls") or [])
        self.card(t, _item_card(item, self.calls))
        if item.get("compaction"):
            self.lanes["git"].append({"turn": self.ti, "t": t, "kind": "compact"})

    def _runs(self, call: str, events: list[_ToolEvent], last: float) -> list[dict]:
        """One segment per run of `call` in this turn, so a call that ran twice shows twice,
        with the gap between: each tool.start paired with the next tool.end, oldest open run
        first.

        Paired in time order, not stored order: after turn.end a tool's events go to the turn
        row's `late` list, where its tool.end can come before its tool.start. A run that never
        ended lasts to the turn's last event (its process died); a tool.end without a start in
        this turn is a mark at its time. `late`: the run was not done by the loop's turn.end."""
        segs: list[dict] = []
        running: list[dict] = []  # started, not ended yet, oldest first

        def new(e: _ToolEvent) -> dict:
            seg = {"turn": self.ti, "t0": e.t, "t1": last, "name": e.data.get("name"),
                   "call": call, "eager": e.eager, "late": e.late, "ok": None}  # fmt: skip
            segs.append(seg)
            return seg

        # At one time a start sorts before an end, and the end closes the oldest open run: that
        # is right whether the end is this start's (a run too short to measure) or an earlier's.
        for e in sorted(events, key=lambda e: (e.t, not e.start)):
            if e.start:
                running.append(new(e))
                continue
            seg = running.pop(0) if running else new(e)  # no start in this turn: a mark
            seg["t1"], seg["ok"] = e.t, bool(e.data.get("ok"))
            seg["name"] = seg["name"] or e.data.get("name")
            seg["late"] = seg["late"] or e.late
        if len(segs) > 1:  # the same call ran again (I2 fails): each segment says which run
            for k, seg in enumerate(segs, 1):
                seg["run"], seg["runs"] = k, len(segs)
        return segs

    def on_tool_start(self, t: float, data: dict, is_late: bool) -> None:
        self.stats["tool_runs"] += 1
        call = data.get("call_id")
        # Eager: a read-only tool started while the model still streams the message that holds
        # its call. A call saved in an earlier turn (it runs after an approval or a crash
        # resume) or started after the stream ended is not eager, whatever its turn.
        eager = (
            not is_late
            and self.req is not None
            and bool(data.get("read_only"))
            and call not in self.calls.saved
        )
        seen = self.calls_seen.setdefault(call, [])
        if not any(e.start for e in seen):  # the stats count calls: its first start decides
            self.stats["eager"] += eager
        seen.append(_ToolEvent(t, True, is_late, data, eager))

    def on_tool_end(self, t: float, data: dict, is_late: bool) -> None:
        self.calls_seen.setdefault(data.get("call_id"), []).append(
            _ToolEvent(t, False, is_late, data)
        )

    def on_permission_asked(self, t: float, data: dict, is_late: bool) -> None:
        call, name = data.get("call_id"), data.get("name")
        self.lanes["perm"].append(
            {"turn": self.ti, "t": t, "kind": "asked", "call": call, "name": name}
        )
        args = clip(_args(data.get("arguments")), CARD_LIMIT)
        self.card(t, {"k": "ask", "call": call, "name": name, "args": args})

    def on_turn_start(self, t: float, data: dict, is_late: bool) -> None:
        resume = data.get("resume")
        if not resume:
            return  # a user turn: its user item is the card
        decisions = dict(sorted((resume.get("decisions") or {}).items()))
        for call, decision in decisions.items():
            perm = {
                "turn": self.ti,
                "t": t,
                "kind": decision,
                "call": call,
                "name": self.calls.names.get(call),
            }
            self.lanes["perm"].append(perm)
        self.card(t, {"k": "resume", "kind": resume.get("kind"), "decisions": decisions,
                      "reason": resume.get("reason")})  # fmt: skip

    def on_turn_end(self, t: float, data: dict, is_late: bool) -> None:
        self.ended, self.stop = True, data.get("stop")
        self.close(t, "cancel" if self.stop == "cancelled" else str(self.stop or "end"))
        self.card(t, {"k": "end", "stop": self.stop, "steps": data.get("steps"),
                      "pending": data.get("pending") or [], "error": data.get("error"),
                      "usage": dict(self.usage)})  # fmt: skip

    def on_commit(self, t: float, data: dict, is_late: bool) -> None:
        files = sorted(map(str, data.get("files") or []))
        sha = data.get("sha")
        self.lanes["git"].append(
            {"turn": self.ti, "t": t, "kind": "commit", "sha": sha, "files": files}
        )
        self.card(t, {"k": "commit", "sha": sha, "files": files})


# --- wire recordings ----------------------------------------------------------------------


def _is_body(path: Path) -> bool:
    return path.suffix == ".json" and path.stem.isdigit()


def find_wire_dir(wire: Path, scenario: str, run_id: str, impl: str) -> Path | None:
    """The folder holding this cursor's NNN.json bodies: `wire/` itself, or the fake server's
    own `<scenario>/<run>/<impl>/` layout below it."""
    if not wire.is_dir():
        return None
    if any(_is_body(p) for p in wire.iterdir()):
        return wire
    nested = wire / scenario / run_id / impl
    if nested.is_dir():
        return nested
    candidates = sorted({p.parent for p in wire.rglob("*.json") if _is_body(p)})
    return next((c for c in candidates if c.name == impl), candidates[0] if candidates else None)


def _clip_deep(value: Any, cuts: list[int]) -> Any:
    """`value` with every long string clipped; appends the cut length to `cuts` for each.

    The clip note names no file: two loops that sent the same long string must still compare
    (and pool) as equal. The request entry says where the full body is."""
    if isinstance(value, str):
        if len(value) > WIRE_LIMIT:
            cuts.append(len(value) - WIRE_LIMIT)
        return clip(value, WIRE_LIMIT)
    if isinstance(value, list):
        return [_clip_deep(v, cuts) for v in value]
    if isinstance(value, dict):
        return {k: _clip_deep(v, cuts) for k, v in value.items()}
    return value


def encode_body(body: Any, pool: Pool, cuts: list[int]) -> Any:
    """A request body with long strings clipped (counted in `cuts`) and its messages and tool
    list moved into `pool` (key order kept)."""
    body = _clip_deep(body, cuts)
    if not isinstance(body, dict):
        return body
    out = {}
    for key, value in body.items():
        if key == "messages" and isinstance(value, list):
            out[key] = {"$refs": [pool.add(m) for m in value]}
        elif key == "tools" and isinstance(value, list):
            out[key] = {"$ref": pool.add(value)}
        else:
            out[key] = value
    return out


def wire_requests(directory: Path, pool: Pool, run_dir: Path) -> list[dict[str, Any]]:
    """Every recorded request of one cursor, in order, with its meta (never headers: the fake
    server does not record them)."""
    out = []
    for path in sorted((p for p in directory.iterdir() if _is_body(p)), key=lambda p: int(p.stem)):
        raw = path.read_bytes()
        meta, _ = read_json(path.with_name(f"{path.stem}.meta.json"))
        meta = meta if isinstance(meta, dict) else {}
        entry: dict[str, Any] = {
            "n": int(path.stem),
            "file": path.relative_to(run_dir.parent.parent).as_posix(),
            "bytes": len(raw),
            "meta": {k: meta[k] for k in ("status", "conn_id", "t_us", "error") if k in meta},
        }
        cuts: list[int] = []
        try:
            entry["body"] = encode_body(json.loads(raw), pool, cuts)
        except ValueError:
            entry["raw"] = _clip_deep(raw.decode("utf-8", "replace"), cuts)
        if cuts:  # the page says how much is missing and in which file the full body is
            entry["clipped"] = {"strings": len(cuts), "chars": sum(cuts)}
        out.append(entry)
    return out


# --- live runs and metrics ----------------------------------------------------------------

_LIVE_KEYS = (
    "run_id", "impl", "model", "base_url", "stops", "steps", "requests", "tool_runs", "usage",
    "latency", "duration_ms", "passed", "error", "invariants",
)  # fmt: skip


# The model settings a live answer depends on (ModelConfig fields), besides the endpoint.
_LIVE_SETTINGS = ("model", "kind", "reasoning", "temperature", "max_tokens", "compat")


def endpoint(base_url: str | None) -> str:
    """The host (and port) of a base URL: never a path, a query or a user:password@ part."""
    if not base_url:
        return ""
    return base_url.split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0].rsplit("@", 1)[-1].lower()


def _thread_setup(log: Path, thread: Any) -> tuple[dict[str, Any], str] | str:
    """The model config and the system prompt a live thread was created with (the runner keeps
    the config, minus the key, in the thread row's meta), or why the log cannot tell."""
    if not log.exists():
        return "missing"
    try:
        with _private_log(log) as db:
            rows = {r["id"]: r for r in db.execute("SELECT id, system, meta FROM threads")}
        row = rows.get(thread) or (next(iter(rows.values())) if len(rows) == 1 else None)
        model = json.loads(row["meta"]).get("model") if row else None
    except (OSError, sqlite3.Error, ValueError, AttributeError) as exc:
        return f"unreadable ({type(exc).__name__}: {exc})"
    if row is None:
        return f"no thread {thread!r} in it"
    if not isinstance(model, dict) or not isinstance(row["system"], str):
        return "its thread has no model settings"
    return model, row["system"]


def live_settings(idir: Path, result: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """What a live run's answer depends on besides the loop and the prompt, or None and why its
    session log cannot tell. That is the model settings and the system prompt from the log
    (result.json has only the model and base_url), with the base_url cut to its endpoint (the
    per-loop path on one server is still one endpoint) and the system prompt to a hash (it
    changes with the harness code, so runs of one user prompt can still differ in it)."""
    setup = _thread_setup(idir / "log.sqlite", result.get("thread"))
    if isinstance(setup, str):
        return None, setup
    model, system = setup
    return {
        "endpoint": endpoint(model.get("base_url") or result.get("base_url")),
        **{k: model.get(k, result.get(k)) for k in _LIVE_SETTINGS},
        "system": hashlib.sha256(system.encode()).hexdigest()[:12],
    }, None


def load_live(live_dir: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    """The newest live runs (`out/live/<run_id>/<impl>/result.json`), newest first.

    Each run's `group` names its prompt and model settings: runs compare (and pool into medians)
    only within a group. It is None when the run's loops differ in either, or when a loop's
    settings are unknown (its session log cannot tell): `unknown` lists those loops, since
    runs whose settings are unknown could differ in any of them."""
    if live_dir is None or not live_dir.is_dir():
        return [], []
    runs, problems = [], []
    folders = sorted(_subdirs(live_dir), key=lambda p: natural_key(p.name), reverse=True)
    for run in folders[:MAX_LIVE_RUNS]:
        results, setups, unknown = {}, set(), []
        for idir in sorted(_subdirs(run), key=lambda p: impl_order(p.name)):
            data, problem = read_json(idir / "result.json")
            if problem:
                problems.append(f"live/{run.name}/{idir.name}/{problem}")
            elif data is not None and not isinstance(data, dict):
                problems.append(f"live/{run.name}/{idir.name}/result.json: not an object")
            if isinstance(data, dict):
                slim = {k: data.get(k) for k in _LIVE_KEYS if k in data}
                slim["prompt"] = clip(str(data.get("prompt") or ""), CARD_LIMIT)
                slim["final_text"] = clip(str(data.get("final_text") or ""), LIVE_LIMIT)
                slim["settings"], why = live_settings(idir, data)
                if why is not None:
                    unknown.append(idir.name)
                    problems.append(
                        f"live/{run.name}/{idir.name}/log.sqlite: {why}; the run's model settings "
                        "are unknown, so it is not in the live medians"
                    )
                results[idir.name] = slim
                # The full prompt: two prompts may differ only after the clipped part.
                setup = [str(data.get("prompt") or ""), slim["settings"]]
                setups.add(json.dumps(setup, sort_keys=True, default=str))
        if results:
            prompt = next((r["prompt"] for r in results.values() if r["prompt"]), "")
            group = (
                hashlib.sha256(setups.pop().encode()).hexdigest()[:16]
                if len(setups) == 1 and not unknown
                else None
            )
            runs.append({"run_id": run.name, "prompt": prompt, "results": results, "group": group,
                         "unknown": unknown})  # fmt: skip
    return runs, problems


def load_metrics(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """out/metrics.json, or None."""
    if path is None:
        return None, None
    data, problem = read_json(path)
    if data is not None and not isinstance(data, dict):
        return None, f"{path.name}: not an object"
    return data, problem

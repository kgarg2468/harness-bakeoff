"""Invariant checks over wire recordings and the session log (DESIGN.md, "Invariants").

Each check is a small function that returns a `Check`. Scenario tests assert `ok`; the
report shows `detail` and `info`.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bakeoff.shared.contract import Item
from bakeoff.shared.sessionlog import SessionLog
from bakeoff.shared.workcopy import GIT_CONFIG, git_env

_REASONING_KEYS = ("type", "text", "signature", "data", "format", "index")
_SYSTEM_ROLES = ("system", "developer")
_QUOTE, _BACKSLASH, _COLON, _COMMA = b'"'[0], b"\\"[0], b":"[0], b","[0]
_OPEN, _CLOSE = b"[{", b"]}"


@dataclass(slots=True, frozen=True)
class Check:
    """The outcome of one invariant check."""

    name: str
    ok: bool
    detail: str
    info: dict[str, Any] = field(default_factory=dict)


def load_wire(directory: Path) -> list[tuple[bytes, dict[str, Any]]]:
    """The recorded requests of one (scenario, run, impl): `(body, meta)` sorted by NNN."""
    bodies = sorted(
        (p for p in directory.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem)
    )
    out = []
    for body in bodies:
        meta = body.with_name(f"{body.stem}.meta.json")
        out.append((body.read_bytes(), json.loads(meta.read_bytes()) if meta.exists() else {}))
    return out


# I1 --------------------------------------------------------------------------------------


def check_prefix(
    bodies: Sequence[bytes], items: Sequence[Item] = (), turns: Sequence[dict[str, Any]] = ()
) -> Check:
    """I1: each request's conversation is a prefix of the next request's.

    The conversation is a chat request's `messages`, or a Responses API request's
    `instructions` (the system prompt, if sent there) followed by its `input` items.
    `ok` is semantic equality; byte equality of the raw elements is reported in
    `info["byte_prefix"]`. A prefix may reset only at a new compaction summary, placed right
    after the unchanged system messages (contract rule 8). Only the compaction `items` of
    the runner's "compact" turns (`turns`, as `SessionLog.turns` returns them) count as
    summaries, so neither a look-alike message nor a loop's own item can fake a reset. They
    are matched in log order: a reset moves to a compaction item after the one the prefix
    starts from, never back to an older one, and two compactions may share a summary text.
    """
    compact_turns = {t["id"] for t in turns if t["kind"] == "compact"}
    summary_messages = [
        item.message for item in items if item.compaction and item.turn_id in compact_turns
    ]
    # A summary is a user message: in the Responses API the same shape is an input message.
    summaries = {
        "chat": [_semantic(m) for m in summary_messages],
        "responses": [_semantic_item(m) for m in summary_messages],
    }
    requests: list[_Conversation] = []
    for i, body in enumerate(bodies):
        try:
            requests.append(_conversation(body))
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            return Check("I1", False, f"request {i}: no readable messages or input ({exc})")
    violations: list[dict[str, int]] = []
    byte_mismatches: list[dict[str, int]] = []
    resets: list[int] = []
    # The index of the compaction item the prefix starts from (-1: none). A recording may
    # start after a compaction, e.g. in a new process.
    used = -1
    if requests:
        first = requests[0].semantic
        n = _system_count(first)
        if len(first) > n:
            used = _next_summary(summaries[requests[0].api], first[n], -1)
    for i in range(1, len(requests)):
        prev, cur = requests[i - 1], requests[i]
        kept = len(prev.semantic)  # the elements that must reach `cur` unchanged
        j = _first_difference(prev.semantic, cur.semantic)
        if j is not None:
            k = _reset_to(prev.semantic, cur.semantic, summaries[cur.api], used)
            if k < 0:
                violations.append({"request": i, "message": j})
                continue
            used = k
            resets.append(i)
            kept = _system_count(prev.semantic)
        k = _first_difference(prev.raw[:kept], cur.raw)
        if k is not None:
            byte_mismatches.append({"request": i, "message": k})
    info = {
        "requests": len(requests),
        "apis": sorted({r.api for r in requests}),
        "resets": resets,
        "violations": violations,
        "byte_prefix": not byte_mismatches,
        "byte_mismatches": byte_mismatches,
    }
    if violations:
        v = violations[0]
        detail = (
            f"request {v['request']} does not extend request {v['request'] - 1}:"
            f" {requests[v['request'] - 1].label(v['message'])} changed or was dropped"
        )
    else:
        detail = (
            f"{len(requests)} requests append-only ({len(resets)} compaction resets);"
            f" byte-identical prefix: {'yes' if not byte_mismatches else 'no'}"
        )
    return Check("I1", not violations, detail, info)


@dataclass(slots=True, frozen=True)
class _Conversation:
    """One request's conversation: each element's semantic form and its raw bytes."""

    api: str  # "chat" or "responses"
    semantic: list[dict[str, Any]]
    raw: list[bytes]
    instructions: bool = False  # a Responses request whose first element is `instructions`

    def label(self, j: int) -> str:
        """How a detail names element `j`."""
        if self.api == "chat":
            return f"message {j}"
        if self.instructions:
            return "instructions" if j == 0 else f"input item {j - 1}"
        return f"input item {j}"


def _conversation(body: bytes) -> _Conversation:
    data = json.loads(body)
    if "messages" in data or "input" not in data:
        return _Conversation(
            "chat", [_semantic(m) for m in data["messages"]], _raw_elements(body, b'"messages"')
        )
    elements = data["input"]
    raw = _raw_elements(body, b'"input"')
    if isinstance(elements, str):  # the API reads a string as one user message
        elements = [{"role": "user", "content": elements}]
    semantic = [_semantic_item(item) for item in elements]
    instructions = data.get("instructions")
    if not instructions:
        return _Conversation("responses", semantic, raw)
    system = _semantic_item({"role": "system", "content": instructions})
    return _Conversation(
        "responses", [system, *semantic], [*_raw_elements(body, b'"instructions"'), *raw], True
    )


def _semantic(msg: dict[str, Any]) -> dict[str, Any]:
    """The fields that matter to a provider, with absent treated as None/empty."""
    return {
        "role": msg.get("role"),
        "content": msg.get("content"),
        "tool_calls": [
            (c.get("id"), (c.get("function") or {}).get("name"), _arguments(c))
            for c in msg.get("tool_calls") or []
        ],
        "tool_call_id": msg.get("tool_call_id"),
        "reasoning_details": [
            {k: d.get(k) for k in _REASONING_KEYS} for d in msg.get("reasoning_details") or []
        ],
    }


def _semantic_item(item: dict[str, Any]) -> dict[str, Any]:
    """A Responses input item's fields that matter to the model. Text is compared as text
    (a string or `input_text`/`output_text` parts); item ids and statuses do not count."""
    kind = item.get("type") or ("message" if "role" in item else None)
    match kind:
        case "message":
            content = _parts(item.get("content"))
            return {
                "type": kind,
                "role": item.get("role"),
                "content": content,
                "phase": item.get("phase"),
            }
        case "function_call":
            return {
                "type": kind,
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "arguments": _json(item.get("arguments")),
            }
        case "function_call_output":
            return {
                "type": kind,
                "call_id": item.get("call_id"),
                "output": _parts(item.get("output")),
            }
        case "reasoning":
            summary = [s.get("text") for s in item.get("summary") or [] if isinstance(s, dict)]
            return {
                "type": kind,
                "id": item.get("id"),
                "summary": summary,
                "encrypted_content": item.get("encrypted_content"),
            }
    return item


def _parts(content: Any) -> Any:
    """Text content as a list of texts, whether a string or a list of text parts."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p.get("text") if isinstance(p, dict) and "text" in p else p for p in content]
    return content


def _arguments(call: dict[str, Any]) -> Any:
    return _json((call.get("function") or {}).get("arguments"))


def _json(raw: Any) -> Any:
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return raw


def _first_difference(prev: Sequence[Any], cur: Sequence[Any]) -> int | None:
    """None if `prev` is a prefix of `cur`, else the first index where they differ."""
    for j, message in enumerate(prev):
        if j >= len(cur) or cur[j] != message:
            return j
    return None


def _system_count(messages: list[dict[str, Any]]) -> int:
    """How many system (or developer) messages `messages` starts with."""
    return next(
        (i for i, m in enumerate(messages) if m.get("role") not in _SYSTEM_ROLES), len(messages)
    )


def _reset_to(
    prev: list[dict[str, Any]],
    cur: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    used: int,
) -> int:
    """Where `cur` starts over as rule 8 says (`prev`'s system messages, then a summary): the
    index of the first compaction item after `used` with that summary, or -1 if none."""
    n = _system_count(prev)
    if len(cur) <= n or cur[:n] != prev[:n]:
        return -1
    return _next_summary(summaries, cur[n], used)


def _next_summary(summaries: list[dict[str, Any]], message: dict[str, Any], after: int) -> int:
    """The index of the first summary after index `after` that equals `message`, or -1."""
    return next((k for k in range(after + 1, len(summaries)) if summaries[k] == message), -1)


def _raw_elements(body: bytes, key: bytes) -> list[bytes]:
    """The raw bytes of each element of the top-level array `key` (e.g. b'"messages"'), or of
    its value as one element if it is not an array (a string `input` or `instructions`)."""
    depth, i = 0, 0
    while i < len(body):
        c = body[i]
        if c == _QUOTE:
            end = _string_end(body, i)
            if depth == 1 and body[i:end] == key:
                j = _skip_space(body, end)
                if body[j] == _COLON:
                    j = _skip_space(body, j + 1)
                    if body[j] == _OPEN[0]:
                        return _array_elements(body, j)
                    return [body[j : _value_end(body, j)].strip()]
            i = end
            continue
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        i += 1
    return []


def _value_end(body: bytes, i: int) -> int:
    """Index just past the JSON value that starts at `body[i]` (inside an object or array)."""
    if body[i] == _QUOTE:
        return _string_end(body, i)
    depth = 0
    while i < len(body):
        c = body[i]
        if c == _QUOTE:
            i = _string_end(body, i)
            continue
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            if depth == 0:
                return i
            depth -= 1
            if depth == 0:
                return i + 1
        elif c == _COMMA and depth == 0:
            return i
        i += 1
    return i


def _array_elements(body: bytes, i: int) -> list[bytes]:
    """Split the array that opens at `body[i]` into its raw elements."""
    elements: list[bytes] = []
    depth, start = 0, i + 1
    while i < len(body):
        c = body[i]
        if c == _QUOTE:
            i = _string_end(body, i)
            continue
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth == 0:
                last = body[start:i].strip()
                if last:
                    elements.append(last)
                return elements
        elif c == _COMMA and depth == 1:
            elements.append(body[start:i].strip())
            start = i + 1
        i += 1
    return elements


def _string_end(body: bytes, i: int) -> int:
    """Index just past the JSON string that opens at `body[i]`."""
    i += 1
    while body[i] != _QUOTE:
        i += 2 if body[i] == _BACKSLASH else 1
    return i + 1


def _skip_space(body: bytes, i: int) -> int:
    while body[i] in b" \t\r\n":
        i += 1
    return i


# I2, I3 ----------------------------------------------------------------------------------


_I2_PROBLEMS = (
    "missing",  # a call without a result
    "extra",  # a call with more than one result
    "orphans",  # a result without a call
    "misplaced",  # a result not in the run of results right after its assistant message
    "duplicate_calls",  # a call id in more than one assistant message
    "reran",  # more than one tool.start for a call id
    "unknown_runs",  # a tool that can have side effects ran for a call id in no assistant message
    "late_runs",  # a tool started after the loop's turn.end (kept in the turn row's `late`)
    "unfinished",  # a tool still running at its turn's turn.end (contract rule 6: no orphans)
    "unrun_results",  # a result no tool run produced, though nobody denied or cut short the call
)
# A turn that stops early may answer calls it never ran: a cancel cuts them short, and the step
# cap or budget leaves the last step's calls unrun (their results could never be sent).
_EARLY_STOPS = ("cancelled", "max_steps", "budget", "error")


def check_tool_results(
    items: Sequence[Item], events: Sequence[dict[str, Any]], turns: Sequence[dict[str, Any]]
) -> Check:
    """I2: every tool call has exactly one result, right after its call; every run belongs
    to a call in history, no call runs twice, and every run ends before its turn does.

    Runs are the `tool.start` events plus those the turn rows (`SessionLog.turns`) keep in
    `late`: tools that started after their loop's `turn.end`. A result must come from a run
    (`ToolHost.run`), unless the user denied the call (`Resume.decisions`, recorded in
    `turn.start`) or the result's turn stopped early (`_EARLY_STOPS`).
    """
    calls: Counter[Any] = Counter()
    results: Counter[Any] = Counter()
    result_turns: dict[Any, Any] = {}  # call id -> the turn of its (last) result
    out_of_place: list[Any] = []
    latest: list[Any] = []  # the calls that the next result may answer
    for item in items:
        role = item.message.get("role")
        if role == "tool":
            call_id = item.message.get("tool_call_id")
            results[call_id] += 1
            result_turns[call_id] = item.turn_id
            if call_id not in latest:
                out_of_place.append(call_id)
        else:  # any other item ends the results that answer the assistant message before it
            tool_calls = item.message.get("tool_calls") if role == "assistant" else None
            latest = [c.get("id") for c in tool_calls or []]
            calls.update(latest)
    start_events = [e["data"] for e in events if e["type"] == "tool.start"]
    starts = Counter(d.get("call_id") for d in start_events)
    read_only = {d.get("call_id") for d in start_events if d.get("read_only")}
    late = [
        x["data"].get("call_id")
        for turn in turns
        for x in turn.get("late") or []
        if x["type"] == "tool.start"
    ]
    starts.update(late)
    stops = {e.get("turn"): e["data"].get("stop") for e in events if e["type"] == "turn.end"}
    denied = {
        call_id
        for e in events
        if e["type"] == "turn.start"
        for call_id, decision in ((e["data"].get("resume") or {}).get("decisions") or {}).items()
        if decision == "deny"
    }
    info = {
        "calls": sum(calls.values()),
        "results": sum(results.values()),
        "missing": [c for c in calls if results[c] == 0],
        "extra": [c for c in calls if results[c] > 1],
        "orphans": [r for r in results if r not in calls],
        "misplaced": [c for c in out_of_place if c in calls],
        "duplicate_calls": [c for c, n in calls.items() if n > 1],
        "reran": [c for c, n in starts.items() if n > 1],
        "unknown_runs": [c for c in starts if c not in calls and c not in read_only],
        # Read-only runs whose call never reached history: a loop started a read early and the
        # response then failed or was cut off. Harmless, so reported but not a problem.
        "speculative_runs": [c for c in starts if c not in calls and c in read_only],
        "late_runs": late,
        "unfinished": _unfinished(events, stops),
        "unrun_results": [
            c
            for c, turn in result_turns.items()
            if c in calls
            and starts[c] == 0
            and c not in denied
            and stops.get(turn) not in _EARLY_STOPS
        ],
    }
    problems = [f"{k}: {info[k]}" for k in _I2_PROBLEMS if info[k]]
    detail = "; ".join(problems) or f"{info['calls']} calls, each with exactly one result"
    return Check("I2", not problems, detail, info)


def _unfinished(events: Sequence[dict[str, Any]], stops: dict[Any, Any]) -> list[Any]:
    """Calls whose tool was still running when its turn's `turn.end` was stamped: its
    `tool.end` came later (in the turn row's `late`) or never. A turn without `turn.end` (its
    process died) has no end to compare with."""
    open_runs: Counter[tuple[Any, Any]] = Counter()
    for e in events:
        key = (e.get("turn"), e["data"].get("call_id"))
        if e["type"] == "tool.start":
            open_runs[key] += 1
        elif e["type"] == "tool.end" and open_runs[key] > 0:
            open_runs[key] -= 1
    return [call for (turn, call), n in open_runs.items() if n > 0 and turn in stops]


def check_seq(events: Sequence[dict[str, Any]], items: Sequence[Item]) -> Check:
    """I3: event seqs are 1..n without gaps, and `item` events match the item rows."""
    seqs = [e["seq"] for e in events]
    seen = set(seqs)
    item_events = [e["data"]["item"]["id"] for e in events if e["type"] == "item"]
    info = {
        "events": len(seqs),
        "gaps": [s for s in range(1, max(seqs, default=0) + 1) if s not in seen],
        "duplicates": [s for s, n in Counter(seqs).items() if n > 1],
        "item_events": len(item_events),
        "items": len(items),
    }
    problems = [f"seq {k}: {info[k]}" for k in ("gaps", "duplicates") if info[k]]
    if item_events != [item.id for item in items]:
        problems.append(f"{len(item_events)} item events do not match {len(items)} item rows")
    detail = "; ".join(problems) or f"{len(seqs)} events, {len(items)} items"
    return Check("I3", not problems, detail, info)


# I7 --------------------------------------------------------------------------------------


def check_commits(log: SessionLog, thread_id: str, wc_path: Path) -> Check:
    """I7: one commit per completed turn, in order; HEAD is the last turn's commit; and each
    commit is on record as its turn's last event (rule 7).

    Completed = done, error or cancelled. Compaction turns change no files and have no
    commit; paused turns are committed by the turn that resumes them. An "error" turn without
    a commit failed to commit: a later turn's commit includes its changes, so a later git turn
    must have a commit. (A revert without a commit recorded nothing, and the runner undid what
    git did for it.)
    """
    all_turns = log.turns(thread_id)
    rows = [t for t in all_turns if t["kind"] != "compact"]
    turns = [
        t
        for t in rows
        if t["status"] in ("done", "cancelled") or (t["status"] == "error" and t["commit_sha"])
    ]
    stranded = [
        t["id"]
        for i, t in enumerate(rows)
        if t["status"] == "error"
        and t["kind"] != "revert"
        and not t["commit_sha"]
        and not any(later["commit_sha"] for later in rows[i + 1 :])
    ]
    expected = [t["commit_sha"] for t in turns]
    wc_path = wc_path.absolute()
    proc = subprocess.run(
        ["git", *GIT_CONFIG, "rev-list", "--first-parent", "--reverse", "HEAD"],
        cwd=wc_path,
        env=git_env(wc_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        return Check("I7", False, f"git rev-list failed: {proc.stderr.strip()}")
    commits = proc.stdout.split()[1:]  # the first commit is the empty initial one
    info = {
        "turns": len(turns),
        "commits": len(commits),
        "uncommitted": [t["id"] for t in turns if not t["commit_sha"]] + stranded,
        "unknown": [t["id"] for t in turns if t["commit_sha"] and t["commit_sha"] not in commits],
        "commit_events": _commit_event_problems(all_turns, log.events(thread_id)),
    }
    if info["uncommitted"] or info["unknown"]:
        detail = f"turns without a commit: {info['uncommitted']}; unknown shas: {info['unknown']}"
    elif commits != expected:
        detail = (
            f"{len(commits)} commits after init for {len(turns)} completed turns"
            " (HEAD or order does not match the turn rows)"
        )
    elif info["commit_events"]:
        detail = (
            "turns whose commit event is missing, repeated, not their last event or for"
            f" another sha: {info['commit_events']}"
        )
    else:
        return Check("I7", True, f"{len(turns)} turns, one commit each; HEAD matches", info)
    return Check("I7", False, detail, info)


def _commit_event_problems(
    turns: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]]
) -> list[str]:
    """The turns whose stored events break rule 7: a turn with a commit sha has exactly one
    `commit` event, for that sha, as its last event; any other turn has none."""
    by_turn: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_turn.setdefault(event["turn"], []).append(event)
    problems = []
    for turn in turns:
        stream = by_turn.get(turn["id"], [])
        commits = [e for e in stream if e["type"] == "commit"]
        if turn["commit_sha"]:
            ok = (
                len(commits) == 1
                and stream[-1] is commits[0]
                and commits[0]["data"].get("sha") == turn["commit_sha"]
            )
        else:
            ok = not commits
        if not ok:
            problems.append(turn["id"])
    return problems

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
from bakeoff.shared.runner import SUMMARY_PREFIX
from bakeoff.shared.sessionlog import SessionLog
from bakeoff.shared.workcopy import GIT_CONFIG, git_env

_REASONING_KEYS = ("type", "text", "signature", "data", "format", "index")
_COMMITTED = ("done", "error", "cancelled")
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


def check_prefix(bodies: Sequence[bytes]) -> Check:
    """I1: each request's `messages` are a prefix of the next request's.

    `ok` is semantic equality; byte equality of the raw message elements is reported in
    `info["byte_prefix"]`. A prefix may reset only where a new compaction summary starts.
    """
    requests: list[tuple[list[dict[str, Any]], list[bytes]]] = []
    for i, body in enumerate(bodies):
        try:
            messages = json.loads(body)["messages"]
            requests.append(([_semantic(m) for m in messages], _raw_messages(body)))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return Check("I1", False, f"request {i}: no readable messages array ({exc})")
    violations: list[dict[str, int]] = []
    byte_mismatches: list[dict[str, int]] = []
    resets: list[int] = []
    for i in range(1, len(requests)):
        (prev, prev_raw), (cur, cur_raw) = requests[i - 1], requests[i]
        j = _first_difference(prev, cur)
        if j is None:
            k = _first_difference(prev_raw, cur_raw)
            if k is not None:
                byte_mismatches.append({"request": i, "message": k})
        elif _is_reset(prev, cur):
            resets.append(i)
        else:
            violations.append({"request": i, "message": j})
    info = {
        "requests": len(requests),
        "resets": resets,
        "violations": violations,
        "byte_prefix": not byte_mismatches,
        "byte_mismatches": byte_mismatches,
    }
    if violations:
        v = violations[0]
        detail = (
            f"request {v['request']} does not extend request {v['request'] - 1}:"
            f" message {v['message']} changed or was dropped"
        )
    else:
        detail = (
            f"{len(requests)} requests append-only ({len(resets)} compaction resets);"
            f" byte-identical prefix: {'yes' if not byte_mismatches else 'no'}"
        )
    return Check("I1", not violations, detail, info)


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


def _arguments(call: dict[str, Any]) -> Any:
    raw = (call.get("function") or {}).get("arguments")
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


def _is_reset(prev: list[dict[str, Any]], cur: list[dict[str, Any]]) -> bool:
    first = _first_conversation_message(cur)
    return (
        first is not None
        and isinstance(first["content"], str)
        and first["content"].startswith(SUMMARY_PREFIX)
        and first != _first_conversation_message(prev)  # a new summary, not the same one again
    )


def _first_conversation_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((m for m in messages if m["role"] not in ("system", "developer")), None)


def _raw_messages(body: bytes) -> list[bytes]:
    """The raw bytes of each element of the top-level "messages" array."""
    depth, i = 0, 0
    while i < len(body):
        c = body[i]
        if c == _QUOTE:
            end = _string_end(body, i)
            if depth == 1 and body[i:end] == b'"messages"':
                j = _skip_space(body, end)
                if body[j] == _COLON:
                    return _array_elements(body, _skip_space(body, j + 1))
            i = end
            continue
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        i += 1
    return []


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


def check_tool_results(items: Sequence[Item], events: Sequence[dict[str, Any]]) -> Check:
    """I2: every tool call has exactly one result, no result lacks a call, no call runs twice."""
    calls: Counter[Any] = Counter()
    results: Counter[Any] = Counter()
    for item in items:
        role = item.message.get("role")
        if role == "assistant":
            calls.update(c.get("id") for c in item.message.get("tool_calls") or [])
        elif role == "tool":
            results[item.message.get("tool_call_id")] += 1
    starts = Counter(e["data"].get("call_id") for e in events if e["type"] == "tool.start")
    info = {
        "calls": sum(calls.values()),
        "results": sum(results.values()),
        "missing": [c for c in calls if results[c] == 0],
        "extra": [c for c in calls if results[c] > 1],
        "orphans": [r for r in results if r not in calls],
        "reran": [c for c, n in starts.items() if n > 1],
    }
    problems = [f"{k}: {info[k]}" for k in ("missing", "extra", "orphans", "reran") if info[k]]
    detail = "; ".join(problems) or f"{info['calls']} calls, each with exactly one result"
    return Check("I2", not problems, detail, info)


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
    """I7: one commit per completed turn, in order, and HEAD is the last turn's commit.

    Completed = done, error or cancelled. Compaction turns change no files and have no
    commit; paused turns are committed by the turn that resumes them.
    """
    turns = [
        t for t in log.turns(thread_id) if t["status"] in _COMMITTED and t["kind"] != "compact"
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
        "uncommitted": [t["id"] for t in turns if not t["commit_sha"]],
        "unknown": [t["id"] for t in turns if t["commit_sha"] and t["commit_sha"] not in commits],
    }
    if info["uncommitted"] or info["unknown"]:
        detail = f"turns without a commit: {info['uncommitted']}; unknown shas: {info['unknown']}"
    elif commits != expected:
        detail = (
            f"{len(commits)} commits after init for {len(turns)} completed turns"
            " (HEAD or order does not match the turn rows)"
        )
    else:
        return Check("I7", True, f"{len(turns)} turns, one commit each; HEAD matches", info)
    return Check("I7", False, detail, info)

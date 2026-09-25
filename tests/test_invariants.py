import json
import subprocess

import pytest

from bakeoff.shared.contract import Item
from bakeoff.shared.invariants import (
    _raw_messages,
    check_commits,
    check_prefix,
    check_seq,
    check_tool_results,
    load_wire,
)
from bakeoff.shared.sessionlog import SessionLog
from bakeoff.shared.workcopy import GIT_CONFIG, WorkCopy, git_env

SYSTEM = {"role": "system", "content": "sys"}
USER = {"role": "user", "content": "build a pipe"}
CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.pipe"}'},
        }
    ],
    "reasoning_details": [
        {"type": "reasoning.text", "text": "hm", "signature": "sig", "format": "x", "index": 0}
    ],
}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": "{}"}
ANSWER = {"role": "assistant", "content": "done"}


def body(*messages, **dumps_kwargs):
    return json.dumps({"model": "m", "messages": list(messages), "stream": True}, **dumps_kwargs)


def encode(*messages):
    return body(*messages).encode()


# load_wire


def test_load_wire_sorts_by_number(tmp_path):
    for n in (2, 10, 1):
        (tmp_path / f"{n:03}.json").write_bytes(b'{"n": %d}' % n)
    (tmp_path / "001.meta.json").write_text('{"status": 200, "conn_id": 1}')
    (tmp_path / "010.meta.json").write_text('{"status": 500}')
    wire = load_wire(tmp_path)
    assert [b for b, _ in wire] == [b'{"n": 1}', b'{"n": 2}', b'{"n": 10}']
    assert [m for _, m in wire] == [{"status": 200, "conn_id": 1}, {}, {"status": 500}]
    assert load_wire(tmp_path / "missing") == []


# I1


def test_prefix_holds():
    bodies = [
        encode(SYSTEM, USER),
        encode(SYSTEM, USER, CALL, RESULT),
        encode(SYSTEM, USER, CALL, RESULT, ANSWER),
    ]
    check = check_prefix(bodies)
    assert check.ok, check.detail
    assert check.info["byte_prefix"]
    assert check.info["resets"] == []
    assert check_prefix([]).ok


def test_semantic_equality_ignores_formatting_but_bytes_do_not():
    reformatted = dict(CALL)
    reformatted["tool_calls"] = [
        {
            "type": "function",
            "id": "c1",
            "function": {"arguments": '{"path":"a.pipe"}', "name": "read_file"},
        }
    ]
    # Unknown keys are ignored and an absent field equals None.
    reformatted["reasoning_details"] = [
        {
            "type": "reasoning.text",
            "text": "hm",
            "signature": "sig",
            "format": "x",
            "index": 0,
            "data": None,
            "id": "extra",
        }
    ]
    first = encode(SYSTEM, USER, CALL, RESULT)
    check = check_prefix([first, encode(SYSTEM, USER, reformatted, RESULT, ANSWER)])
    assert check.ok, check.detail
    assert not check.info["byte_prefix"]
    assert check.info["byte_mismatches"] == [{"request": 1, "message": 2}]
    spacing = body(SYSTEM, USER, CALL, RESULT, separators=(",", ":")).encode()
    assert check_prefix([first, spacing]).info["byte_mismatches"] == [{"request": 1, "message": 0}]


@pytest.mark.parametrize(
    "later",
    [
        (SYSTEM, {"role": "user", "content": "edited"}, CALL, RESULT),
        (SYSTEM, USER, RESULT),  # the assistant message was dropped
        (
            SYSTEM,
            USER,
            {**CALL, "reasoning_details": [{**CALL["reasoning_details"][0], "signature": None}]},
        ),
        (
            SYSTEM,
            USER,
            {
                **CALL,
                "tool_calls": [{"id": "c2", "function": {"name": "read_file", "arguments": "{}"}}],
            },
        ),
        (SYSTEM, USER),
    ],
)
def test_prefix_violations(later):
    check = check_prefix([encode(SYSTEM, USER, CALL), encode(*later)])
    assert not check.ok
    assert check.info["violations"][0]["request"] == 1
    assert "request 1 does not extend request 0" in check.detail


SUMMARY = {"role": "user", "content": "[harness] Conversation summary: wrote a.pipe"}
COMPACTION = [Item("t.1:compact", "t.1", SUMMARY, compaction=True)]
TURNS = [{"id": "t.0", "kind": "user"}, {"id": "t.1", "kind": "compact"}]


def test_reset_only_at_a_new_compaction_summary():
    before = encode(SYSTEM, USER, CALL, RESULT, ANSWER)
    after = encode(SYSTEM, SUMMARY, {"role": "user", "content": "next"})
    check = check_prefix(
        [before, after, encode(SYSTEM, SUMMARY, {"role": "user", "content": "next"}, ANSWER)],
        COMPACTION,
        TURNS,
    )
    assert check.ok, check.detail
    assert check.info["resets"] == [1]
    assert check.info["byte_prefix"]
    # The same summary again is not a new compaction, so the prefix must hold.
    dropped = encode(SYSTEM, SUMMARY, ANSWER)
    assert not check_prefix([before, after, dropped], COMPACTION, TURNS).ok
    # A summary that is not first after the system prompt is no reset.
    assert not check_prefix([before, encode(SYSTEM, USER, SUMMARY)], COMPACTION, TURNS).ok
    # The reset keeps the system messages exactly as they were.
    changed = {"role": "system", "content": "changed"}
    for later in [(changed, SUMMARY), (SUMMARY,), (SYSTEM, changed, SUMMARY)]:
        assert not check_prefix([before, encode(*later)], COMPACTION, TURNS).ok, later


def test_only_a_compaction_item_of_the_log_resets_the_prefix():
    before = encode(SYSTEM, USER, CALL, RESULT, ANSWER)
    after = encode(SYSTEM, SUMMARY, {"role": "user", "content": "next"})
    # The loop dropped history behind an ordinary user message that looks like a summary.
    assert not check_prefix([before, after], items(USER, SUMMARY), TURNS).ok
    assert not check_prefix([before, after]).ok
    edited = {**SUMMARY, "content": SUMMARY["content"] + " and b.pipe"}
    assert not check_prefix([before, after], [Item("x", "t.1", edited, compaction=True)], TURNS).ok
    assert check_prefix([before, after], COMPACTION, TURNS).ok
    # A compaction item that no "compact" turn wrote (e.g. a loop's own) does not count either.
    own = [Item("t.0:sum", "t.0", SUMMARY, compaction=True)]
    assert not check_prefix([before, after], own, TURNS).ok
    assert not check_prefix([before, after], COMPACTION).ok  # without the turns: no reset


def test_a_reset_still_compares_the_system_bytes():
    before = encode(SYSTEM, USER, ANSWER)
    # The frozen system prompt is serialized differently after the reset.
    respaced = body(SYSTEM, SUMMARY, separators=(",", ":")).encode()
    check = check_prefix([before, respaced], COMPACTION, TURNS)
    assert check.ok, check.detail  # the same messages, so the reset itself is valid
    assert check.info["resets"] == [1]
    assert not check.info["byte_prefix"]
    assert check.info["byte_mismatches"] == [{"request": 1, "message": 0}]
    assert check_prefix([before, encode(SYSTEM, SUMMARY)], COMPACTION, TURNS).info["byte_prefix"]


def test_unreadable_body_fails():
    check = check_prefix([encode(SYSTEM, USER), b'{"model": "m"}'])
    assert not check.ok
    assert "request 1" in check.detail
    assert not check_prefix([b"not json"]).ok


def test_raw_messages_scanner_is_string_aware():
    tricky = {"role": "user", "content": 'a "quoted" [bracket], {brace}, \\ and "messages": [1]'}
    raw = (
        b'{"tools": [{"messages": [1, 2]}], "note": "\\"messages\\": [",\n'
        b' "messages" : [ {"role": "system", "content": "s"} ,\n  '
        + json.dumps(tricky).encode()
        + b' ], "stream": true}'
    )
    elements = _raw_messages(raw)
    assert elements == [b'{"role": "system", "content": "s"}', json.dumps(tricky).encode()]
    assert [json.loads(e) for e in elements] == json.loads(raw)["messages"]
    assert _raw_messages(b'{"messages": []}') == []


# I2


def items(*messages):
    return [Item(f"i{n}", "t.0", m) for n, m in enumerate(messages)]


def tool_start(call_id):
    return {"type": "tool.start", "data": {"call_id": call_id, "name": "read_file"}}


def test_tool_results_ok():
    check = check_tool_results(items(USER, CALL, RESULT, ANSWER), [tool_start("c1")], [])
    assert check.ok, check.detail
    assert (check.info["calls"], check.info["results"]) == (1, 1)


@pytest.mark.parametrize(
    ("messages", "starts", "problem"),
    [
        ((USER, CALL, ANSWER), ["c1"], "missing"),
        ((USER, CALL, RESULT, RESULT), ["c1"], "extra"),
        ((USER, RESULT), [], "orphans"),
        ((USER, RESULT, CALL), ["c1"], "misplaced"),  # the result before its call
        ((USER, CALL, ANSWER, RESULT), ["c1"], "misplaced"),  # the model answered without it
        ((USER, CALL, USER, RESULT), ["c1"], "misplaced"),  # a new user message came between
        ((USER, CALL, RESULT, CALL), ["c1"], "duplicate_calls"),  # e.g. re-emitted on resume
        ((USER, CALL, RESULT), ["c1", "c1"], "reran"),
        ((USER,), ["ghost"], "unknown_runs"),  # e.g. an eager tool of a retried stream
    ],
)
def test_tool_result_violations(messages, starts, problem):
    check = check_tool_results(items(*messages), [tool_start(c) for c in starts], [])
    assert not check.ok
    assert check.info[problem]
    assert check.detail.startswith(problem)


def test_a_tool_started_after_turn_end_counts_as_a_run():
    late = {"t_us": 9, "type": "tool.start", "data": {"call_id": "c1", "name": "read_file"}}
    turns = [{"id": "t.0", "late": None}, {"id": "t.1", "late": [late]}]
    check = check_tool_results(items(USER, CALL, RESULT), [tool_start("c1")], turns)
    assert not check.ok
    assert (check.info["reran"], check.info["late_runs"]) == (["c1"], ["c1"])
    # The only run of a paused turn's pending call, started after its turn.end.
    check = check_tool_results(items(USER, CALL), [], turns)
    assert (check.info["missing"], check.info["late_runs"]) == (["c1"], ["c1"])


# I3


def item_event(seq, item_id):
    return {"seq": seq, "type": "item", "data": {"item": {"id": item_id}}}


def test_seq_ok():
    events = [{"seq": 1, "type": "turn.start"}, item_event(2, "i0"), item_event(3, "i1")]
    check = check_seq(events, items(USER, ANSWER))
    assert check.ok, check.detail
    assert check_seq([], []).ok


@pytest.mark.parametrize(
    ("seqs", "problem"),
    [([1, 3], "gaps"), ([2, 3], "gaps"), ([1, 2, 2], "duplicates")],
)
def test_seq_gaps_and_duplicates(seqs, problem):
    check = check_seq([{"seq": s, "type": "text.delta"} for s in seqs], [])
    assert not check.ok
    assert check.info[problem]


def test_item_events_must_match_rows():
    events = [item_event(1, "i0")]
    assert not check_seq(events, items(USER, ANSWER)).ok
    assert not check_seq([item_event(1, "other")], items(USER)).ok


# I7


def git(root, *args):
    subprocess.run(["git", *GIT_CONFIG, *args], cwd=root, env=git_env(root), check=True)


@pytest.fixture
def log(tmp_path):
    log = SessionLog(tmp_path / "log.sqlite")
    log.create_thread("th", impl="our", system="s", meta={})
    yield log
    log.close()


async def committed_turn(log, wc, status="done", kind="user"):
    turn = log.start_turn("th", kind)
    sha, _ = await wc.commit(f"turn {turn['idx']}")
    log.set_turn_status(turn["id"], status, stop="end_turn", commit_sha=sha)
    return sha


async def test_commits_ok(log, tmp_path):
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    assert check_commits(log, "th", wc.root).ok  # no turns yet
    await committed_turn(log, wc)
    paused = log.start_turn("th", "user")
    log.set_turn_status(paused["id"], "paused", stop="paused", pending=["c1"])
    await committed_turn(log, wc, kind="approval")
    await committed_turn(log, wc, status="cancelled")
    log.set_turn_status(log.start_turn("th", "compact")["id"], "done")
    log.start_turn("th", "user")  # crashed: still running, never committed
    await committed_turn(log, wc, kind="crash", status="error")
    check = check_commits(log, "th", wc.root)
    assert check.ok, check.detail
    assert check.info["turns"] == check.info["commits"] == 4


async def test_an_error_turn_that_failed_to_commit_is_not_a_committed_turn(log, tmp_path):
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    await committed_turn(log, wc)
    failed = log.start_turn("th", "user")
    log.set_turn_status(failed["id"], "error", stop="end_turn")  # its commit failed
    await committed_turn(log, wc)  # the next turn's commit includes its changes
    check = check_commits(log, "th", wc.root)
    assert check.ok, check.detail
    assert check.info["turns"] == check.info["commits"] == 2


async def test_extra_commit_fails(log, tmp_path):
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    await committed_turn(log, wc)
    await wc.commit("not a turn")
    check = check_commits(log, "th", wc.root)
    assert not check.ok
    assert "2 commits after init for 1 completed turns" in check.detail


async def test_missing_or_unknown_sha_fails(log, tmp_path):
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    await committed_turn(log, wc)
    log.set_turn_status("th.0", "done", commit_sha="0" * 40)
    assert check_commits(log, "th", wc.root).info["unknown"] == ["th.0"]
    log.set_turn_status("th.0", "done")
    check = check_commits(log, "th", wc.root)
    assert not check.ok
    assert check.info["uncommitted"] == ["th.0"]


async def test_head_and_order_must_match_the_turns(log, tmp_path):
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    first = await committed_turn(log, wc)
    second = await committed_turn(log, wc)
    log.set_turn_status("th.0", "done", commit_sha=second)
    log.set_turn_status("th.1", "done", commit_sha=first)
    check = check_commits(log, "th", wc.root)
    assert not check.ok
    assert "HEAD or order" in check.detail

    log.set_turn_status("th.0", "done", commit_sha=first)
    log.set_turn_status("th.1", "done", commit_sha=second)
    git(wc.root, "reset", "-q", "--hard", "HEAD~1")
    assert check_commits(log, "th", wc.root).info["unknown"] == ["th.1"]


def test_commits_without_repo_fail(log, tmp_path):
    check = check_commits(log, "th", tmp_path)
    assert not check.ok
    assert "git rev-list failed" in check.detail

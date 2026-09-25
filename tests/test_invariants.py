import json
import subprocess

import pytest

from bakeoff.shared.contract import Item
from bakeoff.shared.invariants import (
    _raw_elements,
    check_commits,
    check_prefix,
    check_seq,
    check_tool_results,
    load_wire,
)
from bakeoff.shared.sessionlog import SessionLog, event_row
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


S1 = {"role": "user", "content": "[harness] Conversation summary: one"}
S2 = {"role": "user", "content": "[harness] Conversation summary: two"}


def compactions(*summaries):
    """Compaction items of turns t.1, t.3, ... and the turn rows: t.0, t.2, ... are user turns."""
    log_items = [
        Item(f"t.{2 * n + 1}:compact", f"t.{2 * n + 1}", m, compaction=True)
        for n, m in enumerate(summaries)
    ]
    turns = [
        {"id": f"t.{k}", "kind": "compact" if k % 2 else "user"}
        for k in range(2 * len(summaries) + 1)
    ]
    return log_items, turns


def user(text):
    return {"role": "user", "content": text}


def test_resets_follow_the_compaction_items_in_log_order():
    log_items, turns = compactions(S1, S2)
    one, two, three = (
        encode(SYSTEM, user("u1")),
        encode(SYSTEM, S1, user("u2")),
        encode(SYSTEM, S2, user("u3")),
    )
    check = check_prefix([one, two, three], log_items, turns)
    assert (check.ok, check.info["resets"]) == (True, [1, 2])
    # Back to the older summary: rule 8 says a request starts from the last one.
    back = encode(SYSTEM, S1, user("u2"), ANSWER, user("u3"))
    check = check_prefix([one, two, three, back], log_items, turns)
    assert (check.ok, check.info["violations"]) == (False, [{"request": 3, "message": 1}])


def test_two_compactions_may_have_the_same_summary():
    log_items, turns = compactions(SUMMARY, SUMMARY)
    bodies = [
        encode(SYSTEM, user("u1")),
        encode(SYSTEM, SUMMARY, user("u2")),
        encode(SYSTEM, SUMMARY, user("u3")),  # the second compaction
    ]
    check = check_prefix(bodies, log_items, turns)
    assert (check.ok, check.info["resets"]) == (True, [1, 2]), check.detail
    # A third reset to the same text has no compaction item left to match.
    assert not check_prefix([*bodies, encode(SYSTEM, SUMMARY, user("u4"))], log_items, turns).ok


def test_a_recording_may_start_after_a_compaction():
    log_items, turns = compactions(S1, S2)
    bodies = [encode(SYSTEM, S1, user("u2")), encode(SYSTEM, S2, user("u3"))]
    assert check_prefix(bodies, log_items, turns).info["resets"] == [1]
    started_late = [encode(SYSTEM, S2, user("u3")), encode(SYSTEM, S1, user("u2"))]
    assert not check_prefix(started_late, log_items, turns).ok


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
    elements = _raw_elements(raw, b'"messages"')
    assert elements == [b'{"role": "system", "content": "s"}', json.dumps(tricky).encode()]
    assert [json.loads(e) for e in elements] == json.loads(raw)["messages"]
    assert _raw_elements(b'{"messages": []}', b'"messages"') == []


def test_raw_scanner_reads_a_value_that_is_not_an_array():
    raw = b'{"instructions" : "say \\"hi\\", {x}" , "input": "hello", "n": {"a": [1]}}'
    assert _raw_elements(raw, b'"instructions"') == [b'"say \\"hi\\", {x}"']
    assert _raw_elements(raw, b'"input"') == [b'"hello"']
    assert _raw_elements(raw, b'"n"') == [b'{"a": [1]}']
    assert _raw_elements(raw, b'"missing"') == []


# I1 on Responses API bodies: `instructions`, then the `input` items.

R_USER = {"role": "user", "content": "build a pipe"}
R_REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "plan"}],
    "encrypted_content": "gAAAAB-opaque",
}
R_CALL = {
    "id": "fc_1",
    "type": "function_call",
    "status": "completed",
    "arguments": '{"path": "a.pipe"}',
    "call_id": "c1",
    "name": "read_file",
}
R_OUTPUT = {"type": "function_call_output", "call_id": "c1", "output": "{}"}
R_ANSWER = {
    "id": "msg_1",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "annotations": [], "text": "done"}],
    "phase": "final_answer",
}


def responses_body(*items, instructions="sys", **dumps_kwargs):
    data = {"model": "m", "instructions": instructions, "input": list(items), "stream": True}
    if instructions is None:
        del data["instructions"]
    return json.dumps(data, **dumps_kwargs).encode()


def test_responses_prefix_holds_with_instructions_as_the_system_prompt():
    bodies = [
        responses_body(R_USER),
        responses_body(R_USER, R_REASONING, R_CALL, R_OUTPUT),
        responses_body(R_USER, R_REASONING, R_CALL, R_OUTPUT, R_ANSWER),
    ]
    check = check_prefix(bodies)
    assert check.ok, check.detail
    assert check.info["byte_prefix"]
    assert check.info["apis"] == ["responses"]
    # A string input is one user message.
    hello = {"role": "user", "content": "hi"}
    assert check_prefix([b'{"input": "hi"}', responses_body(hello, R_ANSWER, instructions=None)]).ok


def test_responses_semantics_compare_text_not_representation():
    as_parts = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "build a pipe"}],
    }
    respaced = {**R_CALL, "id": "other", "status": None, "arguments": '{"path":"a.pipe"}'}
    first = responses_body(R_USER, R_REASONING, R_CALL)
    check = check_prefix([first, responses_body(as_parts, R_REASONING, respaced, R_OUTPUT)])
    assert check.ok, check.detail
    assert check.info["byte_mismatches"] == [{"request": 1, "message": 1}]


@pytest.mark.parametrize(
    ("later", "changed"),
    [
        ((R_USER, {**R_REASONING, "encrypted_content": None}, R_CALL), "input item 1"),
        ((R_USER, {**R_REASONING, "summary": []}, R_CALL), "input item 1"),
        ((R_USER, R_CALL), "input item 1"),  # the reasoning item was dropped
        ((R_USER, R_REASONING, {**R_CALL, "call_id": "c2"}), "input item 2"),
        ((R_USER, R_REASONING, R_CALL, R_OUTPUT), None),
    ],
)
def test_responses_prefix_violations(later, changed):
    check = check_prefix([responses_body(R_USER, R_REASONING, R_CALL), responses_body(*later)])
    assert check.ok is (changed is None), check.detail
    if changed:
        assert (
            check.detail == f"request 1 does not extend request 0: {changed} changed or was dropped"
        )


def test_responses_instructions_are_part_of_the_prefix():
    before = responses_body(R_USER)
    check = check_prefix([before, responses_body(R_USER, R_ANSWER, instructions="other")])
    assert (
        check.detail == "request 1 does not extend request 0: instructions changed or was dropped"
    )
    # A system prompt moved from `instructions` into `input` is the same conversation, though
    # not the same bytes.
    moved = responses_body(
        {"role": "system", "content": "sys"}, R_USER, R_ANSWER, instructions=None
    )
    check = check_prefix([before, moved])
    assert check.ok, check.detail
    assert not check.info["byte_prefix"]


def test_responses_reset_at_a_compaction_summary():
    before = responses_body(R_USER, R_REASONING, R_CALL, R_OUTPUT, R_ANSWER)
    summary_parts = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": SUMMARY["content"]}],
    }
    after = responses_body(summary_parts, {"role": "user", "content": "next"})
    check = check_prefix([before, after], COMPACTION, TURNS)
    assert (check.ok, check.info["resets"]) == (True, [1]), check.detail
    assert not check_prefix([before, after]).ok


def test_a_chat_request_does_not_extend_a_responses_request():
    check = check_prefix([responses_body(R_USER), encode(SYSTEM, USER, ANSWER)])
    assert not check.ok
    assert check.info["apis"] == ["chat", "responses"]


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


def turn_events(turn, *types, stop="end_turn", resume=None):
    """Events of one turn: turn.start, then `types` (tool events are about c1), then turn.end."""
    start = {"turn_id": turn} | ({"resume": resume} if resume else {})
    return [
        {"turn": turn, "type": "turn.start", "data": start},
        *({"turn": turn, "type": t, "data": {"call_id": "c1", "name": "x"}} for t in types),
        {"turn": turn, "type": "turn.end", "data": {"stop": stop, "steps": 1}},
    ]


def test_a_run_must_end_before_its_turn_ends():
    done = turn_events("t.0", "tool.start", "tool.end")
    assert check_tool_results(items(USER, CALL, RESULT), done, []).ok
    # The tool.end came after turn.end (it is in the turn row's `late`), or never.
    running = turn_events("t.0", "tool.start")
    check = check_tool_results(items(USER, CALL, RESULT), running, [])
    assert not check.ok and check.info["unfinished"] == ["c1"]
    assert check.detail == "unfinished: ['c1']"
    # A turn without turn.end died with its process: nothing ran on after it.
    crashed = running[:-1]
    assert check_tool_results(items(USER, CALL, RESULT), crashed, []).ok


@pytest.mark.parametrize(
    ("stop", "resume", "ok"),
    [
        ("end_turn", None, False),  # e.g. a loop that invents results instead of running tools
        ("paused", None, False),
        ("end_turn", {"kind": "approval", "decisions": {"c1": "deny"}}, True),
        ("end_turn", {"kind": "approval", "decisions": {"c1": "allow"}}, False),
        ("cancelled", None, True),  # cut short by the cancel
        ("max_steps", None, True),  # the last step's calls never run
        ("budget", None, True),
        ("error", None, True),
    ],
)
def test_a_result_needs_a_run_unless_denied_or_cut_short(stop, resume, ok):
    events = turn_events("t.0", stop=stop, resume=resume)
    check = check_tool_results(items(USER, CALL, RESULT), events, [])
    assert check.ok is ok
    assert check.info["unrun_results"] == ([] if ok else ["c1"])


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


def event(log, turn_id, type_="commit", **data):
    env = {"v": 1, "thread": "th", "turn": turn_id, "impl": "our", "seq": log.next_seq("th")}
    return event_row({**env, "t_us": 0, "type": type_, "data": data})


async def committed_turn(log, wc, status="done", kind="user"):
    """A turn as the runner records it: the sha on the row, a `commit` event last."""
    turn = log.start_turn("th", kind)
    log.append_events([event(log, turn["id"], "turn.start")])
    sha, files = await wc.commit(f"turn {turn['idx']}")
    commit = event(log, turn["id"], sha=sha, files=files)
    log.set_turn_status(turn["id"], status, stop="end_turn", commit_sha=sha, events=[commit])
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


async def test_an_error_turn_needs_a_later_commit(log, tmp_path):
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    await committed_turn(log, wc)
    failed = log.start_turn("th", "user")
    log.set_turn_status(failed["id"], "error", stop="end_turn")  # its commit failed
    (wc.root / "b.pipe").write_text("{}")  # ...and nothing ever committed its change
    check = check_commits(log, "th", wc.root)
    assert (check.ok, check.info["uncommitted"]) == (False, ["th.1"])
    log.set_turn_status(log.start_turn("th", "compact")["id"], "done")
    paused = log.start_turn("th", "user")
    log.set_turn_status(paused["id"], "paused", stop="paused", pending=["c1"])
    assert check_commits(log, "th", wc.root).info["uncommitted"] == ["th.1"]
    await committed_turn(log, wc, kind="approval")  # its commit includes b.pipe
    check = check_commits(log, "th", wc.root)
    assert check.ok, check.detail
    # A failed revert recorded nothing (the runner undid git's part): no commit is owed.
    log.set_turn_status(log.start_turn("th", "revert")["id"], "error")
    check = check_commits(log, "th", wc.root)
    assert check.ok, check.detail


async def test_each_commit_is_its_turns_last_event(log, tmp_path):
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    sha = await committed_turn(log, wc)
    assert check_commits(log, "th", wc.root).ok
    log.append_events([event(log, "th.0", "text.delta", text="after the commit")])
    check = check_commits(log, "th", wc.root)
    assert (check.ok, check.info["commit_events"]) == (False, ["th.0"])
    assert "commit event is missing, repeated, not their last event" in check.detail

    turn = log.start_turn("th", "user")  # the sha is on the row, but no commit event
    (wc.root / "a.pipe").write_text("{}")
    sha, _ = await wc.commit("turn 1")
    log.set_turn_status(turn["id"], "done", stop="end_turn", commit_sha=sha)
    assert check_commits(log, "th", wc.root).info["commit_events"] == ["th.0", "th.1"]
    # A commit event for another sha, or for a turn without one, is wrong too.
    log.append_events([event(log, "th.1", sha="0" * 40)])
    compact = log.start_turn("th", "compact")["id"]
    log.set_turn_status(compact, "done", events=[event(log, compact, sha=sha)])
    assert check_commits(log, "th", wc.root).info["commit_events"] == ["th.0", "th.1", "th.2"]


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


def test_i2_tolerates_speculative_read_only_runs_but_not_side_effects():
    """A read started early whose call never reached history is speculative (reported, not a
    problem); an unrecorded run of a tool with side effects is still a failure."""
    from bakeoff.shared.contract import Item
    from bakeoff.shared.invariants import check_tool_results

    items = [Item("u", "t", {"role": "user", "content": "go"})]

    def start(call_id, read_only):
        return {
            "type": "tool.start",
            "data": {"call_id": call_id, "name": "x", "read_only": read_only},
        }

    ok = check_tool_results(items, [start("spec", True)], [])
    assert ok.ok and ok.info["speculative_runs"] == ["spec"] and ok.info["unknown_runs"] == []
    bad = check_tool_results(items, [start("write", False)], [])
    assert not bad.ok and bad.info["unknown_runs"] == ["write"]

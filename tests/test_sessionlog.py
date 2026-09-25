import sqlite3
import subprocess
import sys

import pytest

from bakeoff.shared.contract import Item
from bakeoff.shared.sessionlog import (
    SessionLog,
    event_row,
    item_from_json,
    item_to_json,
    now_us,
)


@pytest.fixture
def log(tmp_path):
    log = SessionLog(tmp_path / "out" / "log.sqlite")
    log.create_thread("th", impl="our", system="sys", meta={"rules": {"*": "allow"}})
    yield log
    log.close()


def env(seq, type_="text.delta", turn="th.0", **data):
    return {
        "v": 1,
        "thread": "th",
        "turn": turn,
        "impl": "our",
        "seq": seq,
        "t_us": seq * 10,
        "type": type_,
        "data": data,
    }


def row(seq, type_="text.delta", turn="th.0", **data):
    return event_row(env(seq, type_, turn, **data))


def test_wal_mode(log, tmp_path):
    db = sqlite3.connect(tmp_path / "out" / "log.sqlite")
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    db.close()


def test_thread_roundtrip(log):
    thread = log.get_thread("th")
    assert thread["impl"] == "our"
    assert thread["system"] == "sys"
    assert thread["meta"] == {"rules": {"*": "allow"}}
    assert thread["created_us"] > 0
    assert log.get_thread("nope") is None


def test_turns_get_indexes_and_status(log):
    first = log.start_turn("th", "user")
    assert first["id"] == "th.0"
    assert (first["idx"], first["kind"], first["status"], first["ended_us"]) == (
        0,
        "user",
        "running",
        None,
    )
    late = [{"t_us": 5, "type": "tool.start", "data": {"call_id": "c1", "name": "x"}}]
    log.set_turn_status("th.0", "paused", stop="paused", pending=["c1", "c2"], late=late)
    second = log.start_turn("th", "approval")
    assert second["idx"] == 1
    log.set_turn_status("th.1", "done", stop="end_turn", commit_sha="abc")
    turns = log.turns("th")
    assert [t["status"] for t in turns] == ["paused", "done"]
    assert (turns[0]["pending"], turns[0]["late"]) == (["c1", "c2"], late)
    assert turns[0]["ended_us"] >= turns[0]["started_us"]
    assert turns[1]["commit_sha"] == "abc"
    assert turns[1]["pending"] is turns[1]["late"] is None
    assert log.last_turn("th")["id"] == "th.1"
    assert log.last_turn("other") is None


def test_only_a_crash_resume_starts_while_a_turn_runs(log, tmp_path):
    log.start_turn("th", "user")
    other = SessionLog(tmp_path / "out" / "log.sqlite")  # e.g. another process
    for kind in ("user", "approval", "revert", "compact"):
        with pytest.raises(RuntimeError, match=r"running turn: th\.0"):
            other.start_turn("th", kind)
    assert other.start_turn("th", "crash")["id"] == "th.1"
    other.close()
    log.set_turn_status("th.1", "done")
    assert log.start_turn("th", "user")["id"] == "th.2"


def test_discard_turn(log):
    log.start_turn("th", "revert")
    log.discard_turn("th.0")
    assert log.turns("th") == []
    assert log.start_turn("th", "user")["id"] == "th.0"


def test_turn_kind_and_status_are_checked(log):
    with pytest.raises(sqlite3.IntegrityError):
        log.start_turn("th", "bogus")
    log.start_turn("th", "user")
    with pytest.raises(sqlite3.IntegrityError):
        log.set_turn_status("th.0", "finished")


def test_item_json_keeps_every_field():
    item = Item(
        id="i1",
        turn_id="th.0",
        message={"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        status="incomplete",
        native={"parts": [{"kind": "text", "n": 1}]},
        usage={"input_tokens": 3},
        compaction=True,
    )
    data = item_to_json(item)
    assert set(data) == {"id", "turn_id", "message", "status", "native", "usage", "compaction"}
    assert item_from_json(data) == item


def test_items_append_in_order_with_events(log):
    a = Item("a", "th.0", {"role": "user", "content": "hi"})
    b = Item("b", "th.0", {"role": "assistant", "content": "yo"}, native=[1, 2])
    assert log.append_item("th", a) == 0
    assert log.append_item("th", b, [row(1, "turn.start"), row(2, "item")]) == 1
    assert log.items("th") == [a, b]
    assert [e["seq"] for e in log.events("th")] == [1, 2]
    assert log.items("other") == []


def test_events_batch_and_next_seq(log):
    assert log.next_seq("th") == 1
    log.append_events([row(1), row(2, text="x")])
    log.append_events([row(3, "turn.end", turn="th.1", stop="end_turn")])
    events = log.events("th")
    assert events[1] == env(2, text="x")
    assert [e["turn"] for e in events] == ["th.0", "th.0", "th.1"]
    assert log.next_seq("th") == 4


def test_event_row_rejects_data_that_is_not_json():
    with pytest.raises(TypeError):
        row(1, cost=object())


def test_duplicate_event_seq_is_rejected(log):
    log.append_events([row(1)])
    with pytest.raises(sqlite3.IntegrityError):
        log.append_events([row(2), row(1)])
    assert [e["seq"] for e in log.events("th")] == [1]  # the failed batch left nothing behind


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE items SET json = '{}'",
        "DELETE FROM items",
        "UPDATE events SET type = 'x'",
        "DELETE FROM events",
        # REPLACE deletes the old row without firing DELETE triggers.
        "INSERT OR REPLACE INTO items VALUES ('th', 0, 'th.0', 'a', '{}')",
        "REPLACE INTO events VALUES ('th', 1, 'th.0', 'forged', 0, '{}')",
        "INSERT INTO events VALUES ('th', 1, 'th.0', 'forged', 0, '{}')"
        " ON CONFLICT DO UPDATE SET type = 'forged'",
    ],
)
def test_items_and_events_are_append_only(log, tmp_path, sql):
    item = Item("a", "th.0", {"role": "user", "content": "hi"})
    log.append_item("th", item, [row(1, "item")])
    db = sqlite3.connect(tmp_path / "out" / "log.sqlite")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute(sql)
    db.close()
    assert log.items("th") == [item]
    assert [e["type"] for e in log.events("th")] == ["item"]


def test_cancel_requests_since(log):
    started = now_us()
    assert not log.cancel_requested("th", started)
    log.request_cancel("th")
    assert log.cancel_requested("th", started)
    assert not log.cancel_requested("th", now_us() + 1)
    assert not log.cancel_requested("other")


def test_two_connections_share_the_log(log, tmp_path):
    other = SessionLog(tmp_path / "out" / "log.sqlite")
    other.start_turn("th", "user")
    other.append_item("th", Item("a", "th.0", {"role": "user", "content": "hi"}))
    assert log.last_turn("th")["id"] == "th.0"
    assert log.items("th")[0].id == "a"
    other.close()


def test_cancel_from_another_process(log, tmp_path):
    code = (
        "import sys; from bakeoff.shared.sessionlog import SessionLog;"
        " SessionLog(sys.argv[1]).request_cancel('th')"
    )
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "out" / "log.sqlite")], check=True)
    assert log.cancel_requested("th")

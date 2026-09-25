"""SQLite session log: threads, turns, items and events.

Items and events are append-only (triggers reject UPDATE, DELETE and any insert over an
existing row); thread and turn rows may change. The file runs in WAL mode, so several
processes can share it (a worker running a turn plus a separate `approve` or `cancel`
command).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

from bakeoff.shared.contract import Item

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads(
    id TEXT PRIMARY KEY, impl TEXT NOT NULL, system TEXT NOT NULL, meta TEXT NOT NULL,
    created_us INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS turns(
    id TEXT PRIMARY KEY, thread TEXT NOT NULL, idx INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('user', 'approval', 'crash', 'revert', 'compact')),
    status TEXT NOT NULL
        CHECK (status IN ('running', 'paused', 'done', 'error', 'cancelled')),
    stop TEXT, pending TEXT, commit_sha TEXT, started_us INTEGER NOT NULL, ended_us INTEGER,
    late TEXT, UNIQUE (thread, idx));
CREATE TABLE IF NOT EXISTS items(
    thread TEXT NOT NULL, seq INTEGER NOT NULL, turn TEXT NOT NULL, id TEXT NOT NULL,
    json TEXT NOT NULL, PRIMARY KEY (thread, seq));
CREATE TABLE IF NOT EXISTS events(
    thread TEXT NOT NULL, seq INTEGER NOT NULL, turn TEXT NOT NULL, type TEXT NOT NULL,
    t_us INTEGER NOT NULL, json TEXT NOT NULL, PRIMARY KEY (thread, seq));
CREATE TABLE IF NOT EXISTS cancels(thread TEXT NOT NULL, ts INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS cancels_thread ON cancels(thread, ts);
CREATE TRIGGER IF NOT EXISTS items_no_update BEFORE UPDATE ON items
    BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER IF NOT EXISTS items_no_delete BEFORE DELETE ON items
    BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'append-only'); END;
-- REPLACE deletes the old row without firing DELETE triggers, so block it at the insert.
CREATE TRIGGER IF NOT EXISTS items_no_replace BEFORE INSERT ON items
    WHEN EXISTS (SELECT 1 FROM items WHERE thread = NEW.thread AND seq = NEW.seq)
    BEGIN SELECT RAISE(ABORT, 'append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_replace BEFORE INSERT ON events
    WHEN EXISTS (SELECT 1 FROM events WHERE thread = NEW.thread AND seq = NEW.seq)
    BEGIN SELECT RAISE(ABORT, 'append-only'); END;
"""

_ITEM_FIELDS = tuple(f.name for f in fields(Item))

# One stored event: thread, seq, turn, type, t_us and the envelope as JSON.
EventRow = tuple[str, int, str, str, int, str]


def now_us() -> int:
    """Wall-clock time in microseconds (comparable across processes)."""
    return time.time_ns() // 1000


def item_to_json(item: Item) -> dict[str, Any]:
    """Every `Item` field as a JSON-ready dict; `native` is kept as-is."""
    return {name: getattr(item, name) for name in _ITEM_FIELDS}


def item_from_json(data: dict[str, Any]) -> Item:
    """Inverse of `item_to_json`."""
    return Item(**data)


def event_row(envelope: dict[str, Any]) -> EventRow:
    """Serialize an event envelope (`{"v", "thread", "turn", "impl", "seq", "t_us", "type",
    "data"}`) for `append_events` / `append_item`. Raises TypeError if it is not JSON."""
    e = envelope
    return (e["thread"], e["seq"], e["turn"], e["type"], e["t_us"], json.dumps(e))


class SessionLog:
    """The durable record of every thread. Writes are small and synchronous; any number of
    instances, in one or more processes, can share a file."""

    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        # busy_timeout first, so switching to WAL waits for other processes instead of failing.
        self._db.execute("PRAGMA busy_timeout = 5000")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA synchronous = NORMAL")
        self._db.executescript(_SCHEMA)
        # A log created before `turns.late` existed gets the column. Checked first, so opening
        # an up-to-date log never writes to it.
        if "late" not in self._columns("turns"):
            with self._tx() as db:
                if "late" not in self._columns("turns"):  # another process may have added it
                    db.execute("ALTER TABLE turns ADD COLUMN late TEXT")

    def _columns(self, table: str) -> set[str]:
        return {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        # IMMEDIATE takes the write lock up front, so read-then-insert (next seq/idx) is atomic.
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # threads

    def create_thread(
        self, thread_id: str, *, impl: str, system: str, meta: dict[str, Any]
    ) -> None:
        self._db.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?)",
            (thread_id, impl, system, json.dumps(meta), now_us()),
        )

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """`{"id", "impl", "system", "meta", "created_us"}`, or None if unknown."""
        row = self._db.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        return None if row is None else {**dict(row), "meta": json.loads(row["meta"])}

    # turns

    def start_turn(self, thread_id: str, kind: str) -> dict[str, Any]:
        """Insert a running turn with the next index; its id is `<thread>.<idx>`.

        Raises RuntimeError if the thread's last turn is still running, unless `kind` is
        "crash" (the worker running it died). The check and the insert are one transaction,
        so two processes cannot both start a turn.
        """
        with self._tx() as db:
            last = db.execute(
                "SELECT id, idx, status FROM turns WHERE thread = ? ORDER BY idx DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
            if last is not None and last["status"] == "running" and kind != "crash":
                raise RuntimeError(f"thread {thread_id} has a running turn: {last['id']}")
            idx = 0 if last is None else last["idx"] + 1
            turn_id = f"{thread_id}.{idx}"
            db.execute(
                "INSERT INTO turns (id, thread, idx, kind, status, started_us)"
                " VALUES (?, ?, ?, ?, 'running', ?)",
                (turn_id, thread_id, idx, kind, now_us()),
            )
        return self._turn_rows("WHERE id = ?", turn_id)[0]

    def set_turn_status(
        self,
        turn_id: str,
        status: str,
        *,
        stop: str | None = None,
        pending: list[str] | None = None,
        commit_sha: str | None = None,
        events: Iterable[EventRow] = (),
        item: Item | None = None,
    ) -> None:
        """Record how a turn ended (sets `ended_us` unless the status is "running"), in one
        transaction with its last `events` and, if given, a last `item` of its thread."""
        with self._tx() as db:
            if item is not None:
                (thread_id,) = db.execute(
                    "SELECT thread FROM turns WHERE id = ?", (turn_id,)
                ).fetchone()
                self._insert_item(db, thread_id, item)
            self._insert_events(db, events)
            db.execute(
                "UPDATE turns SET status = ?, stop = ?, pending = ?, commit_sha = ?, ended_us = ?"
                " WHERE id = ?",
                (
                    status,
                    stop,
                    None if pending is None else json.dumps(pending),
                    commit_sha,
                    None if status == "running" else now_us(),
                    turn_id,
                ),
            )

    def append_late(self, turn_id: str, event: dict[str, Any]) -> None:
        """Add a tool event that came after the loop's `turn.end` to the turn row's `late` list.
        Contract rule 7 keeps it out of the event stream; it is kept here as evidence, at any
        time (e.g. from a tool task that outlived its turn). A value that is not JSON is stored
        as its str()."""
        self._db.execute(
            "UPDATE turns SET late = json_insert(COALESCE(late, '[]'), '$[#]', json(?))"
            " WHERE id = ?",
            (json.dumps(event, default=str), turn_id),
        )

    def discard_turn(self, turn_id: str) -> None:
        """Delete a turn that recorded no items or events (a revert that git refused)."""
        self._db.execute("DELETE FROM turns WHERE id = ?", (turn_id,))

    def turns(self, thread_id: str) -> list[dict[str, Any]]:
        """All turns of a thread in order, as dicts of the `turns` columns."""
        return self._turn_rows("WHERE thread = ? ORDER BY idx", thread_id)

    def last_turn(self, thread_id: str) -> dict[str, Any] | None:
        rows = self._turn_rows("WHERE thread = ? ORDER BY idx DESC LIMIT 1", thread_id)
        return rows[0] if rows else None

    def _turn_rows(self, where: str, arg: str) -> list[dict[str, Any]]:
        rows = self._db.execute(f"SELECT * FROM turns {where}", (arg,)).fetchall()
        return [
            {
                **dict(r),
                **{k: None if r[k] is None else json.loads(r[k]) for k in ("pending", "late")},
            }
            for r in rows
        ]

    # items and events

    def append_item(self, thread_id: str, item: Item, events: Iterable[EventRow] = ()) -> int:
        """Append an item (seq = its position in history) plus `events`, in one transaction."""
        with self._tx() as db:
            seq = self._insert_item(db, thread_id, item)
            self._insert_events(db, events)
        return seq

    @staticmethod
    def _insert_item(db: sqlite3.Connection, thread_id: str, item: Item) -> int:
        data = json.dumps(item_to_json(item))
        (seq,) = db.execute(
            "SELECT COALESCE(MAX(seq) + 1, 0) FROM items WHERE thread = ?", (thread_id,)
        ).fetchone()
        db.execute(
            "INSERT INTO items VALUES (?, ?, ?, ?, ?)",
            (thread_id, seq, item.turn_id, item.id, data),
        )
        return seq

    def items(self, thread_id: str) -> list[Item]:
        rows = self._db.execute(
            "SELECT json FROM items WHERE thread = ? ORDER BY seq", (thread_id,)
        ).fetchall()
        return [item_from_json(json.loads(r[0])) for r in rows]

    def append_events(self, events: Iterable[EventRow]) -> None:
        """Insert events serialized with `event_row`, in one transaction."""
        with self._tx() as db:
            self._insert_events(db, events)

    @staticmethod
    def _insert_events(db: sqlite3.Connection, events: Iterable[EventRow]) -> None:
        db.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)", events)

    def events(self, thread_id: str) -> list[dict[str, Any]]:
        """All event envelopes of a thread in seq order."""
        rows = self._db.execute(
            "SELECT json FROM events WHERE thread = ? ORDER BY seq", (thread_id,)
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def next_seq(self, thread_id: str) -> int:
        """The seq the thread's next event gets (event seqs start at 1)."""
        (seq,) = self._db.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE thread = ?", (thread_id,)
        ).fetchone()
        return seq

    # cancellation, e.g. from a separate `bakeoff cancel` process

    def request_cancel(self, thread_id: str) -> None:
        self._db.execute("INSERT INTO cancels VALUES (?, ?)", (thread_id, now_us()))

    def cancel_requested(self, thread_id: str, since_us: int = 0) -> bool:
        """True if a cancel was requested for the thread at or after `since_us`."""
        row = self._db.execute(
            "SELECT 1 FROM cancels WHERE thread = ? AND ts >= ? LIMIT 1", (thread_id, since_us)
        ).fetchone()
        return row is not None

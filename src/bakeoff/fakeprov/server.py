"""Scripted OpenAI/OpenRouter-compatible streaming server on 127.0.0.1.

`FakeProvider` serves every scenario in `scenarios_dir` on one port. The URL picks the script
and the cursor: `http://127.0.0.1:<port>/s/<scenario>/<run>/<impl>/v1`. Each request body is
recorded verbatim under `<wire_dir>/<scenario>/<run>/<impl>/`. See fakeprov/README.md.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import re
import select
import shutil
import socket
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

from bakeoff.fakeprov.script import (
    CREATED,
    SCENARIOS_DIR,
    Op,
    Reply,
    Scenario,
    ScenarioError,
    Sleep,
    Stall,
    load_scenario,
    rejected,
    reply,
)

_SEGMENT = r"[A-Za-z0-9_-][A-Za-z0-9_.-]*"  # never "." or "..": segments become directories
_ROUTE = re.compile(
    rf"/s/(?P<scenario>{_SEGMENT})/(?P<run>{_SEGMENT})/(?P<impl>{_SEGMENT})/v1"
    r"/(?P<endpoint>chat/completions|models)"
)


class FakeProvider:
    """Serves scenario scripts over HTTP/1.1 with keep-alive and chunked SSE.

    Every (scenario, run, impl) has its own cursor over the scenario's exchanges, so each loop
    sees the identical script. The server outlives its clients: a killed client process only
    closes its own connections.
    """

    def __init__(
        self,
        scenarios_dir: Path | None = None,
        wire_dir: Path = Path("out/wire"),
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.scenarios_dir = scenarios_dir or SCENARIOS_DIR
        self.wire_dir = wire_dir
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._scenarios: dict[str, Scenario] = {}
        self._cursors: dict[tuple[str, str, str], int] = {}
        self._last_us: dict[tuple[str, str, str], int] = {}  # previous request per cursor
        self._claimed: set[tuple[str, str, str]] = set()
        self._conn_ids = itertools.count(1)
        self._conns: set[socket.socket] = set()
        self._stopping = threading.Event()
        self._t0_ns = time.monotonic_ns()
        self._server: _Server | None = None

    def start(self) -> Self:
        """Bind and serve in a background thread. `port=0` picks a free port.

        Every start is a new server run: cursors restart at the first exchange and recording
        folders are emptied again on first use."""
        with self._lock:
            self._cursors.clear()
            self._last_us.clear()
            self._claimed.clear()
        self._stopping.clear()
        self._server = _Server((self.host, self.port), _Handler)
        self._server.provider = self
        self.port = self._server.server_address[1]
        self._t0_ns = time.monotonic_ns()
        # stop() waits up to one poll interval for serve_forever() to notice.
        threading.Thread(
            target=self._server.serve_forever, args=(0.01,), name="fakeprov", daemon=True
        ).start()
        return self

    def stop(self) -> None:
        """Stop serving, end stalled streams and idle keep-alive connections, join handlers."""
        server, self._server = self._server, None
        if server is None:
            return
        self._stopping.set()
        server.shutdown()
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
        server.server_close()

    def base_url(self, scenario: str, run: str, impl: str) -> str:
        """The OpenAI-compatible base URL of one cursor. Claims its recording folder."""
        for segment in (scenario, run, impl):
            if not re.fullmatch(_SEGMENT, segment):
                raise ValueError(f"invalid URL segment: {segment!r}")
        self._claim((scenario, run, impl))
        return f"http://{self.host}:{self.port}/s/{scenario}/{run}/{impl}/v1"

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _connected(self, conn: socket.socket) -> int:
        with self._lock:
            self._conns.add(conn)
            return next(self._conn_ids)

    def _disconnected(self, conn: socket.socket) -> None:
        with self._lock:
            self._conns.discard(conn)

    def _scenario(self, name: str) -> Scenario:
        with self._lock:
            if name not in self._scenarios:
                self._scenarios[name] = load_scenario(self.scenarios_dir / f"{name}.json")
            return self._scenarios[name]

    def _claim(self, key: tuple[str, str, str]) -> Path:
        """The recording folder of one cursor. The first claim in this server run empties it,
        so no file from an older run survives, even when this run sends no request."""
        wire = self.wire_dir.joinpath(*key)
        with self._lock:
            if key not in self._claimed:
                shutil.rmtree(wire, ignore_errors=True)
                wire.mkdir(parents=True)
                self._claimed.add(key)
        return wire

    def _chat(self, key: tuple[str, str, str], raw: bytes, meta: dict[str, Any]) -> Reply:
        """Answer one chat request from its cursor and record it as NNN.json + NNN.meta.json."""
        t_us = (time.monotonic_ns() - self._t0_ns) // 1000
        with self._lock:
            index = self._cursors.get(key, 0)
            self._cursors[key] = index + 1
            prev_us, self._last_us[key] = self._last_us.get(key), t_us
        gap_ms = None if prev_us is None else (t_us - prev_us) / 1000
        try:
            answer = reply(self._scenario(key[0]), index, raw, gap_ms)
        except ScenarioError as exc:
            answer = rejected(500, str(exc))
        meta |= {"t_us": t_us, "status": answer.status}
        if answer.error:
            meta["error"] = answer.error
        try:
            wire = self._claim(key)
            wire.joinpath(f"{index + 1:03d}.json").write_bytes(raw)
            wire.joinpath(f"{index + 1:03d}.meta.json").write_text(json.dumps(meta) + "\n")
        except OSError as exc:  # an unrecorded request must fail loudly, not look answered
            return rejected(500, f"recording failed: {exc}")
        return answer


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # The default backlog (5) resets connects when every scenario and loop starts at once.
    request_queue_size = 128
    provider: FakeProvider

    def server_bind(self) -> None:
        # HTTPServer.server_bind() also does a reverse DNS lookup of the host; skip it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Clients vanish mid-request by design (cancel, SIGKILL); only report real bugs.
        if not isinstance(sys.exc_info()[1], OSError):
            super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True  # small SSE frames must leave immediately
    server: _Server

    def setup(self) -> None:
        super().setup()
        self.conn_id = self.server.provider._connected(self.connection)

    def finish(self) -> None:
        self.server.provider._disconnected(self.connection)
        super().finish()

    def log_message(self, format: str, *args: Any) -> None:
        """Stay silent: nothing may reach stdout/stderr during a scenario."""

    def do_GET(self) -> None:
        route = _ROUTE.fullmatch(self.path)
        if route is None or route["endpoint"] != "models":
            return self._send(rejected(404, f"no route for GET {self.path}"))
        try:
            model = self.server.provider._scenario(route["scenario"]).model["model"]
        except ScenarioError as exc:
            return self._send(rejected(404, str(exc)))
        entry = {"id": model, "object": "model", "created": CREATED, "owned_by": "fakeprov"}
        self._send(Reply(200, {"object": "list", "data": [entry]}))

    def do_POST(self) -> None:
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            self.close_connection = True  # the body's end is unknown, so the connection is lost
            return self._send(rejected(411, "Content-Length is required"))
        raw = self.rfile.read(int(length))
        if len(raw) < int(length):  # the client died mid-send: answer and record nothing
            self.close_connection = True
            return
        route = _ROUTE.fullmatch(self.path)
        if route is None or route["endpoint"] != "chat/completions":
            return self._send(rejected(404, f"no route for POST {self.path}"))
        key = (route["scenario"], route["run"], route["impl"])
        meta = {"conn_id": self.conn_id, "path": self.path}
        self._send(self.server.provider._chat(key, raw, meta))

    def _send(self, answer: Reply) -> None:
        self.send_response(answer.status)
        for name, value in answer.headers.items():
            self.send_header(name, value)
        if answer.body is not None:
            payload = json.dumps(answer.body).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._play(answer.stream)
        except OSError:  # the client hung up mid-stream (cancel)
            self.close_connection = True

    def _play(self, ops: list[Op]) -> None:
        stopping = self.server.provider._stopping
        for op in ops:
            match op:
                case Sleep(seconds=seconds):
                    if stopping.wait(seconds):
                        self.close_connection = True
                        return
                case Stall():
                    self._stall(stopping)
                    self.close_connection = True
                    return
                case bytes():
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(op), op))
        self.wfile.write(b"0\r\n\r\n")

    def _stall(self, stopping: threading.Event) -> None:
        """Send nothing until the client leaves (or sends anything) or the server stops."""
        while not stopping.is_set():
            readable, _, _ = select.select([self.connection], [], [], 0.05)
            if readable:
                return

"""Tiny stdlib SSE server for the pydantic_version tests.

It answers `POST .../chat/completions` with scripted OpenAI/OpenRouter-style streams, one
`Reply` per request in order, and records every request body verbatim. It speaks plain
HTTP/1.1 and closes each connection after the reply, so it works with any HTTP client the
OpenAI SDK happens to use (httpx or httpx2).
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class Reply:
    """One scripted HTTP response. `chunks` are SSE events: a dict is sent as `data: <json>`,
    a str starting with ':' as a comment line, and a float as a pause in seconds."""

    chunks: list[dict[str, Any] | str | float] = field(default_factory=list)
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None  # sent as a plain JSON body instead of a stream
    stall: bool = False  # after the chunks, keep the connection open until the server stops


def chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A chat.completion.chunk carrying one delta."""
    out: dict[str, Any] = {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    if usage is not None:
        out["usage"] = usage
    return out


def text(*pieces: str) -> list[dict[str, Any] | str | float]:
    """Assistant text split into the given pieces."""
    return [chunk({"role": "assistant", "content": p}) for p in pieces]


def tool_call(
    index: int, call_id: str, name: str, *arg_pieces: str
) -> list[dict[str, Any] | str | float]:
    """One streamed tool call whose arguments arrive in `arg_pieces`."""
    head = {
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": ""},
    }
    return [chunk({"tool_calls": [head]})] + [
        chunk({"tool_calls": [{"index": index, "function": {"arguments": p}}]}) for p in arg_pieces
    ]


def done(
    finish: str = "stop",
    *,
    prompt: int = 10,
    completion: int = 5,
    cost: float | None = None,
    cached: int = 0,
    reasoning: int = 0,
) -> dict[str, Any]:
    """The closing chunk: finish reason plus OpenRouter-style usage."""
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }
    if cost is not None:
        usage["cost"] = cost
    return chunk(finish=finish, usage=usage)


class SSEServer:
    """Threaded scripted server. Use as a context manager; `base_url` ends in `/v1`."""

    def __init__(self, *replies: Reply) -> None:
        self.replies: deque[Reply] = deque(replies)
        self.bodies: list[bytes] = []  # raw request bodies, verbatim
        self.headers: list[dict[str, str]] = []
        self._stop = threading.Event()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self._httpd.daemon_threads = True
        # A short poll interval keeps shutdown (and so each test) fast.
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    @property
    def requests(self) -> list[dict[str, Any]]:
        """The recorded request bodies, parsed."""
        return [json.loads(b) for b in self.bodies]

    def add(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def __enter__(self) -> SSEServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._httpd.shutdown()
        self._httpd.server_close()


def _handler(server: SSEServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # keep test output quiet
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            server.bodies.append(body)
            server.headers.append({k.lower(): v for k, v in self.headers.items()})
            if not server.replies:
                self._send(
                    500, {}, json.dumps({"error": {"message": "no scripted reply"}}).encode()
                )
                return
            reply = server.replies.popleft()
            if reply.body is not None or reply.status != 200:
                self._send(reply.status, reply.headers, json.dumps(reply.body or {}).encode())
                return
            self.send_response(200)
            for key, value in {"Content-Type": "text/event-stream", **reply.headers}.items():
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for item in reply.chunks:
                    if isinstance(item, float):
                        time.sleep(item)
                        continue
                    line = item if isinstance(item, str) else f"data: {json.dumps(item)}"
                    self.wfile.write(f"{line}\n\n".encode())
                    self.wfile.flush()
                if reply.stall:
                    server._stop.wait()
                    return
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client hung up (e.g. a cancelled turn)
            self.close_connection = True

        def _send(self, status: int, headers: dict[str, str], body: bytes) -> None:
            self.send_response(status)
            for key, value in {"Content-Type": "application/json", **headers}.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

    return Handler

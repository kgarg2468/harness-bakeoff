"""Live runs against a real model endpoint: `bakeoff live` and `bakeoff chat`.

Same machinery as the scenarios (Runner, SessionLog, WorkCopy, ToolHost on MockEngine); only the
model endpoint is real. The API key comes from the environment or an env file, lives only in
memory, and is never printed or stored (the runner keeps the model config without it). It is
sent only over https to api.openai.com, or to a host the command line names with `--key-host`.
The network guard allows loopback plus the endpoint's own host, nothing else.

A live run writes `out/live/<run_id>/<impl>/` with log.sqlite, events.ndjson, wc/ and result.json.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import ipaddress
import json
import os
import signal
import socket
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO
from urllib.parse import urlsplit

from bakeoff import loops
from bakeoff.fakeprov.script import SCENARIOS_DIR, load_scenario
from bakeoff.shared import netguard
from bakeoff.shared.contract import Limits, Loop, ModelConfig, Resume
from bakeoff.shared.scenario import (
    Captured,
    Workspace,
    capture_output,
    check_invariants,
    decide,
    observe,
    pending_calls,
    point_latest,
    usage_totals,
    write_json,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_HOST = "api.openai.com"  # the one host that gets the key without --key-host
KEY_NAME = "OPENAI_API_KEY"
ENV_FILE_VAR = "BAKEOFF_ENV_FILE"
LIVE_RULES = {"*": "allow"}
# Interactive runs ask before anything is written, like the approval scenarios.
ASK_RULES = {"*": "allow", "write_file": "ask", "edit_file": "ask"}
_OUTPUT = ("text.delta", "reasoning.delta", "tool_call.ready")  # the model's first "token"

Ask = Callable[[str], Awaitable[str]]


class LiveError(RuntimeError):
    """A live run cannot start (e.g. no API key)."""


# --- key and network -------------------------------------------------------------------------------


def is_loopback(url: str) -> bool:
    """Whether `url` points at this machine (the fake provider)."""
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def read_env_key(path: Path, name: str = KEY_NAME) -> str | None:
    """The value of `name` in a dotenv file (`KEY=value`, optional `export`, quotes and a
    trailing ` # comment`), or None. Only that one key is read; nothing is put into the
    environment."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ").lstrip()
        key, sep, value = line.partition("=")
        if line.startswith("#") or not sep or key.strip() != name:
            continue
        value = value.strip()
        quote = value[:1]
        if quote in ("'", '"') and (end := value.find(quote, 1)) > 0:
            value = value[1:end]  # a quoted value ends at its closing quote; a comment may follow
        else:
            value = value.split(" #", 1)[0].strip()
        return value or None
    return None


def resolve_api_key(
    base_url: str, env_file: Path | None = None, *, key_hosts: Sequence[str] = ()
) -> str:
    """The key for `base_url`: a dummy for the local fake provider; otherwise `--env-file`, then
    OPENAI_API_KEY in the environment, then the file named by BAKEOFF_ENV_FILE.

    A real key goes only over https, and only to api.openai.com or a host in `key_hosts` (from
    the command line), so neither a typo in `--base-url` nor a session log that names another
    endpoint can send it elsewhere. Raises LiveError otherwise."""
    if is_loopback(base_url):
        return "dummy"
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    if host != OPENAI_HOST and host not in {h.lower() for h in key_hosts}:
        raise LiveError(
            f"refusing to send {KEY_NAME} to {host or base_url!r}: it goes only to {OPENAI_HOST}"
            f" unless the command says --key-host {host or '<host>'}"
        )
    if parts.scheme != "https":
        raise LiveError(f"refusing to send {KEY_NAME} without https: {base_url}")
    key = read_env_key(env_file) if env_file is not None else None
    key = key or os.environ.get(KEY_NAME)
    if not key and (path := os.environ.get(ENV_FILE_VAR)):
        key = read_env_key(Path(path))
    if not key:
        raise LiveError(
            f"no {KEY_NAME}: set it in the environment, or pass --env-file / {ENV_FILE_VAR}"
        )
    return key


def guard_network(base_url: str) -> None:
    """Install the loopback-only guard (I6), plus the host of `base_url` if it is remote.

    The guard compares IP addresses, and HTTP clients connect to whatever the resolver returns,
    so every address the resolver gives for that host (now and later) is allowed; other hosts
    stay blocked.
    """
    netguard.install()
    host = urlsplit(base_url).hostname
    if host is None or is_loopback(base_url):
        return
    socket.getaddrinfo = _allowing(socket.getaddrinfo, host)
    with contextlib.suppress(OSError):  # if it does not resolve, the request says so later
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)


def _allowing(resolve: Callable[..., Any], host: str) -> Callable[..., Any]:
    """Wrap `socket.getaddrinfo` so that the addresses it returns for `host` are allowed."""

    def getaddrinfo(name: Any, *args: Any, **kwargs: Any) -> Any:
        infos = resolve(name, *args, **kwargs)
        if (name.decode() if isinstance(name, bytes) else name) == host:
            netguard.install(tuple(str(info[4][0]) for info in infos))
        return infos

    return getaddrinfo


# --- terminal ----------------------------------------------------------------------------------------


def open_terminal() -> TextIO:
    """A stream to the current stdout on its own file descriptor, so output capture (I5, which
    redirects fd 1 during turns) does not swallow what we print for the user."""
    try:
        return os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):  # no real descriptor (e.g. under pytest capsys)
        return sys.stdout


def _short(value: Any, limit: int = 100) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _args(arguments: str) -> str:
    """Tool arguments for one line: long strings become their length."""
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return _short(arguments, 60)
    if not isinstance(args, dict):
        return _short(arguments, 60)
    parts = []
    for key, value in args.items():
        shown = json.dumps(value) if not isinstance(value, str) else repr(value)
        parts.append(f"{key}=<{len(shown)} chars>" if len(shown) > 40 else f"{key}={shown}")
    return ", ".join(parts)


class Printer:
    """A sink that streams a thread's events to a terminal: text inline, tools summarized."""

    def __init__(self, out: TextIO) -> None:
        self.out = out
        self.names: dict[str, str] = {}  # call id -> tool name
        self.ends: dict[str, dict[str, Any]] = {}  # call id -> tool.end data
        self.inline: str | None = None  # the delta kind being printed inline

    def _line(self, text: str) -> None:
        if self.inline is not None:
            self.out.write("\n")
            self.inline = None
        self.out.write(text + "\n")
        self.out.flush()

    def _inline(self, kind: str, text: str) -> None:
        if self.inline != kind:
            if self.inline is not None:
                self.out.write("\n")
            if kind == "reasoning.delta":
                self.out.write("[thinking] ")
            self.inline = kind
        self.out.write(text)
        self.out.flush()

    def __call__(self, envelope: dict[str, Any]) -> None:
        kind, data = envelope["type"], envelope["data"]
        if kind in ("text.delta", "reasoning.delta"):
            self._inline(kind, data["text"])
        elif kind == "tool_call.ready":
            self.names[data["call_id"]] = data["name"]
            self._line(f"-> {data['name']}({_args(data['arguments'])})")
        elif kind == "tool.end":
            self.ends[data["call_id"]] = data
        elif kind == "item" and data["item"]["message"].get("role") == "tool":
            message = data["item"]["message"]
            call_id = message.get("tool_call_id")
            end = self.ends.get(call_id, {})
            took = f" {end['ms']:.0f} ms" if "ms" in end else ""
            mark = "ok" if end.get("ok") else "not ok"
            name = self.names.get(call_id, "tool")
            self._line(f"   <- {name} {mark}{took}: {_short(message.get('content'), 140)}")
        elif kind == "permission.asked":
            self._line(f"?? {data['name']}({_args(data['arguments'])}) needs approval")
        elif kind == "retry":
            self._line(
                f"[retry {data.get('attempt')} in {data.get('wait_ms')} ms: {_short(data.get('reason'))}]"
            )
        elif kind == "error":
            self._line(f"[error {data.get('kind')}: {_short(data.get('message'), 300)}]")
        elif kind == "turn.end":
            self._line(f"[turn end: {data.get('stop')}, {data.get('steps')} steps]")
        elif kind == "commit":
            self._line(f"[commit {data['sha'][:10]}: {', '.join(data['files']) or 'no files'}]")


def stdin_ask(out: TextIO, stdin: TextIO | None = None) -> Ask:
    """Prompt on `out` and read a line from stdin without blocking the event loop.

    End of input and Ctrl-C at the prompt raise EOFError: the caller ends the chat (or the
    approval) cleanly. The read is non-blocking (`add_reader`), so no thread is left stuck in
    `readline()` that `asyncio.run` would wait for at exit."""
    source = stdin or sys.stdin
    pending = bytearray()  # bytes read past the last line, for the next prompt

    async def ask(prompt: str) -> str:
        out.write(prompt)
        out.flush()
        aio = asyncio.get_running_loop()
        stop = aio.create_future()
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            aio.add_signal_handler(signal.SIGINT, _wake, stop)
        try:
            return await _read_line(source.fileno(), pending, stop)
        finally:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                aio.remove_signal_handler(signal.SIGINT)

    return ask


def _wake(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


async def _read_line(fd: int, pending: bytearray, stop: asyncio.Future[None]) -> str:
    """The next line from `fd` (without its newline). Raises EOFError at end of input or when
    `stop` is set first."""
    aio = asyncio.get_running_loop()
    while b"\n" not in pending:
        readable = aio.create_future()
        try:
            aio.add_reader(fd, _wake, readable)
        except (NotImplementedError, OSError, ValueError):
            # Regular files never block; elsewhere (Windows) a thread has to wait for input.
            chunk = await asyncio.to_thread(os.read, fd, 4096)
        else:
            try:
                await asyncio.wait({readable, stop}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                aio.remove_reader(fd)
            if stop.done():
                raise EOFError("interrupted")
            chunk = os.read(fd, 4096)
        if not chunk:
            if not pending:
                raise EOFError("end of input")
            pending += b"\n"  # a last line without its newline
            break
        pending += chunk
    line, _, rest = bytes(pending).partition(b"\n")
    pending[:] = rest
    return line.decode(errors="replace").rstrip("\r")


# --- one live thread ---------------------------------------------------------------------------------


def system_prompt() -> str:
    """The scenarios' system prompt (Rocket Agent builds RocketRide pipelines), plus the skills
    index when `bakeoff.shared.skills` is available."""
    base = load_scenario(SCENARIOS_DIR / "S01.json").system
    if importlib.util.find_spec("bakeoff.shared.skills") is None:
        return base
    from bakeoff.shared import skills  # optional: lands with the skills package

    extra = getattr(skills, "skills_prompt", None)
    return f"{base}\n\n{extra()}" if extra is not None else base


@dataclass(slots=True)
class LiveModel:
    """What `bakeoff live` / `chat` need to build each impl's ModelConfig."""

    model: str
    base_url: str = OPENAI_BASE_URL  # may contain "{impl}" (one fakeprov cursor per loop)
    api_key: str = "dummy"
    kind: Literal["openrouter", "openai_compat"] = "openai_compat"
    reasoning: str | None = None  # effort; "none" is sent as-is (reasoning_effort=none)
    max_tokens: int = 4096

    def config(self, impl: str) -> ModelConfig:
        return ModelConfig(
            base_url=self.base_url.replace("{impl}", impl),
            model=self.model,
            api_key=self.api_key,
            kind=self.kind,
            max_tokens=self.max_tokens,
            temperature=None,  # reasoning models reject a temperature; use the model default
            reasoning=None if self.reasoning is None else {"effort": self.reasoning},
            timeout_s=180.0,
        )


class LiveThread:
    """One loop on one thread in `directory`, with streamed output and approval prompts."""

    def __init__(
        self,
        impl: str,
        directory: Path,
        model: LiveModel,
        term: TextIO,
        *,
        rules: dict[str, Any],
        max_steps: int,
        loop: Loop | None = None,
    ) -> None:
        self.impl, self.dir, self.term = impl, directory, term
        self.model = model.config(impl)
        self.limits = Limits(max_steps=max_steps)
        self.loop = loop or loops.load(impl)()
        self.ws = Workspace(directory / "log.sqlite", sinks=[Printer(term)])
        try:
            self.thread_id = self.ws.runner.new_thread(
                impl=self.loop.name,
                system=system_prompt(),
                rules=rules,
                model=self.model,
                thread_id=f"live-{impl}",
            )
        except BaseException:
            self.ws.close()
            raise
        self.captured: list[Captured] = []

    async def turn(
        self, *, user_text: str | None = None, resume: Resume | None = None
    ) -> dict[str, Any]:
        """Run a turn with I5 capture; Ctrl-C cancels it (stop="cancelled")."""
        cancel = asyncio.Event()
        aio = asyncio.get_running_loop()
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            aio.add_signal_handler(signal.SIGINT, cancel.set)
        captured = Captured()
        try:
            with capture_output() as captured:
                return await self.ws.runner.turn(
                    self.loop,
                    self.thread_id,
                    model=self.model,
                    user_text=user_text,
                    resume=resume,
                    limits=self.limits,
                    cancel=cancel,
                )
        finally:
            self.captured.append(captured)
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                aio.remove_signal_handler(signal.SIGINT)

    async def converse(self, user_text: str, ask: Ask | None) -> dict[str, Any]:
        """A user turn, then approval resumes until it no longer pauses. Without `ask`, a
        paused call is denied (nobody is there to approve it)."""
        summary = await self.turn(user_text=user_text)
        while summary["stop"] == "paused":
            decisions, reason = await self._answers(ask)
            summary = await self.turn(
                resume=Resume(kind="approval", decisions=decisions, reason=reason)
            )
        return summary

    async def _answers(
        self, ask: Ask | None
    ) -> tuple[dict[str, Literal["allow", "deny"]], str | None]:
        pending = pending_calls(self.ws.log, self.thread_id)
        calls = {
            c["id"]: c["function"]
            for item in self.ws.log.items(self.thread_id)
            for c in item.message.get("tool_calls") or ()
        }
        allow, reasons = [], []
        for call_id in pending:
            fn = calls.get(call_id, {})
            answer = (
                ""
                if ask is None
                else await ask(
                    f"allow {fn.get('name')}({_args(fn.get('arguments', ''))})? [y/N or a reason] "
                )
            )
            if answer.strip().lower() in ("y", "yes"):
                allow.append(call_id)
            elif answer.strip().lower() not in ("", "n", "no"):
                reasons.append(answer.strip())
        reason = "; ".join(reasons) or (
            "no approver in a non-interactive run" if ask is None else None
        )
        return decide(pending, allow, [c for c in pending if c not in allow]), reason

    def result(
        self, run_id: str, prompt: str, duration_ms: float, error: str | None
    ) -> dict[str, Any]:
        """The live result.json: the scenario result's fields that apply (no expect, no I1),
        plus model, prompt, final text, latency and steps."""
        workdir = self.ws.runner.workdir(self.thread_id)
        obs = observe(self.ws.log, self.thread_id, self.dir / "wire", workdir)
        out = "".join(c.stdout for c in self.captured)
        err = "".join(c.stderr for c in self.captured)
        checks = check_invariants(self.ws.log, self.thread_id, obs, out, err, wire=False)
        events = self.ws.log.events(self.thread_id)
        return empty_result(self.impl, self.model, run_id, prompt, duration_ms, error) | {
            "final_text": obs.last_text,
            "stops": obs.stops,
            "invariants": {
                n: {"ok": c.ok, "detail": c.detail, "info": c.info} for n, c in checks.items()
            },
            "requests": sum(e["type"] == "request.start" for e in events),
            "steps": sum(
                e["type"] == "request.start" and e["data"].get("attempt") == 1 for e in events
            ),
            "tool_runs": obs.tool_runs,
            "usage": usage_totals(obs.usage),
            "latency": latency(events),
            "files": sorted(p.name for p in workdir.iterdir() if p.name != ".git")
            if workdir.exists()
            else [],
            # A live run passes when it answered: no failure, invariants hold, last stop end_turn.
            "passed": error is None
            and obs.stops[-1:] == ["end_turn"]
            and all(c.ok for c in checks.values()),
            "thread": self.thread_id,
        }

    async def close_loop(self) -> None:
        """Close the loop (its output counts for I5 too); the log stays open for `result`."""
        captured = Captured()
        try:
            with capture_output() as captured:
                await self.loop.aclose()
        finally:
            self.captured.append(captured)

    async def finish(
        self, run_id: str, prompt: str, started: float, error: str | None
    ) -> dict[str, Any]:
        """Close the loop, write result.json and close the log; also after a failure or an
        interrupt, so every live run leaves its result. Returns the result."""
        try:
            try:
                await self.close_loop()
            finally:
                result = self.result(run_id, prompt, _ms_since(started), error)
                write_json(self.dir / "result.json", result)
        finally:
            self.ws.close()
        return result


def empty_result(
    impl: str, model: ModelConfig, run_id: str, prompt: str, duration_ms: float, error: str | None
) -> dict[str, Any]:
    """A live result.json with nothing observed: the base of every result, and all that a run
    whose thread could not be set up leaves."""
    return {
        "v": 1,
        "run_id": run_id,
        "impl": impl,
        "model": model.model,
        "base_url": model.base_url,
        "prompt": prompt,
        "final_text": "",
        "stops": [],
        "invariants": {},
        "requests": 0,
        "steps": 0,
        "tool_runs": {},
        "usage": usage_totals([]),
        "latency": latency([]),
        "files": [],
        "duration_ms": duration_ms,
        "passed": False,
        "error": error,
        "thread": None,
    }


async def _finish(
    thread: LiveThread | None,
    impl: str,
    directory: Path,
    model: LiveModel,
    run_id: str,
    prompt: str,
    started: float,
    error: str | None,
) -> dict[str, Any]:
    """`thread.finish(...)`; or, if the thread could not be set up (the loop, the log or the
    working copy failed), a result.json with the error: the run directory may exist already,
    and it blocks its run id, so it must say why it has nothing else."""
    if thread is not None:
        return await thread.finish(run_id, prompt, started, error)
    result = empty_result(impl, model.config(impl), run_id, prompt, _ms_since(started), error)
    write_json(directory / "result.json", result)
    return result


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def latency(events: list[dict[str, Any]]) -> dict[str, float | None]:
    """Time to first token (first model output of the first turn, from its start) and the total
    time of the loop turns (each from its start to its last event, approvals excluded)."""
    turns: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        turns.setdefault(event["turn"], []).append(event)
    ttft = None
    if turns:
        first = next(iter(turns.values()))
        ttft = next((e["t_us"] / 1000 for e in first if e["type"] in _OUTPUT), None)
    total = sum(stream[-1]["t_us"] / 1000 for stream in turns.values())
    return {"ttft_ms": ttft, "total_ms": round(total, 1)}


async def run_live(
    impls: list[str],
    model: LiveModel,
    prompt: str,
    *,
    out: Path,
    run_id: str,
    term: TextIO,
    max_steps: int = 8,
    ask: Ask | None = None,
) -> list[dict[str, Any]]:
    """Run `prompt` once per impl, one after the other, streaming to `term`. With `ask`,
    writes need approval; without it, the rules allow everything. Returns the results.
    Raises LiveError before anything runs if a run directory already exists. A run that fails,
    even before its thread exists, still writes its result.json, and the next impl runs."""
    impls = list(dict.fromkeys(impls))  # each loop once: its thread id is `live-<impl>`
    directories = {impl: fresh_dir(out / "live" / run_id / impl) for impl in impls}
    results = []
    for impl in impls:
        directory = directories[impl]
        term.write(f"\n== {impl} ({loops.REGISTRY[impl].target}) model {model.model}\n")
        term.write(f"   {directory}\n\n> {prompt}\n")
        term.flush()
        rules = ASK_RULES if ask is not None else LIVE_RULES
        thread: LiveThread | None = None
        started, error = time.perf_counter(), None
        try:
            thread = LiveThread(impl, directory, model, term, rules=rules, max_steps=max_steps)
            await thread.converse(prompt, ask)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            term.write(f"\n[{impl} failed: {error}]\n")
        except BaseException as exc:  # Ctrl-C: record the run so far, then stop
            error = f"interrupted ({type(exc).__name__})"
            raise
        finally:
            results.append(
                await _finish(thread, impl, directory, model, run_id, prompt, started, error)
            )
            point_latest(out / "live", run_id)
    return results


def fresh_dir(directory: Path) -> Path:
    """`directory`, which must not exist yet: a live run never overwrites an earlier one
    (it cost real tokens) and never reuses its thread."""
    if directory.exists():
        raise LiveError(f"{directory} already exists: pick another --run-id")
    return directory


def format_side_by_side(results: list[dict[str, Any]]) -> str:
    """Tokens, latency and steps per impl, in columns."""

    def ms(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"

    rows: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
        ("stop", lambda r: ", ".join(r["stops"]) or "-"),
        ("steps", lambda r: r["steps"]),
        ("requests", lambda r: r["requests"]),
        ("input tokens", lambda r: r["usage"]["input_tokens"]),
        ("cached tokens", lambda r: r["usage"]["cached_tokens"]),
        ("output tokens", lambda r: r["usage"]["output_tokens"]),
        ("cost", lambda r: f"${r['usage']['cost_usd']:.6f} ({r['usage']['cost_source']})"),
        ("first token", lambda r: ms(r["latency"]["ttft_ms"])),
        ("total", lambda r: ms(r["latency"]["total_ms"])),
        ("tool runs", lambda r: sum(r["tool_runs"].values())),
        ("files", lambda r: ", ".join(r["files"]) or "-"),
        (
            "invariants",
            lambda r: (
                "-"  # a run that could not start checked none
                if not r["invariants"]
                else "ok"
                if all(c["ok"] for c in r["invariants"].values())
                else ", ".join(n for n, c in r["invariants"].items() if not c["ok"]) + " failed"
            ),
        ),
    ]
    width = max(24, *(len(str(get(r))) + 2 for r in results for _, get in rows))
    lines = ["".ljust(16) + "".join(r["impl"].ljust(width) for r in results)]
    for label, get in rows:
        lines.append(label.ljust(16) + "".join(str(get(r)).ljust(width) for r in results).rstrip())
    return "\n".join(lines)


async def chat(
    impl: str,
    model: LiveModel,
    *,
    out: Path,
    run_id: str,
    term: TextIO,
    ask: Ask,
    max_steps: int = 12,
) -> dict[str, Any]:
    """An interactive REPL on one thread: each line is a user turn; writes ask for approval.
    `/revert N` undoes turn N, `/compact TEXT` compacts, `/exit`, end of input or Ctrl-C at a
    prompt quits (Ctrl-C during a turn cancels the turn)."""
    directory = fresh_dir(out / "live" / run_id / impl)
    thread: LiveThread | None = None
    started, error = time.perf_counter(), None
    try:
        thread = LiveThread(impl, directory, model, term, rules=ASK_RULES, max_steps=max_steps)
        term.write(f"chat with {impl} on {model.model}; thread {thread.thread_id} in {directory}\n")
        term.write("/revert N, /compact TEXT, /exit\n")
        while True:
            try:
                line = (await ask("\n> ")).strip()
                if line in ("/exit", "/quit"):
                    break
                if line:
                    await _chat_line(thread, line, ask)
            except EOFError:  # end of input, or Ctrl-C at a prompt
                break
            except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                term.write(f"[{exc}]\n")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    except BaseException as exc:
        error = f"interrupted ({type(exc).__name__})"
        raise
    finally:
        result = await _finish(thread, impl, directory, model, run_id, "(chat)", started, error)
        point_latest(out / "live", run_id)
    return result


async def _chat_line(thread: LiveThread, line: str, ask: Ask) -> None:
    command, _, rest = line.partition(" ")
    if command == "/revert":
        turn = thread.ws.log.turns(thread.thread_id)[int(rest) - 1]
        await thread.ws.runner.revert(thread.thread_id, turn["id"])
    elif command == "/compact":
        thread.ws.runner.compact(thread.thread_id, rest)
        thread.term.write("[compacted]\n")
    else:
        await thread.converse(line, ask)

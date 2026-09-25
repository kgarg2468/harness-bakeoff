"""`bakeoff live` / `chat` against a local fake OpenAI-compatible endpoint, the API key and
network helpers, and CLI argument parsing."""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bakeoff import live, loops
from bakeoff.cli import build_parser, main
from bakeoff.fakeprov.server import FakeProvider
from bakeoff.shared.contract import ModelConfig
from bakeoff.shared.scenario import Workspace
from bakeoff.shared.sessionlog import SessionLog
from bakeoff.shared.workcopy import WorkCopy

PIPELINE = {
    "source": "chat_1",
    "components": [
        {"id": "chat_1", "provider": "chat", "config": {}},
        {
            "id": "llm_1",
            "provider": "llm_openai_api",
            "config": {
                "profile": "custom",
                "custom": {"model": "m", "base_url": "u", "apikey": "k"},
            },
            "input": [{"lane": "questions", "from": "chat_1"}],
        },
        {
            "id": "response_1",
            "provider": "response_answers",
            "config": {},
            "input": [{"lane": "answers", "from": "llm_1"}],
        },
    ],
}
USAGE = {"usage": {"prompt_tokens": 1200, "completion_tokens": 30, "cached_tokens": 1024}}


def calls(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"respond": {"stream": [{"tool_calls": list(calls)}, {"finish": "tool_calls"}, USAGE]}}


def says(text: str) -> dict[str, Any]:
    return {"respond": {"stream": [{"text": text, "chunks": 3}, {"finish": "stop"}, USAGE]}}


def write_scenario(folder: Path, sid: str, exchanges: list[dict[str, Any]]) -> None:
    folder.mkdir(exist_ok=True)
    data = {
        "id": sid,
        "title": "a live-mode endpoint",
        "system": "unused: live runs send their own system prompt",
        "model": {"kind": "openai_compat", "model": "fake-live"},
        "rules": {"*": "allow"},
        "limits": {"max_steps": 8},
        "engine": {"delay_ms": 0},
        "driver": [{"user": "x"}],
        "exchanges": exchanges,
        "expect": {"stops": ["end_turn"]},
    }
    (folder / f"{sid}.json").write_text(json.dumps(data))


@pytest.fixture
def endpoint(tmp_path: Path):
    """A fake OpenAI-compatible endpoint: build, validate, answer. `{impl}` gives each loop its
    own cursor."""
    write_scenario(
        tmp_path / "scenarios",
        "L1",
        [
            {
                "expect": {"model": "fake-live", "body_has": ["stream_options"], "messages_len": 2},
                "respond": {
                    "stream": [
                        {"text": "Writing chat.pipe."},
                        {
                            "tool_calls": [
                                {
                                    "id": "call_L1_1",
                                    "name": "write_file",
                                    "arguments": {
                                        "path": "chat.pipe",
                                        "content": json.dumps(PIPELINE),
                                    },
                                }
                            ]
                        },
                        {"finish": "tool_calls"},
                        USAGE,
                    ]
                },
            },
            calls(
                {"id": "call_L1_2", "name": "validate_pipeline", "arguments": {"path": "chat.pipe"}}
            ),
            {
                "expect": {"tool_result_contains": {"call_L1_2": '"ok":true'}},
                **says("chat.pipe is saved and validates."),
            },
        ],
    )
    with FakeProvider(tmp_path / "scenarios", tmp_path / "wire") as provider:
        yield f"http://127.0.0.1:{provider.port}/s/L1/r1/{{impl}}/v1"


def test_live_run_streams_and_writes_the_layout(
    endpoint: str, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    impls = list(loops.available())
    out = tmp_path / "out"
    argv = ["live", "--impl", ",".join(impls), "--model", "fake-live", "--base-url", endpoint]
    code = main(
        [*argv, "--prompt", "Build chat.pipe and validate it.", "--out", str(out), "--run-id", "t1"]
    )
    printed, errors = capfd.readouterr()
    assert code == 0, printed + errors
    assert errors == ""
    assert "Writing chat.pipe." in printed
    assert "-> write_file(path='chat.pipe', content=<" in printed
    assert "<- validate_pipeline ok" in printed
    assert "chat.pipe is saved and validates." in printed
    assert "first token" in printed and "output tokens" in printed
    for impl in impls:
        directory = out / "live" / "t1" / impl
        result = json.loads((directory / "result.json").read_text())
        assert result["passed"], result
        assert (result["impl"], result["model"], result["run_id"]) == (impl, "fake-live", "t1")
        assert result["prompt"] == "Build chat.pipe and validate it."
        assert result["final_text"] == "chat.pipe is saved and validates."
        assert set(result["invariants"]) == {"I2", "I3", "I5", "I7"}  # no wire: no I1
        assert (result["steps"], result["requests"], result["files"]) == (3, 3, ["chat.pipe"])
        assert result["usage"]["input_tokens"] == 3600
        assert result["latency"]["ttft_ms"] > 0
        pipe = json.loads((directory / "wc" / f"live-{impl}" / "chat.pipe").read_text())
        assert pipe == PIPELINE
        assert (directory / "log.sqlite").is_file() and (directory / "events.ndjson").is_file()
    assert (out / "live" / "latest").resolve() == (out / "live" / "t1").resolve()


@pytest.mark.parametrize("impl", ["our", "pydantic"])
def test_live_sends_the_system_prompt_and_reasoning(
    endpoint: str, tmp_path: Path, impl: str
) -> None:
    """Both loops send reasoning_effort "none" (pydantic-ai's unified `thinking` setting has no
    such level, so A passes it through `openai_reasoning_effort`), and an unattended run tells
    the model that approval gates are pre-approved."""
    argv = ["live", "--impl", impl, "--model", "fake-live", "--base-url", endpoint]
    argv += [
        "--reasoning",
        "none",
        "--prompt",
        "p",
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "t1",
    ]
    code = main(argv)
    assert code == 0
    body = json.loads((tmp_path / "wire" / "L1" / "r1" / impl / "001.json").read_text())
    assert body["reasoning_effort"] == "none"
    assert "temperature" not in body  # reasoning models reject it; the model default applies
    system = live.system_prompt(attended=False)
    assert body["messages"][0] == {"role": "system", "content": system}
    assert system.startswith("You are Rocket Agent.") and live.UNATTENDED in system
    assert live.UNATTENDED not in live.system_prompt(attended=True)


async def test_chat_asks_before_writing(tmp_path: Path) -> None:
    write_scenario(
        tmp_path / "scenarios",
        "C1",
        [
            calls(
                {
                    "id": "call_C1_1",
                    "name": "write_file",
                    "arguments": {"path": "notes.md", "content": "# Notes"},
                }
            ),
            {"expect": {"tool_result_contains": {"call_C1_1": "Wrote"}}, **says("Saved notes.md.")},
        ],
    )
    answers = iter(["Create notes.md", "y", "/exit"])
    prompts: list[str] = []

    async def ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    term = io.StringIO()
    with FakeProvider(tmp_path / "scenarios", tmp_path / "wire") as provider:
        model = live.LiveModel(model="fake-live", base_url=provider.base_url("C1", "r1", "our"))
        result = await live.chat(
            "our", model, out=tmp_path / "out", run_id="c1", term=term, ask=ask
        )
    assert prompts[1].startswith("allow write_file(path='notes.md', content='# Notes')?")
    assert result["stops"] == ["paused", "end_turn"] and result["final_text"] == "Saved notes.md."
    assert result["passed"] and result["files"] == ["notes.md"]
    assert "?? write_file(path='notes.md', content='# Notes') needs approval" in term.getvalue()


def fail_git_init(monkeypatch: pytest.MonkeyPatch, thread: str) -> None:
    """Make creating `thread`'s working copy fail, as a broken git would."""
    init_sync = WorkCopy.init_sync

    def failing(self: WorkCopy) -> None:
        if self.root.name == thread:
            raise RuntimeError(f"git init failed in {self.root}: simulated")
        init_sync(self)

    monkeypatch.setattr(WorkCopy, "init_sync", failing)


def test_a_live_run_that_cannot_start_records_why(
    endpoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fail_git_init(monkeypatch, "live-our")
    out = tmp_path / "out"
    argv = ["live", "--impl", "our,pydantic", "--model", "fake-live", "--base-url", endpoint]
    argv += ["--prompt", "Build chat.pipe and validate it.", "--out", str(out), "--run-id", "t1"]
    assert main(argv) == 1
    printed = capsys.readouterr().out
    directory = out / "live" / "t1" / "our"
    result = json.loads((directory / "result.json").read_text())
    error = f"RuntimeError: git init failed in {directory / 'wc' / 'live-our'}: simulated"
    assert (result["passed"], result["error"], result["stops"]) == (False, error, [])
    assert (result["impl"], result["prompt"], result["thread"]) == ("our", argv[-5], None)
    assert f"[our failed: {error}]" in printed
    # The other loop still ran, and both show up side by side.
    assert json.loads((out / "live" / "t1" / "pydantic" / "result.json").read_text())["passed"]
    assert "invariants      -" in printed


async def test_a_chat_that_cannot_start_records_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_git_init(monkeypatch, "live-our")

    async def ask(prompt: str) -> str:
        raise AssertionError("no prompt before the chat has started")

    model = live.LiveModel(model="m", base_url="http://127.0.0.1:9/v1")
    with pytest.raises(RuntimeError, match="simulated"):
        await live.chat("our", model, out=tmp_path, run_id="c1", term=io.StringIO(), ask=ask)
    result = json.loads((tmp_path / "live" / "c1" / "our" / "result.json").read_text())
    assert (result["passed"], result["prompt"]) == (False, "(chat)")
    assert result["error"].startswith("RuntimeError: git init failed in ")


# --- key and network ---------------------------------------------------------------------------


def test_read_env_key(tmp_path: Path) -> None:
    env = tmp_path / ".env.test"
    env.write_text(
        "# comment\nOTHER=1\nexport OPENAI_API_KEY = 'test-key-123'  \nOPENAI_API_KEY=later\n"
    )
    assert live.read_env_key(env) == "test-key-123"
    env.write_text("OPENAI_API_KEY=plain-value # trailing comment\n")
    assert live.read_env_key(env) == "plain-value"
    env.write_text('OPENAI_API_KEY="quoted-value" # personal key\n')
    assert live.read_env_key(env) == "quoted-value"
    env.write_text("OPENAI_API_KEY='has # inside' # comment\n")
    assert live.read_env_key(env) == "has # inside"
    env.write_text("OPENAI_API_KEY=\n")
    assert live.read_env_key(env) is None


def test_resolve_api_key_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = "https://api.openai.com/v1"
    assert live.resolve_api_key("http://127.0.0.1:9/v1") == "dummy"
    with pytest.raises(live.LiveError, match="no OPENAI_API_KEY"):
        live.resolve_api_key(remote)
    named = tmp_path / "named.env"
    named.write_text("OPENAI_API_KEY=from-bakeoff-env-file\n")
    monkeypatch.setenv("BAKEOFF_ENV_FILE", str(named))
    assert live.resolve_api_key(remote) == "from-bakeoff-env-file"
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    assert live.resolve_api_key(remote) == "from-environment"
    explicit = tmp_path / "explicit.env"
    explicit.write_text("OPENAI_API_KEY=from-env-file-option\n")
    assert live.resolve_api_key(remote, explicit) == "from-env-file-option"


def test_the_key_goes_only_to_openai_or_a_named_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    assert live.resolve_api_key("https://api.openai.com/v1") == "from-environment"
    with pytest.raises(live.LiveError, match=r"refusing to send OPENAI_API_KEY to 'evil\.test'"):
        live.resolve_api_key("https://evil.test/v1")
    assert live.resolve_api_key("https://Evil.test/v1", key_hosts=["evil.TEST"]) == (
        "from-environment"
    )
    with pytest.raises(live.LiveError, match="without https"):
        live.resolve_api_key("http://api.openai.com/v1")


def test_a_worker_never_sends_the_key_where_only_the_log_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "log.sqlite"
    ws = Workspace(db)
    model = ModelConfig(base_url="https://collector.example.test/v1", model="m")
    ws.runner.new_thread(impl="our", system="s", rules={}, model=model, thread_id="t1")
    ws.close()
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value-42")
    assert main(["turn", "t1", "--db", str(db), "--user", "hi"]) == 1
    err = capsys.readouterr().err
    assert "refusing to send OPENAI_API_KEY to 'collector.example.test'" in err
    assert "test-secret-value-42" not in err
    log = SessionLog(db)
    assert log.turns("t1") == []  # nothing ran
    log.close()


def test_a_run_id_is_never_reused(endpoint: str, tmp_path: Path) -> None:
    argv = ["live", "--impl", "our,our", "--model", "fake-live", "--base-url", endpoint]
    argv += [
        "--prompt",
        "Build chat.pipe and validate it.",
        "--out",
        str(tmp_path),
        "--run-id",
        "t1",
    ]
    assert build_parser().parse_args(argv).impl == ["our"]  # each loop once
    assert main(argv) == 0
    result = (tmp_path / "live" / "t1" / "our" / "result.json").read_bytes()
    with pytest.raises(live.LiveError, match="already exists: pick another --run-id"):
        live.fresh_dir(tmp_path / "live" / "t1" / "our")
    assert main(argv) == 1  # refused before anything ran
    assert (tmp_path / "live" / "t1" / "our" / "result.json").read_bytes() == result


async def test_stdin_ask_reads_lines_without_a_thread(tmp_path: Path) -> None:
    threads = threading.active_count()
    read_fd, write_fd = os.pipe()
    out = io.StringIO()
    with os.fdopen(read_fd, "r") as stdin:
        ask = live.stdin_ask(out, stdin)
        os.write(write_fd, b"first\nsecond\nthi")
        assert await ask("> ") == "first"
        assert await ask("> ") == "second"
        pending = asyncio.ensure_future(ask("> "))
        await asyncio.sleep(0.05)
        assert not pending.done()  # waits for the rest of the line, without blocking the loop
        os.write(write_fd, b"rd\nlast line without newline")
        assert await pending == "third"
        os.close(write_fd)
        assert await ask("> ") == "last line without newline"
        with pytest.raises(EOFError):
            await ask("> ")
    assert out.getvalue() == "> " * 5
    assert threading.active_count() == threads  # no reader thread left behind


async def test_ctrl_c_at_the_prompt_ends_the_read(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "r") as stdin:
        ask = live.stdin_ask(io.StringIO(), stdin)
        pending = asyncio.ensure_future(ask("> "))
        await asyncio.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)
        with pytest.raises(EOFError, match="interrupted"):
            await asyncio.wait_for(pending, 5)
    os.close(write_fd)


def test_chat_exits_on_ctrl_c_at_the_prompt(endpoint: str, tmp_path: Path) -> None:
    """Ctrl-C while `bakeoff chat` waits for input (stdin open, nothing typed) ends the chat
    at once and still writes result.json."""
    argv = [sys.executable, "-m", "bakeoff.cli", "chat", "--base-url", endpoint]
    argv += ["--out", str(tmp_path / "out"), "--run-id", "c1"]
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        argv, stdin=read_fd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    os.close(read_fd)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().startswith("chat with our on gpt-6-luna")
        time.sleep(0.3)  # at the prompt
        proc.send_signal(signal.SIGINT)
        printed, err = proc.communicate(timeout=10)
    finally:
        os.close(write_fd)
        proc.kill()
    assert (proc.returncode, err) == (0, ""), printed
    result = json.loads((tmp_path / "out" / "live" / "c1" / "our" / "result.json").read_text())
    assert (result["stops"], result["error"], result["prompt"]) == ([], None, "(chat)")


@pytest.mark.parametrize(
    ("exchanges", "lines", "extra", "code", "stops"),
    [
        ([says("Hi.")], ["hello"], [], 0, ["end_turn"]),
        # Any turn that stops short counts, not only the last one.
        (
            [{"respond": {"status": 400}}, says("Hi.")],
            ["hello", "again"],
            [],
            1,
            ["error", "end_turn"],
        ),
        (
            [calls({"id": "call_C2_1", "name": "list_files", "arguments": {}})],
            ["hello"],
            ["--max-steps", "1"],
            1,
            ["max_steps"],
        ),
    ],
    ids=["answered", "a-turn-errors", "max-steps"],
)
def test_chat_exits_1_if_a_turn_stopped_short(
    exchanges: list[dict[str, Any]],
    lines: list[str],
    extra: list[str],
    code: int,
    stops: list[str],
    tmp_path: Path,
) -> None:
    write_scenario(tmp_path / "scenarios", "C2", exchanges)
    with FakeProvider(tmp_path / "scenarios", tmp_path / "wire") as provider:
        argv = [sys.executable, "-m", "bakeoff.cli", "chat", "--model", "fake-live"]
        argv += ["--base-url", provider.base_url("C2", "r1", "our"), *extra]
        argv += ["--out", str(tmp_path / "out"), "--run-id", "c1"]
        stdin = "".join(f"{line}\n" for line in [*lines, "/exit"])
        proc = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=60)
    result = json.loads((tmp_path / "out" / "live" / "c1" / "our" / "result.json").read_text())
    assert (proc.returncode, result["stops"], result["passed"]) == (code, stops, code == 0), (
        proc.stdout + proc.stderr
    )
    if code:
        assert f"bakeoff chat: not every turn ended well: stops {stops}" in proc.stderr
    else:
        assert proc.stderr == ""


def test_the_key_is_never_stored(
    endpoint: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live, "resolve_api_key", lambda *args, **kwargs: "test-secret-value-42")
    out = tmp_path / "out"
    assert (
        main(
            [
                "live",
                "--impl",
                "our",
                "--model",
                "fake-live",
                "--base-url",
                endpoint,
                "--prompt",
                "p",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    for path in [*out.rglob("*"), *(tmp_path / "wire").rglob("*")]:
        if path.is_file() and ".git" not in path.parts:
            assert b"test-secret-value-42" not in path.read_bytes(), path


def test_resolved_addresses_of_the_endpoint_host_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed: list[tuple[str, ...]] = []
    monkeypatch.setattr(live.netguard, "install", lambda hosts=(): allowed.append(hosts))
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.7", 443))]
    resolve = live._allowing(lambda *args, **kwargs: infos, "api.example.test")
    assert resolve(b"api.example.test", 443) == infos  # anyio passes the name as bytes
    assert resolve("other.example.test", 443) == infos
    assert allowed == [("192.0.2.7",)]


def test_is_loopback() -> None:
    assert live.is_loopback("http://127.0.0.1:8787/s/S01/r/our/v1")
    assert live.is_loopback("http://localhost/v1") and live.is_loopback("http://[::1]:9/v1")
    assert not live.is_loopback("https://api.openai.com/v1")


# --- CLI parsing -----------------------------------------------------------------------------------


def test_cli_parses_every_command(tmp_path: Path) -> None:
    db = tmp_path / "log.sqlite"
    db.touch()
    parser = build_parser()
    args = parser.parse_args(["scenario", "S01", "S05", "--impl", "our,pydantic", "--out", "x"])
    assert (args.ids, args.impl, args.out, args.all) == (
        ["S01", "S05"],
        ["our", "pydantic"],
        Path("x"),
        False,
    )
    args = parser.parse_args(["live", "--impl", "our", "--prompt", "hi", "--reasoning", "none"])
    assert (args.impl, args.model, args.max_steps, args.base_url) == (
        ["our"],
        "gpt-6-luna",
        8,
        live.OPENAI_BASE_URL,
    )
    args = parser.parse_args(["chat"])
    assert (args.impl, args.model) == ("our", "gpt-6-luna")
    args = parser.parse_args(
        ["approve", "t1", "--db", str(db), "--allow", "a", "--deny", "b", "--reason", "r"]
    )
    assert (args.thread, args.allow, args.deny, args.reason, args.max_steps) == (
        "t1",
        ["a"],
        ["b"],
        "r",
        12,
    )
    args = parser.parse_args(
        ["turn", "t1", "--db", str(db), "--user", "u", "--crash-after", "item", "--call-id", "c"]
    )
    assert (args.user, args.crash_after, args.call_id) == ("u", "item", "c")
    assert parser.parse_args(["resume", "t1", "--db", str(db)]).command == "resume"
    assert parser.parse_args(["fakeprov", "--port", "0"]).port == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["scenario", "S01", "--impl", "nope"],
        ["live", "--prompt", "p"],  # --impl is required
        ["approve", "t1", "--db", "/no/such/log.sqlite"],
    ],
)
def test_cli_rejects_bad_arguments(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_cli_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scenario", "--impl", "our"]) == 2
    assert "give scenario ids or --all" in capsys.readouterr().err
    assert main(["scenario", "S99", "--impl", "our"]) == 2
    assert "unknown scenarios ['S99']" in capsys.readouterr().err
    assert main([]) == 0
    assert "usage: bakeoff" in capsys.readouterr().out


def test_cli_scenario_prints_the_matrix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["scenario", "S01", "S13", "--impl", "our", "--out", str(tmp_path), "--run-id", "c1"]
    assert main(argv) == 0
    printed = capsys.readouterr()
    assert printed.out.splitlines()[:3] == ["scenario  our", "S01       pass", "S13       pass"]
    assert "S01   our       pass" in printed.err
    assert json.loads((tmp_path / "runs" / "c1" / "summary.json").read_text())["run_id"] == "c1"


def test_a_broken_loop_fails_the_command_and_a_missing_one_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A loop that exists but does not import must not silently drop out of the comparison."""
    (tmp_path / "broken_loop.py").write_text("raise ImportError('pydantic_ai renamed Agent')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    broken = replace(loops.REGISTRY["pydantic"], target="broken_loop:PydanticLoop")
    monkeypatch.setitem(loops.REGISTRY, "pydantic", broken)
    monkeypatch.setitem(loops.REGISTRY, "probe", loops.LoopEntry("probe", "not_built:Loop", ()))
    out = tmp_path / "out"
    assert main(["scenario", "S01", "--impl", "our,pydantic", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "error: broken loop pydantic: ImportError: pydantic_ai renamed Agent" in err
    argv = ["live", "--impl", "our,pydantic", "--prompt", "p", "--out", str(out)]
    assert main([*argv, "--base-url", "http://127.0.0.1:9/v1"]) == 1
    assert "error: broken loop pydantic" in capsys.readouterr().err
    assert not out.exists()  # refused before anything ran
    # A loop that is not built yet is only skipped.
    argv = ["scenario", "S01", "--impl", "our,probe", "--out", str(out), "--run-id", "m1"]
    assert main(argv) == 0
    assert "skipping probe: not installed" in capsys.readouterr().err
    assert list(json.loads((out / "runs" / "m1" / "summary.json").read_text())["loops"]) == ["our"]


def test_cli_scenario_never_reuses_a_run_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["scenario", "S01", "--impl", "our", "--out", str(tmp_path), "--run-id", "c1"]
    assert main(argv) == 0
    run = tmp_path / "runs" / "c1"
    saved = {p: p.read_bytes() for p in (run / "summary.json", run / "S01" / "our" / "result.json")}
    capsys.readouterr()
    assert main(argv) == 1
    assert f"{run} already exists: pick another --run-id" in capsys.readouterr().err
    assert {p: p.read_bytes() for p in saved} == saved


def test_cli_worker_errors_are_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "log.sqlite"
    SessionLog(db).close()
    assert main(["resume", "nope", "--db", str(db)]) == 1
    assert "unknown thread nope" in capsys.readouterr().err
    assert main(["cancel", "nope", "--db", str(db)]) == 1

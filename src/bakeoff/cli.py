"""`bakeoff` command line: scenarios, live runs, chat, and the cross-process thread commands.

The logic lives in `bakeoff.shared.scenario` and `bakeoff.live`; this module only parses
arguments, installs the network guard and prints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from bakeoff import live, loops
from bakeoff.fakeprov.__main__ import main as fakeprov_main
from bakeoff.shared import netguard, scenario
from bakeoff.shared.contract import Limits, Resume
from bakeoff.shared.sessionlog import SessionLog


def _impls(value: str) -> list[str]:
    names = list(dict.fromkeys(n.strip() for n in value.split(",") if n.strip()))
    if unknown := [n for n in names if n not in loops.REGISTRY]:
        raise argparse.ArgumentTypeError(
            f"unknown loop {unknown}; known: {', '.join(loops.REGISTRY)}"
        )
    return names


def _existing(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"no such file: {value}")
    return path


def _thread_args(parser: argparse.ArgumentParser) -> None:
    """Options shared by the commands that act on one thread of a session log."""
    parser.add_argument("thread", help="thread id")
    parser.add_argument("--db", type=_existing, required=True, help="the session log (log.sqlite)")


def _worker_args(parser: argparse.ArgumentParser) -> None:
    """Options of the commands that run a turn in this process."""
    _thread_args(parser)
    parser.add_argument("--wc", type=Path, help="working copy root (default: wc/ next to --db)")
    parser.add_argument(
        "--events", type=Path, help="ndjson mirror (default: events.ndjson next to --db)"
    )
    parser.add_argument("--max-steps", type=int, default=Limits().max_steps)
    parser.add_argument("--engine-delay-ms", type=int, default=0, help="MockEngine delay")
    parser.add_argument("--env-file", type=Path, help="env file with OPENAI_API_KEY (live threads)")
    _key_host_arg(parser)


def _key_host_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--key-host",
        action="append",
        default=[],
        metavar="HOST",
        help=f"a host besides {live.OPENAI_HOST} that may receive {live.KEY_NAME} (repeatable)",
    )


def _model_args(parser: argparse.ArgumentParser, *, impl_default: str | None) -> None:
    if impl_default is None:
        parser.add_argument("--impl", type=_impls, required=True, help="loops, e.g. our,pydantic")
    else:
        parser.add_argument("--impl", default=impl_default, choices=list(loops.REGISTRY))
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning", help="reasoning effort, e.g. none, low, high")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--env-file", type=Path, help=f"env file with {live.KEY_NAME}")
    _key_host_arg(parser)
    parser.add_argument(
        "--base-url",
        default=live.OPENAI_BASE_URL,
        help="the API's base URL (default: OpenAI's); {impl} is replaced by the loop name",
    )
    parser.add_argument(
        "--api",
        choices=["chat", "responses"],
        default="chat",
        help="chat completions, or OpenAI's Responses API (endpoint kind openai_responses)",
    )
    parser.add_argument(
        "--kind",
        choices=["openai_compat", "openrouter"],
        help="the chat completions endpoint's kind (default: openai_compat)",
    )
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--run-id", help="default: a new timestamped id")


def build_parser() -> argparse.ArgumentParser:
    """The `bakeoff` argument parser."""
    parser = argparse.ArgumentParser(prog="bakeoff", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("scenario", help="run scenarios against loops and print the matrix")
    p.add_argument("ids", nargs="*", help="scenario ids, e.g. S01 S05")
    p.add_argument("--all", action="store_true", help="every scenario")
    p.add_argument("--impl", type=_impls, help="loops, e.g. our,pydantic (default: all built)")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--run-id", help="default: a new timestamped id")

    p = sub.add_parser("live", help="one prompt against a real model, per loop, side by side")
    _model_args(p, impl_default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--interactive", action="store_true", help="ask before writes (stdin)")

    p = sub.add_parser("chat", help="interactive REPL on one loop, with approval prompts")
    _model_args(p, impl_default="our")
    p.add_argument(
        "--yes",
        action="store_true",
        help="allow every tool and ask nothing (unattended, as in `live`)",
    )

    p = sub.add_parser("turn", help="run one user turn of a thread (a worker process)")
    _worker_args(p)
    p.add_argument("--user", required=True, help="the user message")
    p.add_argument("--crash-after", help="SIGKILL this process at the first event of this type")
    p.add_argument("--call-id", help="... about this tool call")

    p = sub.add_parser("approve", help="answer a paused turn and resume it")
    _worker_args(p)
    p.add_argument("--allow", action="append", default=[], help="call id (default: all pending)")
    p.add_argument("--deny", action="append", default=[], help="call id to deny")
    p.add_argument("--reason", help="shown to the model for denied calls")

    p = sub.add_parser("deny", help="deny a paused turn's calls and resume it")
    _worker_args(p)
    p.add_argument("--call", action="append", default=[], help="call id (default: all pending)")
    p.add_argument("--reason", help="shown to the model")

    p = sub.add_parser("resume", help="resume a turn whose worker died (crash resume)")
    _worker_args(p)

    p = sub.add_parser("cancel", help="ask the worker running a thread's turn to cancel it")
    _thread_args(p)

    # `report` keeps its own argument parser (bakeoff.report.build); see main().
    sub.add_parser("report", help="build the comparison page (out/report.html)", add_help=False)
    p = sub.add_parser("fakeprov", help="serve the fake model server on 127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--wire-dir", type=Path, default=Path("out/wire"))
    p.add_argument("--check", action="store_true", help="validate the scenarios and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of `bakeoff` and `python -m bakeoff.cli`."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["report"]:  # its options live in the report package
        from bakeoff.report import build

        return build.main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return _COMMANDS[args.command](args)
    except (scenario.DriverError, live.LiveError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"bakeoff {args.command}: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _usable(names: list[str]) -> list[str]:
    """The loops that import here. One that is not built or installed yet is skipped (with a
    note); one that exists but fails to import is an error, because skipping it would let the
    command succeed without the comparison it was asked for."""
    usable, broken = [], []
    for name in names:
        try:
            loops.load(name)
            usable.append(name)
        except loops.LoopUnavailable as exc:
            if not exc.missing:
                broken.append(str(exc))
                continue
            print(f"skipping {exc}", file=sys.stderr)
    if broken:
        raise RuntimeError(f"broken loop {'; '.join(broken)} (fix it, or leave it out of --impl)")
    return usable


def _scenario(args: argparse.Namespace) -> int:
    known = scenario.scenario_ids()
    ids = known if args.all else args.ids
    if not ids:
        print("bakeoff scenario: give scenario ids or --all", file=sys.stderr)
        return 2
    if unknown := [i for i in ids if i not in known]:
        print(f"bakeoff scenario: unknown scenarios {unknown}", file=sys.stderr)
        return 2
    impls = _usable(args.impl or list(loops.REGISTRY))
    if not impls:
        return 2
    netguard.install()

    def progress(result: dict[str, Any]) -> None:
        took = result["duration_ms"] / 1000
        print(
            f"  {result['scenario']:<5} {result['impl']:<9} {scenario.status(result):<5} {took:5.1f} s",
            file=sys.stderr,
        )

    # Not asyncio.run: at exit it cancels the tasks still running and waits for them, which
    # never ends if a loop left one that ignores cancellation (see the stop below).
    with asyncio.Runner() as runner:
        summary = runner.run(
            scenario.run_matrix(ids, impls, out=args.out, run_id=args.run_id, on_result=progress)
        )
        print(scenario.format_matrix(summary))
        print(f"results: {args.out / 'runs' / summary['run_id']}")
        if summary["stopped"]:
            sys.stdout.flush()
            print(f"bakeoff scenario: stopped: {summary['stopped']}", file=sys.stderr, flush=True)
            # The leaked tasks already ignored a cancel. Everything is written: end the process
            # without waiting for them.
            os._exit(1)
    return 1 if scenario.unexpected(summary) else 0


def _kind(args: argparse.Namespace) -> live.Kind:
    """The endpoint kind: `--api responses` is OpenAI's Responses API; `--kind` picks among the
    chat completions kinds, so it does not go with `--api responses`."""
    if args.api == "chat":
        return args.kind or "openai_compat"
    if args.kind is not None:
        raise ValueError(
            f"--kind {args.kind} is a chat completions kind; drop it for --api responses"
        )
    return "openai_responses"


def _live_model(args: argparse.Namespace) -> live.LiveModel:
    kind = _kind(args)
    live.guard_network(args.base_url)  # before anything can connect
    return live.LiveModel(
        model=args.model,
        base_url=args.base_url,
        api_key=live.resolve_api_key(args.base_url, args.env_file, key_hosts=args.key_host),
        kind=kind,
        reasoning=args.reasoning,
        max_tokens=args.max_tokens,
    )


def _live(args: argparse.Namespace) -> int:
    model = _live_model(args)
    impls = _usable(args.impl)
    if not impls:
        return 2
    term = live.open_terminal()
    results = asyncio.run(
        live.run_live(
            impls,
            model,
            args.prompt,
            out=args.out,
            run_id=args.run_id or scenario.new_run_id(),
            term=term,
            max_steps=args.max_steps,
            ask=live.stdin_ask(term) if args.interactive else None,
        )
    )
    term.write("\n" + live.format_side_by_side(results) + "\n")
    term.write(f"results: {args.out / 'live' / results[0]['run_id']}\n")
    _close(term)
    return 0 if all(r["passed"] for r in results) else 1


def _chat(args: argparse.Namespace) -> int:
    model = _live_model(args)
    if not _usable([args.impl]):
        return 2
    term = live.open_terminal()
    result = asyncio.run(
        live.chat(
            args.impl,
            model,
            out=args.out,
            run_id=args.run_id or scenario.new_run_id(),
            term=term,
            ask=live.stdin_ask(term),
            max_steps=args.max_steps,
            yes=args.yes,
        )
    )
    term.write(f"\nresults: {args.out / 'live' / result['run_id'] / args.impl}\n")
    _close(term)
    if result["passed"]:  # also a chat ended at a prompt (Ctrl-C, /exit) before any turn
        return 0
    failed = [f"; {n} failed" for n, c in result["invariants"].items() if not c["ok"]]
    print(
        f"bakeoff chat: not every turn ended well: stops {result['stops']}{''.join(failed)}",
        file=sys.stderr,
    )
    return 1


def _close(term: Any) -> None:
    term.flush()
    if term is not sys.stdout:  # our own descriptor (see live.open_terminal)
        term.close()


def _pending(args: argparse.Namespace) -> list[str]:
    log = SessionLog(args.db)
    try:
        return scenario.pending_calls(log, args.thread)
    finally:
        log.close()


def _worker(args: argparse.Namespace) -> int:
    """turn, approve, deny and resume: one turn of an existing thread, in this process."""
    log = SessionLog(args.db)
    try:
        thread = log.get_thread(args.thread)
    finally:
        log.close()
    if thread is None:
        raise scenario.DriverError(f"unknown thread {args.thread} in {args.db}")
    base_url = thread["meta"]["model"]["base_url"]
    # The log names the endpoint; the key goes there only if this command line allows it.
    api_key = live.resolve_api_key(base_url, args.env_file, key_hosts=args.key_host)
    live.guard_network(base_url)
    sinks = []
    user_text, resume = None, None
    if args.command == "turn":
        user_text = args.user
        if args.crash_after:
            sinks.append(scenario.crash_sink(args.crash_after, args.call_id))
    elif args.command == "approve":
        decisions = scenario.decide(_pending(args), args.allow or "all", args.deny)
        resume = Resume(kind="approval", decisions=decisions, reason=args.reason)
    elif args.command == "deny":
        pending = _pending(args)
        decisions = scenario.decide(pending, [], args.call or pending)
        resume = Resume(kind="approval", decisions=decisions, reason=args.reason)
    else:
        resume = Resume(kind="crash")
    summary = asyncio.run(
        scenario.worker_turn(
            args.db,
            args.thread,
            api_key=api_key,
            user_text=user_text,
            resume=resume,
            max_steps=args.max_steps,
            engine_delay_ms=args.engine_delay_ms,
            wc=args.wc,
            events=args.events,
            sinks=sinks,
        )
    )
    print(json.dumps(summary))  # one line: the driver parses it
    return 0


def _cancel(args: argparse.Namespace) -> int:
    netguard.install()
    log = SessionLog(args.db)
    try:
        if log.get_thread(args.thread) is None:
            raise scenario.DriverError(f"unknown thread {args.thread} in {args.db}")
        log.request_cancel(args.thread)
    finally:
        log.close()
    print(json.dumps({"thread": args.thread, "cancel": "requested"}))
    return 0


def _fakeprov(args: argparse.Namespace) -> int:
    netguard.install()
    argv = [f"--port={args.port}", f"--wire-dir={args.wire_dir}"]
    return fakeprov_main([*argv, "--check"] if args.check else argv)


_COMMANDS = {
    "scenario": _scenario,
    "live": _live,
    "chat": _chat,
    "turn": _worker,
    "approve": _worker,
    "deny": _worker,
    "resume": _worker,
    "cancel": _cancel,
    "fakeprov": _fakeprov,
}


if __name__ == "__main__":
    raise SystemExit(main())

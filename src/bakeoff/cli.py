"""`bakeoff` command line: scenarios and the cross-process thread commands.

The logic lives in `bakeoff.shared.scenario`; this module only parses arguments, installs the
network guard and prints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from bakeoff import loops
from bakeoff.fakeprov.__main__ import main as fakeprov_main
from bakeoff.shared import netguard, scenario
from bakeoff.shared.contract import Limits, Resume
from bakeoff.shared.sessionlog import SessionLog


def _impls(value: str) -> list[str]:
    names = [n.strip() for n in value.split(",") if n.strip()]
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

    p = sub.add_parser("fakeprov", help="serve the fake model server on 127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--wire-dir", type=Path, default=Path("out/wire"))
    p.add_argument("--check", action="store_true", help="validate the scenarios and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of `bakeoff` and `python -m bakeoff.cli`."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return _COMMANDS[args.command](args)
    except (scenario.DriverError, RuntimeError, ValueError) as exc:
        print(f"bakeoff {args.command}: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _usable(names: list[str]) -> list[str]:
    """The loops that import here; says which requested ones are skipped and why."""
    usable = []
    for name in names:
        try:
            loops.load(name)
            usable.append(name)
        except loops.LoopUnavailable as exc:
            print(f"skipping {exc}", file=sys.stderr)
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

    summary = asyncio.run(
        scenario.run_matrix(ids, impls, out=args.out, run_id=args.run_id, on_result=progress)
    )
    print(scenario.format_matrix(summary))
    print(f"results: {args.out / 'runs' / summary['run_id']}")
    return 1 if scenario.unexpected(summary) else 0


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
    netguard.install()
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
            api_key="dummy",  # scenario threads talk to the local fake provider
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
    "turn": _worker,
    "approve": _worker,
    "deny": _worker,
    "resume": _worker,
    "cancel": _cancel,
    "fakeprov": _fakeprov,
}


if __name__ == "__main__":
    raise SystemExit(main())

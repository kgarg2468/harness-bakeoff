"""Write a snapshot of `bakeoff live` runs to commit or share: key fields, one row per loop.

    uv run python scripts/snapshot_live.py OUT.json RUN_ID... [--live-dir out/live] [--what TEXT]

Rows are copied from `<live-dir>/<run_id>/<loop>/result.json`, plus what only the session log
keeps: the thread's model settings (`kind`, `reasoning`) and what each `validate_pipeline` call
returned. The harness never records the API key, headers or request bodies, so neither holds
them. The prompt, the final answer and the validation results are copied as written, though:
key-shaped strings in them are redacted (and counted on stderr), but read a snapshot before you
share it, since anything else sensitive in a prompt stays in.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Key-shaped strings (those the CI secrets check looks for, and any other `sk-` key).
KEY_SHAPED = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
)
REDACTED = "[redacted]"


def redact(value: Any) -> tuple[Any, int]:
    """`value` with every key-shaped string in it replaced, and how many were."""
    if isinstance(value, str):
        return KEY_SHAPED.subn(REDACTED, value)
    if isinstance(value, list):
        pairs = [redact(v) for v in value]
        return [v for v, _ in pairs], sum(n for _, n in pairs)
    if isinstance(value, dict):
        pairs = {k: redact(v) for k, v in value.items()}
        return {k: v for k, (v, _) in pairs.items()}, sum(n for _, n in pairs.values())
    return value, 0


def _from_log(log: Path, thread: str | None) -> tuple[dict[str, Any], list[Any]]:
    """The thread's model settings and its `validate_pipeline` results, from its session log
    (opened read-only); ({}, []) if the log can't tell."""
    try:
        db = sqlite3.connect(f"{log.resolve().as_uri()}?mode=ro", uri=True)
        try:
            threads = dict(db.execute("SELECT id, meta FROM threads").fetchall())
            if thread not in threads and len(threads) == 1:  # as the report reads it
                thread = next(iter(threads))
            items = [
                j
                for (j,) in db.execute(
                    "SELECT json FROM items WHERE thread = ? ORDER BY seq", (thread,)
                )
            ]
        finally:
            db.close()
    except sqlite3.Error:
        return {}, []
    model = json.loads(threads[thread]).get("model") if thread in threads else None
    calls, validations = set(), []
    for item in items:
        message = json.loads(item).get("message") or {}
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] == "validate_pipeline":
                calls.add(call["id"])
        if message.get("role") == "tool" and message.get("tool_call_id") in calls:
            try:
                validations.append(json.loads(message["content"]))
            except (TypeError, ValueError):
                validations.append(message.get("content"))
    return (model if isinstance(model, dict) else {}), validations


def row(run_dir: Path, loop_dir: Path) -> dict[str, Any]:
    r = json.loads((loop_dir / "result.json").read_text())
    model, validations = _from_log(loop_dir / "log.sqlite", r.get("thread"))
    usage, latency = r.get("usage") or {}, r.get("latency") or {}
    invariants = r.get("invariants") or {}
    return {
        "run_id": run_dir.name,
        "loop": loop_dir.name,
        "model": r.get("model"),
        "base_url": r.get("base_url"),
        "kind": model.get("kind"),
        "reasoning": model.get("reasoning"),
        "max_steps": r.get("max_steps"),
        "attended": r.get("attended"),
        "prompt": r.get("prompt"),
        "stops": r.get("stops"),
        "steps": r.get("steps"),
        "requests": r.get("requests"),
        "input_tokens": usage.get("input_tokens"),
        "cached_tokens": usage.get("cached_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "first_token_ms": latency.get("ttft_ms"),
        "duration_ms": r.get("duration_ms"),
        "tool_runs": sum((r.get("tool_runs") or {}).values()),
        "files": r.get("files"),
        "validations": validations,
        "passed": r.get("passed"),
        "error": r.get("error"),
        "invariants": {k: bool(v.get("ok")) for k, v in invariants.items()},
        "final_text": r.get("final_text"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--live-dir", type=Path, default=Path("out/live"))
    ap.add_argument("--what", default="Key fields of `bakeoff live` runs.")
    args = ap.parse_args(argv)
    rows, redacted = [], 0
    for run_id in args.run_ids:
        run_dir = args.live_dir / run_id
        loops = (
            sorted(p for p in run_dir.iterdir() if (p / "result.json").is_file())
            if run_dir.is_dir()
            else []
        )
        if not loops:
            print(f"no loop results in {run_dir}", file=sys.stderr)
            return 1
        for loop_dir in loops:
            clean, n = redact(row(run_dir, loop_dir))
            rows.append(clean)
            redacted += n
    if redacted:
        print(f"redacted {redacted} key-shaped string(s)", file=sys.stderr)
    what = (
        f"{args.what} Copied field by field from out/live/<run_id>/<loop>/result.json, and kind,"
        " reasoning and validations from its session log (no API key, headers or request bodies;"
        " key-shaped strings redacted); written by scripts/snapshot_live.py."
    )
    args.out.write_text(
        json.dumps({"what": what, "runs": rows}, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

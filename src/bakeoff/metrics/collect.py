"""Collect the scorecard metrics into `out/metrics.json`.

Always runs `loc`. Runs `bench` in quick mode by default (`--bench full` or `--bench none` to
change that), and `deps` only with `--deps`, because it installs from PyPI. The file carries a
schema version and the git commit the numbers belong to, with `dirty` set when the working tree
had uncommitted changes.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from bakeoff.metrics import REPO_ROOT, bench, deps, loc
from bakeoff.shared import netguard

SCHEMA_VERSION = 1


def git_state(root: Path) -> dict[str, Any]:
    """The checked-out commit and whether the working tree has uncommitted changes."""

    def git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"sha": sha, "dirty": None if status is None else bool(status)}


def collect(
    root: Path = REPO_ROOT,
    *,
    bench_mode: str = "quick",
    with_deps: bool = False,
    latest: bool = True,
) -> dict[str, Any]:
    """All requested metrics as one JSON-ready dict."""
    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "git": git_state(root),
        "python": platform.python_version(),
        "loc": loc.measure(root),
        "bench": None,
        "deps": None,
    }
    if bench_mode != "none":
        config = bench.QUICK if bench_mode == "quick" else bench.FULL
        report["bench"] = {"mode": bench_mode, **bench.measure(config)}
    if with_deps:
        report["deps"] = deps.measure(
            src=root / "src", pyproject=root / "pyproject.toml", latest=latest
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """Write out/metrics.json and print where it went."""
    parser = argparse.ArgumentParser(prog="python -m bakeoff.metrics.collect", description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--out", type=Path, help="output file (default: <root>/out/metrics.json)")
    parser.add_argument("--bench", choices=("quick", "full", "none"), default="quick")
    parser.add_argument("--deps", action="store_true", help="also measure dependencies (network)")
    parser.add_argument(
        "--no-latest", action="store_true", help="deps: skip the latest pydantic-ai"
    )
    args = parser.parse_args(argv)
    if not args.deps:
        netguard.install()  # I6; only the dependency installs need PyPI
    root = args.root.resolve()
    report = collect(root, bench_mode=args.bench, with_deps=args.deps, latest=not args.no_latest)
    out = args.out or root / "out" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(loc.table(report["loc"], per_file=False))
    if report["bench"] is not None:
        print(bench.summary(report["bench"]))
    if report["deps"] is not None:
        print(deps.table(report["deps"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

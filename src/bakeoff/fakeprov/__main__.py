"""`python -m bakeoff.fakeprov`: serve the scenario scripts on 127.0.0.1, or `--check` them."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from bakeoff.fakeprov.script import SCENARIOS_DIR, ScenarioError, load_scenario
from bakeoff.fakeprov.server import FakeProvider


def main(argv: list[str] | None = None) -> int:
    """Run the fake provider until Ctrl-C, or validate every scenario file with `--check`."""
    parser = argparse.ArgumentParser(prog="python -m bakeoff.fakeprov", description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR)
    parser.add_argument("--wire-dir", type=Path, default=Path("out/wire"))
    parser.add_argument("--check", action="store_true", help="validate the scenarios and exit")
    args = parser.parse_args(argv)
    if args.check:
        failed = False
        for path in sorted(args.scenarios.glob("*.json")):
            try:
                load_scenario(path)
                print(f"ok    {path.name}")
            except ScenarioError as exc:
                failed = True
                print(f"FAIL  {exc}")
        return int(failed)
    with FakeProvider(args.scenarios, args.wire_dir, port=args.port) as provider:
        print(f"serving http://127.0.0.1:{provider.port}/s/<scenario>/<run>/<impl>/v1")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

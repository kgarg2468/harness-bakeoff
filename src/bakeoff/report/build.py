"""Build `out/report.html`: one self-contained page (inline CSS, JS and data; no external
assets), readable from file:// or over HTTP.

    python -m bakeoff.report.build       # out/runs/latest + out/live/* + out/metrics.json
    python -m bakeoff.report.build --out-dir out -o out/report.html

`main(argv)` is ready to be the `bakeoff report` subcommand.

The same inputs give the same bytes, except the "generated" timestamp (set `SOURCE_DATE_EPOCH`
to pin it). The page stays under ~2 MB: long texts are cut with an expand control, and if the
data is still too big, every embedded string is cut shorter until it fits.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from bakeoff.report import data, render

BUDGET_BYTES = 1_900_000
# Tried in order until the page fits the budget: None = the default cuts in `data`.
_SHRINK_STEPS: tuple[int | None, ...] = (None, 1_500, 400, 120)

SECTIONS = (
    ("scorecard", "Scorecard", "The numbers at a glance", render.scorecard),
    ("matrix", "Scenario matrix", "Which loop passes which scenario", render.matrix),
    ("replay", "Side-by-side replay", "Watch both loops handle the same scenario", render.replay),
    ("wire", "Wire diff", "What each loop actually sent to the model", render.wire),
    ("live", "Live runs", "The same prompt on a real model", render.live),
    ("wins", "Where each wins", "Evidence for each side, no verdict", render.wins),
)


def _asset(name: str) -> str:
    return (resources.files("bakeoff.report") / name).read_text(encoding="utf-8")


def _generated_at(now: datetime | None) -> str:
    if now is None:
        epoch = os.environ.get("SOURCE_DATE_EPOCH")
        now = datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)
    return now.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _shrink(value: Any, limit: int) -> Any:
    """Every string in `value` cut to `limit` characters (used only over the size budget)."""
    if isinstance(value, str):
        return data.clip(value, limit)
    if isinstance(value, list):
        return [_shrink(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: _shrink(v, limit) for k, v in value.items()}
    return value


def page_loops(
    run: dict[str, Any] | None,
    live: list[dict[str, Any]],
    metrics: dict[str, Any] | None,
    discovered: list[str],
) -> list[str]:
    """The loops the page compares: every loop with data, plus the loops this checkout has."""
    found = set(discovered) | set((run or {}).get("impls") or [])
    found |= {impl for r in live for impl in r["results"]}
    loc = ((metrics or {}).get("loc") or {}).get("loops") or {}
    # An empty package (a loop not merged yet) has nothing to compare.
    found |= {
        data.impl_of_package(k)
        for k, v in loc.items()
        if ((v or {}).get("total") or {}).get("code")
    }
    for key in ("bench", "deps"):
        section = (metrics or {}).get(key) or {}
        names = section.get("loops", {}) if key == "bench" else section
        found |= {data.impl_of_package(k) for k in names}
    return sorted(found, key=data.impl_order)


def build(
    *,
    runs: Path | None,
    live: Path | None,
    metrics: Path | None,
    now: datetime | None = None,
    discovered: list[str] | None = None,
) -> str:
    """The report page as a string. Any input may be missing; its sections then say so."""
    run = data.load_run(runs)
    live_runs, live_problems = data.load_live(live)
    metrics_data, metrics_problem = data.load_metrics(metrics)
    loops = page_loops(
        run, live_runs, metrics_data, data.discover() if discovered is None else discovered
    )
    page = render.Page(
        loops=loops,
        run=run,
        live=live_runs,
        metrics=metrics_data,
        # Kept on the page, not the run: they must show when there is no scenario run too.
        problems=[*live_problems, *([metrics_problem] if metrics_problem else [])],
    )
    for scenario in page.scenarios:
        page.cells[scenario["id"]] = {
            impl: render.make_cell(page.summary, scenario["id"], impl, scenario["runs"].get(impl))
            for impl in sorted({*loops, *scenario["runs"]}, key=data.impl_order)
        }
    generated = _generated_at(now)
    body = [render.header(page, generated)]
    for number, (key, kicker, title, section) in enumerate(SECTIONS, 1):
        body.append(
            f'<section id="{key}" class="sec" data-section="{key}">'
            f'<div class="kick">{number} · {render.esc(kicker)}</div><h2>{render.esc(title)}</h2>'
            f"{section(page)}</section>"
        )
    body.append(f'<footer class="gl" id="glossary">{render.glossary()}</footer>')
    nav = "".join(
        f'<a href="#{key}">{n} {render.esc(kicker)}</a>'
        for n, (key, kicker, *_) in enumerate(SECTIONS, 1)
    )
    js_data = render.js_data(page)
    parts = {
        "TITLE": render.esc(f"harness-bakeoff report · {run['run_id'] if run else 'no runs'}"),
        "STYLE": _asset("style.css"),
        "NAV": nav,
        "GENERATED": render.esc(generated),
        "BODY": "\n".join(body),
        "SCRIPT": _asset("report.js"),
    }
    template = _asset("template.html")
    html = ""
    for limit in _SHRINK_STEPS:
        parts["DATA"] = render.embed_json(js_data if limit is None else _shrink(js_data, limit))
        # One pass over the template only, so a placeholder-like text in the data stays as is.
        html = re.sub(r"\{\{(\w+)\}\}", lambda m: parts[m[1]], template)
        if len(html.encode()) <= BUDGET_BYTES:
            break
    return html


def main(argv: list[str] | None = None) -> int:
    """Write the page and say where it went (the `bakeoff report` command, once wired)."""
    parser = argparse.ArgumentParser(
        prog="bakeoff report", description="Build the comparison report (one HTML file)."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="where runs, live runs and metrics are"
    )
    parser.add_argument("--runs", type=Path, help="a run folder (default: OUT_DIR/runs/latest)")
    parser.add_argument("--live", type=Path, help="live runs (default: OUT_DIR/live)")
    parser.add_argument("--metrics", type=Path, help="metrics file (default: OUT_DIR/metrics.json)")
    parser.add_argument("-o", "--output", type=Path, help="default: OUT_DIR/report.html")
    args = parser.parse_args(argv)
    out = args.out_dir
    runs = args.runs or out / "runs" / "latest"
    output = args.output or out / "report.html"
    page = build(
        runs=runs, live=args.live or out / "live", metrics=args.metrics or out / "metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    size = len(page.encode())
    print(f"wrote {output} ({size / 1024:.0f} KB)")
    if size > BUDGET_BYTES:
        print(f"warning: over the {BUDGET_BYTES / 1e6:.1f} MB budget", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""HTML for the report's sections, rendered on the server side so the numbers are in the page
itself (readable without JavaScript, and by tests). The replay and the wire diff are drawn by
`report.js` from the embedded data; this module renders their controls.

Everything shown comes from the data passed in; every text goes through `esc`.
"""

from __future__ import annotations

import html
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from bakeoff.report.data import WIRE_LIMIT, impl_of_package, impl_order, loop_info

REPO_URL = "https://github.com/kgarg2468/harness-bakeoff/blob/main"

# Plain-language definitions behind every dotted-underlined term (the reader is new to this).
GLOSSARY: dict[str, tuple[str, str]] = {
    "harness": (
        "harness",
        "Everything around the model that turns it into an agent: the loop that calls the model, the tools, the saved history, approvals and limits.",
    ),
    "loop": (
        "loop",
        "The part of the harness that talks to the model: send the conversation, stream the answer, run the tools it asks for, repeat until it is done. The only part that differs between A and B.",
    ),
    "shared": (
        "shared layer",
        "Code both loops use unchanged: tools, session log, git working copy, permission rules, fake model server. Counted once, and not in either loop's total.",
    ),
    "code": (
        "code lines",
        "Lines with Python code on them. Blank lines, comment lines and docstrings are counted separately.",
    ),
    "docstring": (
        "docstring",
        "Text in triple quotes at the top of a module, class or function that explains it.",
    ),
    "ported": (
        "ported",
        "Lines copied or translated from another MIT-licensed open-source project (Pi, OpenCode, RocketRide). Each such file says so at its top and in THIRD_PARTY_NOTICES.md. We still own and maintain them.",
    ),
    "dependency": (
        "dependency",
        "A third-party package that must be installed for the code to run.",
    ),
    "site-packages": (
        "site-packages",
        "The folder where installed Python packages live; its size is the disk space the dependencies take.",
    ),
    "cold-import": (
        "cold import",
        "How long Python takes to load a package in a fresh process: every file it pulls in, before any work is done.",
    ),
    "third-party-lines": (
        "third-party code loaded",
        "Lines of other people's Python code that are loaded once the loop has run one turn: code you run but did not write.",
    ),
    "overhead": (
        "harness overhead",
        "Time the harness adds on top of the model: parsing the stream, building requests, emitting events. Measured against a bare HTTP client replaying the same requests.",
    ),
    "baseline": (
        "raw baseline",
        "The same requests sent by a minimal HTTP client with no harness at all: the reference point for overhead.",
    ),
    "p50": ("p50", "The median: half of the measurements were faster than this."),
    "p95": ("p95", "95% of the measurements were faster than this: a look at the slow tail."),
    "chunk": ("chunk", "One small piece of a streamed model answer (one server-sent event)."),
    "scenario": (
        "scenario",
        "A scripted test: fixed user messages and fixed fake-model answers, so both loops face exactly the same situation.",
    ),
    "fakeprov": (
        "fake model server",
        "A local stand-in for OpenRouter that plays back scripted answers and records every request. No real model, no internet.",
    ),
    "openrouter": (
        "OpenRouter",
        "A service that forwards one OpenAI-style API to many model providers. The fake model server imitates it.",
    ),
    "session-log": (
        "session log",
        "The SQLite file the shared runner writes for each conversation: every event and every history item, in order. The replay is drawn from it.",
    ),
    "working-copy": (
        "working copy",
        "The folder of files the agent reads and edits in a scenario. It is a git repository, so every turn can be committed and undone.",
    ),
    "turn": (
        "turn",
        "One round: the user says something and the loop works until it answers, pauses for an approval, is cancelled or hits a limit.",
    ),
    "step": (
        "step",
        "One request to the model inside a turn. A turn with two rounds of tool calls has three steps.",
    ),
    "tool-call": (
        "tool call",
        "The model asking the harness to run a tool (for example read_file) with some arguments.",
    ),
    "tool-result": (
        "tool result",
        "What a tool returned. It goes back to the model in the next request. A denied call, or one cut off by a limit, still gets a result that says so, without the tool running.",
    ),
    "tool-run": (
        "tool run",
        "One execution of a tool. A call that was denied, or cut off by a limit, has a result but no run.",
    ),
    "approval": (
        "approval",
        "Some tools need a yes from the user first. The turn pauses; the answer resumes it, possibly in a new process.",
    ),
    "commit": (
        "commit",
        "A git snapshot of the working copy taken after every finished turn, so any turn can be undone.",
    ),
    "revert": ("revert", "Undoing an earlier turn's file changes with a new commit."),
    "compaction": (
        "compaction",
        "Replacing old history with a short summary so the conversation keeps fitting the model's context window.",
    ),
    "crash-resume": (
        "crash resume",
        "The worker process is killed on purpose in the middle of a turn; a new process continues from the saved history without re-running finished tools.",
    ),
    "cancel": (
        "cancel",
        "The user stops a turn. The loop must stop within 200 ms and leave no half-finished tool calls.",
    ),
    "retry": (
        "retry",
        "Sending a failed model request again after a wait, e.g. after HTTP 429 (too many requests) or an error in the middle of a stream.",
    ),
    "invariant": (
        "invariant",
        "A rule that must hold in every scenario. It is checked from the wire recordings and the session log, never from what the loop says about itself.",
    ),
    "I1": (
        "I1 append-only",
        "Each request's messages start with the previous request's messages, unchanged (except right after a compaction). This keeps provider prompt caches warm.",
    ),
    "I2": (
        "I2 one result per call",
        "Every tool call gets exactly one result, and no tool runs twice for the same call.",
    ),
    "I3": (
        "I3 complete log",
        "The event log has no gaps, and every saved history item has its event.",
    ),
    "I5": (
        "I5 silent",
        "The loop prints nothing to stdout or stderr: it will run inside the engine process.",
    ),
    "I7": (
        "I7 one commit per turn",
        "Exactly one git commit per finished turn, matching the session log.",
    ),
    "wire": (
        "wire",
        "The exact bytes sent over the network to the model provider. The fake server records every request body (never headers).",
    ),
    "request-body": (
        "request body",
        "The JSON sent to the model API: model name, settings, the tool list and the whole conversation so far.",
    ),
    "reasoning-details": (
        "reasoning_details",
        "OpenRouter's field that carries the model's reasoning (thinking) blocks, including signatures that must be sent back unchanged.",
    ),
    "prompt-cache": (
        "prompt cache",
        "A provider reuses its work for a request that starts with exactly the same bytes as an earlier one. Cheaper and faster.",
    ),
    "byte-prefix": (
        "byte-identical prefix",
        "Earlier messages are re-sent with exactly the same bytes, not just the same meaning. Prompt caches need this.",
    ),
    "xfail": (
        "XFAIL",
        "Expected failure: a known limitation marked in advance. Reported, but not a regression.",
    ),
    "stop": (
        "stop reason",
        "Why a turn ended: end_turn (it answered), paused (waiting for an approval), max_steps (step cap), cancelled, error or budget.",
    ),
    "tokens": (
        "tokens",
        "Units of text the model reads (input) and writes (output). Cached tokens were served from the provider's prompt cache.",
    ),
    "cost-source": (
        "cost source",
        "provider: the billed cost the provider reported. estimate: from a price table. none: unknown, never guessed.",
    ),
    "byok": (
        "BYOK",
        "Bring your own key: calling an OpenAI-compatible endpoint directly instead of through OpenRouter.",
    ),
    "engine": (
        "engine",
        "The RocketRide engine: the process the agent will run inside. Everything must fit in its one Python environment.",
    ),
    "fit": (
        "dependency fit",
        "Whether a loop's packages can be installed next to every package the engine already ships, with no version conflict.",
    ),
    "crewai": (
        "crewai node",
        "A RocketRide pipeline node built on the CrewAI library. It needs openai < 3.",
    ),
    "latency": ("latency", "Wall-clock time from sending the prompt to the final answer."),
    "lane": ("lane", "A row of the replay timeline that shows one kind of activity."),
    "eager": (
        "eager tool start",
        "Starting a read-only tool as soon as its call has streamed in, while the model is still writing the rest of its answer.",
    ),
    "connection": (
        "connection reuse",
        "Sending several requests over one open network connection instead of opening a new one each time (saves a TLS handshake per step).",
    ),
    "pydantic-ai": (
        "pydantic-ai",
        "An open-source Python agent framework from the Pydantic team. Loop A uses it the way its docs recommend.",
    ),
    "mock-engine": (
        "MockEngine",
        "A stand-in for the RocketRide engine built from the real node definitions, so pipeline tools give realistic answers offline.",
    ),
}

STATUS_LABEL = {
    "pass": "PASS",
    "fail": "FAIL",
    "xfail": "XFAIL",
    "xpass": "XPASS",
    "none": "no run",
}
STATUS_TIP = {
    "pass": "Every final check of the scenario and every invariant held.",
    "fail": "At least one final check or invariant failed; the reason says which.",
    "xfail": GLOSSARY["xfail"][1],
    "xpass": "Marked as an expected failure, but it passed.",
    "none": "This loop has no result for this scenario in this run.",
}
INVARIANTS = ("I1", "I2", "I3", "I5", "I7")

# The engine dependency-fit result (scripts/engine_fit.sh); `metrics.json` may override it.
ENGINE_FIT = (
    ("our", "our_version", "fits", "needs only httpx 0.28.1 and jsonschema 4.26.0, the versions the engine already ships"),
    ("pydantic", "pydantic-ai 2.31.1", "fits", "the newest release that installs next to the engine's packages"),
    ("pydantic", "pydantic-ai ≥ 2.32", "conflict", "needs openai ≥ 3, but the engine's crewai node (crewai ≥ 1.14.1, < 2) needs openai < 3"),
)  # fmt: skip


def esc(value: Any) -> str:
    """HTML-escape any value."""
    return html.escape(str(value), quote=True)


def term(key: str, text: str | None = None) -> str:
    """A glossary term with its hover/focus definition."""
    label, definition = GLOSSARY[key]
    shown = esc(text if text is not None else label)
    return f'<span class="t" tabindex="0" data-term="{esc(label)}" data-tip="{esc(definition)}">{shown}</span>'


def fmt_int(value: Any) -> str:
    """1234 -> "1,234"; None -> "n/a"."""
    return "n/a" if value is None else f"{round(value):,}"


def fmt_ms(value: Any) -> str:
    """Milliseconds for people: "0.42 ms", "12.3 ms", "1.52 s"."""
    if value is None:
        return "n/a"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.2f} ms" if value < 10 else f"{value:.1f} ms"


def fmt_usd(value: Any) -> str:
    """Dollars with enough digits for per-request costs."""
    return "n/a" if value is None else f"${value:.4f}"


def chip(impl: str) -> str:
    """The coloured name tag of a loop: letter plus label (identity never by colour alone)."""
    info = loop_info(impl)
    return f'<span class="lp {esc(info.color)}"><i></i><b>{esc(info.letter)}</b> {esc(info.label)}</span>'


def status_badge(status: str) -> str:
    """A PASS/FAIL/XFAIL badge with an icon and a definition (never colour alone)."""
    icon = {"pass": "✓", "fail": "✕", "xfail": "≈", "xpass": "!", "none": "–"}[status]
    return (
        f'<span class="st {status}" tabindex="0" data-term="{esc(STATUS_LABEL[status])}" '
        f'data-tip="{esc(STATUS_TIP[status])}">{icon} {STATUS_LABEL[status]}</span>'
    )


# --- page context ---------------------------------------------------------------------------


@dataclass(slots=True)
class Page:
    """Everything a section needs, computed once."""

    loops: list[str]  # the loops shown, in registry order
    run: dict[str, Any] | None
    live: list[dict[str, Any]]
    metrics: dict[str, Any] | None
    cells: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)  # scn -> impl -> cell

    @property
    def summary(self) -> dict[str, Any]:
        return (self.run or {}).get("summary") or {}

    @property
    def scenarios(self) -> list[dict[str, Any]]:
        return (self.run or {}).get("scenarios") or []


def summary_cell(summary: dict[str, Any], scenario: str, impl: str) -> dict[str, Any]:
    """summary.json's matrix entry for one cell, whatever its exact shape ({} if none)."""
    matrix = summary.get("matrix")
    cell: Any = None
    if isinstance(matrix, dict):
        cell = (matrix.get(scenario) or {}).get(impl)
    elif isinstance(matrix, list):
        cell = next(
            (
                c
                for c in matrix
                if isinstance(c, dict) and c.get("scenario") == scenario and c.get("impl") == impl
            ),
            None,
        )
    if isinstance(cell, bool):
        return {"passed": cell}
    return cell if isinstance(cell, dict) else {}


def derive_reason(result: dict[str, Any] | None) -> str:
    """A one-line reason from result.json: the first failed check, or what held."""
    if not result:
        return ""
    if result.get("error"):
        return f"error: {result['error']}"
    expect = result.get("expect") or {}
    invariants = result.get("invariants") or {}
    for group in (expect, invariants):
        for name, check in group.items():
            if isinstance(check, dict) and check.get("ok") is False:
                return f"{name}: {check.get('detail', 'failed')}"
    if result.get("passed"):
        return f"all {len(expect)} checks and {len(invariants)} invariants hold"
    return "failed"


def make_cell(
    summary: dict[str, Any], scenario: str, impl: str, run: dict | None
) -> dict[str, str]:
    """`{"status", "reason"}` of one matrix cell (status: pass, fail, xfail, xpass or none)."""
    result = (run or {}).get("result") or {}
    scell = summary_cell(summary, scenario, impl)
    passed = result.get("passed", scell.get("passed"))
    flag = str(scell.get("status") or result.get("status") or "").lower()
    xfail = flag in ("xfail", "xpass") or bool(scell.get("xfail") or result.get("xfail"))
    if passed is None:
        status = "none"
    elif xfail:
        status = "xpass" if passed else "xfail"
    else:
        status = "pass" if passed else "fail"
    reason = str(scell.get("reason") or derive_reason(result) or STATUS_TIP[status])
    return {"status": status, "reason": reason.splitlines()[0][:300] if reason else ""}


# --- header ---------------------------------------------------------------------------------


def _flatten(value: Any) -> list[str]:
    """A summary's loop entry as short "name version" texts; nested dicts (the driver's
    `versions`) are flattened, and a `target` or `class` shows as itself."""
    if not isinstance(value, dict):
        return [str(value)]
    out = []
    for key, item in value.items():
        if key in ("file", "path") or item in (None, "", {}, []):
            continue
        if isinstance(item, dict):
            out += _flatten(item)
        else:
            out.append(str(item) if key in ("target", "class") else f"{key} {item}")
    return out


def header(page: Page, generated: str) -> str:
    """The run's identity: run id, git commit, loop versions, data sources."""
    summary = page.summary
    # Two shapes: {"git": {"sha", "dirty"}} or the driver's top-level git_sha and git_dirty.
    git = summary.get("git") if isinstance(summary.get("git"), dict) else {}
    sha = git.get("sha") or summary.get("git_sha") or summary.get("sha")
    dirty = git.get("dirty") if git else summary.get("git_dirty")
    facts = []
    if page.run:
        facts.append(f"scenario run <code>{esc(page.run['run_id'])}</code>")
    if sha:
        note = " (with uncommitted changes)" if dirty else ""
        facts.append(f"git <code>{esc(str(sha)[:10])}</code>{note}")
    metrics_git = ((page.metrics or {}).get("git") or {}).get("sha")
    if metrics_git:
        facts.append(f"metrics from <code>{esc(str(metrics_git)[:10])}</code>")
    facts.append(f"generated {esc(generated)}")
    loops = summary.get("loops") or summary.get("versions") or {}
    versions = []
    if isinstance(loops, dict):
        for impl in sorted(loops, key=impl_order):
            shown = ", ".join(_flatten(loops[impl]))
            versions.append(f"<li>{chip(impl)} <span class='mono small'>{esc(shown)}</span></li>")
    sources = [
        ("scenario runs", page.run is not None),
        ("metrics.json", page.metrics is not None),
        (f"{len(page.live)} live run{'s' * (len(page.live) != 1)}", bool(page.live)),
    ]
    found = " ".join(
        f'<span class="src {"on" if ok else "off"}">{"✓" if ok else "–"} {esc(name)}</span>'
        for name, ok in sources
    )
    problems = [*((page.run or {}).get("problems") or [])]
    for scenario in page.scenarios:
        for impl, run in scenario["runs"].items():
            problems += [f"{scenario['id']}/{impl}: {p}" for p in run.get("problems") or []]
    notes = ""
    if problems:
        items = "".join(f"<li>{esc(p)}</li>" for p in problems)
        notes = f'<details class="notes"><summary>{len(problems)} data notes</summary><ul>{items}</ul></details>'
    return f"""
<div class="intro">
  <p class="lede">The same agent {term("harness")} built twice: <b>A</b> on {term("pydantic-ai")},
  <b>B</b> as our own {term("loop")}, written without an agent framework. Everything else is
  {term("shared", "shared")} and identical, so every number below is about the loop only. This page
  shows evidence; it names no overall winner.</p>
  <p class="meta">{" · ".join(facts)}</p>
  {f'<ul class="versions">{"".join(versions)}</ul>' if versions else ""}
  <p class="meta">Data found: {found}</p>
  {notes}
</div>"""


# --- 1. scorecard -----------------------------------------------------------------------------


def _loc(page: Page) -> dict[str, Any]:
    return (page.metrics or {}).get("loc") or {}


def _loc_for(page: Page, impl: str) -> dict[str, Any] | None:
    loops = _loc(page).get("loops") or {}
    entry = next((v for k, v in loops.items() if impl_of_package(k) == impl), None)
    # An empty package is a loop that is not merged yet: nothing to count.
    return entry if ((entry or {}).get("total") or {}).get("code") else None


def _deps_for(page: Page, impl: str) -> dict[str, Any] | None:
    """The dependency set of `impl` pinned for the engine (not the `@latest` one)."""
    deps = (page.metrics or {}).get("deps") or {}
    name = next(
        (n for n in sorted(deps) if impl_of_package(n) == impl and "@latest" not in n), None
    )
    return None if name is None else deps[name] or {}


def _bench_for(page: Page, impl: str) -> dict[str, Any] | None:
    loops = ((page.metrics or {}).get("bench") or {}).get("loops") or {}
    return next((v for k, v in loops.items() if impl_of_package(k) == impl), None)


def _overhead(page: Page, impl: str) -> dict[str, Any]:
    """Wall-clock overhead per turn ({"p50", "p95"}, ms) from the benchmark, or {}."""
    return (((_bench_for(page, impl) or {}).get("wall_ms") or {}).get("overhead")) or {}


def passed_counts(page: Page, impl: str) -> tuple[int, int, int]:
    """(passed, xfail, total with a result) for one loop over the run's scenarios."""
    cells = [page.cells[s["id"]].get(impl) for s in page.scenarios]
    statuses = [c["status"] for c in cells if c and c["status"] != "none"]
    passed = sum(s in ("pass", "xpass") for s in statuses)
    return passed, statuses.count("xfail"), len(statuses)


_MISSING = '<span class="muted">not measured</span>'


def _score_cells(page: Page, impl: str) -> list[str]:
    """One loop's column of the scorecard, in `_SCORE_ROWS` order."""
    cells = []
    loc = _loc_for(page, impl)
    if loc:
        ported = sum((c or {}).get("code", 0) for c in (loc.get("ported") or {}).values())
        note = f' <span class="muted small">({fmt_int(ported)} ported)</span>' if ported else ""
        cells.append(f"<b>{fmt_int((loc.get('total') or {}).get('code'))}</b>{note}")
    else:
        cells.append(_MISSING)
    deps = _deps_for(page, impl)
    if deps:
        mb = deps.get("site_packages_mb", "n/a")
        cells.append(f"<b>{fmt_int(deps.get('distributions'))}</b> packages · {esc(mb)} MB")
    else:
        cells.append(_MISSING)
    cells.append(_import_cell(deps) if deps else _MISSING)
    over, error = _overhead(page, impl), (_bench_for(page, impl) or {}).get("error")
    if over.get("p50") is not None:
        p95 = f"<span class='muted small'>p95 {fmt_ms(over.get('p95'))}</span>"
        cells.append(f"<b>{fmt_ms(over['p50'])}</b> {p95}")
    elif error:
        cells.append(_error_cell(error))
    else:
        cells.append(_MISSING)
    passed, xfail, total = passed_counts(page, impl)
    xfails = f' <span class="muted small">+{xfail} {term("xfail")}</span>' if xfail else ""
    cells.append(
        f"<b>{passed} / {total}</b>{xfails}" if total else '<span class="muted">no runs</span>'
    )
    fits = [
        f'<span class="fit {esc(v)}">{"✓" if v == "fits" else "✕"} {esc(what)}</span>'
        for i, what, v, _ in _engine_fit(page)
        if i == impl
    ]
    cells.append(" · ".join(fits) or _MISSING)
    return cells


# (label, figure it links to), in the order `_score_cells` fills a column.
_SCORE_ROWS = (
    (lambda: f"Lines of its own {term('code', 'code')}", "#loc"),
    (lambda: f"{term('dependency', 'Dependencies')} installed", "#deps"),
    (lambda: f"{term('cold-import', 'Cold import')} of the loop", "#deps"),
    (lambda: f"{term('overhead', 'Harness overhead')} per turn ({term('p50')})", "#overhead"),
    (lambda: f"{term('scenario', 'Scenarios')} passed", "#matrix"),
    (lambda: term("fit", "Fits in the engine"), "#fit"),
)


def scorecard(page: Page) -> str:
    """Section 1: the headline table, then the figures behind each row."""
    if page.loops:
        columns = [_score_cells(page, impl) for impl in page.loops]
        head = "".join(f"<th>{chip(impl)}</th>" for impl in page.loops)
        body = "".join(
            f"<tr><th scope='row'>{label()} <a class='to' href='{link}' aria-label='details'>↓</a>"
            f"</th>{''.join(f'<td>{column[i]}</td>' for column in columns)}</tr>"
            for i, (label, link) in enumerate(_SCORE_ROWS)
        )
        table = f'<table class="score"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>'
    else:
        table = _empty("No loops found: no runs, no metrics and no loop package that imports.", "")
    return f"""
<p class="lede">Six numbers per loop. Hover a dotted term for a definition; the ↓ next to each row
jumps to its figure.</p>
{table}
<div class="fig" id="loc">{loc_figure(page)}</div>
<div class="grid2">
  <div class="fig" id="deps">{deps_figure(page)}</div>
  <div class="fig" id="overhead">{overhead_figure(page)}</div>
</div>
<div class="fig" id="fit">{fit_figure(page)}</div>"""


def _hbar(x: float, y: float, w: float, h: float, r: float) -> str:
    """A bar path square at its start and rounded at its data end."""
    r = min(r, w / 2, h / 2)
    return (
        f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
        f"a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z"
    )


_KINDS = (
    ("code", 1.0, "code"),
    ("comment", 0.55, "comment lines"),
    ("docstring", 0.34, "docstring lines"),
    ("blank", 0.16, "blank lines"),
)


def _stack_segments(counts: dict[str, Any], ported: int) -> list[tuple[str, float, int, bool]]:
    """(kind label, opacity, lines, hatched) segments of one bar; ported code first, hatched."""
    code = counts.get("code") or 0
    segments = []
    if ported:
        segments.append(("ported code", 1.0, ported, True))
    segments.append(("code" if not ported else "original code", 1.0, code - ported, False))
    for kind, opacity, label in _KINDS[1:]:
        segments.append((label, opacity, counts.get(kind) or 0, False))
    return [s for s in segments if s[2] > 0]


class _Bar(NamedTuple):
    label: str
    color: str  # a CSS variable
    counts: dict[str, Any]  # code / comment / docstring / blank / files
    ported: int  # code lines ported from other projects
    note: str  # where the ported lines came from


def loc_figure(page: Page) -> str:
    """Stacked horizontal bars: code / comment / docstring / blank lines per loop, ported code
    hatched, and the shared layer once in grey."""
    loc = _loc(page)
    intro = (
        "<h3>Lines of code per loop</h3><p class='small muted'>Only the loop packages are "
        f"counted (FAIRNESS.md rule 1). {term('ported', 'Ported')} lines are hatched. The "
        f"{term('shared')} is drawn once, in grey: both loops use all of it.</p>"
    )
    if not loc:
        return intro + _empty("Not measured yet.", "uv run python -m bakeoff.metrics.collect")

    def bar(label: str, color: str, entry: dict[str, Any]) -> _Bar:
        ported = {k: (v or {}).get("code", 0) for k, v in (entry.get("ported") or {}).items()}
        note = ", ".join(f"{n:,} ported from {k}" for k, n in sorted(ported.items()))
        return _Bar(label, color, entry.get("total") or {}, sum(ported.values()), note)

    bars = []
    for impl in page.loops:
        if (entry := _loc_for(page, impl)) is not None:
            info = loop_info(impl)
            bars.append(bar(f"{info.letter} {info.label}", info.color, entry))
    if shared := loc.get("shared"):
        bars.append(bar("shared (once)", "grey", shared))
    if not bars:
        return intro + _empty("No loop package has line counts yet.", "")
    lines = [sum((bar.counts.get(k) or 0) for k, *_ in _KINDS) for bar in bars]
    scale = max(lines, default=0) or 1
    width, left, plot, row_h, bar_h = 760, 130, 470, 34, 18
    svg = [
        f'<svg class="chart" viewBox="0 0 {width} {row_h * len(bars) + 8}" role="img" '
        'aria-label="Lines of code per loop">',
        _hatch_defs({bar.color for bar in bars}),
    ]
    for i, bar in enumerate(bars):
        y = 6 + i * row_h
        svg.append(
            f'<text class="lab" x="{left - 10}" y="{y + bar_h - 5}" text-anchor="end">'
            f"{esc(bar.label)}</text>"
        )
        x = float(left)
        segments = _stack_segments(bar.counts, bar.ported)
        for j, (kind, opacity, n, hatched) in enumerate(segments):
            w = n / scale * plot
            tip = f"{bar.label}: {n:,} {kind}" + (f" ({bar.note})" if hatched and bar.note else "")
            fill = f"url(#hatch-{bar.color})" if hatched else f"var(--{bar.color})"
            if j == len(segments) - 1:  # the data end is rounded
                shape = f'<path d="{_hbar(x, y, max(w, 1), bar_h, 4)}"'
            else:  # 2px of surface between touching segments
                shape = f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 0.5):.1f}" height="{bar_h}"'
            svg.append(
                f'{shape} style="fill:{fill};fill-opacity:{opacity}" data-tip="{esc(tip)}"/>'
            )
            x += w
        svg.append(
            f'<text class="val" x="{x + 8:.1f}" y="{y + bar_h - 5}">{bar.counts.get("code") or 0:,} '
            f'code<tspan class="muted"> · {lines[i]:,} lines</tspan></text>'
        )
    svg.append("</svg>")
    swatch = '<span><i class="sw" style="background:var(--ink);opacity:{}"></i>{}</span>'
    legend = (
        '<div class="legend">'
        + swatch.format(1, "code")
        + '<span><i class="sw hatch"></i>ported code</span>'
        + swatch.format(0.55, "comments")
        + swatch.format(0.34, term("docstring", "docstrings"))
        + swatch.format(0.16, "blank")
        + "</div>"
    )
    head = "".join(f"<th>{k}</th>" for k in (*(k for k, *_ in _KINDS), "ported code", "files"))
    rows = "".join(
        f"<tr><th>{esc(bar.label)}</th>"
        + "".join(f"<td>{fmt_int(bar.counts.get(k))}</td>" for k, *_ in _KINDS)
        + (
            f"<td data-tip='{esc(bar.note)}'>{fmt_int(bar.ported)}</td>"
            if bar.ported
            else "<td>–</td>"
        )
        + f"<td>{fmt_int(bar.counts.get('files'))}</td></tr>"
        for bar in bars
    )
    table = (
        '<details class="tv"><summary>Show as a table</summary><table class="num"><thead><tr>'
        f"<th></th>{head}</tr></thead><tbody>{rows}</tbody></table></details>"
    )
    return intro + legend + "".join(svg) + table + _outside_notes(page)


def _outside_notes(page: Page) -> str:
    """Code a loop keeps outside its counted package (it would not show in its bar)."""
    notes = []
    for impl in page.loops:
        entry = _loc_for(page, impl) or {}
        for name in entry.get("imports_outside") or []:
            notes.append(
                f"{chip(impl)} imports <code>{esc(name)}</code>, which is outside its counted package"
            )
        for other in entry.get("other_files") or []:
            notes.append(
                f"{chip(impl)} has a non-Python file <code>{esc(other.get('path'))}</code> ({fmt_int(other.get('lines'))} lines, not counted)"
            )
    return "".join(f'<p class="small warn-t">⚠ {n}</p>' for n in notes)


def _hatch_defs(colors: set[str]) -> str:
    patterns = "".join(
        f'<pattern id="hatch-{c}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        f'<rect width="6" height="6" style="fill:var(--{c})"/><rect width="2.4" height="6" style="fill:#fff;fill-opacity:.6"/></pattern>'
        for c in sorted(colors)
    )
    return f"<defs>{patterns}</defs>"


def _empty(text: str, command: str) -> str:
    cmd = f" Run <code>{esc(command)}</code>." if command else ""
    return f'<p class="empty">{esc(text)}{cmd}</p>'


_ALONE = "<span class='muted' data-tip='The library alone: this loop is not in the measured checkout yet.'>*</span>"


def _error_cell(error: str) -> str:
    return f'<span class="bad-t" tabindex="0" data-tip="{esc(error)}">error</span>'


def _import_cell(d: dict[str, Any]) -> str:
    """The loop's cold import time; "error" with the reason on hover; the library's alone
    (marked *) only when the loop is not in the measured checkout: metrics leaves `import_ms`
    empty without an error exactly when the package exports no loop yet."""
    if d.get("import_ms") is not None:
        return fmt_ms(d["import_ms"])
    if error := d.get("import_error"):  # a loop that is there but fails to import
        return _error_cell(error)
    if d.get("framework_import_ms") is not None:
        return f"{fmt_ms(d['framework_import_ms'])}{_ALONE}"
    if error := d.get("framework_import_error"):
        return _error_cell(error)
    return "n/a"


def _third_party_cell(d: dict[str, Any]) -> str:
    """Third-party code lines loaded after one turn; "error" if the import or the turn failed;
    the library's import alone (marked *) when the loop is not in the measured checkout."""
    code = d.get("third_party_code") or {}
    if (turn := (code.get("loop_turn") or {}).get("total")) is not None:
        return fmt_int(turn)
    if error := d.get("import_error") or d.get("turn_error"):
        return _error_cell(error)
    if (alone := (code.get("framework_import") or {}).get("total")) is not None:
        return f"{fmt_int(alone)}{_ALONE}"
    return "n/a"


def deps_figure(page: Page) -> str:
    """What each loop makes you install and load, per dependency set."""
    intro = (
        f"<h3>{term('dependency', 'Dependencies')}</h3><p class='small muted'>Each set installed "
        f"alone in a fresh virtual environment. {term('third-party-lines')} counts other people's "
        "code that actually runs. * the loop package is not in the measured checkout yet, so "
        "only its library was imported.</p>"
    )
    deps = (page.metrics or {}).get("deps") or {}
    if not deps:
        command = "uv run python -m bakeoff.metrics.collect --deps"
        return intro + _empty("Not measured yet (needs network).", command)
    rows = []
    for name in sorted(deps, key=lambda n: (impl_order(impl_of_package(n)), n)):
        d = deps[name] or {}
        installed = d.get("installed") or {}
        versions = ", ".join(
            f"{k} {installed[k]}"
            for k in ("httpx", "jsonschema", "pydantic-ai-slim", "openai", "pydantic")
            if k in installed
        )
        rows.append(
            f"<tr><th>{chip(impl_of_package(name))}<div class='mono small muted'>{esc(name)}</div></th>"
            f"<td>{fmt_int(d.get('distributions'))}</td><td>{esc(d.get('site_packages_mb', 'n/a'))}</td>"
            f"<td>{_import_cell(d)}</td><td>{_third_party_cell(d)}</td></tr>"
            f"<tr class='sub'><td colspan='5' class='mono small muted'>{esc(versions)}</td></tr>"
        )
    head = (
        f"<th>set</th><th>packages</th><th>{term('site-packages', 'MB')}</th>"
        f"<th>{term('cold-import', 'import')}</th><th>{term('third-party-lines', '3rd-party lines')}</th>"
    )
    return f"{intro}<table class='num deps'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def overhead_figure(page: Page) -> str:
    """Per-turn wall time: the raw client's share (grey) plus what each harness adds."""
    bench = (page.metrics or {}).get("bench") or {}
    intro = (
        f"<h3>{term('overhead')}</h3><p class='small muted'>One benchmark turn "
        f"({fmt_int((bench.get('config') or {}).get('chunks'))} streamed {term('chunk', 'chunks')} "
        f"and a tool call) against the local fake server: the grey part is the {term('baseline')}, "
        f"the coloured part is what the loop adds ({term('p50')}; {term('p95')} in the label).</p>"
    )
    entries = [(impl, b) for impl in page.loops if (b := _bench_for(page, impl)) is not None]
    if not entries:
        return intro + _empty("Not measured yet.", "uv run python -m bakeoff.metrics.collect")

    def baseline(b: dict[str, Any]) -> float:
        return ((b.get("wall_ms") or {}).get("baseline") or {}).get("p50") or 0.0

    totals = [
        baseline(b) + _overhead(page, impl)["p50"]
        for impl, b in entries
        if _overhead(page, impl).get("p50") is not None
    ]
    scale = max(totals, default=0.0) or 1e-9  # every entry may be an error: nothing to scale
    width, left, plot, row_h, bar_h = 520, 110, 210, 34, 16
    svg = [
        f'<svg class="chart" viewBox="0 0 {width} {row_h * len(entries) + 8}" role="img" '
        'aria-label="Harness overhead per turn">'
    ]
    notes = []
    for i, (impl, b) in enumerate(entries):
        info, y, text_y = loop_info(impl), 6 + i * row_h, 6 + i * row_h + bar_h - 4
        svg.append(
            f'<text class="lab" x="{left - 10}" y="{text_y}" text-anchor="end">'
            f"{esc(info.letter)} {esc(info.label)}</text>"
        )
        over = _overhead(page, impl)
        if over.get("p50") is None:
            error = esc((b.get("error") or "no numbers")[:80])
            svg.append(f'<text class="val bad-t" x="{left}" y="{text_y}">{error}</text>')
            continue
        base = baseline(b)
        wb, wo = base / scale * plot, max(over["p50"], 0) / scale * plot
        p50, p95 = fmt_ms(over["p50"]), fmt_ms(over.get("p95"))
        svg += [
            f'<rect x="{left}" y="{y}" width="{max(wb - 2, 0.5):.1f}" height="{bar_h}" '
            f'style="fill:#c9ced4" data-tip="raw baseline turn: {fmt_ms(base)} (p50)"/>',
            f'<path d="{_hbar(left + wb, y, max(wo, 1), bar_h, 4)}" style="fill:var(--{info.color})" '
            f'data-tip="{esc(info.label)} adds {p50} per turn (p50), {p95} (p95)"/>',
            f'<text class="val" x="{left + wb + wo + 8:.1f}" y="{text_y}">+{p50}'
            f'<tspan class="muted"> p95 {p95}</tspan></text>',
        ]
        frame = ((b.get("overhead_us_per_frame") or {}).get("wall") or {}).get("p50")
        if frame is not None:
            notes.append(
                f"{chip(impl)} {frame:.1f} µs per streamed {term('chunk')} ({term('p50')}); "
                f"the raw baseline turn takes {fmt_ms(base)}"
            )
    svg.append("</svg>")
    return intro + "".join(svg) + "".join(f'<p class="small muted">{n}</p>' for n in notes)


def _engine_fit(page: Page) -> list[tuple[str, str, str, str]]:
    """(impl, what, verdict, why): metrics' `engine_fit` if it has one, else the static result."""
    fit = (page.metrics or {}).get("engine_fit")
    if isinstance(fit, dict) and fit:
        return [
            (
                impl_of_package(name),
                name,
                str((v or {}).get("verdict", v) if isinstance(v, dict) else v),
                str((v or {}).get("why", "")) if isinstance(v, dict) else "",
            )
            for name, v in sorted(fit.items())
        ]
    return list(ENGINE_FIT)


def fit_figure(page: Page) -> str:
    """The engine dependency-fit result, in words."""
    items = "".join(
        f'<li><span class="fit {esc(verdict)}">{"✓ fits" if verdict == "fits" else "✕ conflicts"}</span> '
        f"{chip(impl)} <b>{esc(what)}</b>: {esc(why)}</li>"
        for impl, what, verdict, why in _engine_fit(page)
    )
    return (
        f"<h3>{term('fit', 'Does it fit in the engine?')}</h3>"
        f"<p class='small muted'>The {term('engine')} installs every node's requirements into one Python "
        f"environment, so a loop can only use package versions that agree with all of them. Today the "
        f"binding constraint is the {term('crewai')}. Checked by <code>scripts/engine_fit.sh</code> "
        "(uv resolves each set against the engine's pins).</p>"
        f"<ul class='fitlist'>{items}</ul>"
    )


# --- 2. scenario matrix -----------------------------------------------------------------------


def matrix(page: Page) -> str:
    """Section 2: scenario × loop, PASS/FAIL/XFAIL with the one-line reason."""
    invariants = ", ".join(term(name, name) for name in INVARIANTS)
    intro = (
        f"<p class='lede'>Every {term('scenario')} against every loop, on the {term('fakeprov')} "
        f"(it imitates {term('openrouter')}); pipeline tools answer from a {term('mock-engine')}. "
        f"The scenarios cover {term('tool-call', 'tool calls')} and their "
        f"{term('tool-result', 'results')}, {term('approval', 'approvals')}, "
        f"{term('retry', 'retries')}, {term('cancel')}, {term('crash-resume')}, "
        f"{term('compaction')} and {term('revert')}. A cell passes when every final check of "
        f"the scenario and all five {term('invariant', 'invariants')} ({invariants}) hold, "
        f"judged only from the {term('wire')} recordings and the {term('session-log')}. "
        "Click a cell to replay that scenario below.</p>"
    )
    if not page.scenarios:
        return intro + _empty("No scenario runs found.", "bakeoff scenario --all")
    head = "".join(f"<th>{chip(impl)}</th>" for impl in page.loops)
    rows = []
    for s in page.scenarios:
        cells = []
        for impl in page.loops:
            c = page.cells[s["id"]].get(impl) or {"status": "none", "reason": ""}
            result = ((s["runs"].get(impl) or {}).get("result")) or {}
            extra = ""
            if result.get("duration_ms") is not None:
                n = result.get("requests")
                extra = f"{n} request{'s' * (n != 1)} · {fmt_ms(result['duration_ms'])}"
            cells.append(
                f'<td class="cell {c["status"]}" id="cell-{esc(s["id"])}-{esc(impl)}" data-scn="{esc(s["id"])}" '
                f'data-impl="{esc(impl)}" tabindex="0" role="button" aria-label="replay {esc(s["id"])} for {esc(impl)}">'
                f'{status_badge(c["status"])}<div class="why">{esc(c["reason"])}</div>'
                f"{f'<div class=meta2>{esc(extra)}</div>' if extra else ''}</td>"
            )
        rows.append(
            f'<tr><th scope="row"><b>{esc(s["id"])}</b><div class="small muted">{esc(s["title"])}</div></th>'
            + "".join(cells)
            + "</tr>"
        )
    legend = " ".join(status_badge(s) for s in ("pass", "fail", "xfail", "none"))
    return (
        intro
        + f'<p class="small">{legend}</p><div class="scroll"><table class="matrix"><thead><tr><th>scenario</th>{head}</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


# --- 3. replay and 4. wire diff (drawn by report.js) -------------------------------------------


def _scenario_options(page: Page) -> str:
    return "".join(
        f'<option value="{esc(s["id"])}">{esc(s["id"])} · {esc(s["title"][:70])}</option>'
        for s in page.scenarios
    )


def _loop_options(impls: list[str], selected: str) -> str:
    return "".join(
        f'<option value="{esc(i)}"{" selected" if i == selected else ""}>{esc(loop_info(i).letter)} · {esc(loop_info(i).label)}</option>'
        for i in impls
    )


def replay_impls(page: Page) -> list[str]:
    """Loops the replay can show: our | pydantic always (the pair being compared), plus any other
    loop with runs."""
    return sorted({"our", "pydantic", *(page.run or {}).get("impls", [])}, key=impl_order)


def replay(page: Page) -> str:
    """Section 3's controls; report.js draws the lanes and the transcript cards."""
    intro = (
        f"<p class='lede'>Two loops, one scenario, one time axis. {term('lane', 'Lanes')} show when each "
        f"loop waits for the model during a {term('step')}, streams, runs tools, asks for an "
        f"{term('approval')} and makes a git {term('commit')} of the {term('working-copy')}. Drag the "
        "scrubber or press play; the cards below each timeline are the transcript up to that moment, "
        f"each turn ending with its {term('stop')}. Idle time between {term('turn', 'turns')} (a user "
        "deciding, a new process starting) is cut, the same way for both loops.</p>"
    )
    if not page.scenarios:
        return intro + _empty("Nothing to replay yet.", "bakeoff scenario --all")
    impls = replay_impls(page)
    return f"""{intro}
<div class="ctrl">
  <label>Scenario <select id="rp-scn">{_scenario_options(page)}</select></label>
  <label>Left <select id="rp-left">{_loop_options(impls, "our")}</select></label>
  <label>Right <select id="rp-right">{_loop_options(impls, "pydantic")}</select></label>
</div>
<div class="ctrl scrub">
  <button class="btn pri" id="rp-play" type="button">▶ Play</button>
  <button class="btn" id="rp-prev" type="button" title="previous event">◀</button>
  <button class="btn" id="rp-next" type="button" title="next event">▶</button>
  <input id="rp-range" type="range" min="0" max="1000" value="1000" aria-label="time">
  <span id="rp-time" class="mono small"></span>
</div>
<div class="rp-key small">
  <span><i class="sw wait"></i>waiting for the first token</span><span><i class="sw strm"></i>streaming</span>
  <span><i class="sw tool"></i>tool running</span><span>◆ approval asked</span><span>● commit</span>
  <span class="muted">{term("eager", "eager")} tools start before the stream ends</span>
</div>
<div class="cols" id="rp-cols"></div>
<noscript><p class="empty">The replay needs JavaScript. The scenario matrix above has the results.</p></noscript>"""


def wire(page: Page) -> str:
    """Section 4's controls; report.js decodes and diffs the recorded bodies."""
    intro = (
        f"<p class='lede'>The exact {term('request-body', 'request bodies')} each loop sent for the "
        f"scenario chosen above, side by side. Changed lines are highlighted, identical stretches "
        f"folded. Compare request 2 of S10a to see how each loop re-sends {term('reasoning-details')}; "
        f"key order and added fields matter for a {term('prompt-cache')}, which needs a "
        f"{term('byte-prefix')}. Each request also shows the network connection that carried it "
        f"({term('connection')}). Strings over {WIRE_LIMIT:,} characters are cut the same way for "
        "both loops; the header names the file with the full body.</p>"
    )
    if not page.scenarios:
        return intro + _empty("No wire recordings yet.", "bakeoff scenario --all")
    return f"""{intro}
<div class="ctrl">
  <label>Request <select id="wd-req"></select></label>
  <span class="seg" role="group" aria-label="comparison mode">
    <button type="button" data-mode="aligned" aria-pressed="true" data-tip="Top-level fields lined up in the same order; everything inside exactly as sent.">aligned</button>
    <button type="button" data-mode="sent" aria-pressed="false" data-tip="Every key in the exact order each loop sent it.">exact order</button>
    <button type="button" data-mode="sorted" aria-pressed="false" data-tip="All keys sorted: only differences in content remain.">keys sorted</button>
  </span>
  <label class="small"><input type="checkbox" id="wd-all"> show identical lines</label>
</div>
<div id="wd-sum" class="small"></div>
<div id="wd-view" class="diff"></div>
<noscript><p class="empty">The wire diff needs JavaScript.</p></noscript>"""


# --- 5. live runs -----------------------------------------------------------------------------


def live(page: Page) -> str:
    """Section 5: real-model runs, if any: prompt, answers, tokens, latency, steps."""
    intro = (
        "<p class='lede'>The same prompt sent to a real model through each loop. One run is one "
        f"sample: {term('latency')} and {term('tokens')} vary between runs.</p>"
    )
    if not page.live:
        return intro + _empty("No live runs yet.", "bakeoff live")
    out = [intro]
    for run in page.live:
        results = run["results"]
        cols = []
        for impl in sorted(results, key=impl_order):
            r = results[impl]
            usage = r.get("usage") or {}
            cost = usage.get("cost_usd")
            tools = (
                sum((r.get("tool_runs") or {}).values())
                if isinstance(r.get("tool_runs"), dict)
                else None
            )
            # `steps` counts first attempts only; `requests` adds retries (older results have
            # only `requests`, which is the same number when nothing was retried).
            steps, requests = r.get("steps", r.get("requests")), r.get("requests")
            first_token = (r.get("latency") or {}).get("ttft_ms")
            stats = [
                (term("latency"), fmt_ms(r.get("duration_ms"))),
                *([("first token", fmt_ms(first_token))] if first_token is not None else []),
                (term("step", "steps"), fmt_int(steps)),
                *(
                    [("requests (with retries)", fmt_int(requests))]
                    if requests is not None and requests != steps
                    else []
                ),
                (term("tool-run", "tool runs"), fmt_int(tools)),
                (term("tokens", "input tokens"), fmt_int(usage.get("input_tokens"))),
                (term("tokens", "output tokens"), fmt_int(usage.get("output_tokens"))),
                (term("prompt-cache", "cached tokens"), fmt_int(usage.get("cached_tokens"))),
                ("cost", _cost(cost, usage.get("cost_source"))),
            ]
            dl = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in stats)
            error = f'<p class="bad-t small">error: {esc(r["error"])}</p>' if r.get("error") else ""
            stops = ", ".join(map(str, r.get("stops") or []))
            stop = f" · {term('stop', 'stop')} {esc(stops)}" if stops else ""
            cols.append(
                f'<div class="live-col"><div class="lh">{chip(impl)} <span class="mono small muted">'
                f"{esc(r.get('model') or '')}{_endpoint(r.get('base_url'))}{stop}</span></div>"
                f'<dl class="stats">{dl}</dl>{error}<div class="answer"><div class="k">final answer</div>'
                f"{_expandable(r.get('final_text') or '', 700)}</div></div>"
            )
        out.append(
            f'<div class="fig live"><div class="small muted">live run <code>{esc(run["run_id"])}</code></div>'
            f'<div class="prompt"><span class="k">prompt</span> {_expandable(run.get("prompt") or "", 400)}</div>'
            f'<div class="live-cols">{"".join(cols)}</div></div>'
        )
    return "".join(out)


def _cost(cost: float | None, source: str | None) -> str:
    """A cost with where it came from; "unknown" when the provider reported none."""
    where = term("cost-source", source or "n/a")
    if source in (None, "none") or cost is None:
        return f"unknown <span class='muted small'>({where})</span>"
    return f"{fmt_usd(cost)} <span class='muted small'>({where})</span>"


def _endpoint(base_url: str | None) -> str:
    """ " · via OpenRouter" or " · BYOK api.openai.com": where a live run's requests went."""
    if not base_url:
        return ""
    # The host only: never a path, a query or a user:password@ part.
    host = base_url.split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0].rsplit("@", 1)[-1]
    if "openrouter" in host:
        return f" · via {term('openrouter')}"
    return f" · {term('byok')} {esc(host)}"


def _expandable(text: str, preview: int) -> str:
    """Text with a "show all" control when it is longer than `preview` characters."""
    if len(text) <= preview:
        return f'<div class="txt">{esc(text) or "<span class=muted>(empty)</span>"}</div>'
    label = f"show all ({len(text):,} characters)"
    return (
        f'<div class="txt exp"><span class="pv">{esc(text[:preview])}…</span>'
        f'<span class="full" hidden>{esc(text)}</span>'
        f'<button type="button" class="more" data-label="{esc(label)}">{esc(label)}</button></div>'
    )


# --- 6. where each wins -----------------------------------------------------------------------

TIMING_MARGIN = 0.10  # a timing is a win only when it is at least 10% better: less is noise
MIN_LIVE_SAMPLES = 2  # one live run is an anecdote: run order and prompt caches swing it


class Claim(NamedTuple):
    """One comparison in section 6. `winner` is None when it is too close to call."""

    winner: str | None
    text: str  # HTML
    link: str  # the figure or cell that is its evidence


@dataclass(slots=True)
class Evidence:
    """Counts from the scenario runs, taken only over scenarios every compared loop ran, so a
    loop never loses a count for having run fewer scenarios."""

    loops: list[str]  # the loops with at least one scenario result: the ones compared
    common: int = 0  # scenarios all of them ran
    replayed: int = 0  # ... of which every loop has a replay (eager starts are counted there)
    multi: int = 0  # ... in which every loop sent 2+ requests (a prefix can be compared)
    same_requests: int = 0  # ... in which every loop sent the same number of requests
    facts: dict[str, dict[str, int]] = field(default_factory=dict)  # impl -> counts


def evidence(page: Page) -> Evidence:
    """Per loop: passes, eager tool starts, byte-identical prefixes, and connections opened (the
    last only where every loop sent the same number of requests)."""

    def status(s: dict[str, Any], impl: str) -> str:
        return (page.cells[s["id"]].get(impl) or {}).get("status", "none")

    ran = [i for i in page.loops if any(status(s, i) != "none" for s in page.scenarios)]
    keys = ("passed", "eager", "byte_ok", "requests", "conns")
    ev = Evidence(ran, facts={i: dict.fromkeys(keys, 0) for i in ran})
    if len(ran) < 2:
        return ev
    for s in page.scenarios:
        if any(status(s, i) == "none" for i in ran):
            continue
        runs = {i: s["runs"].get(i) or {} for i in ran}
        i1 = {
            i: (((r.get("result") or {}).get("invariants") or {}).get("I1") or {}).get("info") or {}
            for i, r in runs.items()
        }
        replays = {i: r.get("replay") for i, r in runs.items()}
        wire = {i: r.get("wire") or [] for i, r in runs.items()}
        multi = all((i1[i].get("requests") or 0) >= 2 for i in ran)
        same = (
            len({len(w) for w in wire.values()}) == 1
            and all(wire.values())
            and all(r["meta"].get("conn_id") is not None for w in wire.values() for r in w)
        )
        ev.common += 1
        ev.replayed += all(replays.values())
        ev.multi += multi
        ev.same_requests += same
        for i in ran:
            facts = ev.facts[i]
            facts["passed"] += status(s, i) in ("pass", "xpass")
            if all(replays.values()):
                facts["eager"] += (replays[i].get("stats") or {}).get("eager", 0)
            if multi:
                facts["byte_ok"] += bool(i1[i].get("byte_prefix"))
            if same:
                facts["requests"] += len(wire[i])
                facts["conns"] += len({r["meta"]["conn_id"] for r in wire[i]})
    return ev


def _better(values: dict[str, Any], lower: bool = True, margin: float = 0.0) -> str | None:
    """The loop with the best value; None on a tie, with fewer than two values, or when the lead
    is under `margin` (a share of the larger of the two best values)."""
    known = sorted(((v, k) for k, v in values.items() if v is not None), reverse=not lower)
    if len(known) < 2:
        return None
    (best, winner), (second, _) = known[0], known[1]
    if best == second or abs(second - best) < margin * max(abs(best), abs(second)):
        return None
    return winner


def _vs(values: dict[str, Any], fmt: Callable[[Any], str]) -> str:
    """ "A 512 vs B 698" """
    return " vs ".join(
        f"{esc(loop_info(k).letter)} {fmt(v)}" for k, v in values.items() if v is not None
    )


def _median(values: list[Any]) -> float | None:
    known = sorted(v for v in values if v is not None)
    if not known:
        return None
    mid = len(known) // 2
    return known[mid] if len(known) % 2 else (known[mid - 1] + known[mid]) / 2


def _claims(page: Page) -> list[Claim]:
    """Every comparison the data supports: a win where one loop is better by a clear margin,
    a "too close to call" note for measured timings that are not."""
    claims: list[Claim] = []
    pct = f"{TIMING_MARGIN:.0%}"

    def compare(
        values: dict[str, Any],
        win: Callable[[dict[str, Any]], str],
        link: str,
        *,
        lower: bool = True,
        margin: float = 0.0,
        close: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        known = {k: v for k, v in values.items() if v is not None}
        if len(known) < 2:
            return
        winner = _better(known, lower, margin)
        if winner is not None and (lower or known[winner]):  # 0 eager starts wins nothing
            claims.append(Claim(winner, win(known), link))
        elif close is not None:
            claims.append(Claim(None, close(known), link))

    def per_loop(get: Callable[[str], Any]) -> dict[str, Any]:
        return {i: get(i) for i in page.loops}

    deps = per_loop(lambda i: _deps_for(page, i) or {})
    mb = per_loop(lambda i: deps[i].get("site_packages_mb"))
    compare(
        per_loop(lambda i: ((_loc_for(page, i) or {}).get("total") or {}).get("code")),
        lambda v: f"Fewer lines of its own code to maintain: {_vs(v, fmt_int)}.",
        "#loc",
    )
    compare(
        per_loop(lambda i: deps[i].get("distributions")),
        lambda v: f"Installs less: {_vs(v, fmt_int)} packages ({_vs(mb, lambda x: f'{x} MB')}).",
        "#deps",
    )
    compare(
        per_loop(lambda i: deps[i].get("import_ms")),
        lambda v: f"Starts faster in a fresh process ({term('cold-import')}): {_vs(v, fmt_ms)}.",
        "#deps",
        margin=TIMING_MARGIN,
        close=lambda v: f"{term('cold-import', 'Cold import')}: {_vs(v, fmt_ms)}, within {pct}.",
    )
    # Overhead: a win needs the median and the slow tail to agree, each by the margin.
    p50 = per_loop(lambda i: _overhead(page, i).get("p50"))
    p95 = per_loop(lambda i: _overhead(page, i).get("p95"))
    both = f"{_vs(p50, fmt_ms)} ({term('p50')}); {_vs(p95, fmt_ms)} ({term('p95')})"
    winner = _better(p50, margin=TIMING_MARGIN)
    if winner is not None and winner == _better(p95, margin=TIMING_MARGIN):
        claims.append(Claim(winner, f"Adds less time on top of the model per turn: {both}.", "#overhead"))  # fmt: skip
    elif sum(v is not None for v in p50.values()) >= 2:
        claims.append(Claim(None, f"{term('overhead', 'Harness overhead')} per turn: {both}; the median and the slow tail do not both differ by {pct}.", "#overhead"))  # fmt: skip

    ev = evidence(page)
    who = "both loops" if len(ev.loops) == 2 else "every loop"
    if ev.common:
        compare(
            {i: f["passed"] for i, f in ev.facts.items()},
            lambda v: f"Passes more scenarios: {_vs(v, str)}, of the {ev.common} {who} ran.",
            "#matrix",
            lower=False,
            close=lambda v: (
                f"Passes as many scenarios: {_vs(v, str)}, of the {ev.common} {who} ran."
            ),
        )
    if ev.replayed:
        compare(
            {i: f["eager"] for i, f in ev.facts.items()},
            lambda v: (
                f"Starts read-only tools while the model is still streaming ({term('eager', 'eager starts')}): {_vs(v, fmt_int)}, in the {ev.replayed} scenarios {who} ran."
            ),
            "#replay",
            lower=False,
        )
    if ev.multi:
        compare(
            {i: f["byte_ok"] for i, f in ev.facts.items()},
            lambda v: (
                f"Keeps a {term('byte-prefix')} in more multi-request scenarios: {_vs(v, str)}, of {ev.multi}."
            ),
            "#wire",
            lower=False,
        )
    if ev.same_requests:
        requests = next(iter(ev.facts.values()))["requests"]
        compare(
            {i: f["conns"] for i, f in ev.facts.items()},
            lambda v: (
                f"Opens fewer connections ({term('connection')}) for the same requests: {_vs(v, fmt_int)} for {requests} requests each, in the {ev.same_requests} scenarios where {who} sent the same number of requests."
            ),
            "#wire",
        )
    claims += _live_claims(page)
    for s in page.scenarios:
        cells = {i: page.cells[s["id"]].get(i) or {} for i in page.loops}
        for impl, other in itertools.product(page.loops, repeat=2):
            if cells[impl].get("status") == "pass" and cells[other].get("status") == "fail":
                why = esc(cells[other]["reason"])
                text = f"Passes {esc(s['id'])}, where {esc(loop_info(other).letter)} fails ({why})."
                claims.append(Claim(impl, text, f"#cell-{esc(s['id'])}-{esc(other)}"))
    return claims


def _answered(result: dict[str, Any] | None) -> bool:
    """The loop finished the live run: its last turn ended with end_turn and nothing failed."""
    if not result or result.get("error"):
        return False
    stops = result.get("stops") or []
    return bool(stops) and stops[-1] == "end_turn"


def _live_claims(page: Page) -> list[Claim]:
    """Latency and input tokens over the live runs every live loop answered: medians, and a win
    only with enough samples and a clear margin. A run counts only if every loop finished it
    (last stop end_turn, no error): a loop that failed at once would otherwise look fast and
    cheap."""
    loops = [i for i in page.loops if any(i in run["results"] for run in page.live)]
    samples = [
        run["results"] for run in page.live if all(_answered(run["results"].get(i)) for i in loops)
    ]
    if len(loops) < 2 or not samples:
        return []
    n = len(samples)
    runs = f"{n} live run{'s' * (n != 1)}"
    latency = {i: _median([r[i].get("duration_ms") for r in samples]) for i in loops}
    tokens = {
        i: _median([(r[i].get("usage") or {}).get("input_tokens") for r in samples]) for i in loops
    }
    if n < MIN_LIVE_SAMPLES:
        shown = (
            f"{term('latency', 'Answer time')} {_vs(latency, fmt_ms)}; "
            f"{term('tokens', 'input tokens')} {_vs(tokens, fmt_int)}"
        )
        why = "the order of runs and the provider's prompt cache swing it"
        return [Claim(None, f"{shown}: one live run, too few to call ({why}).", "#live")]
    claims = []
    for values, win, name, fmt in (
        (latency, "Answered faster", term("latency", "Answer time"), fmt_ms),
        (tokens, "Sent fewer input tokens", term("tokens", "Input tokens"), fmt_int),
    ):
        shown = f"median {_vs(values, fmt)} over {runs}"
        if (winner := _better(values, margin=TIMING_MARGIN)) is not None:
            claims.append(Claim(winner, f"{win}: {shown}.", "#live"))
        else:
            claims.append(Claim(None, f"{name}: {shown}, within {TIMING_MARGIN:.0%}.", "#live"))
    return claims


# Properties each side has by design, not measured on this page (FAIRNESS.md rule 5: A's too).
DESIGN_WINS = {
    "pydantic": (
        f"Retries, usage limits, {term('approval', 'approvals')} (deferred tools), cancellation and the message history format come from the library: fixes and new features arrive with upgrades, and there is less of our own code to review.",
        "One typed interface to many model providers (OpenAI, Anthropic, Gemini and more), should the engine need more than OpenAI-compatible endpoints.",
        "Tool arguments are validated by the library, and a bad call gets the library's own retry prompt.",
    ),
    "our": (
        f"Needs no package the engine does not already ship, so it fits whatever pydantic-ai does next (see {term('fit', 'engine fit')}).",
        f"Sends {term('reasoning-details')} back exactly as received and builds each request from cached bytes (DESIGN.md), which is what a {term('prompt-cache')} needs.",
        "No library behaviour to work around: A_CHECKLIST.md lists the workarounds A needed (OpenRouter's string error code, retrying a stream that fails midway).",
    ),
}  # fmt: skip


def wins(page: Page) -> str:
    """Section 6: measured wins per loop (each linked to its evidence), then design properties
    that are not measured here, for both sides. No overall verdict (FAIRNESS.md rule 7)."""
    claims = _claims(page)
    cards = []
    for impl in page.loops:
        measured = "".join(
            f'<li>{c.text} <a href="{c.link}" class="ev">evidence</a></li>'
            for c in claims
            if c.winner == impl
        )
        measured = (
            measured or '<li class="muted">No measured advantage in the data on this page.</li>'
        )
        design = "".join(f"<li>{text}</li>" for text in DESIGN_WINS.get(impl, ()))
        if design:
            design = (
                f'<div class="k">by design (not measured here)</div><ul class="ev">{design}</ul>'
            )
        cards.append(
            f'<div class="win {esc(loop_info(impl).color)}"><h3>{chip(impl)} wins on</h3>'
            f'<div class="k">measured</div><ul class="ev">{measured}</ul>{design}</div>'
        )
    close = "".join(
        f'<li>{c.text} <a href="{c.link}" class="ev">evidence</a></li>'
        for c in claims
        if c.winner is None
    )
    if close:
        close = f'<div class="fig close"><div class="k">too close to call</div><ul class="ev">{close}</ul></div>'
    sources = " · ".join(
        f'<a href="{REPO_URL}/{name}">{name}</a>'
        for name in ("FAIRNESS.md", "A_CHECKLIST.md", "PREDICTIONS.md", "DESIGN.md")
    )
    return (
        "<p class='lede'>Only what the data on this page shows, plus a few design properties for "
        "each side, labelled as such. Counts compare only the scenarios every loop ran; a timing "
        f"counts only when it is at least {TIMING_MARGIN:.0%} better, and a live result only over "
        f"{MIN_LIVE_SAMPLES} or more runs. Scenarios that favour A (cancel repair, built-in "
        "features) are in the matrix. People decide; this page names no overall winner.</p>"
        f'<div class="wins">{"".join(cards)}</div>{close}'
        f'<p class="small muted">Sources: {sources}</p>'
    )


def glossary() -> str:
    """Every term the page defines, as a list (the hover texts, readable in one place)."""
    items = "".join(
        f"<div><dt>{esc(label)}</dt><dd>{esc(text)}</dd></div>"
        for label, text in sorted(GLOSSARY.values(), key=lambda v: v[0].lower())
    )
    return f'<details><summary>Glossary: every term on this page</summary><dl class="gloss">{items}</dl></details>'


_JS_TERMS = ("stop", "step", "tokens", "tool-result", "approval", "retry", "crash-resume",
             "commit", "turn", "byte-prefix")  # fmt: skip


def js_data(page: Page) -> dict[str, Any]:
    """The data report.js draws the replay and the wire diff from."""
    scenarios = []
    for s in page.scenarios:
        runs = {}
        for impl, run in s["runs"].items():
            result = run.get("result") or {}
            runs[impl] = {
                **page.cells[s["id"]].get(impl, {"status": "none", "reason": ""}),
                "stops": result.get("stops"),
                "error": result.get("error"),
                "expect": {
                    k: {"ok": v.get("ok"), "detail": v.get("detail")}
                    for k, v in (result.get("expect") or {}).items()
                    if isinstance(v, dict)
                },
                "invariants": {
                    k: {"ok": v.get("ok"), "detail": v.get("detail")}
                    for k, v in (result.get("invariants") or {}).items()
                },
                "byte_prefix": (
                    ((result.get("invariants") or {}).get("I1") or {}).get("info") or {}
                ).get("byte_prefix"),
                "replay": run.get("replay"),
                "wire": run.get("wire") or [],
            }
        scenarios.append({"id": s["id"], "title": s["title"], "runs": runs})
    return {
        "loops": {
            impl: {
                "letter": loop_info(impl).letter,
                "label": loop_info(impl).label,
                "color": loop_info(impl).color,
            }
            for impl in replay_impls(page)
        },
        # (label, definition) of the terms report.js draws itself.
        "terms": {k: GLOSSARY[k] for k in (*INVARIANTS, *_JS_TERMS)},
        "scenarios": scenarios,
        "pool": (page.run or {}).get("pool") or [],
    }


def embed_json(data: Any) -> str:
    """JSON that is safe inside a <script> element (no "</script>", no comment openers)."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

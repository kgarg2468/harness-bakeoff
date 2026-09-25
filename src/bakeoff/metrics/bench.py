"""Harness overhead: what a loop adds on top of reading the model's stream, model time excluded.

The real fake provider (`python -m bakeoff.fakeprov`) serves a generated scenario on 127.0.0.1
with zero delay. Each turn is a tool step (`chunks` text chunks, then one `list_files` call) and
a one-chunk final answer. A loop runs `turns` measured turns, after `warmup` discarded ones,
with a no-op ToolHost. Every turn is the first turn of a new thread, so all turns do the same
work.

After each loop turn, the baseline sends the same request bodies (read back from the server's
wire recording) with a bare pooled `httpx.AsyncClient` and only reads the SSE lines. The server
does the same work for both, so the paired difference (loop turn minus baseline turn) is the
harness's own cost: JSON parsing, events, tool dispatch and request building.

Both wall time and client CPU time are measured. With zero delay the baseline is bound by the
server's speed, and that can hide some of the loop's work in wall time. CPU time is not bound by
the server, so the regression check uses it.

The server runs in its own process on purpose. A server thread shares the GIL with the loop
under test, and on a 16-core laptop that made every frame 3-4x slower for the loop and the
baseline alike, with p95 overhead noise of about 40 us per frame.

Reported per loop: p50/p95 of turn time (loop, baseline, overhead) in wall and CPU time,
overhead per SSE frame, events per second, and peak RSS growth. RSS is only meaningful in a
fresh process, so the command benchmarks each loop in its own subprocess. `--quick` is the CI
mode: few turns and a generous ceiling that fails only on large regressions. Unit tests assert
sanity, never timing.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import math
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from bakeoff.metrics import LOOP_PACKAGES
from bakeoff.shared import netguard
from bakeoff.shared.contract import (
    Decision,
    Item,
    Limits,
    Loop,
    ModelConfig,
    ToolCall,
    ToolResult,
    ToolSpec,
    TurnInput,
)
from bakeoff.shared.toolhost import TOOLS

SCENARIO = "BENCH"
MODEL = "bench/model"
WORD = "lorem "  # one text chunk: 2000 chunks make a 12 kB answer
FINAL = "Done."
SYSTEM = "You are Rocket Agent. You build and fix RocketRide pipelines."
# CPU overhead per SSE frame (p50, microseconds) above which --quick fails. A laptop measures
# about 8 for our_version and 250 for pydantic_version; the ceilings are 6-10x that, so only a
# large regression trips them on a slow, noisy CI runner.
MAX_OVERHEAD_US = {"our_version": 50.0}
DEFAULT_MAX_OVERHEAD_US = 2500.0


class BenchError(RuntimeError):
    """A loop did not complete a benchmark turn the way the script requires."""


@dataclass(frozen=True, slots=True)
class Config:
    """How much to run: measured turns, discarded warm-up turns, text chunks per turn."""

    turns: int
    warmup: int
    chunks: int


QUICK = Config(turns=20, warmup=2, chunks=500)
FULL = Config(turns=100, warmup=5, chunks=2000)


class NoopTools:
    """A ToolHost whose tools do nothing. The specs are the real ones, so request bodies have a
    realistic size."""

    def specs(self) -> list[ToolSpec]:
        """The shared tool specs, in the usual order."""
        return [tool.spec for tool in TOOLS]

    def check(self, call: ToolCall) -> Decision:
        """Everything is allowed."""
        return "allow"

    async def run(self, call: ToolCall) -> ToolResult:
        """Return at once."""
        return ToolResult(call.id, True, "ok")


def scenario(turns: int, chunks: int) -> dict[str, Any]:
    """A fake-provider scenario with the two exchanges of a benchmark turn, `turns` times."""
    usage = {"usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.001}}
    text = {"text": WORD * chunks, "chunks": chunks}
    answer = {"respond": {"stream": [{"text": FINAL}, {"finish": "stop"}, usage]}}
    exchanges: list[dict[str, Any]] = []
    for n in range(1, turns + 1):  # the scenario loader wants unique call ids
        call = {"id": f"call_{SCENARIO}_{n}", "name": "list_files", "arguments": "{}"}
        stream = [text, {"tool_calls": [call]}, {"finish": "tool_calls"}, usage]
        exchanges += [{"respond": {"stream": stream}}, answer]
    return {
        "id": SCENARIO,
        "title": "harness overhead benchmark (generated)",
        "system": SYSTEM,
        "model": {"kind": "openrouter", "model": MODEL},
        "rules": {"*": "allow"},
        "limits": {"max_steps": 4},
        "engine": {"delay_ms": 0},
        "driver": [{"user": "bench"}],
        "exchanges": exchanges,
        "expect": {"stops": ["end_turn"]},
    }


def discover_one(package: str) -> type[Loop] | None:
    """The loop class exported (in `__all__`) by `bakeoff.<package>`, or None when the package is
    missing, still empty, or its dependencies are not installed."""
    try:
        module = importlib.import_module(f"bakeoff.{package}")
    except ImportError:
        return None
    for name in getattr(module, "__all__", ()):
        if hasattr(cls := getattr(module, name), "run_turn"):
            return cls
    return None


def discover() -> dict[str, type[Loop]]:
    """The loop classes that import in this environment, by package name."""
    return {p: cls for p in LOOP_PACKAGES if (cls := discover_one(p)) is not None}


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile, interpolating linearly between closest ranks (numpy's default)."""
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    low = math.floor(pos)
    high = min(low + 1, len(xs) - 1)
    return xs[low] + (xs[high] - xs[low]) * (pos - low)


def _spread(values: list[float], scale: float) -> dict[str, float]:
    return {f"p{q}": round(percentile(values, q) * scale, 3) for q in (50, 95)}


def _rss_mb() -> float:
    """Current resident set size (Linux). Elsewhere the peak so far, which is coarser: a spike
    before the start mark hides later growth."""
    with contextlib.suppress(OSError):
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**20 if sys.platform == "darwin" else 2**10)  # bytes on macOS, else KiB


@contextlib.contextmanager
def fake_provider(scenarios: Path, wire: Path) -> Iterator[int]:
    """Serve `scenarios` with `python -m bakeoff.fakeprov` in a child process; yield its port."""
    command = [sys.executable, "-u", "-m", "bakeoff.fakeprov", "--port", "0"]
    command += ["--scenarios", str(scenarios), "--wire-dir", str(wire)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()  # "serving http://127.0.0.1:<port>/s/..."
        if (match := re.search(r"127\.0\.0\.1:(\d+)/", line)) is None:
            raise BenchError(f"the fake provider did not start: {line!r}")
        yield int(match[1])
    finally:
        proc.send_signal(signal.SIGINT)  # its main() stops the server cleanly on Ctrl-C
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


@dataclass(frozen=True, slots=True)
class _Sample:
    wall: float  # seconds
    cpu: float  # seconds of this process's CPU time
    count: int  # loop: events yielded; baseline: SSE data frames read


async def _loop_turn(
    loop: Loop, n: int, base_url: str, tools: NoopTools, expected_text: int
) -> _Sample:
    """Run one turn to its end. Raises unless it ended cleanly with every text chunk seen."""
    user = Item(uuid.uuid4().hex, f"turn-{n}", {"role": "user", "content": "bench"})
    turn = TurnInput(
        thread_id=f"bench-{n}",
        turn_id=f"turn-{n}",
        system=SYSTEM,
        history=[user],
        resume=None,
        limits=Limits(max_steps=4),
        model=ModelConfig(base_url=base_url, model=MODEL),
    )
    events, text, end = 0, 0, None
    wall, cpu = time.perf_counter(), time.process_time()
    async for event in loop.run_turn(turn, tools, asyncio.Event()):
        events += 1
        if event.type == "text.delta":
            text += len(event.data["text"])
        elif event.type == "turn.end":
            end = event.data
    sample = _Sample(time.perf_counter() - wall, time.process_time() - cpu, events)
    if end is None or end.get("stop") != "end_turn":
        raise BenchError(f"turn {n} ended with {end}")
    if text != expected_text:
        raise BenchError(f"turn {n} streamed {text} text characters, not {expected_text}")
    return sample


async def _raw_turn(client: httpx.AsyncClient, url: str, bodies: list[bytes]) -> _Sample:
    """Send the bodies in order and read every SSE line, nothing more."""
    frames = 0
    headers = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}
    wall, cpu = time.perf_counter(), time.process_time()
    for body in bodies:
        async with client.stream("POST", url, content=body, headers=headers) as response:
            if response.status_code != 200:
                raise BenchError(f"baseline request failed: HTTP {response.status_code}")
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    frames += 1
    return _Sample(time.perf_counter() - wall, time.process_time() - cpu, frames)


async def bench_loop(package: str, config: Config, workdir: Path) -> dict[str, Any]:
    """Benchmark one loop package; the scenario and wire recordings go under `workdir`."""
    if config.turns < 1 or config.warmup < 0 or config.chunks < 1:
        raise ValueError(f"nothing to measure with {config}")
    rss_start = _rss_mb()
    loop_cls = discover_one(package)
    if loop_cls is None:
        raise BenchError(f"bakeoff.{package} has no importable loop")
    total = config.warmup + config.turns
    scenarios = workdir / "scenarios"
    scenarios.mkdir(parents=True, exist_ok=True)
    (scenarios / f"{SCENARIO}.json").write_text(json.dumps(scenario(total, config.chunks)))
    expected_text = len(WORD) * config.chunks + len(FINAL)
    loop, tools = loop_cls(), NoopTools()
    pairs: list[tuple[_Sample, _Sample]] = []
    rss: list[float] = []  # after each turn, outside the timed parts
    try:
        with fake_provider(scenarios, workdir / "wire") as port:
            base = f"http://127.0.0.1:{port}/s/{SCENARIO}"
            wire = workdir / "wire" / SCENARIO / "loop" / package
            async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
                for n in range(total):
                    turn = await _loop_turn(
                        loop, n, f"{base}/loop/{package}/v1", tools, expected_text
                    )
                    # Both cursors walk the same exchanges, so each turn must send exactly two.
                    if (wire / f"{2 * n + 3:03d}.json").exists():
                        raise BenchError(f"turn {n} sent more than 2 requests")
                    bodies = [(wire / f"{2 * n + i:03d}.json").read_bytes() for i in (1, 2)]
                    raw = await _raw_turn(
                        client, f"{base}/raw/{package}/v1/chat/completions", bodies
                    )
                    if n >= config.warmup:
                        pairs.append((turn, raw))
                    rss.append(_rss_mb())
    finally:
        await loop.aclose()
    rss_warm = rss[config.warmup - 1] if config.warmup else rss_start
    frames = pairs[-1][1].count
    result: dict[str, Any] = {
        "frames_per_turn": frames,
        "events_per_turn": round(percentile([t.count for t, _ in pairs], 50)),
    }
    per_frame = {}
    for clock in ("wall", "cpu"):
        loop_s = [getattr(t, clock) for t, _ in pairs]
        raw_s = [getattr(r, clock) for _, r in pairs]
        over = [a - b for a, b in zip(loop_s, raw_s, strict=True)]  # paired: same turn
        result[f"{clock}_ms"] = {
            "loop": _spread(loop_s, 1e3),
            "baseline": _spread(raw_s, 1e3),
            "overhead": _spread(over, 1e3),
        }
        per_frame[clock] = _spread(over, 1e6 / frames)
    return result | {
        "overhead_us_per_frame": per_frame,
        "events_per_s": round(percentile([t.count / t.wall for t, _ in pairs], 50)),
        # From before the loop was imported: import, warm-up and measured turns.
        "peak_rss_delta_mb": round(max(rss) - rss_start, 1),
        # Measured turns only: a number that grows with --turns is a leak.
        "turns_rss_delta_mb": round(max(rss[config.warmup :]) - rss_warm, 1),
    }


def _run_here(package: str, config: Config) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="bakeoff-bench-") as tmp:
        return asyncio.run(bench_loop(package, config, Path(tmp)))


def _run_isolated(package: str, config: Config) -> dict[str, Any]:
    """Run `bench_loop` in a fresh interpreter, so imports and RSS start from nothing."""
    command = [sys.executable, "-m", "bakeoff.metrics.bench", "--no-isolate", "--json"]
    command += ["--loop", package, "--turns", str(config.turns)]
    command += ["--warmup", str(config.warmup), "--chunks", str(config.chunks)]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise BenchError(f"{package}: benchmark process failed:\n{proc.stderr.strip()}")
    # The report is the last stdout line, whatever a library may have printed before it.
    return json.loads(proc.stdout.strip().splitlines()[-1])["loops"][package]


def measure(
    config: Config, packages: list[str] | None = None, *, isolate: bool = True
) -> dict[str, Any]:
    """Benchmark each loop package (default: every loop that imports here)."""
    names = list(discover()) if packages is None else packages
    run = _run_isolated if isolate else _run_here
    return {"config": asdict(config), "loops": {name: run(name, config) for name in names}}


def regressions(report: dict[str, Any], ceiling_us: float | None = None) -> list[str]:
    """Loops whose p50 CPU overhead per frame exceeds their ceiling (`ceiling_us` for all)."""
    failures = []
    for name, result in report["loops"].items():
        limit = MAX_OVERHEAD_US.get(name, DEFAULT_MAX_OVERHEAD_US)
        limit = limit if ceiling_us is None else ceiling_us
        got = result["overhead_us_per_frame"]["cpu"]["p50"]
        if got > limit:
            failures.append(f"{name}: CPU overhead {got} us per frame (p50) > {limit}")
    return failures


def _row(label: str, wall: dict[str, float], cpu: dict[str, float]) -> str:
    cells = (wall["p50"], wall["p95"], cpu["p50"], cpu["p95"])
    return f"  {label:<28}" + "".join(f"{c:>12.3f}" for c in cells)


def summary(report: dict[str, Any]) -> str:
    """A short human-readable block per loop."""
    config = report["config"]
    heads = "".join(f"{h:>12}" for h in ("wall p50", "wall p95", "cpu p50", "cpu p95"))
    out = []
    for name, r in report["loops"].items():
        wall, cpu = r["wall_ms"], r["cpu_ms"]
        out += [
            f"{name}: {config['turns']} turns (+{config['warmup']} warm-up), per turn "
            f"{r['frames_per_turn']} SSE frames and {r['events_per_turn']} loop events",
            f"  {'':<28}{heads}",
            _row("loop turn (ms)", wall["loop"], cpu["loop"]),
            _row("raw httpx baseline (ms)", wall["baseline"], cpu["baseline"]),
            _row("harness overhead (ms)", wall["overhead"], cpu["overhead"]),
            _row("overhead per frame (us)", *r["overhead_us_per_frame"].values()),
            f"  {'events per second (p50)':<28}{r['events_per_s']:>12}",
            f"  {'peak RSS growth':<28}{r['peak_rss_delta_mb']:>+12.1f} MB "
            f"(measured turns only: {r['turns_rss_delta_mb']:+.1f} MB)",
        ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Benchmark the importable loops; with --quick, fail on large regressions."""
    parser = argparse.ArgumentParser(prog="python -m bakeoff.metrics.bench", description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="CI mode: small N, fail on regressions"
    )
    parser.add_argument("--loop", action="append", help="loop package (repeatable; default: all)")
    parser.add_argument("--turns", type=int, help="measured turns per loop")
    parser.add_argument("--warmup", type=int, help="discarded warm-up turns")
    parser.add_argument("--chunks", type=int, help="text chunks in each turn's tool step")
    parser.add_argument("--max-overhead-us", type=float, help="CPU overhead ceiling per frame")
    parser.add_argument("--no-isolate", action="store_true", help="no fresh process per loop")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    netguard.install()  # I6: the benchmark only ever talks to 127.0.0.1
    base = QUICK if args.quick else FULL
    config = Config(
        turns=args.turns or base.turns,
        warmup=base.warmup if args.warmup is None else args.warmup,
        chunks=args.chunks or base.chunks,
    )
    report = measure(config, args.loop, isolate=not args.no_isolate)
    print(json.dumps(report) if args.json else summary(report))
    if args.quick or args.max_overhead_us is not None:
        for failure in (failures := regressions(report, args.max_overhead_us)):
            print(f"REGRESSION {failure}", file=sys.stderr)
        return int(bool(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

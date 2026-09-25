"""The overhead benchmark: scenario, statistics, and a tiny real run on our_version.

Timing is never asserted beyond sanity: CI machines are too noisy for thresholds here."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from bakeoff.fakeprov.script import load_scenario
from bakeoff.metrics import bench
from bakeoff.shared.contract import Event, ToolHost, TurnInput

TINY = bench.Config(turns=3, warmup=1, chunks=40)


def test_percentile() -> None:
    assert bench.percentile([5.0], 95) == 5.0
    assert bench.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert bench.percentile([4.0, 1.0, 3.0, 2.0], 0) == 1.0
    assert bench.percentile(list(map(float, range(101))), 95) == 95.0


def test_scenario_passes_the_fake_provider_loader(tmp_path: Path) -> None:
    path = tmp_path / f"{bench.SCENARIO}.json"
    path.write_text(json.dumps(bench.scenario(turns=3, chunks=10)))
    loaded = load_scenario(path)
    assert len(loaded.exchanges) == 6
    first = loaded.exchanges[0]["respond"]["stream"]
    assert first[0] == {"text": bench.WORD * 10, "chunks": 10}


def test_discover_finds_our_version() -> None:
    loops = bench.discover()
    assert "our_version" in loops
    assert bench.discover_one("no_such_package") is None


def test_quick_run_on_our_version_gives_sane_numbers() -> None:
    report = bench.measure(TINY, ["our_version"], isolate=False)
    assert report["config"] == {"turns": 3, "warmup": 1, "chunks": 40}
    result = report["loops"]["our_version"]
    # 40 text chunks, tool call head + arguments, finish, usage, [DONE]; then text, finish,
    # usage, [DONE] for the answer.
    assert result["frames_per_turn"] == TINY.chunks + 9
    assert result["events_per_turn"] >= TINY.chunks
    for clock in ("wall_ms", "cpu_ms"):
        for part in ("loop", "baseline"):
            spread = result[clock][part]
            assert 0 < spread["p50"] <= spread["p95"] < 10_000, (clock, part)
    assert result["events_per_s"] > 0
    assert set(result["overhead_us_per_frame"]) == {"wall", "cpu"}
    assert "our_version" in bench.summary(report)


class _FailingLoop:
    name = "failing"

    async def run_turn(
        self, turn: TurnInput, tools: ToolHost, cancel: asyncio.Event
    ) -> AsyncIterator[Event]:
        yield Event("turn.end", {"stop": "error", "steps": 0, "error": "boom"})

    async def aclose(self) -> None:
        pass


async def test_a_failing_loop_gets_no_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "discover_one", lambda package: _FailingLoop)
    with pytest.raises(bench.BenchError, match=r"turn 0 ended with .*boom"):
        await bench.bench_loop("failing", TINY, tmp_path)


def test_regressions() -> None:
    def report(us: float) -> dict:
        return {"loops": {"our_version": {"overhead_us_per_frame": {"cpu": {"p50": us}}}}}

    assert bench.regressions(report(5.0)) == []
    assert bench.regressions(report(500.0)) == [
        "our_version: CPU overhead 500.0 us per frame (p50) > 50.0"
    ]
    assert bench.regressions(report(5.0), ceiling_us=1.0) != []

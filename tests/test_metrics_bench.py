"""The overhead benchmark: scenario, statistics, and a tiny real run on our_version.

Timing is never asserted beyond sanity: CI machines are too noisy for thresholds here."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

import bakeoff
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
    loops, errors = bench.discover()
    assert "our_version" in loops and "our_version" not in errors
    assert bench.discover_one("no_such_package") is None


@pytest.fixture
def extra_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """`bakeoff.bench_broken` (exists, import fails) and `bakeoff.bench_empty` (no loop yet)."""
    packages = {
        "bench_broken": "from bakeoff.shared.contract import NoSuchNameAfterUpgrade\n",
        "bench_empty": '"""Nothing here yet."""\n',
    }
    for name, source in packages.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(source)
    monkeypatch.setattr(bakeoff, "__path__", [*bakeoff.__path__, str(tmp_path)])
    importlib.invalidate_caches()
    yield
    for name in packages:
        sys.modules.pop(f"bakeoff.{name}", None)
        with contextlib.suppress(AttributeError):
            delattr(bakeoff, name)


@pytest.mark.usefixtures("extra_packages")
def test_a_broken_loop_package_is_an_error_not_a_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    assert bench.discover_one("bench_empty") is None
    with pytest.raises(bench.LoopImportError, match=r"bench_broken does not import: ImportError"):
        bench.discover_one("bench_broken")
    monkeypatch.setattr(bench, "LOOP_PACKAGES", ("bench_broken", "bench_empty", "bench_nope"))
    assert bench.discover()[0] == {}
    assert set(bench.discover()[1]) == {"bench_broken"}
    report = bench.measure(TINY, isolate=False)
    assert list(report["loops"]) == ["bench_broken"]
    assert "NoSuchNameAfterUpgrade" in report["loops"]["bench_broken"]["error"]
    assert bench.failed_loops(report) == [
        f"bench_broken: {report['loops']['bench_broken']['error']}"
    ]
    assert bench.regressions(report) == []
    assert "bench_broken: ERROR" in bench.summary(report)


def test_quick_run_on_our_version_gives_sane_numbers() -> None:
    report = bench.measure(TINY, ["our_version"], isolate=False)
    assert report["config"] == {"turns": 3, "warmup": 1, "chunks": 40}
    assert report["environment"]["cpu_count"] >= 1
    assert report["environment"]["versions"]["httpx"] == "0.28.1"
    result = report["loops"]["our_version"]
    assert result["loop_class"] == "bakeoff.our_version.loop.OurLoop"
    # 40 text chunks, tool call head + arguments, finish, usage, [DONE]; then text, finish,
    # usage, [DONE] for the answer.
    assert result["frames_per_turn"] == TINY.chunks + 9
    assert result["events_per_turn"] >= TINY.chunks
    for clock in ("wall_ms", "cpu_ms"):
        for part in ("loop", "baseline"):
            spread = result[clock][part]
            assert 0 < spread["p50"] <= spread["p95"] < 10_000, (clock, part)
    assert result["events_per_s"]["wall"] > 0 and result["events_per_s"]["cpu"] > 0
    assert set(result["overhead_us_per_frame"]) == {"wall", "cpu"}
    assert result["peak_rss_delta_mb"] >= 0  # a high-water mark never ends below its start
    assert "our_version" in bench.summary(report)


def test_command_runs_each_loop_in_a_fresh_process(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI path CI uses: an isolated child per loop, then the regression gate."""
    argv = ["--loop", "our_version", "--turns", "2", "--warmup", "0", "--chunks", "20", "--json"]
    assert bench.main([*argv, "--max-overhead-us=-1000000"]) == 1  # every loop is over that
    out, err = capsys.readouterr()
    result = json.loads(out)["loops"]["our_version"]
    assert result["frames_per_turn"] == 20 + 9
    assert "REGRESSION our_version: CPU overhead" in err


def test_command_rejects_an_empty_config() -> None:
    with pytest.raises(SystemExit):
        bench.main(["--turns", "0"])


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


def test_rss_is_optional_where_the_platform_cannot_measure_it(monkeypatch):
    """Windows has no `resource` and no /proc: RSS numbers become None, never a crash."""
    from pathlib import Path

    from bakeoff.metrics import bench

    monkeypatch.setattr(bench, "resource", None)
    real = Path.read_text

    def no_proc(self, *a, **k):
        if str(self).startswith("/proc/"):
            raise OSError("no /proc here")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", no_proc)
    assert bench._rss_mb() is None and bench._peak_rss_mb() is None
    assert bench._delta(None, 1.0) is None and bench._max_or_none([1.0, None]) is None
    assert "n/a" in bench._rss_row(None, None)


def test_an_in_process_loop_crash_is_recorded_not_raised(monkeypatch):
    from bakeoff.metrics import bench

    def boom(package, config):
        raise FileNotFoundError("wire/003.json")

    monkeypatch.setattr(bench, "_run_here", boom)
    report = bench.measure(
        bench.Config(turns=1, warmup=0, chunks=10), ["our_version"], isolate=False
    )
    assert report["loops"]["our_version"] == {"error": "FileNotFoundError: wire/003.json"}

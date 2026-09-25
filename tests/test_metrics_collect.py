"""`collect` writes one versioned JSON file for the report."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bakeoff.metrics import REPO_ROOT, bench, collect


@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="not a git checkout")
def test_git_state() -> None:
    state = collect.git_state(REPO_ROOT)
    assert re.fullmatch(r"[0-9a-f]{40}", state["sha"])
    assert isinstance(state["dirty"], bool)


def test_git_state_outside_a_repository(tmp_path: Path) -> None:
    assert collect.git_state(tmp_path) == {"sha": None, "dirty": None}


def test_git_state_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    assert collect.git_state(REPO_ROOT) == {"sha": None, "dirty": None}


def test_main_writes_metrics_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "metrics.json"
    assert collect.main(["--bench", "none", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["schema"] == collect.SCHEMA_VERSION
    assert report["git"]["sha"] == collect.git_state(REPO_ROOT)["sha"]
    assert report["loc"]["loops"]["our_version"]["total"]["code"] > 0
    assert report["bench"] is None and report["deps"] is None
    assert f"wrote {out}" in capsys.readouterr().out


def test_a_loop_without_bench_numbers_fails_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def measure(config: bench.Config) -> dict:
        error = "bakeoff.pydantic_version does not import: ImportError: boom"
        loops = {"pydantic_version": {"error": error}}
        return {"config": {}, "environment": bench.environment(), "loops": loops}

    monkeypatch.setattr(bench, "measure", measure)
    out = tmp_path / "metrics.json"
    assert collect.main(["--out", str(out)]) == 1
    assert json.loads(out.read_text())["bench"]["loops"]["pydantic_version"]["error"].endswith(
        "boom"
    )
    assert (
        "ERROR pydantic_version: bakeoff.pydantic_version does not import"
        in capsys.readouterr().err
    )

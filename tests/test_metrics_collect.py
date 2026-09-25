"""`collect` writes one versioned JSON file for the report."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bakeoff.metrics import REPO_ROOT, collect


def test_git_state() -> None:
    state = collect.git_state(REPO_ROOT)
    assert re.fullmatch(r"[0-9a-f]{40}", state["sha"])
    assert isinstance(state["dirty"], bool)


def test_git_state_outside_a_repository(tmp_path: Path) -> None:
    assert collect.git_state(tmp_path) == {"sha": None, "dirty": None}


def test_main_writes_metrics_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "metrics.json"
    assert collect.main(["--bench", "none", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["schema"] == collect.SCHEMA_VERSION
    assert report["git"]["sha"] == collect.git_state(REPO_ROOT)["sha"]
    assert report["loc"]["loops"]["our_version"]["total"]["code"] > 0
    assert report["bench"] is None and report["deps"] is None
    assert f"wrote {out}" in capsys.readouterr().out

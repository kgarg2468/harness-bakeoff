"""Dependency footprint: parsing and aggregation from fixtures (the venv builds need PyPI)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from bakeoff.metrics import REPO_ROOT, bench, deps

PYPROJECT = """
[project]
name = "fixture"
dependencies = ["httpx==0.28.1", "jsonschema==4.26.0"]

[project.optional-dependencies]
pydantic = ["pydantic-ai-slim[openai,openrouter]==2.31.1", "openai==2.54.0"]
"""

# `uv pip list --format json`, abridged, in uv's order and spelling.
LISTING = json.dumps(
    [
        {"name": "annotated-types", "version": "0.7.0"},
        {"name": "httpx", "version": "0.28.1"},
        {"name": "Jsonschema", "version": "4.26.0"},
        {"name": "pydantic_ai_slim", "version": "2.31.1"},
        {"name": "typing.extensions", "version": "4.16.0"},
    ]
)


def test_dep_sets_from_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT)
    sets = deps.dep_sets(pyproject, latest="2.40.0")
    assert [s.name for s in sets] == [
        "our_version",
        "pydantic_version@2.31.1",
        "pydantic_version@latest",
    ]
    assert sets[0].requirements == ("httpx==0.28.1", "jsonschema==4.26.0")
    assert sets[1].requirements[2:] == (
        "pydantic-ai-slim[openai,openrouter]==2.31.1",
        "openai==2.54.0",
    )
    assert sets[2].requirements[2:] == ("pydantic-ai-slim[openai,openrouter]==2.40.0",)
    assert [s.locked_extras for s in sets] == [(), ("pydantic",), None]  # latest: fresh resolve
    assert {s.package for s in sets} == {"our_version", "pydantic_version"}
    assert len(deps.dep_sets(pyproject)) == 2


def test_dep_sets_follow_the_real_pyproject() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    ours = deps.dep_sets(REPO_ROOT / "pyproject.toml")[0]
    assert list(ours.requirements) == project["dependencies"]


def test_parse_listing_normalizes_and_sorts() -> None:
    assert deps.parse_listing(LISTING) == {
        "annotated-types": "0.7.0",
        "httpx": "0.28.1",
        "jsonschema": "4.26.0",
        "pydantic-ai-slim": "2.31.1",
        "typing-extensions": "4.16.0",
    }


def test_tree_size_skips_pycache(tmp_path: Path) -> None:
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_bytes(b"x" * 100)
    (tmp_path / "pkg" / "data.bin").write_bytes(b"x" * 23)
    (tmp_path / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x" * 1000)
    assert deps.tree_size(tmp_path) == 123


def _probe_run(
    seconds: float, loop: bool, files: dict[str, str], turn_files: dict[str, str] | None = None
) -> dict:
    return {
        "import_s": seconds,
        "error": None,
        "loop": loop,
        "files": files,
        "turn_files": turn_files,
        "turn_error": None,
        "python": "3.12.3",
    }


PYDANTIC_PIN = deps.DepSet(
    "pydantic_version@2.31.1", "pydantic_version", ("x==1",), ("pydantic_ai",), ("pydantic",)
)
PYDANTIC_LATEST = deps.DepSet(
    "pydantic_version@latest", "pydantic_version", (), ("pydantic_ai",), None
)


def test_summarize(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_text('"""Doc."""\nA = 1\nB = 2\n\n# c\n')
    small = tmp_path / "small.py"
    small.write_text("C = 3\n")
    lazy = tmp_path / "lazy.py"
    lazy.write_text("D = 4\nE = 5\nF = 6\nG = 7\n")
    files = {str(big): "openai", str(small): "httpx"}
    turn_files = files | {str(lazy): "anyio"}
    loop_runs = [_probe_run(s, True, files, turn_files) for s in (0.30, 0.10, 0.20, 0.50, 0.40)]
    framework_runs = [_probe_run(s, False, {str(small): "httpx"}) for s in (0.2, 0.1, 0.3)]
    listing = deps.parse_listing(LISTING)
    report = deps.summarize(PYDANTIC_PIN, listing, 2_345_678, loop_runs, framework_runs)
    assert report["distributions"] == 5
    assert report["installed"] == listing  # the full listing, not just the key versions
    assert report["resolved_from"] == "uv.lock"
    assert report["site_packages_mb"] == 2.3
    assert report["import_ms"] == 300.0  # median of 5
    assert report["framework_import_ms"] == 200.0
    code = report["third_party_code"]
    assert code["loop_import"] == {
        "total": 3,
        "by_distribution": {"openai": {"files": 1, "code": 2}, "httpx": {"files": 1, "code": 1}},
    }
    assert code["loop_turn"]["total"] == 7  # the lazily imported file counts after the turn
    assert code["loop_turn"]["by_distribution"]["anyio"] == {"files": 1, "code": 4}
    assert code["framework_import"]["total"] == 1
    table = deps.table({PYDANTIC_PIN.name: report})
    assert "pydantic_version@2.31.1" in table
    assert (
        "httpx 0.28.1, jsonschema 4.26.0, pydantic-ai-slim 2.31.1 (resolved from uv.lock)" in table
    )
    assert "loop import 3, after one turn 7; framework import alone 1" in table


def test_summarize_without_a_loop_keeps_the_framework_apart() -> None:
    loop_runs = [_probe_run(0.001, False, {})]
    framework_runs = [_probe_run(0.25, False, {})]
    report = deps.summarize(PYDANTIC_PIN, {}, 0, loop_runs, framework_runs)
    assert report["import_ms"] is None
    assert report["framework_import_ms"] == 250.0
    code = report["third_party_code"]
    assert (code["loop_import"], code["loop_turn"]) == (None, None)
    assert code["framework_import"] == {"total": 0, "by_distribution": {}}
    table = deps.table({PYDANTIC_PIN.name: report})
    assert "loop import n/a, after one turn n/a; framework import alone 0" in table


def test_summarize_reports_a_failed_import() -> None:
    failed = _probe_run(0.1, False, {}) | {"error": "ImportError: cannot import name 'X'"}
    report = deps.summarize(PYDANTIC_LATEST, {}, 0, [failed], [_probe_run(0.25, False, {})])
    assert (report["import_ms"], report["import_error"]) == (None, failed["error"])
    assert report["framework_import_ms"] == 250.0
    assert report["resolved_from"] == "PyPI"
    table = deps.table({PYDANTIC_LATEST.name: report})
    assert "error" in table and "cannot import name 'X'" in table


def test_clean_env_drops_what_changes_an_install(monkeypatch: pytest.MonkeyPatch) -> None:
    scrubbed = ("UV_OVERRIDE", "UV_INDEX_URL", "PIP_INDEX_URL", "PYTHONPATH", "VIRTUAL_ENV")
    for name in (*scrubbed, "CONDA_PREFIX"):
        monkeypatch.setenv(name, "/somewhere")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    env = deps.clean_env()
    assert not {*scrubbed, "CONDA_PREFIX"} & set(env)
    assert env["HTTPS_PROXY"] == "http://proxy.invalid:3128"  # the install may need it


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_locked_sets_install_exactly_the_lock() -> None:
    """`uv export --frozen` reads uv.lock only: no network."""
    uv = shutil.which("uv")
    assert uv is not None
    ours, pinned = deps.dep_sets(REPO_ROOT / "pyproject.toml")
    lines = deps.requirements(uv, ours, REPO_ROOT).splitlines()
    assert "httpx==0.28.1" in lines and "jsonschema==4.26.0" in lines
    assert all("==" in line for line in lines)  # transitive dependencies are pinned too
    assert not any(line.startswith(("pytest", "ruff", "pydantic", "-e", ".")) for line in lines)
    assert "pydantic-ai-slim==2.31.1" in deps.requirements(uv, pinned, REPO_ROOT).splitlines()
    assert deps.requirements(uv, PYDANTIC_LATEST, REPO_ROOT) == "\n"


def test_probe_in_this_environment(tmp_path: Path) -> None:
    """The probe script itself, run offline against this venv, the repository's source and the
    fake provider: the turn loads the modules httpx imports lazily."""
    (tmp_path / "scenarios").mkdir()
    scenario = bench.scenario(turns=1, chunks=deps.TURN_CHUNKS)
    (tmp_path / "scenarios" / f"{bench.SCENARIO}.json").write_text(json.dumps(scenario))
    with bench.fake_provider(tmp_path / "scenarios", tmp_path / "wire") as port:
        url = f"http://127.0.0.1:{port}/s/{bench.SCENARIO}/deps/0/v1"
        command = [sys.executable, "-I", "-c", deps.PROBE, str(REPO_ROOT / "src"), url]
        command += [str(deps.TURN_CHUNKS), "bakeoff.our_version"]
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(proc.stdout.splitlines()[-1])
    assert result["loop"] is True and result["error"] is None and result["turn_error"] is None
    assert 0 < result["import_s"] < 30
    assert "httpx" in set(result["files"].values())
    assert all(path.endswith(".py") for path in result["files"])
    assert set(result["files"]) < set(result["turn_files"])
    assert {"httpcore", "h11", "anyio"} <= set(result["turn_files"].values())
    # The driver's tool host imports jsonschema; the loop does not, so it is not counted.
    assert "jsonschema" not in set(result["turn_files"].values())

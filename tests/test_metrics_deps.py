"""Dependency footprint: parsing and aggregation from fixtures (the venv builds need PyPI)."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from bakeoff.metrics import REPO_ROOT, deps

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


def _probe_run(seconds: float, loop: bool, files: dict[str, str]) -> dict:
    return {"import_s": seconds, "error": None, "loop": loop, "files": files, "python": "3.12.3"}


def test_summarize(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_text('"""Doc."""\nA = 1\nB = 2\n\n# c\n')
    small = tmp_path / "small.py"
    small.write_text("C = 3\n")
    files = {str(big): "openai", str(small): "httpx"}
    dep = deps.DepSet("pydantic_version@2.31.1", "pydantic_version", ("x==1",), ("pydantic_ai",))
    loop_runs = [_probe_run(s, True, files) for s in (0.30, 0.10, 0.20, 0.50, 0.40)]
    framework_runs = [_probe_run(s, False, {}) for s in (0.2, 0.1, 0.3)]
    report = deps.summarize(dep, deps.parse_listing(LISTING), 2_345_678, loop_runs, framework_runs)
    assert report["distributions"] == 5
    assert report["site_packages_mb"] == 2.3
    assert report["import_ms"] == 300.0  # median of 5
    assert report["framework_import_ms"] == 200.0
    assert report["versions"] == {
        "httpx": "0.28.1",
        "jsonschema": "4.26.0",
        "pydantic-ai-slim": "2.31.1",
    }
    assert report["imported_code"] == {
        "total": 3,
        "by_distribution": {"openai": {"files": 1, "code": 2}, "httpx": {"files": 1, "code": 1}},
    }
    assert "pydantic_version@2.31.1" in deps.table({dep.name: report})


def test_summarize_without_a_loop_uses_the_framework_import() -> None:
    dep = deps.DepSet("pydantic_version@2.31.1", "pydantic_version", (), ("pydantic_ai",))
    loop_runs = [_probe_run(0.001, False, {})]
    framework_runs = [_probe_run(0.25, False, {})]
    report = deps.summarize(dep, {}, 0, loop_runs, framework_runs)
    assert report["import_ms"] is None
    assert report["framework_import_ms"] == 250.0
    assert "n/a" in deps.table({dep.name: report})


def test_summarize_reports_a_failed_import() -> None:
    dep = deps.DepSet("pydantic_version@latest", "pydantic_version", (), ("pydantic_ai",))
    failed = _probe_run(0.1, False, {}) | {"error": "ImportError: cannot import name 'X'"}
    report = deps.summarize(dep, {}, 0, [failed], [_probe_run(0.25, False, {})])
    assert (report["import_ms"], report["import_error"]) == (None, failed["error"])
    assert report["framework_import_ms"] == 250.0
    table = deps.table({dep.name: report})
    assert "error" in table and "cannot import name 'X'" in table


def test_probe_in_this_environment() -> None:
    """The probe script itself, run offline against this venv and the repository's source."""
    proc = subprocess.run(
        [sys.executable, "-I", "-c", deps.PROBE, str(REPO_ROOT / "src"), "bakeoff.our_version"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout.splitlines()[-1])
    assert result["loop"] is True and result["error"] is None
    assert 0 < result["import_s"] < 30
    assert "httpx" in set(result["files"].values())
    assert all(path.endswith(".py") for path in result["files"])

"""Dependency footprint of each loop: what installing, importing and running it costs.

Each dependency set gets a throwaway uv venv (Python 3.12) in a temp dir with exactly its
runtime requirements, read from pyproject.toml so they never drift:

- `our_version`: the project's core dependencies, at the exact versions in uv.lock.
- `pydantic_version@<pin>`: core plus the `pydantic` extra (the version that fits the engine),
  at the exact versions in uv.lock.
- `pydantic_version@latest`: core plus the newest `pydantic-ai-slim[openai,openrouter]` on PyPI,
  pinned exactly, so the resolver cannot quietly fall back to an older release. Its other
  dependencies are resolved fresh: that is what "latest" means. The full listing is recorded.

uv runs with `--no-config` and without the caller's `UV_*`, `PIP_*`, `PYTHON*`, `VIRTUAL_ENV`
and `CONDA_*` variables, so user settings (overrides, indexes, constraints) cannot change what
gets installed.

Reported per set:

- `distributions` and `installed`: how many distributions are installed, and each version.
- `site_packages_mb`: size of site-packages in MB (10^6 bytes), `__pycache__` excluded.
  Bytecode is compiled at install, so the import timings below include no compiling.
- `import_ms`: cold import of the loop package: median of `runs` fresh `python -I` processes,
  timed with perf_counter around the import, before anything else is imported. The source tree
  is copied into the temp dir, so nothing is written to the repository, and one extra run first
  compiles its bytecode. This is null when the package in that tree does not export a loop yet
  (e.g. still an empty package).
- `framework_import_ms`: the same for the third-party modules the loop is built on, alone.
- `third_party_code`: code lines (the `loc` counter) of the third-party `.py` files loaded, per
  distribution: by the loop's import, after one real turn (text, a tool call and the answer,
  against the fake provider on 127.0.0.1), and by the framework's import alone. The turn
  matters: libraries import a lot lazily (httpx loads httpcore, h11 and anyio when it first
  connects). Compiled extensions such as pydantic-core are not counted.

This needs PyPI, so it is a command and never runs in unit tests. Only uv's own cache (outside
the repository) is reused between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bakeoff.metrics import REPO_ROOT, bench
from bakeoff.metrics.loc import count_source, read_source

PYTHON = "3.12"
PYDANTIC_AI = "pydantic-ai-slim[openai,openrouter]"
# Distributions whose versions identify what a set measured (the table shows these).
KEY_DISTRIBUTIONS = bench.KEY_DISTRIBUTIONS
# Environment variables that change what uv installs or which Python a process uses.
_SCRUBBED = ("UV_", "PIP_", "PYTHON", "VIRTUAL_ENV", "CONDA_")
TURN_CHUNKS = 10  # text chunks in the probe's one turn

# Runs in the venv's interpreter: `python -I -c PROBE <src> <turn URL or -> <chunks> <module>...`.
# Prints one JSON line. Only sys and time (both loaded at startup) come before the timed import.
PROBE = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
before = set(sys.modules)
modules, error = [], None
start = time.perf_counter()
try:
    for name in sys.argv[4:]:
        __import__(name)
        modules.append(sys.modules[name])
except Exception as exc:  # e.g. the loop does not work with this library release
    modules, error = [], f"{type(exc).__name__}: {exc}"
elapsed = time.perf_counter() - start
imported = set(sys.modules) - before

import asyncio, json, sysconfig
from importlib.metadata import packages_distributions

loop_cls = next(
    (
        cls
        for module in modules
        for name in getattr(module, "__all__", ())
        if hasattr(cls := getattr(module, name), "run_turn")
    ),
    None,
)
turned, turn_error = None, None
if loop_cls is not None and sys.argv[2] != "-":
    # One real turn loads what the loop imports lazily. The driver's own imports (the no-op
    # tool host pulls in jsonschema) are not the loop's, so they are left out.
    pre = set(sys.modules)
    from bakeoff.metrics import bench
    driver = set(sys.modules) - pre

    async def one_turn():
        loop = loop_cls()
        try:
            text = bench.expected_text(int(sys.argv[3]))
            await bench.loop_turn(loop, 0, sys.argv[2], bench.NoopTools(), text)
        finally:
            await loop.aclose()

    try:
        asyncio.run(one_turn())
        turned = set(sys.modules) - before - driver
    except Exception as exc:
        turn_error = f"{type(exc).__name__}: {exc}"

purelib = sysconfig.get_paths()["purelib"]
owners = packages_distributions()


def third_party(names):
    files = {}
    for name in names:
        path = getattr(sys.modules.get(name), "__file__", None) or ""
        if path.startswith(purelib) and path.endswith(".py"):
            top = path[len(purelib):].lstrip("/\\").replace("\\", "/").split("/")[0]
            if dist := owners.get(top.removesuffix(".py")):
                files[path] = dist[0]
    return files


print(json.dumps({
    "import_s": elapsed, "error": error, "loop": loop_cls is not None,
    "files": third_party(imported),
    "turn_files": None if turned is None else third_party(turned), "turn_error": turn_error,
    "python": sys.version.split()[0], "purelib": purelib,
}))
"""


@dataclass(frozen=True, slots=True)
class DepSet:
    """One loop's runtime requirements, and the third-party modules the loop is built on."""

    name: str
    package: str  # loop package under src/bakeoff
    requirements: tuple[str, ...]  # as declared
    framework: tuple[str, ...]
    # Install the exact versions uv.lock has for core plus these extras; None: resolve fresh.
    locked_extras: tuple[str, ...] | None


_OUR_FRAMEWORK = ("httpx",)
_PYDANTIC_FRAMEWORK = ("pydantic_ai.models.openrouter", "pydantic_ai.providers.openrouter")


def dep_sets(pyproject: Path, latest: str | None = None) -> list[DepSet]:
    """The dependency sets from pyproject.toml; with `latest`, also that pydantic-ai release."""
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    core = tuple(project["dependencies"])
    extra = tuple(project["optional-dependencies"]["pydantic"])
    pin = next(re.search(r"==(\S+)", r)[1] for r in extra if r.startswith("pydantic-ai-slim"))
    pydantic = f"pydantic_version@{pin}"
    sets = [
        DepSet("our_version", "our_version", core, _OUR_FRAMEWORK, locked_extras=()),
        DepSet(pydantic, "pydantic_version", core + extra, _PYDANTIC_FRAMEWORK, ("pydantic",)),
    ]
    if latest is not None:
        requirements = (*core, f"{PYDANTIC_AI}=={latest}")
        name = "pydantic_version@latest"  # resolved fresh: locked_extras=None
        sets.append(DepSet(name, "pydantic_version", requirements, _PYDANTIC_FRAMEWORK, None))
    return sets


def latest_release(project: str = "pydantic-ai-slim") -> str:
    """The newest release of `project` on PyPI (network)."""
    url = f"https://pypi.org/pypi/{project}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        version = json.load(response)["info"]["version"]
    if not re.fullmatch(r"\d+(\.\d+)*", version):
        raise ValueError(f"unexpected version for {project} on PyPI: {version!r}")
    return version


def parse_listing(text: str) -> dict[str, str]:
    """`uv pip list --format json` output as {normalized name: version}, sorted by name."""
    listing = {re.sub(r"[-_.]+", "-", d["name"]).lower(): d["version"] for d in json.loads(text)}
    return dict(sorted(listing.items()))


def tree_size(root: Path) -> int:
    """Total bytes of the files under `root`, `__pycache__` directories excluded."""
    total = 0
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d != "__pycache__"]
        total += sum(os.lstat(os.path.join(directory, f)).st_size for f in files)
    return total


def imported_code(files: dict[str, str]) -> dict[str, Any]:
    """Code lines per distribution of the imported third-party files ({path: distribution})."""
    per_dist: dict[str, dict[str, int]] = {}
    for path, dist in files.items():
        entry = per_dist.setdefault(dist, {"files": 0, "code": 0})
        entry["files"] += 1
        entry["code"] += count_source(read_source(Path(path))).code
    ordered = dict(sorted(per_dist.items(), key=lambda kv: (-kv[1]["code"], kv[0])))
    return {"total": sum(e["code"] for e in ordered.values()), "by_distribution": ordered}


def summarize(
    dep: DepSet,
    listing: dict[str, str],
    site_bytes: int,
    loop_runs: list[dict[str, Any]],
    framework_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """One set's report from its listing, site-packages size and import probe outputs."""
    has_loop = all(run["loop"] for run in loop_runs)
    first = loop_runs[0]

    def median_ms(runs: list[dict[str, Any]]) -> float | None:
        if any(run["error"] for run in runs):
            return None
        return round(statistics.median(run["import_s"] for run in runs) * 1e3, 1)

    return {
        "requirements": list(dep.requirements),
        "resolved_from": "PyPI" if dep.locked_extras is None else "uv.lock",
        "python": first["python"],
        "distributions": len(listing),
        "installed": listing,
        "site_packages_mb": round(site_bytes / 1e6, 1),
        "import_ms": median_ms(loop_runs) if has_loop else None,
        "import_error": first["error"],
        "framework": list(dep.framework),
        "framework_import_ms": median_ms(framework_runs),
        "framework_import_error": framework_runs[0]["error"],
        "import_runs": len(loop_runs),
        "turn_error": first["turn_error"],
        "third_party_code": {
            "loop_import": imported_code(first["files"]) if has_loop else None,
            "loop_turn": None
            if first["turn_files"] is None
            else imported_code(first["turn_files"]),
            "framework_import": imported_code(framework_runs[0]["files"]),
        },
    }


def clean_env() -> dict[str, str]:
    """This environment without the variables that change what uv installs or which Python and
    site-packages a process uses."""
    return {k: v for k, v in os.environ.items() if not k.startswith(_SCRUBBED)}


def _run(*command: str | Path, cwd: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        [str(c) for c in command], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        shown = " ".join("<probe>" if c == PROBE else str(c) for c in command)
        raise RuntimeError(f"{shown} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def requirements(uv: str, dep: DepSet, project: Path) -> str:
    """A requirements file for `dep`: every distribution at its uv.lock version when the set is
    locked, else the declared requirements (resolved fresh at install)."""
    if dep.locked_extras is None:
        return "\n".join(dep.requirements) + "\n"
    command = [uv, "export", "--no-config", "--frozen", "--no-dev", "--no-emit-project"]
    command += ["--no-hashes", "--no-header", "--no-annotate"]
    for extra in dep.locked_extras:
        command += ["--extra", extra]
    return _run(*command, cwd=project, env=clean_env())


def _probe(
    python: Path, src: Path, modules: tuple[str, ...], runs: int, turn_url: str | None = None
) -> list[dict[str, Any]]:
    """Import `modules` in `runs` fresh interpreters, after one discarded run that compiles.
    With `turn_url` (a fake-provider cursor base), each run also plays one loop turn."""
    # The probe only talks to 127.0.0.1: keep any proxy of the caller out of it.
    env = clean_env() | {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
    outputs = []
    for n in range(runs + 1):
        url = "-" if turn_url is None else f"{turn_url}/{n}/v1"
        command = (python, "-I", "-c", PROBE, src, url, str(TURN_CHUNKS), *modules)
        outputs.append(json.loads(_run(*command, cwd=src, env=env).splitlines()[-1]))
    return outputs[1:]


def measure_set(dep: DepSet, src: Path, project: Path, runs: int = 5) -> dict[str, Any]:
    """Build a throwaway venv for `dep` and measure it. `src` holds the `bakeoff` package and
    `project` the pyproject.toml and uv.lock the set comes from."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH")
    with tempfile.TemporaryDirectory(prefix="bakeoff-deps-") as tmp:
        work = Path(tmp)
        venv, code, scenarios = work / "venv", work / "src", work / "scenarios"
        python = venv / "bin" / "python"
        # --no-config, a scrubbed environment and cwd=work: no uv.toml, pyproject.toml or UV_*
        # setting of the caller or of this repository applies.
        env = clean_env()
        _run(uv, "venv", "--no-config", "--quiet", "--python", PYTHON, venv, cwd=work, env=env)
        (work / "requirements.txt").write_text(requirements(uv, dep, project))
        install = ("pip", "install", "--no-config", "--quiet", "--compile-bytecode")
        _run(uv, *install, "--python", python, "-r", "requirements.txt", cwd=work, env=env)
        listing = _run(uv, "pip", "list", "--python", python, "--format", "json", cwd=work, env=env)
        shutil.copytree(
            src / "bakeoff", code / "bakeoff", ignore=shutil.ignore_patterns("__pycache__")
        )
        scenarios.mkdir()
        scenario = bench.scenario(turns=1, chunks=TURN_CHUNKS)
        (scenarios / f"{bench.SCENARIO}.json").write_text(json.dumps(scenario))
        with bench.fake_provider(scenarios, work / "wire") as port:
            turn_url = f"http://127.0.0.1:{port}/s/{bench.SCENARIO}/deps"
            loop_runs = _probe(python, code, (f"bakeoff.{dep.package}",), runs, turn_url)
        framework_runs = _probe(python, code, dep.framework, runs)
        site_bytes = tree_size(Path(loop_runs[0]["purelib"]))
        return summarize(dep, parse_listing(listing), site_bytes, loop_runs, framework_runs)


def measure(
    names: list[str] | None = None,
    *,
    src: Path = REPO_ROOT / "src",
    pyproject: Path = REPO_ROOT / "pyproject.toml",
    latest: bool = True,
    runs: int = 5,
) -> dict[str, Any]:
    """Measure every dependency set (or only `names`). Needs network."""
    want_latest = latest and (names is None or "pydantic_version@latest" in names)
    sets = dep_sets(pyproject, latest_release() if want_latest else None)
    chosen = [d for d in sets if names is None or d.name in names]
    if unknown := set(names or ()) - {d.name for d in chosen}:
        raise ValueError(
            f"unknown dependency sets {sorted(unknown)}; known: {[d.name for d in sets]}"
        )
    return {d.name: measure_set(d, src, pyproject.parent, runs) for d in chosen}


def _ms(value: float | None, error: str | None) -> str:
    if value is not None:
        return f"{value:.1f}"
    return "error" if error else "n/a"  # n/a: the package exports no loop yet


def _lines(code: dict[str, Any] | None) -> str:
    return "n/a" if code is None else str(code["total"])


def table(report: dict[str, Any]) -> str:
    """The report as a fixed-width text table."""
    heads = ("dists", "site-pkgs MB", "import ms", "framework ms")
    out = [f"{'':<32}" + "".join(f"{h:>14}" for h in heads)]
    for name, r in report.items():
        cells = (
            r["distributions"],
            f"{r['site_packages_mb']:.1f}",
            _ms(r["import_ms"], r["import_error"]),
            _ms(r["framework_import_ms"], r["framework_import_error"]),
        )
        out.append(f"{name:<32}" + "".join(f"{c:>14}" for c in cells))
        installed = r["installed"]
        versions = ", ".join(f"{k} {installed[k]}" for k in KEY_DISTRIBUTIONS if k in installed)
        out.append(f"  python {r['python']}; {versions} (resolved from {r['resolved_from']})")
        code = r["third_party_code"]
        out.append(
            f"  3rd-party code lines loaded: loop import {_lines(code['loop_import'])}, "
            f"after one turn {_lines(code['loop_turn'])}; "
            f"framework import alone {_lines(code['framework_import'])}"
        )
        errors = (r["import_error"], r["turn_error"], r["framework_import_error"])
        out += [f"  {error}" for error in errors if error]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Build a venv per dependency set and print the footprint table (or JSON)."""
    parser = argparse.ArgumentParser(prog="python -m bakeoff.metrics.deps", description=__doc__)
    parser.add_argument("--set", action="append", dest="names", help="set name (repeatable)")
    parser.add_argument("--no-latest", action="store_true", help="skip the latest pydantic-ai")
    parser.add_argument("--runs", type=int, default=5, help="import timings per set (median)")
    parser.add_argument(
        "--src", type=Path, default=REPO_ROOT / "src", help="directory holding the bakeoff package"
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)
    report = measure(args.names, src=args.src.resolve(), latest=not args.no_latest, runs=args.runs)
    print(json.dumps(report, indent=2) if args.json else table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

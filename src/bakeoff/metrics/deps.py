"""Dependency footprint of each loop: what installing and importing it costs.

Each dependency set gets a throwaway uv venv (Python 3.12) in a temp dir with exactly its
runtime requirements, read from pyproject.toml so they never drift:

- `our_version`: the project's core dependencies.
- `pydantic_version@<pin>`: core plus the `pydantic` extra (the version that fits the engine).
- `pydantic_version@latest`: core plus the newest `pydantic-ai-slim[openai,openrouter]` on PyPI,
  pinned exactly, so the resolver cannot quietly fall back to an older release.

Reported per set:

- `distributions`: how many distributions are installed (`uv pip list`).
- `site_packages_mb`: size of site-packages in MB (10^6 bytes), `__pycache__` excluded.
  Bytecode is compiled at install, so the import timings below include no compiling.
- `import_ms`: cold import of the loop package: median of `runs` fresh `python -I` processes,
  timed with perf_counter around the import. The source tree is copied into the temp dir, so
  nothing is written to the repository, and one extra run first compiles its bytecode. This is
  null when the package in that tree does not export a loop yet (e.g. still an empty package).
- `framework_import_ms`: the same for the third-party modules the loop is built on, alone.
- `imported_code`: code lines (the `loc` counter) of the third-party `.py` files that the import
  of the loop (or else of the framework) loaded, per distribution. Modules a library imports
  lazily (httpx loads httpcore, h11 and anyio when its first client is built) and compiled
  extensions such as pydantic-core are not counted.

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

from bakeoff.metrics import REPO_ROOT
from bakeoff.metrics.loc import count_source, read_source

PYTHON = "3.12"
PYDANTIC_AI = "pydantic-ai-slim[openai,openrouter]"
# Distributions whose versions identify what a set measured.
KEY_DISTRIBUTIONS = ("httpx", "jsonschema", "openai", "pydantic", "pydantic-ai-slim")

# Runs in the venv's interpreter: `python -I -c PROBE <src> <module>...`. Prints one JSON line.
PROBE = r"""
import importlib, json, sys, sysconfig, time
sys.path.insert(0, sys.argv[1])
before = set(sys.modules)
error = None
start = time.perf_counter()
try:
    modules = [importlib.import_module(name) for name in sys.argv[2:]]
except Exception as exc:  # e.g. the loop does not work with this library release
    modules, error = [], f"{type(exc).__name__}: {exc}"
elapsed = time.perf_counter() - start
loaded = [m for name, m in list(sys.modules.items()) if name not in before]
purelib = sysconfig.get_paths()["purelib"]
loop = any(
    hasattr(getattr(module, name, None), "run_turn")
    for module in modules
    for name in getattr(module, "__all__", ())
)
from importlib.metadata import packages_distributions
owners = packages_distributions()
files = {}
for module in loaded:
    path = getattr(module, "__file__", None) or ""
    if path.startswith(purelib) and path.endswith(".py"):
        top = path[len(purelib):].lstrip("/\\").replace("\\", "/").split("/")[0]
        if dist := owners.get(top.removesuffix(".py")):
            files[path] = dist[0]
print(json.dumps({
    "import_s": elapsed, "error": error, "loop": loop, "files": files,
    "python": sys.version.split()[0], "purelib": purelib,
}))
"""


@dataclass(frozen=True, slots=True)
class DepSet:
    """One loop's runtime requirements, and the third-party modules the loop is built on."""

    name: str
    package: str  # loop package under src/bakeoff
    requirements: tuple[str, ...]
    framework: tuple[str, ...]


_OUR_FRAMEWORK = ("httpx",)
_PYDANTIC_FRAMEWORK = ("pydantic_ai.models.openrouter", "pydantic_ai.providers.openrouter")


def dep_sets(pyproject: Path, latest: str | None = None) -> list[DepSet]:
    """The dependency sets from pyproject.toml; with `latest`, also that pydantic-ai release."""
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    core = tuple(project["dependencies"])
    extra = tuple(project["optional-dependencies"]["pydantic"])
    pin = next(re.search(r"==(\S+)", r)[1] for r in extra if r.startswith("pydantic-ai-slim"))
    sets = [
        DepSet("our_version", "our_version", core, _OUR_FRAMEWORK),
        DepSet(f"pydantic_version@{pin}", "pydantic_version", core + extra, _PYDANTIC_FRAMEWORK),
    ]
    if latest is not None:
        requirements = (*core, f"{PYDANTIC_AI}=={latest}")
        sets.append(
            DepSet("pydantic_version@latest", "pydantic_version", requirements, _PYDANTIC_FRAMEWORK)
        )
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

    def median_ms(runs: list[dict[str, Any]]) -> float | None:
        if any(run["error"] for run in runs):
            return None
        return round(statistics.median(run["import_s"] for run in runs) * 1e3, 1)

    return {
        "requirements": list(dep.requirements),
        "python": loop_runs[0]["python"],
        "versions": {name: listing[name] for name in KEY_DISTRIBUTIONS if name in listing},
        "distributions": len(listing),
        "site_packages_mb": round(site_bytes / 1e6, 1),
        "import_ms": median_ms(loop_runs) if has_loop else None,
        "import_error": loop_runs[0]["error"],
        "framework": list(dep.framework),
        "framework_import_ms": median_ms(framework_runs),
        "framework_import_error": framework_runs[0]["error"],
        "import_runs": len(loop_runs),
        "imported_code": imported_code((loop_runs if has_loop else framework_runs)[0]["files"]),
    }


def _run(*command: str | Path, cwd: Path) -> str:
    proc = subprocess.run(
        [str(c) for c in command], cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, command))} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def _probe(python: Path, src: Path, modules: tuple[str, ...], runs: int) -> list[dict[str, Any]]:
    """Import `modules` in `runs` fresh interpreters, after one discarded run that compiles."""
    outputs = [
        json.loads(_run(python, "-I", "-c", PROBE, src, *modules, cwd=src).splitlines()[-1])
        for _ in range(runs + 1)
    ]
    return outputs[1:]


def measure_set(dep: DepSet, src: Path, runs: int = 5) -> dict[str, Any]:
    """Build a throwaway venv for `dep` and measure it. `src` holds the `bakeoff` package."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH")
    with tempfile.TemporaryDirectory(prefix="bakeoff-deps-") as tmp:
        work = Path(tmp)
        venv, code = work / "venv", work / "src"
        python = venv / "bin" / "python"
        # cwd=work: no pyproject.toml or uv.toml of this repository applies.
        _run(uv, "venv", "--quiet", "--python", PYTHON, venv, cwd=work)
        install = ("pip", "install", "--quiet", "--compile-bytecode", "--python", python)
        _run(uv, *install, *dep.requirements, cwd=work)
        listing = _run(uv, "pip", "list", "--python", python, "--format", "json", cwd=work)
        shutil.copytree(
            src / "bakeoff", code / "bakeoff", ignore=shutil.ignore_patterns("__pycache__")
        )
        loop_runs = _probe(python, code, (f"bakeoff.{dep.package}",), runs)
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
    return {d.name: measure_set(d, src, runs) for d in chosen}


def _ms(value: float | None, error: str | None) -> str:
    if value is not None:
        return f"{value:.1f}"
    return "error" if error else "n/a"  # n/a: the package exports no loop yet


def table(report: dict[str, Any]) -> str:
    """The report as a fixed-width text table."""
    heads = ("dists", "site-pkgs MB", "import ms", "framework ms", "3rd-party code lines")
    out = [f"{'':<32}" + "".join(f"{h:>14}" for h in heads[:4]) + f"{heads[4]:>22}"]
    for name, r in report.items():
        cells = (
            r["distributions"],
            f"{r['site_packages_mb']:.1f}",
            _ms(r["import_ms"], r["import_error"]),
            _ms(r["framework_import_ms"], r["framework_import_error"]),
        )
        out.append(
            f"{name:<32}"
            + "".join(f"{c:>14}" for c in cells)
            + f"{r['imported_code']['total']:>22}"
        )
        versions = ", ".join(f"{k} {v}" for k, v in r["versions"].items())
        out.append(f"  python {r['python']}; {versions}")
        out += [f"  {error}" for error in (r["import_error"], r["framework_import_error"]) if error]
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

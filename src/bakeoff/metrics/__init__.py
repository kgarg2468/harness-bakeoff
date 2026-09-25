"""Scorecard metrics: lines of code, dependency footprint and harness overhead.

- `loc`: code / comment / docstring / blank lines and statements per loop and for `shared/`.
- `deps`: installed distributions, site-packages size, cold import time and third-party code
  loaded per loop, each in a throwaway venv (network).
- `bench`: harness overhead per turn and per chunk against the fake provider on 127.0.0.1.
- `collect`: runs them and writes `out/metrics.json`.

Each module is also a command: `python -m bakeoff.metrics.<name> --help`. Submodules are not
imported here, so `python -m` runs each of them exactly once.
"""

from __future__ import annotations

from pathlib import Path

# The counted loop packages under src/bakeoff, in scorecard order (FAIRNESS.md rule 1).
LOOP_PACKAGES = ("our_version", "pydantic_version", "hybrid_version")
# src/bakeoff/metrics/__init__.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

"""Scorecard metrics: lines of code, dependency footprint and harness overhead.

- `loc`: code / comment / docstring / blank lines and statements per loop and for `shared/`.
- `deps`: installed distributions, site-packages size and cold import time per loop (network).
- `bench`: harness overhead per turn and per chunk against the in-process fake provider.
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

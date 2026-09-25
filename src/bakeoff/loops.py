"""Registry of loop implementations, imported lazily.

A loop that is not built yet (its package is empty, or its optional dependencies are not
installed) is reported as missing instead of breaking everything else. Each entry also lists the
scenarios the loop is known to fail: exactly which checks fail, and why. The scenario matrix
records that reality (`xfail` in tests and in `bakeoff scenario`) instead of hiding it, and any
other failure in the same cell still counts as a failure.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass, field

from bakeoff.shared.contract import Loop


@dataclass(frozen=True, slots=True)
class KnownFailure:
    """A documented failure of one scenario: the checks that fail (final `expect` keys and
    invariant names, e.g. {"tool_runs"}), and why."""

    checks: frozenset[str]
    why: str


@dataclass(frozen=True, slots=True)
class LoopEntry:
    """One loop implementation: where it lives and what we know about it."""

    name: str  # the loop's `name` and the thread's `impl`
    target: str  # "module:Class"
    packages: tuple[str, ...]  # distributions whose versions describe the loop
    # scenario id -> how and why this loop fails it (documented, e.g. in A_CHECKLIST.md)
    known_failures: Mapping[str, KnownFailure] = field(default_factory=dict)


REGISTRY: dict[str, LoopEntry] = {
    entry.name: entry
    for entry in (
        LoopEntry(
            "our",
            "bakeoff.our_version:OurLoop",
            ("httpx",),
            known_failures={
                # The step cap is checked before the next request, after the capped step's
                # calls ran; a read-only call even starts while its response streams.
                "S11": KnownFailure(
                    frozenset({"tool_runs"}),
                    "runs call_S11_3, the call of the step that hits max_steps, whose result"
                    " can never be sent (S11 expects it not to run)",
                ),
            },
        ),
        LoopEntry(
            "pydantic",
            "bakeoff.pydantic_version:PydanticLoop",
            ("pydantic-ai-slim", "openai", "httpx"),
        ),
        LoopEntry(
            "hybrid", "bakeoff.hybrid_version:HybridLoop", ("pydantic-ai-slim", "openai", "httpx")
        ),
    )
}


class LoopUnavailable(RuntimeError):
    """A loop cannot be loaded. `missing` is True when it simply is not built or installed."""

    def __init__(self, name: str, reason: str, *, missing: bool) -> None:
        super().__init__(f"{name}: {reason}")
        self.name, self.reason, self.missing = name, reason, missing


def load(name: str) -> type[Loop]:
    """The loop class registered as `name`. Raises LoopUnavailable."""
    entry = REGISTRY.get(name)
    if entry is None:
        raise LoopUnavailable(name, f"unknown loop (known: {', '.join(REGISTRY)})", missing=True)
    module_name, _, attr = entry.target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        absent = getattr(exc, "name", None) if isinstance(exc, ModuleNotFoundError) else None
        missing = absent is not None and (absent == module_name or not _installed(absent))
        reason = f"not installed ({exc})" if missing else f"{type(exc).__name__}: {exc}"
        raise LoopUnavailable(name, reason, missing=missing) from exc
    cls = getattr(module, attr, None)
    if cls is None:
        raise LoopUnavailable(name, f"{entry.target} is not built yet", missing=True)
    return cls


def _installed(module: str) -> bool:
    """Whether the top-level package of `module` is installed at all.

    A loop is missing (skipped) only when its own module or a whole dependency is absent (an
    extra that is not installed). A missing submodule of an installed package, e.g. after a
    library renamed it, means the loop exists but is broken, which must fail loudly."""
    try:
        return importlib.util.find_spec(module.partition(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def available(names: list[str] | None = None) -> dict[str, type[Loop]]:
    """The loops (all registered ones by default) that import here, in registry order."""
    loops = {}
    for name in names if names is not None else list(REGISTRY):
        try:
            loops[name] = load(name)
        except LoopUnavailable:
            continue
    return loops


def unavailable() -> dict[str, LoopUnavailable]:
    """Every registered loop that does not import here, with the reason."""
    problems = {}
    for name in REGISTRY:
        try:
            load(name)
        except LoopUnavailable as exc:
            problems[name] = exc
    return problems


def known_failure(name: str, scenario: str) -> KnownFailure | None:
    """How and why loop `name` is expected to fail `scenario`, or None if it should pass."""
    entry = REGISTRY.get(name)
    return None if entry is None else entry.known_failures.get(scenario)


def versions(name: str) -> dict[str, str]:
    """Installed versions of the distributions that describe loop `name`."""
    found = {}
    for dist in REGISTRY[name].packages:
        try:
            found[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found

"""Tool building blocks: `Tool`, the context a tool runs in, and working-copy path rules.

The tools themselves live in `engine_tools`, `file_tools` and `skill_tools`; `toolhost` puts
them in order.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bakeoff.shared.contract import ToolSpec
from bakeoff.shared.engine.base import Engine


@dataclass(slots=True, frozen=True)
class ToolContext:
    """What a running tool may use."""

    workdir: Path  # the working copy, already resolved
    engine: Engine
    unattended: bool = False  # no call can ask a person, so nobody can answer approval gates


@dataclass(slots=True, frozen=True)
class Tool:
    """A tool: the spec the model sees and the coroutine that runs it.

    `check_args` adds argument rules the provider-facing schema cannot state. It returns what is
    wrong, or None; the `ToolHost` applies it with the schema, before anything runs.
    """

    spec: ToolSpec
    fn: Callable[[dict[str, Any], ToolContext], Awaitable[str]]
    check_args: Callable[[dict[str, Any]], str | None] | None = None


class ToolError(Exception):
    """An expected tool failure. Its message is exactly what the model sees."""


class PathError(ToolError):
    """A path the tools refuse: absolute, outside the working copy, or a write into `.git`."""


def in_git(root: Path, full: Path) -> bool:
    """Whether `full`, a resolved path inside `root`, is a `.git` entry or inside one.

    Any depth counts: a nested `.git` directory or gitfile breaks the runner's `git add -A`.
    Case-insensitive, because `.GIT` is `.git` on case-insensitive file systems (macOS default).
    """
    return any(part.casefold() == ".git" for part in full.relative_to(root).parts)


def resolve_path(root: Path, path: str, *, write: bool = False) -> Path:
    """Resolve `path` against the working copy `root` (already resolved).

    Symlinks are followed, so a link pointing outside the working copy is refused too. With
    `write`, any `.git` path is refused as well.
    """
    if Path(path).is_absolute():
        raise PathError(f"Path must be relative to the working copy: {path}")
    try:
        full = (root / path).resolve()
    except (OSError, ValueError, RuntimeError) as e:  # NUL bytes, symlink loops
        raise PathError(f"Invalid path {path!r}: {e}") from e
    if not full.is_relative_to(root):
        raise PathError(f"Path escapes the working copy: {path}")
    if write and in_git(root, full):
        raise PathError(f"Writing inside .git is not allowed: {path}")
    return full

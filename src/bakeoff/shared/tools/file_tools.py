"""File tools over the thread's git working copy: list, read, write and edit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from bakeoff.shared.contract import ToolSpec

from . import Tool, ToolContext, ToolError, in_git, resolve_path
from .edit_match import EditError, replace

MAX_ENTRIES = 500

_PATH = {"type": "string", "description": "File path relative to the working copy"}


def _rel(ctx: ToolContext, full: Path) -> str:
    return full.relative_to(ctx.workdir).as_posix()


def _read(ctx: ToolContext, path: str, *, write: bool = False) -> tuple[Path, str]:
    """The file's text. For an edit (`write=True`) the file must be valid UTF-8, because the
    whole file is written back: replacement characters would corrupt bytes outside the edit."""
    full = resolve_path(ctx.workdir, path, write=write)
    if full.is_dir():
        raise ToolError(f"Not a file: {path} (use list_files for directories)")
    try:
        # Bytes, not read_text: keep line endings exactly as they are on disk.
        data = full.read_bytes()
    except FileNotFoundError:
        raise ToolError(f"File not found: {path}") from None
    try:
        return full, data.decode("utf-8", errors="strict" if write else "replace")
    except UnicodeDecodeError:
        raise ToolError(
            f"Cannot edit {path}: it is not UTF-8 text. Use write_file to replace it."
        ) from None


def _write(full: Path, content: str) -> int:
    data = content.encode("utf-8")
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)
    return len(data)


async def list_files(args: dict[str, Any], ctx: ToolContext) -> str:
    """Every file and directory below `path`, relative to the working copy, sorted."""
    path = args.get("path", ".")
    base = resolve_path(ctx.workdir, path)
    if in_git(ctx.workdir, base):
        raise ToolError(f"Cannot list {path}: .git is hidden")
    if not base.is_dir():
        raise ToolError(f"Not a directory: {path}")
    entries: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d.casefold() != ".git"]
        here = Path(dirpath)
        entries += [f"{_rel(ctx, here / d)}/" for d in dirnames]
        entries += [_rel(ctx, here / f) for f in filenames if f.casefold() != ".git"]
    entries.sort()
    if len(entries) > MAX_ENTRIES:
        more = len(entries) - MAX_ENTRIES
        entries = [*entries[:MAX_ENTRIES], f"... [{more} more entries]"]
    return "\n".join(entries) or "(empty)"


async def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    """The file's text."""
    return _read(ctx, args["path"])[1]


async def write_file(args: dict[str, Any], ctx: ToolContext) -> str:
    """Create or overwrite a file, creating parent directories."""
    full = resolve_path(ctx.workdir, args["path"], write=True)
    if full.is_dir():
        raise ToolError(f"Is a directory: {args['path']}")
    size = _write(full, args["content"])
    return f"Wrote {size} bytes to {_rel(ctx, full)}"


async def edit_file(args: dict[str, Any], ctx: ToolContext) -> str:
    """Replace `old_string` with `new_string` using the forgiving matcher."""
    full, content = _read(ctx, args["path"], write=True)
    # As OpenCode does: match in the file's own line endings.
    ending = "\r\n" if "\r\n" in content else "\n"
    old, new = (
        args[key].replace("\r\n", "\n").replace("\n", ending)
        for key in ("old_string", "new_string")
    )
    try:
        updated = replace(content, old, new, args.get("replace_all", False))
    except EditError as e:
        raise ToolError(str(e)) from None
    _write(full, updated)
    return f"Edited {_rel(ctx, full)}"


FILE_TOOLS = (
    Tool(
        ToolSpec(
            name="list_files",
            description=(
                "List files and directories below a directory of the working copy, recursively "
                f"and sorted. Directories end with '/'. At most {MAX_ENTRIES} entries."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory relative to the working copy (default '.')",
                    }
                },
            },
            read_only=True,
        ),
        list_files,
    ),
    Tool(
        ToolSpec(
            name="read_file",
            description="Read a text file from the working copy.",
            parameters={"type": "object", "properties": {"path": _PATH}, "required": ["path"]},
            read_only=True,
        ),
        read_file,
    ),
    Tool(
        ToolSpec(
            name="write_file",
            description=(
                "Create or overwrite a file in the working copy. Parent directories are created."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": _PATH,
                    "content": {"type": "string", "description": "The full new file content"},
                },
                "required": ["path", "content"],
            },
        ),
        write_file,
    ),
    Tool(
        ToolSpec(
            name="edit_file",
            description=(
                "Replace old_string with new_string in a file of the working copy. old_string "
                "must match one place in the file (add surrounding lines to make it unique) "
                "unless replace_all is true. Small whitespace and indentation differences are "
                "tolerated."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": _PATH,
                    "old_string": {"type": "string", "description": "The text to replace"},
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with (must differ from old_string)",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_string (default false)",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
        edit_file,
    ),
)

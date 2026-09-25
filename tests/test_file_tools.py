import os

import pytest

from bakeoff.shared.engine.mock import MockEngine
from bakeoff.shared.tools import PathError, ToolContext, ToolError, resolve_path
from bakeoff.shared.tools.file_tools import (
    MAX_ENTRIES,
    edit_file,
    list_files,
    read_file,
    write_file,
)


@pytest.fixture
def wc(tmp_path):
    root = tmp_path / "wc"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def ctx(wc):
    return ToolContext(workdir=wc, engine=MockEngine())


@pytest.mark.parametrize("path", ["..", "../x", "a/../../x", "/etc/passwd"])
def test_escaping_paths_are_refused(wc, path):
    with pytest.raises(PathError):
        resolve_path(wc, path)


def test_symlink_out_of_the_working_copy_is_refused(wc, tmp_path):
    (tmp_path / "secret.txt").write_text("s")
    (wc / "link").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(PathError, match="escapes"):
        resolve_path(wc, "link")


def test_paths_inside_are_resolved(wc):
    (wc / "real.txt").write_text("r")
    (wc / "inner").symlink_to(wc / "real.txt")
    assert resolve_path(wc, "inner") == wc / "real.txt"
    assert resolve_path(wc, "a/../b.txt") == wc / "b.txt"
    assert resolve_path(wc, ".") == wc


@pytest.mark.parametrize("path", [".git/config", ".git", "x/../.git/hooks/pre-commit"])
def test_git_is_never_writable(wc, path):
    (wc / ".git").mkdir()
    assert resolve_path(wc, path) == (wc / path).resolve()  # reading is fine
    with pytest.raises(PathError, match=r"\.git"):
        resolve_path(wc, path, write=True)


def test_invalid_path_is_refused(wc):
    with pytest.raises(PathError, match="Invalid path"):
        resolve_path(wc, "a\x00b")


async def test_list_files_recursive_sorted_hides_git(wc, ctx):
    (wc / ".git").mkdir()
    (wc / ".git" / "HEAD").write_text("ref")
    (wc / "b").mkdir()
    (wc / "b" / "c.txt").write_text("c")
    (wc / "a.pipe").write_text("{}")
    (wc / ".gitignore").write_text("out/")
    assert await list_files({}, ctx) == ".gitignore\na.pipe\nb/\nb/c.txt"
    assert await list_files({"path": "b"}, ctx) == "b/c.txt"


async def test_list_files_caps_entries(wc, ctx):
    for i in range(MAX_ENTRIES + 7):
        (wc / f"f{i:04}.txt").write_text("")
    lines = (await list_files({"path": "."}, ctx)).splitlines()
    assert len(lines) == MAX_ENTRIES + 1
    assert lines[-1] == "... [7 more entries]"


async def test_list_files_errors(wc, ctx):
    (wc / "f.txt").write_text("")
    assert await list_files({}, ctx) == "f.txt"
    with pytest.raises(ToolError, match="Not a directory"):
        await list_files({"path": "f.txt"}, ctx)
    (wc / "f.txt").unlink()
    assert await list_files({}, ctx) == "(empty)"


async def test_read_file(wc, ctx):
    (wc / "a.txt").write_bytes(b"one\r\ntwo\n")
    assert await read_file({"path": "a.txt"}, ctx) == "one\r\ntwo\n"
    with pytest.raises(ToolError, match="File not found: nope"):
        await read_file({"path": "nope"}, ctx)
    (wc / "d").mkdir()
    with pytest.raises(ToolError, match="Not a file: d"):
        await read_file({"path": "d"}, ctx)


async def test_write_file_creates_parents(wc, ctx):
    out = await write_file({"path": "./x/y/z.pipe", "content": "é\n"}, ctx)
    assert out == "Wrote 3 bytes to x/y/z.pipe"
    assert (wc / "x" / "y" / "z.pipe").read_bytes() == "é\n".encode()


async def test_write_onto_a_directory(wc, ctx):
    (wc / "d").mkdir()
    with pytest.raises(ToolError, match="Is a directory: d"):
        await write_file({"path": "d", "content": ""}, ctx)


async def test_write_into_git_is_refused(wc, ctx):
    with pytest.raises(PathError):
        await write_file({"path": ".git/config", "content": ""}, ctx)
    assert not (wc / ".git").exists()


async def test_edit_file(wc, ctx):
    (wc / "p.pipe").write_text('{"lane": "text"}\n')
    out = await edit_file(
        {"path": "p.pipe", "old_string": '"lane": "text"', "new_string": '"lane": "questions"'}, ctx
    )
    assert out == "Edited p.pipe"
    assert (wc / "p.pipe").read_text() == '{"lane": "questions"}\n'


async def test_edit_file_keeps_crlf(wc, ctx):
    (wc / "a.txt").write_bytes(b"one\r\ntwo\r\n")
    await edit_file({"path": "a.txt", "old_string": "one\ntwo", "new_string": "1\n2"}, ctx)
    assert (wc / "a.txt").read_bytes() == b"1\r\n2\r\n"


async def test_edit_file_replace_all(wc, ctx):
    (wc / "a.txt").write_text("x x x")
    await edit_file(
        {"path": "a.txt", "old_string": "x", "new_string": "y", "replace_all": True}, ctx
    )
    assert (wc / "a.txt").read_text() == "y y y"


@pytest.mark.parametrize(
    ("old", "message"),
    [("nope", "Could not find old_string"), ("x", "multiple matches")],
)
async def test_edit_file_errors_leave_the_file_alone(wc, ctx, old, message):
    (wc / "a.txt").write_text("x x")
    with pytest.raises(ToolError, match=message):
        await edit_file({"path": "a.txt", "old_string": old, "new_string": "y"}, ctx)
    assert (wc / "a.txt").read_text() == "x x"


async def test_edit_missing_file(ctx):
    with pytest.raises(ToolError, match="File not found"):
        await edit_file({"path": "a.txt", "old_string": "a", "new_string": "b"}, ctx)


async def test_symlinked_directory_is_listed_not_followed(wc, ctx, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    os.symlink(outside, wc / "out")
    assert await list_files({}, ctx) == "out/"

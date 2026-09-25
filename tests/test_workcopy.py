import asyncio
import subprocess

import pytest

from bakeoff.shared.workcopy import GIT_CONFIG, WorkCopy, git_env


def git(root, *args, text=True):
    return subprocess.run(
        ["git", *GIT_CONFIG, *args], cwd=root, env=git_env(root), capture_output=True, text=text
    )


@pytest.fixture
async def wc(tmp_path):
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    return wc


async def test_init_makes_one_empty_commit_on_main(wc):
    assert git(wc.root, "branch", "--show-current").stdout.strip() == "main"
    assert git(wc.root, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    head = await wc.head()
    await wc.init()  # idempotent
    assert await wc.head() == head


async def test_init_repairs_a_repo_without_its_initial_commit(tmp_path):
    root = tmp_path / "wc"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")  # then the process died
    wc = WorkCopy(root)
    wc.init_sync()
    assert git(root, "log", "--format=%s").stdout.split() == ["init"]
    (root / "a.pipe").write_text("{}")
    _, files = await wc.commit("turn 0")
    assert files == ["a.pipe"]


async def test_commit_reports_changed_files(wc):
    (wc.root / "a.txt").write_text("a")
    (wc.root / "dir").mkdir()
    (wc.root / "dir" / "b c.pipe").write_text("{}")
    sha, files = await wc.commit("turn 0: end_turn")
    assert sha == await wc.head()
    assert files == ["a.txt", "dir/b c.pipe"]
    assert git(wc.root, "log", "-1", "--format=%s %an <%ae>").stdout.strip() == (
        "turn 0: end_turn bakeoff <bakeoff@localhost>"
    )

    (wc.root / "a.txt").unlink()
    (wc.root / "dir" / "b c.pipe").write_text('{"x": 1}')
    _, files = await wc.commit("turn 1: end_turn")
    assert files == ["a.txt", "dir/b c.pipe"]


async def test_empty_commit(wc):
    before = await wc.head()
    sha, files = await wc.commit("turn 0: error")
    assert sha != before
    assert files == []


async def test_revert_is_a_new_commit(wc):
    (wc.root / "a.txt").write_text("a")
    first, _ = await wc.commit("turn 0")
    (wc.root / "b.txt").write_text("b")
    await wc.commit("turn 1")
    sha, files = await wc.revert(first)
    assert files == ["a.txt"]
    assert sha == await wc.head()
    assert not (wc.root / "a.txt").exists()
    assert (wc.root / "b.txt").exists()
    assert git(wc.root, "rev-list", "--count", "HEAD").stdout.strip() == "4"


async def test_revert_of_an_empty_commit(wc):
    empty, _ = await wc.commit("turn 0")
    sha, files = await wc.revert(empty)
    assert (sha != empty, files) == (True, [])


async def test_revert_conflict_aborts(wc):
    (wc.root / "a.txt").write_text("1")
    first, _ = await wc.commit("turn 0")
    (wc.root / "a.txt").write_text("2")
    head, _ = await wc.commit("turn 1")
    with pytest.raises(RuntimeError, match="git revert failed"):
        await wc.revert(first)
    assert await wc.head() == head
    assert (wc.root / "a.txt").read_text() == "2"
    assert git(wc.root, "status", "--porcelain").stdout == ""


async def test_cancel_stops_git(wc):
    """A cancelled commit must not land later, nor leave git's index lock behind."""
    hook = wc.root / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\ntouch started\nsleep 0.3\n")
    hook.chmod(0o755)
    before = await wc.head()
    task = asyncio.create_task(wc.commit("turn 0"))
    for _ in range(500):
        if (wc.root / "started").exists():
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.4)  # past the hook: an orphaned git would have committed by now
    assert await wc.head() == before
    assert not (wc.root / ".git" / "index.lock").exists()


async def test_recover_drops_unrecorded_commits_and_a_stale_lock(wc):
    (wc.root / "a.txt").write_text("a")
    recorded, _ = await wc.commit("turn 0")
    (wc.root / "b.txt").write_text("b")
    await wc.commit("orphan")  # landed after its worker died
    (wc.root / ".git" / "index.lock").write_text("")
    await wc.recover(recorded)
    assert await wc.head() == recorded
    _, files = await wc.commit("turn 1")
    assert files == ["b.txt"]

    await wc.recover(None)  # no turn recorded a commit: back to the initial one
    assert git(wc.root, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    _, files = await wc.commit("turn 0")
    assert files == ["a.txt", "b.txt"]


async def test_user_git_config_and_env_do_not_leak(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    (home / ".gitconfig").write_text(
        "[user]\n\tname = leaked\n[commit]\n\tgpgsign = true\n[core]\n\thooksPath = /nonexistent\n"
    )
    # Read by default even without a global config file:
    (home / ".config" / "git" / "ignore").write_text("*.pipe\nout/\n")
    (home / ".config" / "git" / "attributes").write_text("*.txt text\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # As inside a git hook of an enclosing repository:
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere" / ".git"))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "leaked")
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    (wc.root / "a.txt").write_bytes(b"a\r\n")
    (wc.root / "b.pipe").write_text("{}")
    (wc.root / "out").mkdir()
    (wc.root / "out" / "c.json").write_text("{}")
    _, files = await wc.commit("turn 0")
    monkeypatch.undo()
    assert files == ["a.txt", "b.pipe", "out/c.json"]
    assert git(wc.root, "log", "-1", "--format=%an").stdout.strip() == "bakeoff"
    blob = git(wc.root, "cat-file", "blob", "HEAD:a.txt", text=False).stdout
    assert blob == b"a\r\n"  # no line-ending conversion from the user's attributes
    assert not (tmp_path / "elsewhere").exists()


async def test_never_uses_an_enclosing_repository(tmp_path):
    outer = WorkCopy(tmp_path)
    await outer.init()
    inner = WorkCopy(tmp_path / "inner")  # never initialised
    inner.root.mkdir()
    with pytest.raises(RuntimeError, match="git rev-parse failed"):  # not the outer repo's HEAD
        await inner.head()

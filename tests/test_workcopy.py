import subprocess

import pytest

from bakeoff.shared.workcopy import GIT_CONFIG, WorkCopy, git_env


def git(root, *args):
    return subprocess.run(
        ["git", *GIT_CONFIG, *args], cwd=root, env=git_env(root), capture_output=True, text=True
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


async def test_user_git_config_and_env_do_not_leak(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text(
        "[user]\n\tname = leaked\n[commit]\n\tgpgsign = true\n[core]\n\thooksPath = /nonexistent\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    # As inside a git hook of an enclosing repository:
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere" / ".git"))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "leaked")
    wc = WorkCopy(tmp_path / "wc")
    await wc.init()
    (wc.root / "a.txt").write_text("a")
    await wc.commit("turn 0")
    monkeypatch.undo()
    assert git(wc.root, "log", "-1", "--format=%an").stdout.strip() == "bakeoff"
    assert not (tmp_path / "elsewhere").exists()


async def test_never_uses_an_enclosing_repository(tmp_path):
    outer = WorkCopy(tmp_path)
    await outer.init()
    inner = WorkCopy(tmp_path / "inner")  # never initialised
    inner.root.mkdir()
    with pytest.raises(RuntimeError, match="not a git repository"):
        await inner.head()

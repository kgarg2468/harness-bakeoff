"""Git working copy per thread: one commit per completed turn; a revert is a new commit."""

from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import suppress
from pathlib import Path

# Fixed identity, no signing, and none of the user's global ignore or attributes files: git
# reads those from ~/.config/git even when there is no global config file.
GIT_CONFIG = (
    "-c",
    "user.name=bakeoff",
    "-c",
    "user.email=bakeoff@localhost",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "core.excludesFile=/dev/null",
    "-c",
    "core.attributesFile=/dev/null",
)


def git_env(root: Path) -> dict[str, str]:
    """Environment for git in `root`: no system/global config, no inherited GIT_* variables
    (e.g. GIT_DIR from a hook), and no walking up into an enclosing repository."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CEILING_DIRECTORIES=str(root.parent),
    )
    return env


def _failed(root: Path, args: tuple[str, ...], stderr: bytes) -> RuntimeError:
    return RuntimeError(f"git {args[0]} failed in {root}: {stderr.decode().strip()}")


class WorkCopy:
    """A git repository at `root` on branch `main`, starting with an empty commit."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self._env = git_env(self.root)

    async def _git(self, *args: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *GIT_CONFIG,
            *args,
            cwd=self.root,
            env=self._env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await proc.communicate()
        except asyncio.CancelledError:
            # Stop git before propagating, or it finishes on its own (e.g. a commit that no
            # turn records). SIGTERM, not SIGKILL, so git removes its lock files.
            with suppress(ProcessLookupError):
                proc.terminate()
            await proc.wait()
            raise
        if proc.returncode:
            raise _failed(self.root, args, err)
        return out.decode()

    def _git_sync(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(
            ["git", *GIT_CONFIG, *args],
            cwd=self.root,
            env=self._env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        if check and proc.returncode:
            raise _failed(self.root, args, proc.stderr)
        return proc.stdout.decode()

    def init_sync(self) -> None:
        """Create the repository with an empty initial commit (blocking).

        Idempotent. It also repairs a repository whose initial commit never happened (a crash
        right after `git init`), so every turn commit has a parent.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / ".git").exists():
            self._git_sync("init", "-q", "-b", "main")
        if not self._git_sync("rev-parse", "--verify", "-q", "HEAD", check=False):
            self._git_sync("commit", "-q", "--allow-empty", "-m", "init")

    async def init(self) -> None:
        """`init_sync` without blocking the event loop."""
        await asyncio.to_thread(self.init_sync)

    async def commit(self, message: str) -> tuple[str, list[str]]:
        """Commit everything in the working tree (even if nothing changed)."""
        await self._git("add", "-A")
        await self._git("commit", "-q", "--allow-empty", "-m", message)
        return await self._head_change()

    async def revert(self, sha: str) -> tuple[str, list[str]]:
        """Undo commit `sha` with a new commit; on a conflict, abort and raise."""
        # `revert --no-commit` + `commit --allow-empty` also works for an empty commit,
        # where a plain `git revert` fails with "nothing to commit".
        try:
            await self._git("revert", "--no-commit", sha)
        except RuntimeError:
            with suppress(RuntimeError):  # nothing to abort if the revert never started
                await self._git("revert", "--abort")
            raise
        await self._git("commit", "-q", "--allow-empty", "--no-edit")
        return await self._head_change()

    async def recover(self, sha: str | None) -> None:
        """Clean up after a worker that died while git was running.

        Removes a stale index lock and moves HEAD back to `sha` (None: the initial commit) if
        commits landed after it that no turn recorded. Their changes stay staged, so the next
        commit includes them. Call it only when no other git process can be using the repo.
        """
        (self.root / ".git" / "index.lock").unlink(missing_ok=True)
        target = sha or (await self._git("rev-list", "--max-parents=0", "HEAD")).split()[0]
        if await self.head() != target:
            await self._git("reset", "-q", "--soft", target)

    async def head(self) -> str:
        return (await self._git("rev-parse", "HEAD")).strip()

    async def _head_change(self) -> tuple[str, list[str]]:
        # With --always, diff-tree prints the commit id even when the commit changes nothing;
        # --root lists the files of a root commit too.
        out = await self._git("diff-tree", "-r", "--root", "--always", "--name-only", "-z", "HEAD")
        sha, *files = out.rstrip("\0").split("\0")
        return sha, files

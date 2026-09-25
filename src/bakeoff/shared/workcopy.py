"""Git working copy per thread: one commit per completed turn; a revert is a new commit."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from pathlib import Path

# Fixed identity and no signing, whatever the user's git config says.
GIT_CONFIG = (
    "-c",
    "user.name=bakeoff",
    "-c",
    "user.email=bakeoff@localhost",
    "-c",
    "commit.gpgsign=false",
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
        out, err = await proc.communicate()
        if proc.returncode:
            raise RuntimeError(f"git {args[0]} failed in {self.root}: {err.decode().strip()}")
        return out.decode()

    async def init(self) -> None:
        """Create the repository with an empty initial commit. No-op if it already exists."""
        if (self.root / ".git").exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        await self._git("init", "-q", "-b", "main")
        await self._git("commit", "-q", "--allow-empty", "-m", "init")

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

    async def head(self) -> str:
        return (await self._git("rev-parse", "HEAD")).strip()

    async def _head_change(self) -> tuple[str, list[str]]:
        # With --always, diff-tree prints the commit id even when the commit changes nothing.
        out = await self._git("diff-tree", "-r", "--always", "--name-only", "-z", "HEAD")
        sha, *files = out.rstrip("\0").split("\0")
        return sha, files

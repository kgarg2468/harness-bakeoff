"""The shared `ToolHost`: tool registry, argument validation, permission rules, tool events."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from bakeoff.shared import permissions
from bakeoff.shared.contract import Decision, Event, ToolCall, ToolResult, ToolSpec
from bakeoff.shared.contract import ToolError as ToolErrorKind
from bakeoff.shared.engine.base import Engine
from bakeoff.shared.engine.mock import MockEngine
from bakeoff.shared.tools import PathError, Tool, ToolContext, ToolError, resolve_path
from bakeoff.shared.tools.engine_tools import ENGINE_TOOLS
from bakeoff.shared.tools.file_tools import FILE_TOOLS
from bakeoff.shared.tools.skill_tools import SKILL_TOOLS

MAX_OUTPUT = 30_000
MAX_ERROR = 300  # schema messages quote the offending value, which can be a whole pipeline

# DESIGN.md order; never reorder, and append new tools at the end.
TOOLS: tuple[Tool, ...] = (*ENGINE_TOOLS, *FILE_TOOLS, *SKILL_TOOLS)
_BY_NAME = {tool.spec.name: tool for tool in TOOLS}
_VALIDATORS = {tool.spec.name: Draft202012Validator(tool.spec.parameters) for tool in TOOLS}


def build_toolhost(
    workdir: Path,
    rules: dict,
    emit: Callable[[Event], None],
    engine: Engine | None = None,
) -> ToolHostImpl:
    """A `ToolHost` over the working copy `workdir` (engine defaults to `MockEngine()`)."""
    return ToolHostImpl(workdir, rules, emit, MockEngine() if engine is None else engine)


class ToolHostImpl:
    """Implements `contract.ToolHost` for one thread's working copy and permission rules.

    `run_counts` maps each call id to how many times its tool actually executed.
    """

    def __init__(
        self, workdir: Path, rules: dict, emit: Callable[[Event], None], engine: Engine
    ) -> None:
        permissions.validate_rules(rules)
        self._ctx = ToolContext(workdir=workdir.resolve(), engine=engine)
        self._rules = rules
        self._emit = emit
        self.run_counts: dict[str, int] = {}
        self.emit_errors: list[Exception] = []

    def specs(self) -> list[ToolSpec]:
        """All tools, always in the same order."""
        return [tool.spec for tool in TOOLS]

    def check(self, call: ToolCall) -> Decision:
        """allow / ask / deny for a call. Never raises.

        Calls that `run()` will reject without executing (unknown tool, bad arguments) are
        "allow", so the loop runs them and the model sees the error. A path outside the
        working copy is always "deny".
        """
        prepared = self._prepare(call)
        if isinstance(prepared, str):
            return "allow"
        return self._decide(*prepared)[0]

    async def run(self, call: ToolCall) -> ToolResult:
        """Validate, enforce deny, execute. Never raises, except `asyncio.CancelledError`."""
        self._safe_emit(Event("tool.start", {"call_id": call.id, "name": call.name}))
        start = time.perf_counter()
        ok = False
        try:
            error, content = await self._execute(call)
            ok = error is None
            if len(content) > MAX_OUTPUT:
                content = (
                    f"{content[:MAX_OUTPUT]}\n... [truncated {len(content) - MAX_OUTPUT} chars]"
                )
            return ToolResult(call_id=call.id, ok=ok, content=content, error=error)
        finally:
            ms = round((time.perf_counter() - start) * 1000, 3)
            self._safe_emit(
                Event("tool.end", {"call_id": call.id, "name": call.name, "ok": ok, "ms": ms})
            )

    def _safe_emit(self, event: Event) -> None:
        """Emit without letting a failing callback break run()'s never-raises guarantee.

        Failures are kept in `emit_errors` (tests and the runner can inspect them)."""
        try:
            self._emit(event)
        except Exception as e:
            self.emit_errors.append(e)

    async def _execute(self, call: ToolCall) -> tuple[ToolErrorKind | None, str]:
        """(error kind or None on success, content for the model)."""
        prepared = self._prepare(call)
        if isinstance(prepared, str):
            return "invalid_args", prepared
        tool, args = prepared
        decision, reason = self._decide(tool, args)
        if decision == "deny":
            return "denied", reason
        self.run_counts[call.id] = self.run_counts.get(call.id, 0) + 1
        try:
            return None, await tool.fn(args, self._ctx)
        except ToolError as e:
            return "failed", str(e)
        except Exception as e:  # run() never raises; the model gets a short message instead
            return "failed", f"{call.name} failed: {type(e).__name__}: {e}"[:MAX_ERROR]

    def _prepare(self, call: ToolCall) -> tuple[Tool, dict[str, Any]] | str:
        """The tool and parsed arguments, or the error message for the model."""
        tool = _BY_NAME.get(call.name)
        if tool is None:
            return f"Unknown tool: {call.name}. Available tools: {', '.join(_BY_NAME)}"
        try:
            args = json.loads(call.arguments or "{}")
        except (ValueError, RecursionError) as e:
            return f"Invalid arguments for {call.name}: not valid JSON ({e})"[:MAX_ERROR]
        if not isinstance(args, dict):
            return f"Invalid arguments for {call.name}: expected a JSON object"
        error = best_match(_VALIDATORS[call.name].iter_errors(args))
        if error is not None:
            where = "/".join(map(str, error.absolute_path))
            message = f"{where}: {error.message}" if where else error.message
            return f"Invalid arguments for {call.name}: {message}"[:MAX_ERROR]
        if tool.check_args and (problem := tool.check_args(args)):
            return f"Invalid arguments for {call.name}: {problem}"
        return tool, args

    def _decide(self, tool: Tool, args: dict[str, Any]) -> tuple[Decision, str]:
        """The decision for a valid call, and the message the model sees if it is denied."""
        name, path = tool.spec.name, args.get("path")
        if isinstance(path, str):
            try:
                full = resolve_path(self._ctx.workdir, path, write=not tool.spec.read_only)
            except PathError as e:
                return "deny", f"Denied: {e}"
            path = full.relative_to(self._ctx.workdir).as_posix()
        decision = permissions.evaluate(self._rules, name, path)
        return decision, f"Denied by permission rules: {name}" + (f" {path}" if path else "")

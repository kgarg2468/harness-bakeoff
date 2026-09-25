# Ported from OpenCode (MIT): packages/opencode/src/util/wildcard.ts, packages/opencode/src/permission/index.ts @ 16c56fe5ecc3305028d1f0a9cff5806e51c9d480
# Changes: `match` and `evaluate` only, over this repo's rule format (first match wins; the tool key beats "*").
"""Allow/ask/deny rules for tool calls, with OpenCode's wildcard patterns.

Rules are per thread (see DESIGN.md):

    {"*": "allow", "write_file": {"*.pipe": "allow", "*": "ask"}, "edit_file": "ask"}

A tool maps to a decision or to `{glob on path: decision}`, where the first matching glob wins.
The tool's own key is tried before `"*"`. When nothing matches, the decision is `"ask"`.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from typing import Any, cast, get_args

from bakeoff.shared.contract import Decision

DECISIONS: tuple[str, ...] = get_args(Decision)


def match(text: str, pattern: str) -> bool:
    """OpenCode's `Wildcard.match`: `*` matches any run of characters (including `/`), `?` one.

    A pattern ending in `" *"` also matches without the suffix: `"ls *"` matches `"ls"`.
    """
    text = text.replace("\\", "/")
    escaped = (
        re.sub(r"[.+^${}()|[\]\\]", r"\\\g<0>", pattern.replace("\\", "/"))
        .replace("*", ".*")
        .replace("?", ".")
    )
    if escaped.endswith(" .*"):
        escaped = escaped[:-3] + "( .*)?"
    flags = re.S | (re.I if sys.platform == "win32" else 0)
    return re.fullmatch(escaped, text, flags) is not None


def evaluate(rules: Mapping[str, Any], tool: str, path: str | None = None) -> Decision:
    """The decision for calling `tool` (on `path`, relative to the working copy, if any)."""
    for key in (tool, "*"):
        rule = rules.get(key)
        if isinstance(rule, str):
            return cast(Decision, rule)  # values are checked by validate_rules
        for pattern, decision in (rule or {}).items():
            if match(path or "", pattern):
                return decision
    return "ask"


def validate_rules(rules: Mapping[str, Any]) -> None:
    """Raise ValueError unless every decision in `rules` is allow, ask or deny."""
    for tool, rule in rules.items():
        for decision in rule.values() if isinstance(rule, Mapping) else [rule]:
            if decision not in DECISIONS:
                raise ValueError(f"Invalid permission rule for {tool!r}: {decision!r}")


def asks(rules: Mapping[str, Any]) -> bool:
    """Whether some call can evaluate to "ask", i.e. a person may have to approve something in
    this thread (so it is not unattended).

    Follows `evaluate`: a tool without a key of its own gets the "*" rule, and a path that no
    glob of the tool's mapping matches falls through to "*", then to "ask". It errs toward True:
    without a "*" key some tool is taken to have no key of its own, and a glob counts as
    reachable unless an earlier glob of its mapping matches every path.
    """
    default = _reachable(rules.get("*"), {"ask"})  # what a tool without its own key gets
    return "ask" in default or any(
        "ask" in _reachable(rule, default) for tool, rule in rules.items() if tool != "*"
    )


def _reachable(rule: Any, unmatched: set[str]) -> set[str]:
    """The decisions `rule` can give, where `unmatched` is what a path no glob matches gets."""
    if isinstance(rule, str):
        return {rule}
    found: set[str] = set()
    for pattern, decision in (rule or {}).items():
        found.add(decision)
        if not pattern.strip("*"):  # "*" (or "**") matches every path: later globs never run
            return found
    return found | unmatched

"""`load_skill`: read a RocketRide skill's playbook or reference file (see `shared/skills.py`)."""

from __future__ import annotations

from typing import Any

from bakeoff.shared import skills
from bakeoff.shared.contract import ToolSpec

from . import Tool, ToolContext


def _check(args: dict[str, Any]) -> str | None:
    """An unknown skill or file is a bad argument: the model can retry with a listed name."""
    try:
        skills.resolve(args["name"], args.get("file"))
    except ValueError as e:  # SkillError, or a broken bundle: the host must never raise
        return str(e)
    return None


def harness_note(name: str) -> str:
    """The paragraph in front of every `load_skill` result.

    The skills were written for RocketRide's MCP server and IDE agents, which have more tools
    (run_pipeline, monitor, WebFetch, their `tools/*.py` shims). This tells the model which of
    their instructions it can follow here.
    """
    # Imported here: toolhost imports this module to build its tool list.
    from bakeoff.shared.toolhost import TOOLS

    names = ", ".join(tool.spec.name for tool in TOOLS)
    files = ", ".join(skills.load_skills().skills[name].files)
    return (
        f"[harness] The only tools in this harness are: {names}. Where this skill says to run a "
        "script (tools/*.py, SDK or CLI calls) or to call a tool not in that list (run_pipeline, "
        "monitor, WebFetch, ...), that instruction does not apply here: use the closest tool "
        "listed, or tell the user what you could not do. The skill's Write and Edit tools are "
        "write_file and edit_file. Files the skill mentions are read with load_skill(name, "
        f"file), not read_file. Files of {name}: {files}."
    )


# RocketRide's skills stop at approval gates and wait for a person. Said here, right before the
# gate protocol, because a note in the system prompt alone did not stop models from waiting.
# Only gates a person answers are pre-approved: Gate C is a check (validate() returns zero
# errors), and approving it unchecked would ship a broken pipeline.
UNATTENDED_NOTE = (
    "This run is unattended: nobody can answer a gate that waits for a person. Treat each such "
    "gate in this skill as approved: state your choice in one line and continue to the end of "
    "the task. Gates that are checks still apply and must pass: Gate C (validation) passes only "
    "when validate_pipeline returns zero errors, so fix and re-validate until it does. Do not "
    "write gate state files."
)


async def load_skill(args: dict[str, Any], ctx: ToolContext) -> str:
    """The harness note (plus `UNATTENDED_NOTE` when nobody approves), then the requested file
    of the skill (default `SKILL.md`)."""
    name = args["name"]
    path = skills.resolve(name, args.get("file"))
    note = harness_note(name) + (f" {UNATTENDED_NOTE}" if ctx.unattended else "")
    return f"{note}\n\n--- {path} ---\n{skills.read(path)}"


# The names are in the tool description too, so a thread whose system prompt lacks
# `skills_prompt()` can still find them without first triggering the unknown-skill error.
_NAMES = ", ".join(skills.load_skills().skills)

SKILL_TOOLS = (
    Tool(
        ToolSpec(
            name="load_skill",
            description=(
                "Load a RocketRide skill, a step-by-step playbook for pipeline work: its SKILL.md, "
                f"or a reference file it mentions. Skills: {_NAMES}. Call it before acting on a "
                "task that matches a skill."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "One of the skill names"},
                    "file": {
                        # null too: models often send null for an optional argument.
                        "type": ["string", "null"],
                        "description": (
                            "A file the skill mentions, as it writes the path (e.g. "
                            "GATE_PROTOCOL.md, ../MCP_TOOL_CONTRACT.md). Default: SKILL.md"
                        ),
                    },
                },
                "required": ["name"],
            },
            read_only=True,
        ),
        load_skill,
        _check,
    ),
)

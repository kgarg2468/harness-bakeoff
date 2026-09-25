"""RocketRide engine tools: `list_components`, `describe_component`, `validate_pipeline`.

Names, descriptions and argument schemas follow the engine's built-in MCP tool registry
(rocketride-server `packages/ai/src/ai/modules/mcp/tools/introspection.py`, MIT).
`validate_pipeline` also accepts `path`, a pipeline file in the working copy.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bakeoff.shared.contract import ToolSpec

from . import Tool, ToolContext, ToolError, resolve_path


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _summary(description: Any) -> str:
    """The first sentence: every component's full description would overrun the output cap."""
    if isinstance(description, list):  # the engine keeps descriptions as lists of fragments
        description = " ".join(description)
    text = str(description or "").strip()
    found = re.match(r"(.+?[.!?])(\s|$)", text, re.S)
    return found.group(1) if found else text


async def list_components(args: dict[str, Any], ctx: ToolContext) -> str:
    """Every component the engine knows: name, category (classType) and a one-line summary."""
    services = await ctx.engine.get_services()
    components = [
        {
            "name": name,
            "category": service.get("classType"),
            "summary": _summary(service.get("description")),
        }
        for name, service in services.items()
    ]
    return _json({"ok": True, "components": components})


async def describe_component(args: dict[str, Any], ctx: ToolContext) -> str:
    """One component's definition: lanes, description and config fields."""
    name = args["name"]
    service = await ctx.engine.get_service(name)
    if service is None:
        raise ToolError(f"Unknown component: {name}. Call list_components for valid names.")
    return _json({"ok": True, "name": name, **service})


async def validate_pipeline(args: dict[str, Any], ctx: ToolContext) -> str:
    """Validate an inline pipeline or a pipeline file from the working copy."""
    if ("pipeline" in args) == ("path" in args):
        raise ToolError(
            "Invalid arguments for validate_pipeline: pass exactly one of 'pipeline' or 'path'"
        )
    pipeline = args.get("pipeline")
    if pipeline is None:
        path = args["path"]
        try:
            pipeline = json.loads(resolve_path(ctx.workdir, path).read_bytes())
        except (FileNotFoundError, IsADirectoryError):
            raise ToolError(f"No pipeline file at {path}") from None
        except ValueError as e:
            raise ToolError(f"{path} is not valid JSON: {e}") from None
        if not isinstance(pipeline, dict):
            raise ToolError(f"{path} does not contain a JSON object")
    return _json(await ctx.engine.validate(pipeline))


ENGINE_TOOLS = (
    Tool(
        ToolSpec(
            name="list_components",
            description=(
                "List RocketRide components ready to use now (zero-config plus integrations you "
                "have configured). Call describe_component for a config schema, "
                "list_integrations for integrations needing setup."
            ),
            parameters={"type": "object", "properties": {}},
            read_only=True,
        ),
        list_components,
    ),
    Tool(
        ToolSpec(
            name="describe_component",
            description=(
                "Describe a single RocketRide component: full metadata, lanes, and config schema."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Component name from list_components",
                    },
                },
                "required": ["name"],
            },
            read_only=True,
        ),
        describe_component,
    ),
    Tool(
        ToolSpec(
            name="validate_pipeline",
            description=(
                "Validate a pipeline against the engine's own rules (zero client-side rules -- "
                "zero drift)."
            ),
            # The engine's schema requires `pipeline`; `path` is the alternative. "Exactly one"
            # is checked in the tool: providers reject a top-level oneOf in tool schemas.
            parameters={
                "type": "object",
                "properties": {
                    "pipeline": {"type": "object", "description": "Inline pipeline definition"},
                    "path": {
                        "type": "string",
                        "description": "Pipeline file in the working copy (instead of pipeline)",
                    },
                },
            },
            read_only=True,
        ),
        validate_pipeline,
    ),
)

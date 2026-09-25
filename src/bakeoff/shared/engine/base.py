"""The `Engine` protocol behind the RocketRide tools, plus helpers every engine shares."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class Engine(Protocol):
    """The engine calls behind `list_components`, `describe_component` and `validate_pipeline`."""

    async def get_services(self) -> dict[str, dict]:
        """All service definitions, keyed by provider name."""
        ...

    async def get_service(self, name: str) -> dict | None:
        """One service definition, or None if the engine has no such provider."""
        ...

    async def validate(self, pipeline: dict) -> dict:
        """Validate a pipeline, bare or wrapped as `{"pipeline": {...}}`.

        Returns `{"ok": bool, "errors": [str], "warnings": [str]}`.
        """
        ...


def envelope(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Wrap a pipeline for the engine the way its MCP `validate_pipeline` tool does.

    An existing `{"pipeline": ...}` wrapper is unwrapped first. A missing version becomes 1,
    because the engine treats an unversioned pipeline as legacy v0.
    """
    body = pipeline.get("pipeline", pipeline)
    if isinstance(body, dict):
        body = {**body, "version": body.get("version", 1)}
    return {"pipeline": body}


def unknown_provider_errors(body: Any, services: Mapping[str, Any]) -> list[str]:
    """Errors for components whose provider the engine has no service for.

    The engine's pipeline validation never looks providers up, so the MCP tool adds this
    check from the engine's own catalog; every engine here does the same.
    """
    if not services or not isinstance(body, dict) or not isinstance(body.get("components"), list):
        return []
    return [
        f"Component {c.get('id')}: unknown provider {c['provider']!r}"
        for c in body["components"]
        if isinstance(c, dict) and isinstance(c.get("provider"), str)
        if c["provider"] and c["provider"] not in services
    ]

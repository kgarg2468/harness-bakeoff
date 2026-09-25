# Ported from RocketRide (MIT): packages/server/engine-lib/engLib/store/pipeline/pipeline_config.cpp @ a1fa4f15b5c61c51e450ffb6d24057b7edcf13d3
# Changes: the structural rules of PipelineConfig::validate(false) in Python; plus a lane check.
"""`MockEngine`: the RocketRide node catalog and pipeline rules, without an engine process."""

from __future__ import annotations

import asyncio
import functools
import json
from pathlib import Path
from typing import Any

from .base import envelope, unknown_provider_errors

# Package data (src/bakeoff/data/), so source checkouts and installed wheels both have it.
DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "data" / "catalog.json"
VERSION = 2  # IServices::VERSION (engLib/store/headers/services.hpp)
# Binder::MethodNames (engLib/store/headers/binder.hpp): every lane a connection may name.
METHOD_NAMES = frozenset(
    {
        "open",
        "tags",
        "text",
        "table",
        "words",
        "json",
        "audio",
        "video",
        "questions",
        "answers",
        "image",
        "classifications",
        "classificationContext",
        "documents",
        "closing",
        "close",
    }
)


@functools.cache
def _load_services(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))["providers"]


class MockEngine:
    """An in-process `Engine` over `bakeoff/data/catalog.json`.

    `validate` applies the engine's structural rules, the MCP tool's unknown-provider check and
    a lane check: each input lane must be one the target component accepts and one the source
    component produces. The lane check is stricter than the engine's `validate`, which only
    reports lane errors when a pipeline is opened with `use()`. Returned definitions are shared;
    treat them as read-only.
    """

    def __init__(self, delay_ms: int = 0, catalog_path: Path | None = None) -> None:
        self.delay_ms = delay_ms
        self._catalog_path = catalog_path or DEFAULT_CATALOG

    @property
    def _services(self) -> dict[str, dict]:
        # Read on first use, so building a ToolHost never needs the catalog (a wheel does
        # not ship data/); only the engine tools do.
        return _load_services(self._catalog_path)

    async def get_services(self) -> dict[str, dict]:
        """All service definitions, keyed by provider name."""
        await self._delay()
        return self._services

    async def get_service(self, name: str) -> dict | None:
        """One service definition, or None for an unknown provider."""
        await self._delay()
        return self._services.get(name)

    async def validate(self, pipeline: dict) -> dict:
        """Validate a pipeline; see the class docstring for the rules."""
        await self._delay()
        root = envelope(pipeline)
        structural = structure_error(root)
        errors = [structural] if structural else []
        errors += unknown_provider_errors(root["pipeline"], self._services)
        if not structural:
            errors += self._lane_errors(root["pipeline"]["components"])
        return {"ok": not errors, "errors": errors, "warnings": []}

    async def _delay(self) -> None:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)

    def _lane_errors(self, components: list[dict[str, Any]]) -> list[str]:
        providers = {c["id"]: c["provider"] for c in components}
        errors = []
        for component in components:
            target = self._services.get(component["provider"])
            for edge in component.get("input", []):
                lane, source_id = edge["lane"], edge["from"]
                source = self._services.get(providers[source_id])
                if target is None or source is None:
                    continue  # reported as an unknown provider
                # "_source" is not an input: it lists what a source component emits.
                accepts = [name for name in target["lanes"] if name != "_source"]
                if lane not in accepts:
                    errors.append(
                        f"Component {component['id']} input lane {lane} not found in service "
                        f"definition ({component['provider']} accepts: {_names(accepts)})"
                    )
                    continue
                produces = {out for outs in source["lanes"].values() for out in outs}
                if lane not in produces:
                    errors.append(
                        f"Component {component['id']} input lane {lane} is not produced by "
                        f"component {source_id} ({providers[source_id]} produces: {_names(produces)})"
                    )
        return errors


def _names(lanes: Any) -> str:
    return ", ".join(sorted(lanes)) or "none"


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    """JsonCpp's `Value::isInt`: a 32-bit integer, also when written as a real such as `1.0`."""
    return _is_number(value) and -(2**31) <= value < 2**31 and float(value).is_integer()


def structure_error(root: dict[str, Any]) -> str | None:
    """The first rule `PipelineConfig::validate(false)` breaks, in the engine's words, or None.

    `root` is an enveloped pipeline (see `envelope`), so it is always an object (rule 1).
    """
    pipeline = root.get("pipeline")
    if not isinstance(pipeline, dict):
        return "'pipeline' is missing or invalid"
    if "version" in pipeline:
        version = pipeline["version"]
        if not _is_int(version):
            return "'pipeline.version' must be a number"
        if not 1 <= version <= VERSION:
            return "'pipeline.version' is unsupported"
    components = pipeline.get("components")
    if not isinstance(components, list):
        return "'pipeline.components' must be an array"

    ids: set[str] = set()
    for component in components:
        if error := _component_error(component):
            return error
        if component["id"] in ids:
            return f"Duplicate component {component['id']}"
        ids.add(component["id"])

    # validate(false): the source is optional, but a named source must exist. The engine reads
    # it as text first, and a number converts to a name that is not a string.
    source = pipeline.get("source")
    if _is_number(source):
        return "'pipeline.source' must be a non-empty string"
    if _is_text(source) and source not in ids:
        return f"'pipeline.source' references unknown component id: {source}"

    for component in components:
        cid = component["id"]
        for key, field in (("input", "lane"), ("control", "classType")):
            if key not in component:
                continue
            entries = component[key]
            if not isinstance(entries, list):
                return f"Component {cid} {key} must be an array"
            for entry in entries:
                if not isinstance(entry, dict):
                    return f"Component {cid} {key} entries must be objects"
                if not _is_text(entry.get(field)):
                    return f"Component {cid} {key} '{field}' must be a non-empty string"
                if key == "input" and entry["lane"] not in METHOD_NAMES:
                    return f"Component {cid} input has unknown lane {entry['lane']}"
                if not _is_text(entry.get("from")):
                    return f"Component {cid} {key} 'from' must be a non-empty string"
                if entry["from"] not in ids:
                    return f"Component {cid} {key} references unknown component id: {entry['from']}"
    return None


def _component_error(component: Any) -> str | None:
    """`PipelineConfig::validateComponent`."""
    if not isinstance(component, dict):
        return "Component must be an object"
    cid = component.get("id")
    if not _is_text(cid):
        return "Component 'id' must be a non-empty string"
    if not _is_text(component.get("provider")):
        return f"Component {cid} 'provider' must be a non-empty string"
    config = component.get("config")
    if not isinstance(config, dict):
        return f"Component {cid} missing 'config' object"
    if "profile" in config:
        profile = config["profile"]
        if not _is_text(profile):
            return f"Component {cid} config 'profile' must be a non-empty string"
        if profile in config:
            if not isinstance(config[profile], dict):
                return f"Component {cid} config missing profile object '{profile}'"
            if error := _secure_error(config[profile]):
                return f"Component {cid} config profile '{profile}': {error}"
    if "parameters" in config:
        if not isinstance(config["parameters"], dict):
            return f"Component {cid} config 'parameters' must be an object"
        if error := _secure_error(config["parameters"]):
            return f"Component {cid} config 'parameters': {error}"
    return None


def _secure_error(section: dict[str, Any]) -> str | None:
    """`PipelineConfig::validateSecureParameters`."""
    if "secureParameters" not in section:
        return None
    secure = section["secureParameters"]
    if not isinstance(secure, dict):
        return "'secureParameters' must be an object"
    if not isinstance(secure.get("secure"), list):
        return "'secureParameters' must have a 'secure' array"
    if not isinstance(secure.get("token"), str):
        return "'secureParameters' must have a 'token' string"
    return None

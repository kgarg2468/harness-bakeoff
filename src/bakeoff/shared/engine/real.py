"""`RealEngine`: the `Engine` protocol over a running local (OSS-mode) RocketRide engine.

Optional. `rocketride` (the RocketRide Python client) is not a dependency of this project;
it is imported when a `RealEngine` is created.
"""

from __future__ import annotations

from typing import Any

from .base import envelope, unknown_provider_errors


class RealEngine:
    """Wraps `RocketRideClient(uri=..., auth="local", env={})`.

    `uri` is required: nothing here may default to a production engine, and `env={}` stops the
    client from reading `ROCKETRIDE_*` variables or a `.env` file. Connects on first use; call
    `aclose()` when done.
    """

    def __init__(self, uri: str) -> None:
        if not uri:
            raise ValueError("RealEngine needs an explicit engine uri, e.g. ws://127.0.0.1:5565")
        try:
            from rocketride import RocketRideClient
        except ImportError as e:
            raise ImportError(
                "RealEngine needs the RocketRide client package `rocketride`, which is not a "
                "dependency of harness-bakeoff; install it to use a real engine"
            ) from e
        self._client = RocketRideClient(uri=uri, auth="local", env={})
        self._connected = False

    async def _call(self) -> Any:
        if not self._connected:
            await self._client.connect()
            self._connected = True
        return self._client

    async def get_services(self) -> dict[str, dict]:
        """All service definitions, keyed by provider name."""
        response = await (await self._call()).get_services()
        return response.get("services") or {}

    async def get_service(self, name: str) -> dict | None:
        """One service definition, or None (the client raises for an unknown name)."""
        if name not in await self.get_services():
            return None
        return await (await self._call()).get_service(name)

    async def validate(self, pipeline: dict) -> dict:
        """The engine's validation plus the MCP tool's unknown-provider check."""
        root = envelope(pipeline)
        result = await (await self._call()).validate(root)
        errors = [_message(e) for e in result.get("errors") or []]
        errors += unknown_provider_errors(root["pipeline"], await self.get_services())
        warnings = [_message(w) for w in result.get("warnings") or []]
        return {"ok": not errors, "errors": errors, "warnings": warnings}

    async def aclose(self) -> None:
        """Disconnect from the engine."""
        if self._connected:
            self._connected = False
            await self._client.disconnect()


def _message(entry: Any) -> str:
    """The engine reports `{"id"?, "ccode", "message"}` objects."""
    if not isinstance(entry, dict):
        return str(entry)
    message = str(entry.get("message", entry))
    return f"Component {entry['id']}: {message}" if entry.get("id") else message

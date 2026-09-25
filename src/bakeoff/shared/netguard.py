"""Loopback-only network guard (invariant I6).

`install()` patches socket connects so any connection to a non-loopback address raises.
Tests install it for the whole session; the CLI installs it unless `--live` is given.
"""

from __future__ import annotations

import ipaddress
import socket

_installed = False
_allowed_hosts: set[str] = set()


class NetworkBlocked(RuntimeError):
    pass


def _is_allowed(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX paths and other non-IP families
    host = str(address[0])
    if host in _allowed_hosts or host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False  # an unresolved hostname


def install(allow_hosts: tuple[str, ...] = ()) -> None:
    """Block every non-loopback connection. `allow_hosts` adds exact IPs/hostnames."""
    global _installed
    _allowed_hosts.update(allow_hosts)
    if _installed:
        return
    _installed = True
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: object) -> None:
        if not _is_allowed(address):
            raise NetworkBlocked(f"non-loopback connection blocked: {address!r}")
        return real_connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: socket.socket, address: object) -> int:
        if not _is_allowed(address):
            raise NetworkBlocked(f"non-loopback connection blocked: {address!r}")
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]

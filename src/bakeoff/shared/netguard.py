"""Loopback-only network guard (invariant I6).

`install()` patches socket connects and datagram sends so any traffic to a non-loopback
address raises.
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
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_sendmsg = getattr(socket.socket, "sendmsg", None)  # absent on Windows

    def connect(self: socket.socket, address: object) -> None:
        if not _is_allowed(address):
            raise NetworkBlocked(f"non-loopback connection blocked: {address!r}")
        return real_connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: socket.socket, address: object) -> int:
        if not _is_allowed(address):
            raise NetworkBlocked(f"non-loopback connection blocked: {address!r}")
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    def sendto(self: socket.socket, data: bytes, *args: object) -> int:
        # sendto(data, address) or sendto(data, flags, address)
        if args and not _is_allowed(args[-1]):
            raise NetworkBlocked(f"non-loopback datagram blocked: {args[-1]!r}")
        return real_sendto(self, data, *args)  # type: ignore[arg-type]

    def sendmsg(self: socket.socket, buffers: object, *args: object) -> int:
        # sendmsg(buffers[, ancdata[, flags[, address]]])
        if len(args) >= 3 and args[2] is not None and not _is_allowed(args[2]):
            raise NetworkBlocked(f"non-loopback datagram blocked: {args[2]!r}")
        return real_sendmsg(self, buffers, *args)  # type: ignore[misc]

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = sendto  # type: ignore[method-assign]
    if real_sendmsg is not None:
        socket.socket.sendmsg = sendmsg  # type: ignore[method-assign]
    _installed = True

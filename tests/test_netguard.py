import socket

import pytest

from bakeoff.shared.netguard import NetworkBlocked


def test_external_connection_is_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlocked):
            s.connect(("1.1.1.1", 443))
    finally:
        s.close()


def test_loopback_is_allowed():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(server.getsockname())
    finally:
        client.close()
        server.close()


def test_external_datagram_is_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(NetworkBlocked):
            s.sendto(b"x", ("1.1.1.1", 53))
        with pytest.raises(NetworkBlocked):
            s.sendto(b"x", 0, ("1.1.1.1", 53))
        if hasattr(s, "sendmsg"):  # absent on Windows
            with pytest.raises(NetworkBlocked):
                s.sendmsg([b"x"], [], 0, ("1.1.1.1", 53))
    finally:
        s.close()


def test_loopback_datagram_is_allowed():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(b"ping", receiver.getsockname())
        assert receiver.recv(16) == b"ping"
    finally:
        sender.close()
        receiver.close()

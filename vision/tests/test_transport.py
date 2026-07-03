import socket
import threading

import pytest

from vision.transport import SocketTransport, open_transport

try:
    import serial  # noqa: F401
    _HAS_PYSERIAL = True
except ImportError:
    _HAS_PYSERIAL = False


def test_open_transport_rejects_unknown_kind():
    with pytest.raises(ValueError, match="serial_transport"):
        open_transport({"serial_transport": "carrier-pigeon"})


@pytest.mark.skipif(
    not _HAS_PYSERIAL, reason="pyserial not installed"
)
def test_open_transport_usb_requires_locatable_port():
    # No port/vid/pid given -> SerialTransport should refuse clearly
    # rather than guessing or blocking.
    with pytest.raises(RuntimeError, match="serial port"):
        open_transport({"serial_transport": "usb"})


def _echo_server(sock):
    conn, _ = sock.accept()
    with conn:
        while True:
            data = conn.recv(4096)
            if not data:
                return
            conn.sendall(data)


def test_socket_transport_round_trip():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    thread = threading.Thread(target=_echo_server, args=(srv,), daemon=True)
    thread.start()
    try:
        transport = SocketTransport(host="127.0.0.1", port=port)
        try:
            transport.write_line("P:1.00 T:2.00")

            lines = []
            for _ in range(200):
                lines.extend(transport.read_lines())
                if lines:
                    break
            assert lines == ["P:1.00 T:2.00"]
        finally:
            transport.close()
    finally:
        srv.close()
        thread.join(timeout=1.0)


def test_socket_transport_write_is_best_effort_after_close():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    transport = SocketTransport(host="127.0.0.1", port=port)
    # The TCP handshake completes in-kernel as soon as connect() returns,
    # even before accept() is called -- dequeue then actively close from
    # the server side so the client sees a genuinely dead peer.
    conn, _addr = srv.accept()
    conn.close()
    srv.close()

    # Peer gone: write_line and read_lines must not raise -- the tracking
    # loop can never be allowed to block or crash on a transport hiccup.
    for _ in range(20):
        transport.write_line("PING")
    assert transport.read_lines() == []
    transport.close()

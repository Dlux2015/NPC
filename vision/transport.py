"""Serial transport implementing write_line(str)/read_lines() for both
real hardware (USB serial to the ESP32-S3) and the sim (TCP socket to
sim/servo_sim.py). Product code (vision/tracking.py, vision/calibrate.py)
goes through this module only -- no raw `serial.Serial` or `socket.socket`
calls anywhere else in vision/.

open_transport(profile) picks the implementation from
profile["serial_transport"]:
  "usb"    -> pyserial. Port resolved from profile["serial_port"], or by
              VID/PID (profile["serial_vid"]/["serial_pid"]).
  "socket" -> TCP to localhost:$CBOT_SIM_PORT (default 8735), speaking the
              exact bytes defined in shared/serial_protocol.py.

Both writes and reads are best-effort / non-blocking: a transport hiccup
must never stall the tracking frame loop (SS3.1 hard rule).
"""
import os
import socket

DEFAULT_SIM_PORT = 8735


class _LineBuffer:
    """Accumulates raw bytes and yields complete newline-terminated lines."""

    def __init__(self):
        self._buf = b""

    def feed(self, data):
        if not data:
            return []
        self._buf += data
        lines = self._buf.split(b"\n")
        self._buf = lines.pop()  # last chunk may be a partial line
        return [ln.decode("ascii", errors="replace") for ln in lines]


class SocketTransport:
    """TCP client transport used by the "socket" profile (sim)."""

    def __init__(self, host="127.0.0.1", port=None, connect_timeout=2.0):
        if port is None:
            port = int(os.environ.get("CBOT_SIM_PORT", DEFAULT_SIM_PORT))
        self._sock = socket.create_connection((host, port), timeout=connect_timeout)
        self._sock.setblocking(False)
        self._buf = _LineBuffer()

    def write_line(self, line):
        """Best-effort, non-blocking. Never raises into the caller."""
        if not line.endswith("\n"):
            line = line + "\n"
        try:
            self._sock.sendall(line.encode("ascii"))
        except (BlockingIOError, InterruptedError, OSError):
            pass  # drop this write, keep the tracking loop alive

    def read_lines(self):
        """Return whatever complete lines are currently available (may be
        empty). Never blocks."""
        try:
            data = self._sock.recv(4096)
        except (BlockingIOError, InterruptedError):
            return []
        except OSError:
            return []
        if data == b"":
            return []  # peer closed; treat as "nothing new" (best-effort)
        return self._buf.feed(data)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


class SerialTransport:
    """pyserial transport used by the "usb" profile (real ESP32-S3)."""

    def __init__(self, port=None, baud=None, vid=None, pid=None, timeout=0):
        try:
            import serial
            from serial.tools import list_ports
        except ImportError as exc:
            raise RuntimeError(
                "SerialTransport requires pyserial. Install with "
                "`pip install pyserial`."
            ) from exc

        from shared import serial_protocol
        baud = baud or serial_protocol.BAUD

        resolved = port or self._resolve_port(list_ports, vid, pid)
        if resolved is None:
            raise RuntimeError(
                "Could not find a serial port (port=%r vid=%r pid=%r); "
                "pass profile['serial_port'] explicitly." % (port, vid, pid)
            )
        # timeout=0 -> non-blocking reads (returns immediately, possibly empty)
        self._ser = serial.Serial(resolved, baud, timeout=timeout, write_timeout=0)
        self._buf = _LineBuffer()

    @staticmethod
    def _resolve_port(list_ports, vid, pid):
        if vid is None and pid is None:
            return None
        for info in list_ports.comports():
            if vid is not None and info.vid != vid:
                continue
            if pid is not None and info.pid != pid:
                continue
            return info.device
        return None

    def write_line(self, line):
        if not line.endswith("\n"):
            line = line + "\n"
        try:
            self._ser.write(line.encode("ascii"))
        except Exception:
            pass  # best-effort: never block/raise out of the tracking loop

    def read_lines(self):
        try:
            n = self._ser.in_waiting
            if not n:
                return []
            data = self._ser.read(n)
        except Exception:
            return []
        return self._buf.feed(data)

    def close(self):
        try:
            self._ser.close()
        except Exception:
            pass


def open_transport(profile):
    """profile: dict loaded from profiles/<name>/profile.yaml."""
    kind = profile.get("serial_transport", "socket")
    if kind == "usb":
        return SerialTransport(
            port=profile.get("serial_port"),
            baud=profile.get("serial_baud"),
            vid=profile.get("serial_vid"),
            pid=profile.get("serial_pid"),
        )
    if kind == "socket":
        return SocketTransport(
            host=profile.get("serial_host", "127.0.0.1"),
            port=profile.get("serial_port_num"),
        )
    raise ValueError(
        "unknown serial_transport %r (expected 'usb' or 'socket')" % kind
    )

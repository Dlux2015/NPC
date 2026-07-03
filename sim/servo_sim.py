"""Virtual ESP32 — the digital twin of firmware/main.py.

Speaks the exact wire format from shared/serial_protocol.py. Imports the
real firmware/easing.py (HeadController, ServoAxis) — the sim runs the
actual easing/limit/idle-scan logic, never a re-implementation.

Two usage modes:
  - In-process (fast, deterministic): ServoSim.step(dt) / inject_line() /
    read_lines(). No socket, no threads — this is what scenarios/tests use.
  - TCP server (matches the real serial_transport=socket profile):
    `python sim/servo_sim.py`, or import SimServoServer. Listens on
    127.0.0.1:$CBOT_SIM_PORT (default 8735).

Sim time vs wall time: ServoSim.step(dt) advances *sim* time by exactly dt
regardless of how long that took to compute — callers can sub-step at 50Hz
for physical fidelity, or take larger strides to run scenarios faster than
real time. The TCP server ticks the sim at 50Hz of wall time (optionally
scaled via time_scale) since it has no other clock to follow.
"""
import os
import socket
import threading
import time

from firmware.easing import HeadController, ServoAxis
from shared.serial_protocol import (
    PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX, HEARTBEAT_TIMEOUT_S, ANGLE_REPORT_HZ,
    parse_line, encode_angles, encode_pong,
)

DEFAULT_PORT = 8735
DEFAULT_LATENCY_S = 0.03   # modeled host<->MCU command latency
TICK_HZ = 50.0
TICK_DT = 1.0 / TICK_HZ


class ServoSim(object):
    """In-process virtual ESP32: the HeadController plus wire-format I/O."""

    def __init__(self, latency_s=DEFAULT_LATENCY_S,
                 heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S):
        pan = ServoAxis(PAN_MIN, PAN_MAX)
        tilt = ServoAxis(TILT_MIN, TILT_MAX)
        self.head = HeadController(pan, tilt, heartbeat_timeout_s)
        self.latency_s = latency_s
        self.now = 0.0
        self._inbox = []          # [(arrival_time_s, line)], unordered
        self._outbox = []         # pending MCU->host lines
        self._report_period = 1.0 / ANGLE_REPORT_HZ
        self._next_report_s = 0.0

    def inject_line(self, line):
        """Queue one host->MCU line (as sent over the wire); applied once
        modeled latency elapses. Garbage/malformed lines are queued too and
        silently dropped at apply time — that's the protocol-robustness
        contract, not a special case here."""
        self._inbox.append((self.now + self.latency_s, line))

    def _apply(self, line):
        parsed = parse_line(line)
        if parsed is None:
            return  # garbage: ignored
        kind = parsed[0]
        if kind == "target":
            # ServoAxis.set_target (inside HeadController.command) clamps to
            # PAN_MIN/MAX, TILT_MIN/MAX regardless of what was requested.
            self.head.command(parsed[1], parsed[2], self.now)
        elif kind == "ping":
            self.head.heartbeat(self.now)
            self._outbox.append(encode_pong())
        # "angles"/"pong" are MCU->host only; a host would never send them,
        # but if one arrives inbound just ignore it rather than special-case.

    def step(self, dt):
        """Advance dt seconds of sim time: apply due inbound lines, run the
        HeadController, emit an angle report if due. Returns (pan, tilt)."""
        self.now += dt
        due, pending = [], []
        for arrival, line in self._inbox:
            (due if arrival <= self.now else pending).append((arrival, line))
        self._inbox = pending
        due.sort(key=lambda item: item[0])
        for _, line in due:
            self._apply(line)
        pan, tilt = self.head.step(dt, self.now)
        if self.now >= self._next_report_s:
            self._outbox.append(encode_angles(pan, tilt))
            self._next_report_s = self.now + self._report_period
        return pan, tilt

    def read_lines(self):
        """Pop and return all pending MCU->host lines."""
        out = self._outbox
        self._outbox = []
        return out


class SimServoServer(object):
    """TCP front-end for ServoSim: real socket, real 50Hz wall-clock tick.

    Multiple clients may connect (all see the same twin, broadcast angle
    reports); each client's inbound lines feed the one shared ServoSim.
    """

    def __init__(self, port=None, latency_s=DEFAULT_LATENCY_S, time_scale=1.0):
        self.port = port if port is not None else int(
            os.environ.get("CBOT_SIM_PORT", DEFAULT_PORT))
        self.time_scale = max(time_scale, 1e-6)  # >1 = faster than real time
        self.sim = ServoSim(latency_s=latency_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._clients = []
        self._clients_lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", self.port))
        self._sock.listen(5)

    def _tick_loop(self):
        period = TICK_DT / self.time_scale
        next_t = time.time()
        while not self._stop.is_set():
            with self._lock:
                self.sim.step(TICK_DT)
                out = self.sim.read_lines()
            if out:
                self._broadcast(out)
            next_t += period
            sleep_s = next_t - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.time()  # fell behind; don't spiral

    def _broadcast(self, lines):
        data = "".join(lines).encode("ascii")
        with self._clients_lock:
            dead = []
            for c in self._clients:
                try:
                    c.sendall(data)
                except OSError:
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)

    def _recv_loop(self, conn):
        buf = b""
        conn.settimeout(0.5)
        try:
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    with self._lock:
                        self.sim.inject_line(line.decode("ascii", "ignore"))
        except OSError:
            pass
        finally:
            with self._clients_lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            conn.close()

    def serve_forever(self):
        print("servo_sim listening on 127.0.0.1:%d (latency=%dms, time_scale=%.1fx)"
              % (self.port, int(self.sim.latency_s * 1000), self.time_scale))
        threading.Thread(target=self._tick_loop, daemon=True).start()
        self._sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # listener closed by stop()
            with self._clients_lock:
                self._clients.append(conn)
            threading.Thread(target=self._recv_loop, args=(conn,), daemon=True).start()

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    server = SimServoServer()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()

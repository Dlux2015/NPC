"""Serial protocol between Jetson/sim (host) and ESP32-S3 (MCU).

MicroPython-safe: no typing, dataclasses, enum, or f-strings. Imported by
firmware/main.py, vision/tracking.py, and sim/servo_sim.py — one wire
format, one parser, no hand-formatted strings anywhere else.

Wire format (ASCII, newline-terminated, degrees as floats):
  host -> MCU   "P:<pan> T:<tilt>\n"   target angles
  host -> MCU   "PING\n"               heartbeat / liveness probe
  MCU  -> host  "A:<pan> <tilt>\n"     current-angle report
  MCU  -> host  "PONG\n"               heartbeat reply

Provisional limits below are bench-safe defaults; calibration (§3.5 step 3)
provisions the real per-shell numbers into the active profile, and firmware
is flashed/configured with the same values.
"""

PAN_MIN = -90.0
PAN_MAX = 90.0
TILT_MIN = -45.0
TILT_MAX = 45.0

HEARTBEAT_TIMEOUT_S = 2.0   # MCU enters idle scan after this much silence
ANGLE_REPORT_HZ = 2         # MCU angle-report rate

BAUD = 115200


def clamp(value, lo, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def encode_target(pan_deg, tilt_deg):
    """Host->MCU target command. Clamps to protocol limits."""
    p = clamp(pan_deg, PAN_MIN, PAN_MAX)
    t = clamp(tilt_deg, TILT_MIN, TILT_MAX)
    return "P:%.2f T:%.2f\n" % (p, t)


def encode_ping():
    return "PING\n"


def encode_angles(pan_deg, tilt_deg):
    """MCU->host current-angle report."""
    return "A:%.2f %.2f\n" % (pan_deg, tilt_deg)


def encode_pong():
    return "PONG\n"


def parse_line(line):
    """Parse one stripped line from either direction.

    Returns a tuple or None if malformed:
      ("target", pan, tilt) | ("angles", pan, tilt) | ("ping",) | ("pong",)
    """
    line = line.strip()
    if line == "PING":
        return ("ping",)
    if line == "PONG":
        return ("pong",)
    try:
        if line.startswith("P:"):
            parts = line.split()
            if len(parts) != 2 or not parts[1].startswith("T:"):
                return None
            return ("target", float(parts[0][2:]), float(parts[1][2:]))
        if line.startswith("A:"):
            parts = line[2:].split()
            if len(parts) != 2:
                return None
            return ("angles", float(parts[0]), float(parts[1]))
    except ValueError:
        return None
    return None

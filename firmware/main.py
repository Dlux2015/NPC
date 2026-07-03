"""CBot head firmware — ESP32-S3-DevKitC-1, MicroPython (Phase 1).

Drives the pan/tilt servos from serial angle commands using the shared
easing/limit logic in easing.py and the shared wire format in
serial_protocol.py. MicroPython-safe: no typing, dataclasses, nested
f-strings, or numpy.

Deployment (see firmware/README.md): shared/serial_protocol.py and
firmware/easing.py are copied to the board's filesystem ROOT as
serial_protocol.py and easing.py, so this file imports them flat. The
try/except fallbacks below let the very same file import on CPython from
the repo checkout, which is how firmware/test_firmware_logic.py tests the
logic on the dev PC.

Serial transport: USB-CDC. The DevKitC-1's native USB port exposes the
MicroPython REPL as a CDC serial device; sys.stdin/sys.stdout ARE that
port, so no UART wiring is needed — the same cable that flashes the board
carries the protocol. We read stdin non-blocking via select.poll (the
documented MicroPython pattern for USB-CDC input) and write replies to
sys.stdout.

Control loop: 50Hz via time.ticks_ms/ticks_diff (wrap-safe). Steady state
is allocation-free: the read buffer is pre-allocated and no objects are
created on ticks where no serial line completes (string allocs happen only
when a full line arrives or a 2Hz angle report goes out), so GC pauses
cannot stutter the servos.

Idle scan on heartbeat silence is owned by easing.HeadController
(contract §4.3) — nothing here duplicates it; we just keep feeding it
step(dt, now_s).
"""

try:
    import serial_protocol            # board root (deployed flat)
except ImportError:
    from shared import serial_protocol   # CPython dev checkout
try:
    import easing                     # board root (deployed flat)
except ImportError:
    from firmware import easing          # CPython dev checkout

# ---------------------------------------------------------------- pins/PWM
# Wiring (document of record until calibration provisions a profile):
#   pan  servo signal -> GPIO4
#   tilt servo signal -> GPIO5
# Servo V+ comes from the buck-converter rail, NOT the DevKit's 5V pin;
# grounds must be common. See firmware/README.md.
PAN_PIN = 4
TILT_PIN = 5

PWM_FREQ_HZ = 50          # standard analog-servo frame rate (20ms period)

# Angle -> pulse-width map. Full protocol range -90..+90 deg spans the
# conventional 500..2500us envelope; calibration (§3.5 step 3) provisions
# the real per-shell limits later — these are bench-safe defaults.
PULSE_MIN_US = 500        # -90 deg
PULSE_MAX_US = 2500       # +90 deg
ANGLE_MIN_DEG = -90.0
ANGLE_MAX_DEG = 90.0

LOOP_HZ = 50
LOOP_MS = 1000 // LOOP_HZ                       # 20ms tick
REPORT_MS = 1000 // serial_protocol.ANGLE_REPORT_HZ  # 2Hz angle reports

LINE_BUF_SIZE = 64        # longest legal line is ~30 bytes; 64 is roomy


def angle_to_pulse_us(deg):
    """Map angle in degrees to servo pulse width in microseconds.

    Linear over ANGLE_MIN..ANGLE_MAX -> PULSE_MIN..PULSE_MAX and clamped,
    so -90 -> 500us, 0 -> 1500us, +90 -> 2500us. Returns an int (us).
    """
    if deg < ANGLE_MIN_DEG:
        deg = ANGLE_MIN_DEG
    elif deg > ANGLE_MAX_DEG:
        deg = ANGLE_MAX_DEG
    span_us = PULSE_MAX_US - PULSE_MIN_US
    span_deg = ANGLE_MAX_DEG - ANGLE_MIN_DEG
    return int(PULSE_MIN_US + (deg - ANGLE_MIN_DEG) * span_us / span_deg
               + 0.5)


class LineBuffer(object):
    """Accumulates single characters into newline-terminated lines.

    Pre-allocated buffer; feed() allocates nothing until a full line
    completes. Oversized garbage lines are discarded rather than parsed.
    """

    def __init__(self, size=LINE_BUF_SIZE):
        self.buf = bytearray(size)
        self.size = size
        self.n = 0
        self.overflow = False

    def feed(self, ch):
        """Feed one 1-char string; returns a completed line str or None."""
        if ch == "\n":
            if self.overflow:
                self.overflow = False
                self.n = 0
                return None
            line = str(bytes(memoryview(self.buf)[0:self.n]), "utf-8")
            self.n = 0
            return line
        if self.n >= self.size:
            self.overflow = True          # drop rest of runaway line
            return None
        self.buf[self.n] = ord(ch)
        self.n += 1
        return None


class CommandHandler(object):
    """Pure serial-command logic — no hardware, testable on CPython.

    Owns the mapping from parsed protocol messages to HeadController
    actions. Parsing is ONLY serial_protocol.parse_line; replies are ONLY
    serial_protocol encoders.
    """

    def __init__(self, head):
        self.head = head

    def handle_line(self, line, now_s):
        """Process one received line. Returns a reply str to transmit,
        or None. Malformed input is ignored (returns None)."""
        msg = serial_protocol.parse_line(line)
        if msg is None:
            return None
        kind = msg[0]
        if kind == "target":
            # command() both sets targets and refreshes the heartbeat.
            self.head.command(msg[1], msg[2], now_s)
            return None
        if kind == "ping":
            self.head.heartbeat(now_s)
            return serial_protocol.encode_pong()
        return None                       # "angles"/"pong" are MCU->host


def make_head():
    """Build the HeadController with protocol-limit-clamped axes."""
    pan = easing.ServoAxis(serial_protocol.PAN_MIN, serial_protocol.PAN_MAX)
    tilt = easing.ServoAxis(serial_protocol.TILT_MIN,
                            serial_protocol.TILT_MAX)
    return easing.HeadController(
        pan, tilt,
        heartbeat_timeout_s=serial_protocol.HEARTBEAT_TIMEOUT_S)


# ------------------------------------------------------------ hardware side
# Everything below touches machine/select/sys/time and runs only on the
# board (start() is called from the __main__ guard). Tests import this
# module but never call start().

def start():
    import sys
    import time
    import select
    import machine

    pan_pwm = machine.PWM(machine.Pin(PAN_PIN), freq=PWM_FREQ_HZ)
    tilt_pwm = machine.PWM(machine.Pin(TILT_PIN), freq=PWM_FREQ_HZ)

    head = make_head()
    handler = CommandHandler(head)
    lines = LineBuffer()

    poller = select.poll()
    poller.register(sys.stdin, select.POLLIN)

    stdin = sys.stdin
    stdout = sys.stdout

    # Center both axes immediately (duty_ns: exact us, no rounding games).
    pan_pwm.duty_ns(angle_to_pulse_us(head.pan.current) * 1000)
    tilt_pwm.duty_ns(angle_to_pulse_us(head.tilt.current) * 1000)

    last_tick = time.ticks_ms()
    report_due_ms = 0                 # countdown to next 2Hz angle report
    now_s = 0.0                       # wrap-safe accumulated seconds

    while True:
        # --- drain serial (non-blocking; poll(0) returns immediately)
        while poller.poll(0):
            ch = stdin.read(1)
            if not ch:
                break
            line = lines.feed(ch)
            if line is not None:
                reply = handler.handle_line(line, now_s)
                if reply is not None:
                    stdout.write(reply)

        # --- fixed-rate control step (wrap-safe via ticks_diff)
        now = time.ticks_ms()
        dt_ms = time.ticks_diff(now, last_tick)
        last_tick = now
        dt = dt_ms * 0.001
        now_s += dt

        pan_deg, tilt_deg = head.step(dt, now_s)
        pan_pwm.duty_ns(angle_to_pulse_us(pan_deg) * 1000)
        tilt_pwm.duty_ns(angle_to_pulse_us(tilt_deg) * 1000)

        # --- 2Hz angle report
        report_due_ms -= dt_ms
        if report_due_ms <= 0:
            report_due_ms = REPORT_MS
            stdout.write(serial_protocol.encode_angles(pan_deg, tilt_deg))

        # --- sleep out the remainder of the 20ms tick
        elapsed = time.ticks_diff(time.ticks_ms(), now)
        if elapsed < LOOP_MS:
            time.sleep_ms(LOOP_MS - elapsed)


if __name__ == "__main__":
    start()

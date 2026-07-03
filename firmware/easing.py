"""Servo easing + limit logic. THE seam between firmware and sim.

MicroPython-safe (no typing/dataclasses/f-strings/numpy). Imported by BOTH
firmware/main.py (on the ESP32-S3) and sim/servo_sim.py (on the dev PC) —
the sim tests this exact code, not a re-implementation.

Model: slew-rate-limited exponential approach to target. Smooth start/stop
without position feedback (DS3218MG is open-loop); the deadband keeps a
converged axis from buzzing. Allocation-free: step() creates no objects.
"""


class ServoAxis(object):
    def __init__(self, lo_deg, hi_deg, max_dps=120.0, smoothing=6.0,
                 deadband_deg=0.5, start_deg=0.0):
        """max_dps: slew limit, deg/s. smoothing: 1/s time constant of the
        exponential approach (higher = snappier). Values are provisional
        until calibration provisions per-shell numbers."""
        self.lo = lo_deg
        self.hi = hi_deg
        self.max_dps = max_dps
        self.k = smoothing
        self.deadband = deadband_deg
        self.current = start_deg
        self.target = start_deg

    def set_target(self, deg):
        if deg < self.lo:
            deg = self.lo
        elif deg > self.hi:
            deg = self.hi
        self.target = deg

    def step(self, dt):
        """Advance dt seconds; returns new current angle (also stored)."""
        err = self.target - self.current
        if err < 0.0:
            aerr = -err
        else:
            aerr = err
        if aerr <= self.deadband:
            return self.current
        v = err * self.k                 # proportional approach velocity
        vmax = self.max_dps
        if v > vmax:
            v = vmax
        elif v < -vmax:
            v = -vmax
        self.current += v * dt
        return self.current


class HeadController(object):
    """Pan+tilt axes + heartbeat tracking + idle-scan fallback.

    Firmware and sim both drive this at ~50Hz via step(dt, now_s).
    """

    IDLE_PAN_SPAN = 60.0    # degrees each side of center while scanning
    IDLE_PAN_DPS = 20.0     # slow sweep speed
    IDLE_TILT = 0.0

    def __init__(self, pan_axis, tilt_axis, heartbeat_timeout_s=2.0):
        self.pan = pan_axis
        self.tilt = tilt_axis
        self.timeout = heartbeat_timeout_s
        self.last_cmd_s = 0.0
        self._scan_dir = 1.0
        self._scan_pos = 0.0

    def command(self, pan_deg, tilt_deg, now_s):
        self.pan.set_target(pan_deg)
        self.tilt.set_target(tilt_deg)
        self.last_cmd_s = now_s

    def heartbeat(self, now_s):
        self.last_cmd_s = now_s

    def is_idle(self, now_s):
        return (now_s - self.last_cmd_s) > self.timeout

    def step(self, dt, now_s):
        """Returns (pan_deg, tilt_deg) after advancing dt seconds."""
        if self.is_idle(now_s):
            self._scan_pos += self._scan_dir * self.IDLE_PAN_DPS * dt
            span = self.IDLE_PAN_SPAN
            if self._scan_pos > span:
                self._scan_pos = span
                self._scan_dir = -1.0
            elif self._scan_pos < -span:
                self._scan_pos = -span
                self._scan_dir = 1.0
            self.pan.set_target(self._scan_pos)
            self.tilt.set_target(self.IDLE_TILT)
        return (self.pan.step(dt), self.tilt.step(dt))

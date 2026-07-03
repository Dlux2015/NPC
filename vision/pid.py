"""Pure-Python PID controller (no numpy) for pixel/degree error correction.

Used by vision/tracking.py to turn a degree error (already converted from
pixel error via the calibrated deg_per_px) into a commanded angular
correction. Deliberately dependency-free so it is trivially unit-testable
and safe to import from calibrate.py / tests without cv2 or numpy.
"""


def _clamp(value, lo, hi):
    if lo is not None and value < lo:
        return lo
    if hi is not None and value > hi:
        return hi
    return value


class PID:
    """Standard PID with an integral anti-windup clamp, an error deadband,
    and an output clamp.

    - deadband: |error| <= deadband -> output is exactly 0.0 and neither
      the integral nor derivative history is advanced, so a converged
      axis doesn't creep or build up integral from sensor noise.
    - The integral term is clamped to integral_limits (defaults to
      output_limits) independently of the final output clamp -> this is
      the anti-windup: a long-saturated output can't leave a huge integral
      that overshoots once the error reverses.
    - update() accepts either an explicit dt, or a monotonically
      increasing `now` from which dt is derived versus the `now` passed on
      the previous call.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0,
                 output_limits=(-90.0, 90.0), integral_limits=None,
                 deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min, self.out_max = output_limits
        if integral_limits is None:
            integral_limits = output_limits
        self.i_min, self.i_max = integral_limits
        self.deadband = deadband

        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def reset(self):
        """Clear integral/derivative history (call when the target is lost)."""
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def update(self, error, dt=None, now=None):
        """Advance the controller one tick and return the clamped output.

        error: current error in the same units the gains expect (degrees,
               for tracking.py).
        dt: seconds since the last update. If omitted, it is derived from
            `now` versus the `now` passed on the previous call (the first
            call with only `now` yields dt=0 -> no integral/derivative
            contribution yet).
        """
        if dt is None:
            if now is not None and self._prev_time is not None:
                dt = now - self._prev_time
            else:
                dt = 0.0
        if now is not None:
            self._prev_time = now
        if dt < 0:
            dt = 0.0

        if abs(error) <= self.deadband:
            # Inside the deadband: hold still. Track error so a future
            # exit doesn't see a stale derivative spike.
            self._prev_error = error
            return 0.0

        self._integral += error * dt
        self._integral = _clamp(self._integral, self.i_min, self.i_max)

        derivative = 0.0
        if self._prev_error is not None and dt > 0:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        return _clamp(output, self.out_min, self.out_max)

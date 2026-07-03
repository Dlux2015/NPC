"""Proves the SHIPPING controller (vision.tracking.TrackingApp: absolute
pan_center + PID(pixel_error), calibrated deg/px, target persistence, ...)
closes the loop against the digital twin end-to-end -- not a bespoke
scenario-only P-controller (see conftest.py's note at the top of this
directory). Exercises the exact pieces the product uses at runtime:

  SimWorld ground truth -> SyntheticDetector -> TrackingApp -> a
  write_line()/read_lines() transport adapter -> in-process ServoSim
  (firmware/easing.py's real HeadController) -> SimCamera reads the twin's
  CURRENT (eased) angle back for the next frame.

Runs against a real profiles/sim/calibration.json (generated via
`vision.calibrate --auto` in the `sim_calibration` fixture below if this
checkout doesn't already have one) -- vision/tracking.py refuses to start
without a measured calibration (SS3.1), and this test is supposed to be
proof of the shipping path, calibration included.
"""
import json
import os

import pytest

from shared import ipc
from sim.servo_sim import ServoSim
from sim.world import SimCamera, SimWorld
from vision import calibrate
from vision.detector import SyntheticDetector
from vision.paths import profile_dir
from vision.tracking import TrackingApp

FRAME_HZ = 30.0
FRAME_DT = 1.0 / FRAME_HZ
PHYSICS_DT = 0.02  # 50Hz, matches firmware/servo_sim's real tick rate

STATIC_CONVERGED_PX = 30.0
MOVING_SETTLE_PX = 80.0
MOVING_ACQUIRE_GRACE_S = 2.0  # time to let the loop first lock/settle


class _ServoSimTransport:
    """write_line()/read_lines() adapter over an in-process ServoSim --
    the exact contract vision/transport.py's real transports implement,
    just routed straight to the twin instead of a socket (mirrors how
    vision/tests/test_tracking.py injects a FakeTransport)."""

    def __init__(self, servo_sim):
        self._sim = servo_sim

    def write_line(self, line):
        self._sim.inject_line(line)

    def read_lines(self):
        return self._sim.read_lines()


def _synthetic_detector(world, servo_sim):
    """SyntheticDetector wrapping the world's ground truth at the twin's
    CURRENT (eased) angles -- what a real detector would see through
    SimCamera.read(), converted to detect()'s (x, y, w, h, score) API."""
    def _ground_truth(frame_bgr):
        pan = servo_sim.head.pan.current
        tilt = servo_sim.head.tilt.current
        return [(x, y, w, h, 0.95)
                for (_face_id, x, y, w, h) in world.ground_truth(pan, tilt)]
    return SyntheticDetector(_ground_truth)


class _SimClock:
    """Advances TrackingApp's notion of 'now' in lockstep with sim time
    instead of wall-clock sleeps, so the scenario runs fast and
    deterministically while still exercising the PID's real dt-based math
    (same idea as vision/tests/test_tracking.py's _clock_box)."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture(scope="module")
def sim_calibration():
    """A genuinely measured profiles/sim/calibration.json (F1): generated
    via `vision.calibrate --auto` against the real sim world/servo twin if
    this checkout doesn't already have one. The product loop must run
    against real calibration, not a hand-typed fixture."""
    calib_path = os.path.join(profile_dir("sim"), "calibration.json")
    if not os.path.isfile(calib_path):
        calibrate.run("sim", auto=True)
    with open(calib_path) as f:
        return json.load(f)


def _run_product_loop(world, calibration, state_path, duration_s, move=None):
    """Drives the real TrackingApp.step() at FRAME_HZ, interleaved with
    ServoSim.step() at its real 50Hz physics rate -- same two-rate
    structure as conftest.py's run_tracking(), just with the shipping
    controller instead of the bespoke one. Returns [(t, (ex_px, ey_px) |
    None)] per physics tick, ground-truth error at the twin's actual pose.
    """
    servo = ServoSim()
    camera = SimCamera(world, servo)
    detector = _synthetic_detector(world, servo)
    transport = _ServoSimTransport(servo)
    state = ipc.SharedState(state_path)
    clock = _SimClock()

    app = TrackingApp(camera, detector, transport, state, calibration,
                       clock=clock)
    try:
        samples = []
        t = 0.0
        next_frame = 0.0
        while t < duration_s:
            if move is not None:
                move(t)
            if t >= next_frame:
                clock.now = t
                app.step()
                next_frame += FRAME_DT
            servo.step(PHYSICS_DT)
            t += PHYSICS_DT

            pan, tilt = servo.head.pan.current, servo.head.tilt.current
            boxes = world.ground_truth(pan, tilt)
            err = None
            if boxes:
                _face_id, x, y, w, h = boxes[0]
                ex = (x + w / 2.0) - world.frame_w / 2.0
                ey = (y + h / 2.0) - world.frame_h / 2.0
                err = (ex, ey)
            samples.append((t, err))
    finally:
        app.close()
    return samples


def _mag(err):
    return (err[0] ** 2 + err[1] ** 2) ** 0.5


def test_static_face_converges_within_30px_in_5s(tmp_path, sim_calibration):
    world = SimWorld()
    world.add_face(azimuth_deg=20.0, elevation_deg=0.0, face_id=0)

    samples = _run_product_loop(
        world, sim_calibration, str(tmp_path / "state.json"), duration_s=5.0)

    errs = [(t, e) for t, e in samples if e is not None]
    assert errs, "face never detected"

    converged_at = next(
        (t for t, e in errs if _mag(e) <= STATIC_CONVERGED_PX), None)
    assert converged_at is not None, (
        "TrackingApp (the shipping product controller) never converged "
        "within %spx in 5 sim-seconds" % STATIC_CONVERGED_PX)

    # No oscillation blowup: once converged, error stays converged.
    after = [_mag(e) for t, e in errs if t >= converged_at]
    assert max(after) <= STATIC_CONVERGED_PX, (
        "error grew again after convergence (oscillation)")


def test_moving_face_tracked_stably(tmp_path, sim_calibration):
    world = SimWorld()
    world.add_face(azimuth_deg=0.0, elevation_deg=0.0, face_id=0)

    def move(t):
        world.move_face(0, azimuth_deg=5.0 * t, elevation_deg=0.0)

    samples = _run_product_loop(
        world, sim_calibration, str(tmp_path / "state.json"),
        duration_s=10.0, move=move)

    errs = [(t, e) for t, e in samples if e is not None]
    assert errs and errs[-1][0] > 9.0, "lost the face before the end"

    settled = [_mag(e) for t, e in errs if t >= MOVING_ACQUIRE_GRACE_S]
    assert max(settled) <= MOVING_SETTLE_PX, (
        "tracking lag exceeded %spx" % MOVING_SETTLE_PX)

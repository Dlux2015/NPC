"""Shared fixtures for sim scenarios. Deterministic, in-process (no
sockets), numpy + stdlib only.

NOTE (integration review, F2): the `track` fixture below and the scenarios
built on it (test_closed_loop.py, test_moving_target.py) close the loop
with a bespoke, ad hoc incremental P-controller -- useful sim-
infrastructure tests that exercise sim/world.py + sim/servo_sim.py
themselves, but they never call vision/tracking.py's TrackingApp, so they
do NOT prove the shipping product controller (absolute pan_center +
PID(pixel_error), vision/tracking.py) closes the loop against the servo
model. That proof lives in test_product_loop.py, which drives the real
TrackingApp end-to-end instead. Keep both: these stay as sim-
infrastructure tests, test_product_loop.py is the product-loop proof.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # repo root, so `pytest sim/` works anywhere

import pytest

from shared.serial_protocol import encode_target

PHYSICS_DT = 0.02   # 50Hz controller tick
CONTROL_DT = 0.05   # host control loop, one target per 50ms


def run_tracking(world, servo, duration_s, gain=0.05, move=None):
    """Ground-truth-detector P-control loop against the in-process sim.

    Each control tick: read ground-truth bbox at the servo's CURRENT
    angles, convert pixel error to a new absolute target, send it as a
    real protocol line. Returns [(t, (ex_px, ey_px) | None)] per tick.
    `move(t)` lets a scenario reposition sprites on sim time.
    """
    samples = []
    t, next_control = 0.0, 0.0
    while t < duration_s:
        if move is not None:
            move(t)
        if t >= next_control:
            pan = servo.head.pan.current
            tilt = servo.head.tilt.current
            boxes = world.ground_truth(pan, tilt)
            err = None
            if boxes:
                _, x, y, w, h = boxes[0]
                ex = (x + w / 2.0) - world.frame_w / 2.0
                ey = (y + h / 2.0) - world.frame_h / 2.0
                # +ey = face below center = look down = decrease tilt
                servo.inject_line(encode_target(pan + gain * ex,
                                                tilt - gain * ey))
                err = (ex, ey)
            samples.append((t, err))
            next_control += CONTROL_DT
        servo.step(PHYSICS_DT)
        t += PHYSICS_DT
    return samples


@pytest.fixture
def track():
    return run_tracking

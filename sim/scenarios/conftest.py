"""Shared fixtures for sim scenarios. Deterministic, in-process (no
sockets), numpy + stdlib only."""
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

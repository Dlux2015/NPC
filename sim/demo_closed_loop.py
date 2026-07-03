"""Runnable closed-loop demo: python sim/demo_closed_loop.py

10s scripted scenario against the in-process digital twin: face appears at
azimuth 25 deg, P-control tracks it, face jumps to -10 deg at t=5s.
Prints per-second tracking error; exits 0 if converged (<30px) at the end.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.serial_protocol import encode_target
from sim.servo_sim import ServoSim
from sim.world import SimWorld

GAIN = 0.05        # deg per px error
PHYSICS_DT = 0.02  # 50Hz
CONTROL_DT = 0.05
DURATION_S = 10.0
CONVERGED_PX = 30.0


def main():
    world = SimWorld()
    world.add_face(azimuth_deg=25.0, elevation_deg=5.0, face_id=0)
    servo = ServoSim()

    t, next_control, next_print = 0.0, 0.0, 1.0
    last_err = None
    while t < DURATION_S:
        if 5.0 <= t < 5.0 + PHYSICS_DT:
            world.move_face(0, azimuth_deg=-10.0, elevation_deg=0.0)
            print("t=%4.1fs  [face jumps to azimuth -10 deg]" % t)
        if t >= next_control:
            pan, tilt = servo.head.pan.current, servo.head.tilt.current
            boxes = world.ground_truth(pan, tilt)
            if boxes:
                _, x, y, w, h = boxes[0]
                ex = (x + w / 2.0) - world.frame_w / 2.0
                ey = (y + h / 2.0) - world.frame_h / 2.0
                servo.inject_line(encode_target(pan + GAIN * ex,
                                                tilt - GAIN * ey))
                last_err = (ex ** 2 + ey ** 2) ** 0.5
            next_control += CONTROL_DT
        servo.step(PHYSICS_DT)
        t += PHYSICS_DT
        if t >= next_print:
            err = "%6.1f px" % last_err if last_err is not None else "  no face"
            print("t=%4.1fs  error=%s  pan=%6.2f  tilt=%6.2f"
                  % (t, err, servo.head.pan.current, servo.head.tilt.current))
            next_print += 1.0

    ok = last_err is not None and last_err <= CONVERGED_PX
    print("converged" if ok else "FAILED to converge (last error: %s px)"
          % ("%.1f" % last_err if last_err is not None else "n/a"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

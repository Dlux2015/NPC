"""Interactive visual sim: watch the REAL TrackingApp chase a face you
drive with the keyboard, through the virtual camera's eyes.

    python sim/demo_visual.py            # opens an OpenCV window
    python sim/demo_visual.py --selftest # headless 5 sim-seconds, exit 0

Keys:
  a / d   move face left / right (azimuth)
  w / s   move face up / down (elevation)
  j       jump face to a random azimuth
  x       person leaves / returns (watch idle scan kick in after 2s)
  n       NEW person walks in (new identity -> enroll, then recognized)
  q / Esc quit

The window shows exactly what the virtual camera sees at the servo head's
current angles: green box = detection, cross = frame center, HUD = state.
Recognition uses the same deterministic sim embedder as the e2e test, so
person identity (enroll on first sight, recognized on return) is real
shared/people.py behavior.
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from shared import ipc
from shared.people import PeopleStore
from sim.servo_sim import ServoSim
from sim.world import SimWorld, SimCamera
from sim.scenarios.test_full_robot import (
    _ServoSimTransport, _SimClock, _synthetic_detector,
    _make_fake_embed_face, FRAME_DT, PHYSICS_DT)
from vision.paths import profile_dir
from vision.tracking import TrackingApp, crop_face

MAX_IDS = 10          # pre-registered identity palette for the sim embedder
FACE_SIZE_PX = 140    # big enough that centered => person_in_range (>=0.25)
AWAY_AZ = 400.0       # parked far outside any view = "person left"


def load_sim_calibration():
    path = os.path.join(profile_dir("sim"), "calibration.json")
    if not os.path.exists(path):
        sys.exit("No sim calibration. Run: python -m vision.calibrate "
                 "--profile sim --auto")
    with open(path) as f:
        return json.load(f)


class VisualRig:
    def __init__(self):
        self.world = SimWorld()
        self.servo = ServoSim()
        self.camera = SimCamera(self.world, self.servo)
        self.clock = _SimClock()
        run_dir = tempfile.mkdtemp(prefix="cbot_visual_")
        self.state = ipc.SharedState(os.path.join(run_dir, "state.json"))
        self.people = PeopleStore(os.path.join(run_dir, "people.db"))
        self.app = TrackingApp(
            self.camera, _synthetic_detector(self.world, self.servo),
            _ServoSimTransport(self.servo), self.state,
            load_sim_calibration(), clock=self.clock,
            hold_s=1.0, recognition_interval_s=0.5,
            face_crop_cb=crop_face, people_store=self.people,
            embed_cb=_make_fake_embed_face(range(MAX_IDS)))
        # Face the user drives around
        self.face_id = 0
        self.az, self.el = 25.0, 0.0
        self.present = True
        self.world.add_face(self.az, self.el, size_px=FACE_SIZE_PX,
                            face_id=self.face_id)
        self.t = 0.0
        self._next_frame = 0.0

    def advance(self, sim_dt):
        end = self.t + sim_dt
        while self.t < end:
            if self.t >= self._next_frame:
                self.clock.now = self.t
                self.app.step()
                self._next_frame += FRAME_DT
            self.servo.step(PHYSICS_DT)
            self.t += PHYSICS_DT

    def move_face(self, daz, del_):
        if not self.present:
            return
        self.az = max(-80.0, min(80.0, self.az + daz))
        self.el = max(-18.0, min(18.0, self.el + del_))
        self.world.move_face(self.face_id, self.az, self.el)

    def jump(self):
        if self.present:
            self.az = random.uniform(-60, 60)
            self.world.move_face(self.face_id, self.az, self.el)

    def toggle_presence(self):
        self.present = not self.present
        self.world.move_face(self.face_id,
                             self.az if self.present else AWAY_AZ, self.el)

    def new_person(self):
        if self.face_id + 1 >= MAX_IDS:
            return
        self.world.move_face(self.face_id, AWAY_AZ, 0)  # old one leaves
        self.face_id += 1
        self.az, self.el, self.present = random.uniform(-50, 50), 0.0, True
        self.world.add_face(self.az, self.el, size_px=FACE_SIZE_PX,
                            face_id=self.face_id)

    def render_hud(self, cv2):
        pan = self.servo.head.pan.current
        tilt = self.servo.head.tilt.current
        frame = self.world.render(pan, tilt).copy()
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 24, 1)
        err_px = None
        for _fid, x, y, bw, bh in self.world.ground_truth(pan, tilt):
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
            err_px = ((x + bw / 2 - w / 2) ** 2
                      + (y + bh / 2 - h / 2) ** 2) ** 0.5
        st = self.state.read()
        pid = st.get("person_id")
        name = None
        if pid is not None:
            rec = self.people.get(int(pid))
            name = rec and rec.get("name")
        idle = self.servo.head.is_idle(self.servo.now)
        lines = [
            "t=%6.1fs   pan=%+6.1f  tilt=%+6.1f%s" % (
                self.t, pan, tilt, "   [IDLE SCAN]" if idle else ""),
            "err=%s px   present=%s  in_range=%s" % (
                "%5.1f" % err_px if err_px is not None else "  -- ",
                st.get("person_present"), st.get("person_in_range")),
            "person_id=%s%s   enrolled=%d" % (
                pid, " (%s)" % name if name else "", self.people.count()),
            "a/d w/s move  j jump  x leave/return  n new person  q quit",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (8, 20 + 18 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame

    def close(self):
        self.app.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="headless: run 5 sim-seconds, verify, exit")
    args = ap.parse_args()
    import cv2  # after argparse so --selftest failure messages stay clean

    rig = VisualRig()
    try:
        if args.selftest:
            rig.advance(5.0)
            frame = rig.render_hud(cv2)
            assert frame.shape == (rig.world.frame_h, rig.world.frame_w, 3)
            err = None
            for _f, x, y, bw, bh in rig.world.ground_truth(
                    rig.servo.head.pan.current, rig.servo.head.tilt.current):
                err = abs(x + bw / 2 - rig.world.frame_w / 2)
            assert err is not None and err < 30, err
            print("selftest OK: converged to %.1fpx, HUD renders" % err)
            return 0

        step_az = 4.0
        last = time.monotonic()
        while True:
            now = time.monotonic()
            rig.advance(min(now - last, 0.1))
            last = now
            cv2.imshow("CBot sim - virtual camera", rig.render_hud(cv2))
            k = cv2.waitKey(15) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("a"):
                rig.move_face(-step_az, 0)
            elif k == ord("d"):
                rig.move_face(step_az, 0)
            elif k == ord("w"):
                rig.move_face(0, -3.0)
            elif k == ord("s"):
                rig.move_face(0, 3.0)
            elif k == ord("j"):
                rig.jump()
            elif k == ord("x"):
                rig.toggle_presence()
            elif k == ord("n"):
                rig.new_person()
        cv2.destroyAllWindows()
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    sys.exit(main())

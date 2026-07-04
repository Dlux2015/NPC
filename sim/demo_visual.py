"""Interactive visual sim: watch the REAL TrackingApp chase a face,
through the camera's eyes.

    python sim/demo_visual.py             # virtual world, keyboard-driven
    python sim/demo_visual.py --camera 0  # YOUR webcam: real detection,
                                          # you are the face being tracked
    python sim/demo_visual.py --selftest  # headless 5 sim-seconds, exit 0

Keys (sim world only; webcam mode just uses q):
  a / d   move face left / right (azimuth)
  w / s   move face up / down (elevation)
  j       jump face to a random azimuth
  x       person leaves / returns (watch idle scan kick in after 2s)
  n       NEW person walks in (new identity -> enroll, then recognized)
  q / Esc quit

Sim mode: the window shows what the virtual camera sees at the servo
head's current angles; recognition uses the deterministic sim embedder,
so identity (enroll / recognize-on-return) is real shared/people.py
behavior.

Webcam mode (--camera <index>): frames come from your real webcam and
detection is REAL (YuNet if vision/models/face_detection_yunet_2023mar
.onnx exists -- download one-liner in vision/detector.py -- otherwise
OpenCV's bundled Haar cascade, zero downloads). The servo head stays
virtual: it can't move your webcam, so the HUD's pan/tilt shows where the
head WOULD point. Keep your face off-center and the head walks toward its
soft limit chasing you; center yourself and it settles. deg/px comes from
the sim calibration (a real rig would use its own measured profile).
Recognition is off (real SFace embedder lands at Phase 6).
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


class _Webcam:
    """cv2.VideoCapture wrapper: mirrored (natural for self-view), caches
    the last frame so the HUD can redraw what TrackingApp just consumed
    without stealing a second frame from the device."""

    def __init__(self, index):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(int(index))
        if not self._cap.isOpened():
            sys.exit("Could not open webcam index %s. Try another index "
                     "(0, 1, ...) or close apps using the camera." % index)
        self.last_frame = None

    def read(self):
        ok, frame = self._cap.read()
        if ok and frame is not None:
            frame = self._cv2.flip(frame, 1)
            self.last_frame = frame
        return ok, frame

    def release(self):
        self._cap.release()


class _HaarDetector:
    """Zero-download fallback detector (OpenCV's bundled Haar cascade),
    implementing the pinned detect(frame_bgr) -> [(x,y,w,h,score)] API.
    Noisier than YuNet -- fine for eyeballing the loop, not for tuning."""

    name = "haar (bundled fallback)"

    def __init__(self):
        import cv2
        self._cv2 = cv2
        self._c = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame_bgr):
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        faces = self._c.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
        return [(float(x), float(y), float(w), float(h), 0.8)
                for (x, y, w, h) in faces]


def _pick_real_detector():
    from vision.detector import YuNetDetector, DEFAULT_MODEL_PATH
    candidates = [DEFAULT_MODEL_PATH,
                  DEFAULT_MODEL_PATH.replace("2023mar", "2022mar")]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            det = YuNetDetector(model_path=path)
            # Probe on a dummy frame: some model/OpenCV version pairs only
            # fail at first inference (e.g. the 2023mar model needs
            # cv2 >= 4.8), and a crash mid-demo is worse than a fallback.
            det.detect(np.zeros((120, 160, 3), dtype=np.uint8))
            det.name = "yunet (%s)" % os.path.basename(path)
            return det
        except Exception as exc:
            print("YuNet unusable (%s): %s -- trying next option"
                  % (os.path.basename(path), exc), file=sys.stderr)
    return _HaarDetector()


def load_sim_calibration():
    path = os.path.join(profile_dir("sim"), "calibration.json")
    if not os.path.exists(path):
        sys.exit("No sim calibration. Run: python -m vision.calibrate "
                 "--profile sim --auto")
    with open(path) as f:
        return json.load(f)


class VisualRig:
    def __init__(self, camera_index=None):
        self.servo = ServoSim()
        self.clock = _SimClock()
        run_dir = tempfile.mkdtemp(prefix="cbot_visual_")
        self.state = ipc.SharedState(os.path.join(run_dir, "state.json"))
        self.last_step = None

        if camera_index is None:
            self.world = SimWorld()
            self.camera = SimCamera(self.world, self.servo)
            detector = _synthetic_detector(self.world, self.servo)
            self.detector_name = "sim ground truth"
            self.people = PeopleStore(os.path.join(run_dir, "people.db"))
            recog = dict(recognition_interval_s=0.5,
                         face_crop_cb=crop_face, people_store=self.people,
                         embed_cb=_make_fake_embed_face(range(MAX_IDS)))
        else:
            self.world = None
            self.camera = _Webcam(camera_index)
            detector = _pick_real_detector()
            self.detector_name = detector.name
            self.people = None
            recog = {}  # real SFace embedder lands at Phase 6

        self.app = TrackingApp(
            self.camera, detector, _ServoSimTransport(self.servo),
            self.state, load_sim_calibration(), clock=self.clock,
            hold_s=1.0, **recog)
        # Face the user drives around (sim world only)
        self.face_id = 0
        self.az, self.el = 25.0, 0.0
        self.present = True
        if self.world is not None:
            self.world.add_face(self.az, self.el, size_px=FACE_SIZE_PX,
                                face_id=self.face_id)
        self.t = 0.0
        self._next_frame = 0.0

    def advance(self, sim_dt):
        end = self.t + sim_dt
        while self.t < end:
            if self.t >= self._next_frame:
                self.clock.now = self.t
                self.last_step = self.app.step()
                self._next_frame += FRAME_DT
            self.servo.step(PHYSICS_DT)
            self.t += PHYSICS_DT

    def move_face(self, daz, del_):
        if self.world is None or not self.present:
            return
        self.az = max(-80.0, min(80.0, self.az + daz))
        self.el = max(-18.0, min(18.0, self.el + del_))
        self.world.move_face(self.face_id, self.az, self.el)

    def jump(self):
        if self.world is not None and self.present:
            self.az = random.uniform(-60, 60)
            self.world.move_face(self.face_id, self.az, self.el)

    def toggle_presence(self):
        if self.world is None:
            return
        self.present = not self.present
        self.world.move_face(self.face_id,
                             self.az if self.present else AWAY_AZ, self.el)

    def new_person(self):
        if self.world is None or self.face_id + 1 >= MAX_IDS:
            return
        self.world.move_face(self.face_id, AWAY_AZ, 0)  # old one leaves
        self.face_id += 1
        self.az, self.el, self.present = random.uniform(-50, 50), 0.0, True
        self.world.add_face(self.az, self.el, size_px=FACE_SIZE_PX,
                            face_id=self.face_id)

    def render_hud(self, cv2):
        pan = self.servo.head.pan.current
        tilt = self.servo.head.tilt.current
        if self.world is not None:
            frame = self.world.render(pan, tilt).copy()
        elif self.camera.last_frame is not None:
            frame = self.camera.last_frame.copy()
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 24, 1)
        err_px = None
        target = self.last_step and self.last_step.get("target")
        if target is not None:
            x, y, bw, bh = (int(target[0]), int(target[1]),
                            int(target[2]), int(target[3]))
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
            err_px = ((x + bw / 2.0 - w / 2.0) ** 2
                      + (y + bh / 2.0 - h / 2.0) ** 2) ** 0.5
        st = self.state.read()
        pid = st.get("person_id")
        name = None
        if pid is not None and self.people is not None:
            rec = self.people.get(int(pid))
            name = rec and rec.get("name")
        idle = self.servo.head.is_idle(self.servo.now)
        if self.people is not None:
            id_line = "person_id=%s%s   enrolled=%d" % (
                pid, " (%s)" % name if name else "", self.people.count())
            keys_line = ("a/d w/s move  j jump  x leave/return  "
                         "n new person  q quit")
        else:
            id_line = ("detector=%s   recognition off (Phase 6)"
                       % self.detector_name)
            keys_line = ("webcam mode: head is virtual -- center your "
                         "face and watch it settle.  q quit")
        lines = [
            "t=%6.1fs   pan=%+6.1f  tilt=%+6.1f%s" % (
                self.t, pan, tilt, "   [IDLE SCAN]" if idle else ""),
            "err=%s px   present=%s  in_range=%s" % (
                "%5.1f" % err_px if err_px is not None else "  -- ",
                st.get("person_present"), st.get("person_in_range")),
            id_line,
            keys_line,
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (8, 20 + 18 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame

    def close(self):
        self.app.close()
        if self.world is None:
            self.camera.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="headless: run 5 sim-seconds, verify, exit")
    ap.add_argument("--camera", default="sim", metavar="sim|INDEX",
                    help="'sim' (default) for the virtual world, or a "
                         "webcam index like 0 for real detection on your "
                         "own camera")
    args = ap.parse_args()
    import cv2  # after argparse so --selftest failure messages stay clean

    cam = None if args.camera == "sim" else int(args.camera)
    rig = VisualRig(camera_index=cam)
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

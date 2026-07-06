"""The tracking process: camera -> detector -> target persistence -> PID
-> shared/serial_protocol.py -> transport, plus a throttled IPC update and
a low-frequency recognition hook.

Loads profiles/$CBOT_PROFILE/{profile.yaml, calibration.json} (env var,
default "sim"). REFUSES to start without calibration.json -- there are no
hardcoded pixel->degree fallbacks (SS3.1 hard rule).

Hard rules enforced here (ORCHESTRATION.md SS3.1 / skills/vision-face-tracking):
  - The frame loop never blocks and makes ZERO filesystem calls: no
    LLM/disk/network calls in step(); transport writes are best-effort/
    non-blocking (vision/transport.py); IPC updates go through
    shared.ipc.ThreadedStateWriter.publish() (in-memory hand-off; a
    background daemon thread owns the actual disk write, coalesced to
    <=10Hz) instead of touching SharedState directly from step().
  - Pixel->degree conversion only via the measured deg_per_px from
    calibration.json.
  - PID outputs target angles, never PWM; serial only via
    shared/serial_protocol.py.
  - Target persistence holds one face for >= hold_s before allowing a
    switch to a different one (no crowd-snapping).
  - Recognition (people.py match/enroll, and eventually Phase 6's SFace
    inference + SQLite scan) runs on a dedicated background worker thread,
    never in the frame thread: step() only crops the face (cheap array
    slicing, no I/O) and hands it off non-blocking (dropped if the worker
    is still busy on a previous crop) at ~1Hz.
"""
import argparse
import json
import os
import sys
import threading
import time

from shared import ipc, serial_protocol
from vision.camera import open_camera
from vision.paths import profile_dir, load_profile_yaml
from vision.pid import PID
from vision.transport import open_transport

# Retuned during the F2 integration fix (sim/scenarios/test_product_loop.py):
# this loop recomputes an ABSOLUTE target every tick from the instantaneous
# pixel offset (pan_center + PID(error)), so a pure-P term settles at a
# fixed FRACTION of the way to center (steady-state error = azimuth/(1+kp)
# -- kp=0.6 leaves ~60% of the offset uncorrected) and the old ki=0.05 was
# too small to close that gap in anything under ~15s. The previous values
# were never validated end-to-end (that was the whole point of F2) --
# kp=0.8/ki=0.7/kd=0.0 was chosen empirically against sim/world.py's
# digital twin: monotonic convergence, no oscillation, well inside 30px by
# 5s. kd=0 because the per-frame error is recomputed fresh from a
# pixel-quantized detection each tick (not a smooth continuous signal), and
# any nonzero kd here amplified that quantization into oscillation rather
# than damping it.
DEFAULT_PID_GAINS = {"kp": 0.8, "ki": 0.7, "kd": 0.0}


class CalibrationError(RuntimeError):
    pass


def load_profile(name, root=None):
    """Returns (profile_dict, calibration_dict). Raises CalibrationError
    if profiles/<name>/calibration.json is missing -- no fallback."""
    profile = load_profile_yaml(name, root)
    calib_path = os.path.join(profile_dir(name, root), "calibration.json")
    if not os.path.isfile(calib_path):
        raise CalibrationError(
            "Profile %r has no calibration.json (%s).\n"
            "Run vision/calibrate.py first: "
            "`python -m vision.calibrate --profile %s` "
            "(see ORCHESTRATION.md SS3.5) -- tracking.py refuses to start "
            "without a measured calibration." % (name, calib_path, name)
        )
    with open(calib_path, "r") as f:
        calibration = json.load(f)
    return profile, calibration


# ---------------------------------------------------------------------------
# Target persistence
# ---------------------------------------------------------------------------

def _center(bbox):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return (x + w / 2.0, y + h / 2.0)


def _dist(a, b):
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class TargetTracker:
    """Holds one target across frames: nearest-bbox association to follow
    the same physical face as it moves, and a >= hold_s commitment before
    a different face is allowed to take over (no crowd-snapping).

    A detection is only accepted as "the same face" if its center falls
    within `gate_mult` * the larger of the two bbox sizes of the previous
    position -- a plausible single-frame motion. Anything farther away is
    treated as a candidate face, only adopted once the current lock has
    been held for >= hold_s (whether still detected or lost).
    """

    def __init__(self, hold_s=3.0, gate_mult=1.5, clock=time.monotonic):
        self.hold_s = hold_s
        self.gate_mult = gate_mult
        self._clock = clock
        self.current = None
        self._locked_since = None

    def _same_face(self, prev, cand):
        d = _dist(prev, cand)
        size = max(prev[2], prev[3], cand[2], cand[3])
        return d <= self.gate_mult * size

    def _commit(self, bbox, now):
        self.current = bbox
        self._locked_since = now

    def update(self, detections):
        now = self._clock()

        if self.current is None:
            if detections:
                self._commit(max(detections, key=lambda d: d[4]), now)
            return self.current

        match = None
        if detections:
            nearest = min(detections, key=lambda d: _dist(d, self.current))
            if self._same_face(self.current, nearest):
                match = nearest

        if match is not None:
            self.current = match
            return self.current

        # Current face not (plausibly) present this frame.
        held_for = now - self._locked_since
        if held_for >= self.hold_s:
            if detections:
                self._commit(max(detections, key=lambda d: d[4]), now)
            else:
                self.current = None
                self._locked_since = None
        # else: keep holding the last known position -- no snap yet.
        return self.current


# ---------------------------------------------------------------------------
# Recognition hook (structure only; real embeddings land in Phase 6)
# ---------------------------------------------------------------------------

def crop_face(frame, bbox):
    """Default face_crop_cb: slice bbox out of frame. Returns None if the
    bbox doesn't overlap the frame."""
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    ih, iw = frame.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(iw, int(x + w)), min(ih, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def embed_face(face_crop_bgr):
    """Compute a face embedding for people.py match/enroll.

    TODO(Phase 6): replace with SFace (cv2.FaceRecognizerSF) inference per
    SS3.1/skills/vision-face-tracking. Returns None (recognition inert)
    until then -- the hook is fully wired, just behind this stub.
    """
    return None


# ---------------------------------------------------------------------------
# Tracking loop
# ---------------------------------------------------------------------------

class TrackingApp:
    """One iteration of the tracking loop lives in step() so it can be
    unit-tested without threads, real time, or a real camera/transport."""

    def __init__(self, camera, detector, transport, state, calibration,
                 hold_s=3.0, in_range_frac=0.25, pid_gains=None,
                 recognition_interval_s=1.0, face_crop_cb=None,
                 people_store=None, embed_cb=None, clock=time.monotonic,
                 ipc_min_interval=0.1):
        self.camera = camera
        self.detector = detector
        self.transport = transport
        self.state = state
        self.calibration = calibration
        self.in_range_frac = in_range_frac
        self.clock = clock

        # SS3.1: the frame loop makes zero filesystem calls. publish() is a
        # non-blocking in-memory hand-off; the writer's own daemon thread
        # owns the actual SharedState.update() disk write, coalesced to
        # <=1/ipc_min_interval Hz (see shared/ipc.py).
        self._writer = ipc.ThreadedStateWriter(state, interval_s=ipc_min_interval)

        self.tracker = TargetTracker(hold_s=hold_s, clock=clock)

        axes = calibration["axes"]
        deg_per_px = calibration["deg_per_px"]
        deadband_deg = calibration.get("deadband_deg", 0.0)
        gains = pid_gains or {}
        pan_gains = dict(DEFAULT_PID_GAINS, **gains.get("pan", {}))
        tilt_gains = dict(DEFAULT_PID_GAINS, **gains.get("tilt", {}))

        self.pan_sign = axes["pan"].get("sign", 1)
        self.tilt_sign = axes["tilt"].get("sign", 1)
        self.pan_center = axes["pan"].get("center", 0.0)
        self.tilt_center = axes["tilt"].get("center", 0.0)
        self.deg_per_px_pan = deg_per_px["pan"]
        self.deg_per_px_tilt = deg_per_px["tilt"]

        # PID output is the correction *around* the calibrated center, so
        # its limits are the soft limits shifted into that frame.
        pan_out_limits = (axes["pan"]["min"] - self.pan_center,
                           axes["pan"]["max"] - self.pan_center)
        tilt_out_limits = (axes["tilt"]["min"] - self.tilt_center,
                            axes["tilt"]["max"] - self.tilt_center)

        self.pid_pan = PID(output_limits=pan_out_limits,
                            deadband=deadband_deg, **pan_gains)
        self.pid_tilt = PID(output_limits=tilt_out_limits,
                             deadband=deadband_deg, **tilt_gains)

        self.recognition_interval_s = recognition_interval_s
        self.face_crop_cb = face_crop_cb
        self.people_store = people_store
        # Injectable embedder (sim/tests inject a deterministic one);
        # defaults to module-level embed_face (real SFace at Phase 6).
        self.embed_cb = embed_cb if embed_cb is not None else embed_face
        self._last_recognition = float("-inf")
        self._was_present = False

        # Recognition worker (F5): the frame thread only crops (cheap array
        # slicing, no I/O) and hands the crop off non-blocking; a dedicated
        # background thread owns people.py access (match/enroll, and
        # eventually Phase 6's SFace inference) and publishes results via
        # the ThreadedStateWriter above -- never the frame thread.
        self._recognition_job = None
        self._recognition_lock = threading.Lock()
        self._recognition_wake = threading.Event()
        self._recognition_busy = threading.Event()
        self._recognition_stop = threading.Event()
        self._recognition_thread = None
        if self.face_crop_cb is not None and self.people_store is not None:
            self._recognition_thread = threading.Thread(
                target=self._recognition_worker, daemon=True)
            self._recognition_thread.start()

    def step(self):
        """One frame. Returns a small status dict, or None if the camera
        read failed (caller decides whether/how to retry)."""
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return None

        now = self.clock()
        h, w = frame.shape[0], frame.shape[1]

        detections = self.detector.detect(frame)
        target = self.tracker.update(detections)

        person_present = target is not None
        person_in_range = False
        pan_out = 0.0
        tilt_out = 0.0

        if target is not None:
            x, y, bw, bh, _score = target
            cx, cy = x + bw / 2.0, y + bh / 2.0
            err_x_px = cx - w / 2.0
            err_y_px = cy - h / 2.0

            err_pan_deg = self.pan_sign * err_x_px * self.deg_per_px_pan
            err_tilt_deg = self.tilt_sign * err_y_px * self.deg_per_px_tilt

            pan_out = self.pid_pan.update(err_pan_deg, now=now)
            tilt_out = self.pid_tilt.update(err_tilt_deg, now=now)

            person_in_range = (bh / float(h)) >= self.in_range_frac

            self.transport.write_line(serial_protocol.encode_target(
                self.pan_center + pan_out, self.tilt_center + tilt_out))
        else:
            # No face: stop sending -- the ESP32 owns idle scan on silence.
            self.pid_pan.reset()
            self.pid_tilt.reset()

        # Non-blocking hand-off; the ThreadedStateWriter's own thread owns
        # the disk write (SS3.1: no filesystem calls in the frame loop).
        self._writer.publish(person_present=person_present,
                              person_in_range=person_in_range)
        if self._was_present and not person_present:
            # Departure clears identity, else person_id goes stale and the
            # next visitor could be greeted as the previous one.
            self._writer.publish(person_id=None)
        self._was_present = person_present

        if (target is not None
                and now - self._last_recognition >= self.recognition_interval_s):
            self._last_recognition = now
            self._submit_recognition(frame, target)

        return {
            "target": target,
            "detections": detections,
            "person_present": person_present,
            "person_in_range": person_in_range,
            "pan_out": pan_out,
            "tilt_out": tilt_out,
        }

    def _submit_recognition(self, frame, target):
        """Frame-thread side of the recognition hand-off: crop (cheap, no
        I/O) and wake the worker, dropping the crop if the worker is still
        busy on a previous one -- never blocks the frame loop."""
        if self.face_crop_cb is None or self.people_store is None:
            return
        if self._recognition_busy.is_set():
            return  # worker still busy; drop this frame's crop
        crop = self.face_crop_cb(frame, target)
        if crop is None:
            return
        self._recognition_busy.set()
        with self._recognition_lock:
            self._recognition_job = crop
        self._recognition_wake.set()

    def _recognition_worker(self):
        """Background worker thread (F5): owns all people.py access."""
        while not self._recognition_stop.is_set():
            self._recognition_wake.wait()
            self._recognition_wake.clear()
            if self._recognition_stop.is_set():
                break
            with self._recognition_lock:
                crop = self._recognition_job
                self._recognition_job = None
            try:
                if crop is not None:
                    self._run_recognition(crop)
            finally:
                self._recognition_busy.clear()

    def _run_recognition(self, crop):
        """Runs on the recognition worker thread ONLY -- never the frame
        thread. Publishes results via the ThreadedStateWriter, never a
        direct SharedState write."""
        embedding = self.embed_cb(crop)
        if embedding is None:
            return  # stub until Phase 6
        match = self.people_store.match(embedding)
        if match is not None:
            self._writer.publish(person_id=str(match[0]))
        else:
            new_id = self.people_store.enroll(embedding)
            seq = self.state.get("new_person_seq") + 1
            self._writer.publish(person_id=str(new_id), new_person_seq=seq)

    def _wait_recognition_idle(self, timeout=1.0):
        """Test helper: blocks until any in-flight recognition job the
        worker picked up has finished. The frame loop never calls this."""
        deadline = time.time() + timeout
        while self._recognition_busy.is_set() and time.time() < deadline:
            time.sleep(0.001)

    def close(self):
        """Stops background threads. Both are daemons (harmless to skip in
        short scripts/tests), but call this for a clean shutdown of a
        long-running process."""
        if self._recognition_thread is not None:
            self._recognition_stop.set()
            self._recognition_wake.set()
            self._recognition_thread.join(timeout=1.0)
        self._writer.stop()

    def run_forever(self, target_hz=30.0, stop_flag=None):
        period = 1.0 / target_hz
        while stop_flag is None or not stop_flag():
            t0 = self.clock()
            self.step()
            elapsed = self.clock() - t0
            if elapsed < period:
                time.sleep(period - elapsed)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="CBot vision tracking process")
    parser.add_argument("--profile", default=None,
                         help="defaults to $CBOT_PROFILE, then 'sim'")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)

    name = args.profile or os.environ.get("CBOT_PROFILE", "sim")
    profile, calibration = load_profile(name)

    camera = open_camera(profile)
    transport = open_transport(profile)

    from vision.detector import YuNetDetector
    detector = YuNetDetector(model_path=profile.get("yunet_model_path"))

    state_path = profile.get("ipc_state_path") or os.path.join(
        profile_dir(name), "..", "..", "run", name, "state.json")
    state_path = os.path.normpath(state_path)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    state = ipc.SharedState(state_path)

    people_store = None
    try:
        from shared.people import PeopleStore
        people_db_path = profile.get("people_db_path") or os.path.normpath(
            os.path.join(profile_dir(name), "..", "..", "run", name, "people.db"))
        os.makedirs(os.path.dirname(people_db_path), exist_ok=True)
        people_store = PeopleStore(people_db_path)
    except Exception as exc:  # pragma: no cover - defensive only
        print("warning: recognition disabled (%s)" % exc, file=sys.stderr)

    app = TrackingApp(camera, detector, transport, state, calibration,
                       face_crop_cb=crop_face, people_store=people_store)
    try:
        app.run_forever(target_hz=args.fps)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Virtual camera world: synthetic panorama with face sprites.

No image assets — flat background + rectangular face sprites placed by
known azimuth/elevation, numpy only. This is a pinned contract: other
agents (vision, calibrate.py) code against SimWorld/SimCamera as-is.

Convention: azimuth/pan increase together (both "turn right"); elevation/
tilt increase together (both "look up"). A face is centered in frame when
pan == face.azimuth and tilt == face.elevation.

Module-level factory functions (used by vision/camera.py's
camera_source == "sim" and vision/calibrate.py's --auto mode) so the sim
profile works with no separately-launched process:

  open_camera(profile)      -> vision/camera.py's camera_source=="sim" hook.
  ground_truth_faces(frame) -> SyntheticDetector-compatible ground_truth_fn
                                for calibrate.py --auto.

Both lazily build-or-reuse one process-wide default harness: a SimWorld
with one face plus a sim/servo_sim.py SimServoServer -- the same
TCP-socket-speaking digital twin the "socket" serial_transport profile
already expects on 127.0.0.1:$CBOT_SIM_PORT (default 8735). Building the
harness on first use of *either* function (camera or ground truth) and
binding the server's listening socket synchronously in its constructor
means the port is already accepting connections before anything tries to
connect to it, regardless of which of camera/transport a caller opens
first.
"""
import threading

import numpy as np

BG_COLOR = (40, 40, 40)      # BGR flat background
FACE_COLOR = (170, 190, 210)  # BGR pale sprite fill


class SimWorld(object):
    def __init__(self, pan_fov_deg=62.0, frame_w=640, frame_h=480):
        self.pan_fov_deg = pan_fov_deg
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.deg_per_px_x = pan_fov_deg / float(frame_w)
        # Vertical FOV scaled by aspect ratio (square-pixel assumption) so
        # deg/px is the same on both axes.
        self.tilt_fov_deg = pan_fov_deg * frame_h / float(frame_w)
        self.deg_per_px_y = self.tilt_fov_deg / float(frame_h)
        self._faces = {}  # face_id -> {"azimuth", "elevation", "size"}

    def add_face(self, azimuth_deg, elevation_deg, size_px=80, face_id=0):
        self._faces[face_id] = {
            "azimuth": azimuth_deg, "elevation": elevation_deg, "size": size_px,
        }
        return face_id

    def move_face(self, face_id, azimuth_deg, elevation_deg):
        f = self._faces[face_id]
        f["azimuth"] = azimuth_deg
        f["elevation"] = elevation_deg

    def _project(self, pan_deg, tilt_deg, face):
        """(sprite angle - head angle) -> pixel center via linear deg/px."""
        dx_deg = face["azimuth"] - pan_deg
        dy_deg = face["elevation"] - tilt_deg
        x = self.frame_w / 2.0 + dx_deg / self.deg_per_px_x
        y = self.frame_h / 2.0 - dy_deg / self.deg_per_px_y  # up = -y on screen
        return x, y

    def render(self, pan_deg, tilt_deg):
        """-> uint8 BGR frame (frame_h, frame_w, 3), cv2.imshow-compatible."""
        frame = np.empty((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        frame[:, :] = BG_COLOR
        for face in self._faces.values():
            x, y = self._project(pan_deg, tilt_deg, face)
            half = face["size"] / 2.0
            x0, x1 = int(round(x - half)), int(round(x + half))
            y0, y1 = int(round(y - half)), int(round(y + half))
            xa, xb = max(0, x0), min(self.frame_w, x1)
            ya, yb = max(0, y0), min(self.frame_h, y1)
            if xa < xb and ya < yb:
                frame[ya:yb, xa:xb] = FACE_COLOR
        return frame

    def ground_truth(self, pan_deg, tilt_deg):
        """-> list of (face_id, x, y, w, h) for faces overlapping the frame
        at this head pose. Stands in for a real detector in scenarios."""
        out = []
        for face_id, face in self._faces.items():
            x, y = self._project(pan_deg, tilt_deg, face)
            size = face["size"]
            half = size / 2.0
            x0, y0 = x - half, y - half
            if x0 + size <= 0 or x0 >= self.frame_w:
                continue
            if y0 + size <= 0 or y0 >= self.frame_h:
                continue
            out.append((face_id, int(round(x0)), int(round(y0)), int(size), int(size)))
        return out


class SimCamera(object):
    """cv2.VideoCapture-compatible: read() -> (True, frame), using the
    servo sim's CURRENT (eased, not commanded) angles."""

    def __init__(self, world, servo_sim):
        self.world = world
        self.servo_sim = servo_sim

    def read(self):
        pan = self.servo_sim.head.pan.current
        tilt = self.servo_sim.head.tilt.current
        return True, self.world.render(pan, tilt)

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Default out-of-the-box harness for vision/camera.py + vision/calibrate.py
# ---------------------------------------------------------------------------

DEFAULT_FACE_AZIMUTH_DEG = 15.0
DEFAULT_FACE_ELEVATION_DEG = 0.0
DEFAULT_FACE_SIZE_PX = 80
DEFAULT_GROUND_TRUTH_SCORE = 0.95

_harness_lock = threading.Lock()
_default_world = None
_default_server = None


def _default_harness():
    """Builds, once per process, a SimWorld (one face) + SimServoServer
    (real TCP twin, listening on 127.0.0.1:$CBOT_SIM_PORT) and returns
    (world, server) on every call thereafter -- reused by both
    open_camera() and ground_truth_faces() so they see the same twin."""
    global _default_world, _default_server
    with _harness_lock:
        if _default_world is None:
            _default_world = SimWorld()
            _default_world.add_face(
                azimuth_deg=DEFAULT_FACE_AZIMUTH_DEG,
                elevation_deg=DEFAULT_FACE_ELEVATION_DEG,
                size_px=DEFAULT_FACE_SIZE_PX,
                face_id=0,
            )
        if _default_server is None:
            from sim.servo_sim import SimServoServer
            # Binding/listen()ing happens synchronously inside __init__, so
            # the port is already accepting connections as soon as this
            # returns -- before the accept/tick thread below even starts.
            _default_server = SimServoServer()
            threading.Thread(
                target=_default_server.serve_forever, daemon=True
            ).start()
    return _default_world, _default_server


def open_camera(profile):
    """vision/camera.py's camera_source == "sim" hook: profile is unused
    (the default harness is a fixed single-face bench world) -- accepted
    for symmetry with the other _open_* factories in vision/camera.py.
    Returns a SimCamera (read()/release()) backed by the default harness.
    """
    world, server = _default_harness()
    return SimCamera(world, server.sim)


def ground_truth_faces(frame_bgr):
    """SyntheticDetector-compatible ground_truth_fn for
    vision/calibrate.py --auto: (x, y, w, h, score) tuples for the default
    harness's faces, at the harness's servo twin's CURRENT (eased) angles
    -- matching what a real detector would see through SimCamera.read()."""
    world, server = _default_harness()
    pan = server.sim.head.pan.current
    tilt = server.sim.head.tilt.current
    return [
        (x, y, w, h, DEFAULT_GROUND_TRUTH_SCORE)
        for (_face_id, x, y, w, h) in world.ground_truth(pan, tilt)
    ]

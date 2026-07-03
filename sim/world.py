"""Virtual camera world: synthetic panorama with face sprites.

No image assets — flat background + rectangular face sprites placed by
known azimuth/elevation, numpy only. This is a pinned contract: other
agents (vision, calibrate.py) code against SimWorld/SimCamera as-is.

Convention: azimuth/pan increase together (both "turn right"); elevation/
tilt increase together (both "look up"). A face is centered in frame when
pan == face.azimuth and tilt == face.elevation.
"""
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

"""Camera sources implementing read() -> (ok, frame_bgr), matching the
calling convention of cv2.VideoCapture so callers never branch on source.

open_camera(profile) picks the implementation from profile["camera_source"]:
  "csi"  -> GStreamer pipeline for a Jetson CSI camera (IMX219 / Arducam).
  "usb"  -> cv2.VideoCapture by index or device string
            (profile["camera_device"], default 0).
  "sim"  -> sim.world's virtual camera (imported lazily; sim-engineer owns
            sim/world.py internals -- this module only threads the profile
            dict through to it, per the SS3.6 swap-point contract: product
            code runs unmodified, only the camera_source config changes).
"""


class _CaptureWrapper:
    """Thin read()/release() wrapper around a cv2.VideoCapture."""

    def __init__(self, capture):
        self._cap = capture

    def read(self):
        ok, frame = self._cap.read()
        return bool(ok), (frame if ok else None)

    def release(self):
        self._cap.release()


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python (cv2) is required to open a real camera. "
            "Install with `pip install opencv-python`."
        ) from exc
    return cv2


def _csi_pipeline(profile):
    sensor_id = profile.get("csi_sensor_id", 0)
    width = profile.get("csi_width", 1280)
    height = profile.get("csi_height", 720)
    fps = profile.get("csi_fps", 30)
    flip_method = profile.get("csi_flip_method", 0)
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
        % (sensor_id, width, height, fps, flip_method, width, height)
    )


def _open_csi(profile):
    cv2 = _require_cv2()
    pipeline = profile.get("csi_pipeline") or _csi_pipeline(profile)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open CSI camera via GStreamer pipeline: %s" % pipeline
        )
    return _CaptureWrapper(cap)


def _resolve_usb_device(device):
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    s = str(device).strip()
    return int(s) if s.isdigit() else s


def _open_usb(profile):
    cv2 = _require_cv2()
    device = _resolve_usb_device(profile.get("camera_device", 0))
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera %r" % (device,))
    return _CaptureWrapper(cap)


def _open_sim(profile):
    try:
        from sim import world
    except ImportError as exc:
        raise RuntimeError(
            "camera_source: sim requires a running sim (sim/world.py, "
            "owned by sim-engineer). Import failed: %s" % exc
        ) from exc
    # Thin pass-through: sim.world owns how a camera attaches to a running
    # virtual world/panorama. This call's contract is stable for
    # sim-engineer to implement: a factory that accepts the profile dict
    # and returns a read() -> (ok, frame_bgr) object.
    return world.open_camera(profile)


_OPENERS = {
    "csi": _open_csi,
    "usb": _open_usb,
    "sim": _open_sim,
}


def open_camera(profile):
    """profile: dict loaded from profiles/<name>/profile.yaml.

    Returns an object with read() -> (ok, frame_bgr) and release().
    """
    source = profile.get("camera_source", "sim")
    opener = _OPENERS.get(source)
    if opener is None:
        raise ValueError(
            "unknown camera_source %r (expected one of %s)"
            % (source, sorted(_OPENERS))
        )
    return opener(profile)

"""Face detectors implementing the pinned API:

    detect(frame_bgr) -> list of (x, y, w, h, score)

x, y, w, h are pixel units (top-left corner + size) in `frame_bgr`'s own
coordinate space; score is a float confidence, roughly in [0, 1]. This
module has exactly two implementations and neither talks to serial/IPC —
callers (vision/tracking.py, vision/calibrate.py, sim/world.py) are the
only things that know what to do with the boxes.

- YuNetDetector: cv2.FaceDetectorYN (OpenCV's YuNet ONNX model). The real
  camera-facing detector. CPU-only by design (the GPU belongs to the LLM,
  see ORCHESTRATION.md SS3.4) -- YuNet is built for CPU-speed inference.
- SyntheticDetector: wraps a ground-truth callable. Used by tests and by
  the sim (sim/world.py) so the full detect -> PID -> serial loop can be
  exercised deterministically without cv2 model weights.

Downloading the YuNet model (one-time; not committed to the repo):

    mkdir -p vision/models
    curl -L -o vision/models/face_detection_yunet_2023mar.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

Or point YuNetDetector(model_path=...) at any YuNet-format .onnx file.
"""
import os


class SyntheticDetector:
    """Test/sim detector wrapping a ground-truth-bbox callable.

    ground_truth_fn(frame_bgr) -> iterable of (x, y, w, h, score)

    This is the seam sim/world.py and vision/tests/ use to drive the
    tracking loop and calibrate.py's --auto mode without any real model.
    """

    def __init__(self, ground_truth_fn):
        self._fn = ground_truth_fn

    def detect(self, frame_bgr):
        return [tuple(d) for d in self._fn(frame_bgr)]


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_detection_yunet_2023mar.onnx",
)


class YuNetDetector:
    """cv2.FaceDetectorYN wrapper implementing the pinned detect() API."""

    def __init__(self, model_path=None, input_size=(320, 320),
                 score_threshold=0.6, nms_threshold=0.3, top_k=5000):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "YuNetDetector requires opencv-python (cv2), which is not "
                "installed. Install with `pip install opencv-python` (see "
                "vision/detector.py's module docstring for the model "
                "download step)."
            ) from exc

        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(path):
            raise RuntimeError(
                "YuNet model not found at %r. Download it (see "
                "vision/detector.py's module docstring), or pass an "
                "explicit model_path=." % path
            )

        self._cv2 = cv2
        self._input_size = (int(input_size[0]), int(input_size[1]))
        self._detector = cv2.FaceDetectorYN.create(
            path, "", self._input_size, score_threshold, nms_threshold, top_k,
        )

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._input_size:
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self._detector.detect(frame_bgr)
        results = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
                score = float(f[-1])
                results.append((x, y, bw, bh, score))
        return results

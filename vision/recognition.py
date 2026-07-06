"""SFace face embeddings (Phase 6 seam of ORCHESTRATION.md SS3.1):
the real `embed_cb` for vision/tracking.py's TrackingApp, replacing the
inert `embed_face` stub. cv2.FaceRecognizerSF inference on a plain BGR
face crop -> 128-d float32 embedding for shared/people.py match/enroll.

CPU-only by design -- the GPU belongs to the LLM (SS3.4). SFace is a
small model built for exactly this.

Downloading the model (one-time; .onnx files are gitignored):

    curl -L -o vision/models/face_recognition_sface_2021dec.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

Alignment note: FaceRecognizerSF's best accuracy comes from alignCrop()
with the detector's 5 facial landmarks, but the pinned detect() API
(vision/detector.py) deliberately returns plain (x, y, w, h, score)
boxes, so this embedder runs feature() on the raw crop resized to the
model's 112x112 input instead. That costs some accuracy across
pose/lighting changes -- compensated by SFACE_MATCH_THRESHOLD below
being tuned for unaligned crops, and good enough for same-camera
re-recognition. If cross-session recognition proves flaky at Phase 6
bench, extend detector.py to surface landmarks and switch to
alignCrop() rather than lowering the threshold further.
"""
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_recognition_sface_2021dec.onnx",
)

# Cosine-similarity match threshold for people.match() when embeddings
# come from THIS embedder. SFace's published verification threshold is
# 0.363 (aligned crops); unaligned same-camera crops of the same person
# in practice land well above it, and different people well below, so
# 0.363 stays a sound decision boundary. people.py's default
# MATCH_THRESHOLD (0.55) predates the real embedder and is too strict
# for unaligned crops -- pass this explicitly (TrackingApp's
# match_threshold param).
SFACE_MATCH_THRESHOLD = 0.363

_INPUT_SIZE = (112, 112)  # SFace fixed input


class SFaceEmbedder:
    """embed_cb-compatible callable: (face_crop_bgr) -> 1-D float32
    embedding, or None for a crop too degenerate to embed."""

    def __init__(self, model_path=None):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "SFaceEmbedder requires opencv-python (cv2)."
            ) from exc
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(path):
            raise RuntimeError(
                "SFace model not found at %r. Download it (see "
                "vision/recognition.py's module docstring), or pass an "
                "explicit model_path=." % path
            )
        self._cv2 = cv2
        self._sf = cv2.FaceRecognizerSF.create(path, "")
        self.model_path = path

    def __call__(self, face_crop_bgr):
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        h, w = face_crop_bgr.shape[:2]
        if h < 16 or w < 16:
            return None  # too small to carry identity; skip this tick
        resized = self._cv2.resize(face_crop_bgr, _INPUT_SIZE)
        feature = self._sf.feature(resized)
        return np.asarray(feature, dtype=np.float32).reshape(-1)


def make_embedder(model_path=None):
    """Factory mirroring conversation.llm.make_llm's spirit: the real
    SFaceEmbedder if cv2 + the model file are available, else None
    (recognition stays inert, exactly like the pre-Phase-6 stub) --
    always logged loudly, never silent."""
    try:
        embedder = SFaceEmbedder(model_path=model_path)
    except RuntimeError as exc:
        logger.warning(
            "make_embedder: SFace unavailable (%s) -- recognition will be "
            "inert (detect/track still fine).", exc
        )
        return None
    logger.info("make_embedder: SFace ready (%s)", embedder.model_path)
    return embedder

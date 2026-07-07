"""SFace face embeddings (Phase 6 seam of ORCHESTRATION.md SS3.1):
the real `embed_cb` for vision/tracking.py's TrackingApp, replacing the
inert `embed_face` stub. cv2.FaceRecognizerSF inference on a plain BGR
face crop -> 128-d float32 embedding for shared/people.py match/enroll.

CPU-only by design -- the GPU belongs to the LLM (SS3.4). SFace is a
small model built for exactly this.

Downloading the model (one-time; .onnx files are gitignored):

    curl -L -o vision/models/face_recognition_sface_2021dec.onnx \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

Alignment (added 2026-07-06 after live testing confirmed the unaligned
path mixes people up): FaceRecognizerSF's accuracy depends on
alignCrop() with the detector's 5 facial landmarks. The pinned detect()
API (vision/detector.py) deliberately returns plain (x, y, w, h, score)
boxes, so rather than widen that contract, the embedder is
self-sufficient: it runs its OWN cv2.FaceDetectorYN pass on the crop it
receives (cheap -- crops are small, and embed_cb already runs on the
recognition worker thread at ~1Hz, never the frame loop), takes the
best face's landmarks, and aligns with alignCrop() before feature().
If no face is found in the crop (degenerate/clipped), it falls back to
the old resize-to-112 path -- embeddings from that path are noisier,
which the threshold accounts for.

For alignment to work, the crop handed to embed_cb needs margin around
the face (a tight bbox crop clips the landmarks YuNet needs):
vision/tracking.py's crop_face_padded is the matching face_crop_cb.
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
# 0.363 (aligned crops, LFW-tuned). Live tuning history (2026-07-06):
# 0.363 unaligned produced false matches between different people;
# alignment landed and the threshold went to 0.45 -- which then MISSED
# the same person across a lighting change (matched in afternoon light,
# fell below 0.45 in the evening). 0.40 splits the difference: aligned
# different-person scores sit far lower, so mix-up risk stays small,
# and people.match()'s embedding refresh (DEFAULT_REFRESH_ALPHA) now
# tracks appearance drift so cross-session scores stop decaying.
# people.match() logs near-miss scores -- use those numbers, not
# guesswork, for the final Phase 6 bench tuning.
SFACE_MATCH_THRESHOLD = 0.40

_INPUT_SIZE = (112, 112)  # SFace fixed input


class SFaceEmbedder:
    """embed_cb-compatible callable: (face_crop_bgr) -> 1-D float32
    embedding, or None for a crop too degenerate to embed.

    Landmark-aligned when possible (see module docstring): an internal
    YuNet pass on the crop finds the 5 landmarks alignCrop() needs;
    falls back to plain resize when no face is found in the crop."""

    def __init__(self, model_path=None, yunet_model_path=None):
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

        # Internal landmark detector for alignCrop() -- optional: if the
        # YuNet model is missing, alignment is skipped (resize fallback)
        # rather than failing construction.
        from vision.detector import DEFAULT_MODEL_PATH as YUNET_DEFAULT
        yunet_path = yunet_model_path or YUNET_DEFAULT
        self._yunet = None
        if os.path.isfile(yunet_path):
            self._yunet = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320), 0.6, 0.3, 5000)
        else:
            logger.warning(
                "SFaceEmbedder: YuNet model missing (%s) -- landmark "
                "alignment disabled, falling back to raw-crop embeddings "
                "(noisier; expect weaker cross-session recognition).",
                yunet_path,
            )

    def _best_face_row(self, crop):
        """YuNet on the crop -> the highest-scoring raw face row (bbox +
        5 landmarks, the exact format alignCrop() expects), or None."""
        if self._yunet is None:
            return None
        h, w = crop.shape[:2]
        if h < 32 or w < 32:
            return None  # too small for the landmark pass
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(crop)
        if faces is None or len(faces) == 0:
            return None
        return faces[np.argmax(faces[:, -1])]

    def __call__(self, face_crop_bgr):
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        h, w = face_crop_bgr.shape[:2]
        if h < 16 or w < 16:
            return None  # too small to carry identity; skip this tick

        face_row = self._best_face_row(face_crop_bgr)
        if face_row is not None:
            aligned = self._sf.alignCrop(face_crop_bgr, face_row)
            feature = self._sf.feature(aligned)
        else:
            # No landmarks available: legacy path, noisier embeddings.
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

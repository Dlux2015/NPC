"""vision/recognition.py: the real SFace embedder. Runs only when cv2 +
the model file are present (they are on the dev PC; CI without the
gitignored .onnx skips). Deterministic checks only -- accuracy tuning is
a Phase 6 bench activity, not a unit test.
"""
import os

import numpy as np
import pytest

from vision.recognition import (
    DEFAULT_MODEL_PATH,
    SFACE_MATCH_THRESHOLD,
    SFaceEmbedder,
    make_embedder,
)


def _sface_available():
    if not os.path.isfile(DEFAULT_MODEL_PATH):
        return False
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _sface_available(),
    reason="SFace model or cv2 not available (model is gitignored; "
           "download per vision/recognition.py docstring)",
)


def _fake_face(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(160, 130, 3), dtype=np.uint8)


def test_embedder_returns_1d_float32_vector():
    embedder = SFaceEmbedder()
    emb = embedder(_fake_face(1))
    assert emb is not None
    assert emb.dtype == np.float32
    assert emb.ndim == 1
    assert emb.size == 128  # SFace embedding dimension


def test_embedder_is_deterministic_for_the_same_crop():
    embedder = SFaceEmbedder()
    crop = _fake_face(2)
    a, b = embedder(crop), embedder(crop)
    assert np.allclose(a, b)


def test_same_crop_matches_itself_above_threshold_different_noise_does_not():
    """Sanity anchor for SFACE_MATCH_THRESHOLD: identical input is a
    perfect match; two unrelated noise images shouldn't clear the
    threshold. (Real inter-person separation is Phase 6 bench work.)"""
    embedder = SFaceEmbedder()

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    e1 = embedder(_fake_face(3))
    assert cos(e1, e1) > SFACE_MATCH_THRESHOLD  # == 1.0

    e2 = embedder(_fake_face(4))
    assert cos(e1, e2) < 0.99  # genuinely different inputs differ


def test_degenerate_crops_return_none_not_crash():
    embedder = SFaceEmbedder()
    assert embedder(None) is None
    assert embedder(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert embedder(np.zeros((8, 8, 3), dtype=np.uint8)) is None  # <16px


def test_make_embedder_missing_model_is_loud_none(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        embedder = make_embedder(model_path=str(tmp_path / "nope.onnx"))
    assert embedder is None
    assert any("SFace unavailable" in r.message for r in caplog.records)

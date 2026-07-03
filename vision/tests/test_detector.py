import builtins

import pytest

from vision.detector import SyntheticDetector, YuNetDetector

try:
    import cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def test_synthetic_detector_passthrough():
    calls = []

    def ground_truth(frame):
        calls.append(frame)
        return [(10, 20, 30, 40, 0.9), (1, 2, 3, 4, 0.5)]

    det = SyntheticDetector(ground_truth)
    result = det.detect("fake-frame")

    assert result == [(10, 20, 30, 40, 0.9), (1, 2, 3, 4, 0.5)]
    assert calls == ["fake-frame"]


def test_synthetic_detector_returns_a_fresh_list_each_call():
    det = SyntheticDetector(lambda frame: [(0, 0, 1, 1, 1.0)])
    a = det.detect(None)
    b = det.detect(None)
    assert a == b
    assert a is not b


def test_yunet_missing_cv2_raises_clear_error(monkeypatch):
    # Force the ImportError branch deterministically, regardless of
    # whether cv2 is actually installed in this environment.
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "cv2":
            raise ImportError("simulated missing cv2")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="opencv-python"):
        YuNetDetector()


@pytest.mark.skipif(not _HAS_CV2, reason="cv2 not installed")
def test_yunet_missing_model_raises_clear_error():
    with pytest.raises(RuntimeError, match="model not found"):
        YuNetDetector(model_path="/nonexistent/path/does_not_exist.onnx")

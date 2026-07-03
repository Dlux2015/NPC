import numpy as np
import pytest

from shared import ipc, serial_protocol
from vision.detector import SyntheticDetector
from vision.tracking import TrackingApp

FRAME_W, FRAME_H = 640, 480


class FakeCamera:
    """Always returns the same frame; detections are driven separately
    through the SyntheticDetector's ground-truth callable."""

    def __init__(self, frame=None):
        self.frame = frame if frame is not None else _blank_frame()
        self.reads = 0

    def read(self):
        self.reads += 1
        return True, self.frame


class DeadCamera:
    def read(self):
        return False, None


class FakeTransport:
    def __init__(self):
        self.lines = []

    def write_line(self, line):
        self.lines.append(line)


def _blank_frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _calibration():
    return {
        "version": 1,
        "axes": {
            "pan": {"sign": 1, "center": 0.0, "min": -60.0, "max": 60.0},
            "tilt": {"sign": 1, "center": 0.0, "min": -30.0, "max": 30.0},
        },
        "deg_per_px": {"pan": 0.05, "tilt": 0.05},
        "deadband_deg": 0.5,
        "latency_s": 0.05,
    }


def _clock_box(start=0.0):
    box = [start]

    def clock():
        return box[0]

    return box, clock


def test_step_returns_none_on_camera_read_failure(tmp_path):
    state = ipc.SharedState(str(tmp_path / "state.json"))
    app = TrackingApp(DeadCamera(), SyntheticDetector(lambda f: []),
                       FakeTransport(), state, _calibration())
    assert app.step() is None


def test_face_present_sends_target_and_updates_ipc(tmp_path):
    # bbox tall enough to be "in range" (h/frame_h >= 0.25) and off-center
    # so the PID has a nonzero error to act on.
    bbox = (250, 150, 150, 150, 0.9)
    camera = FakeCamera()
    detector = SyntheticDetector(lambda f: [bbox])
    transport = FakeTransport()
    state = ipc.SharedState(str(tmp_path / "state.json"))

    box, clock = _clock_box(0.0)
    app = TrackingApp(camera, detector, transport, state, _calibration(),
                       clock=clock)

    result = app.step()

    assert result["person_present"] is True
    assert result["person_in_range"] is True  # 150/480 = 0.3125

    assert len(transport.lines) == 1
    parsed = serial_protocol.parse_line(transport.lines[0])
    assert parsed[0] == "target"

    ipc_state = state.read()
    assert ipc_state["person_present"] is True
    assert ipc_state["person_in_range"] is True


def test_no_face_sends_nothing_and_reports_absent(tmp_path):
    camera = FakeCamera()
    detector = SyntheticDetector(lambda f: [])
    transport = FakeTransport()
    state = ipc.SharedState(str(tmp_path / "state.json"))

    app = TrackingApp(camera, detector, transport, state, _calibration())
    result = app.step()

    assert result["person_present"] is False
    assert result["person_in_range"] is False
    assert transport.lines == []  # ESP32 owns idle scan on silence
    assert state.read()["person_present"] is False


def test_small_face_is_present_but_not_in_range(tmp_path):
    bbox = (300, 200, 40, 40, 0.9)  # 40/480 = 0.083 < 0.25
    camera = FakeCamera()
    detector = SyntheticDetector(lambda f: [bbox])
    transport = FakeTransport()
    state = ipc.SharedState(str(tmp_path / "state.json"))

    app = TrackingApp(camera, detector, transport, state, _calibration())
    result = app.step()

    assert result["person_present"] is True
    assert result["person_in_range"] is False


def test_ipc_updates_are_throttled_to_min_interval(tmp_path):
    detections = {"boxes": [(250, 150, 150, 150, 0.9)]}
    camera = FakeCamera()
    detector = SyntheticDetector(lambda f: detections["boxes"])
    transport = FakeTransport()
    state = ipc.SharedState(str(tmp_path / "state.json"))

    box, clock = _clock_box(0.0)
    # hold_s kept tiny and distinct from ipc_min_interval so this test
    # isolates IPC throttling from vision.tracking.TargetTracker's
    # separate (and much longer, 3s default) target-persistence hold.
    app = TrackingApp(camera, detector, transport, state, _calibration(),
                       clock=clock, ipc_min_interval=0.1, hold_s=0.05)

    app.step()  # t=0.0: first update always happens
    assert state.read()["person_present"] is True

    detections["boxes"] = []  # face "disappears"...
    app.step()  # ...but clock hasn't advanced -> IPC write is throttled
    assert state.read()["person_present"] is True  # stale, not yet updated

    box[0] = 0.2  # advance past ipc_min_interval
    app.step()
    assert state.read()["person_present"] is False  # now it catches up


def test_recognition_hook_runs_at_configured_interval_and_is_inert_stub(tmp_path):
    bbox = (250, 150, 150, 150, 0.9)
    camera = FakeCamera()
    detector = SyntheticDetector(lambda f: [bbox])
    transport = FakeTransport()
    state = ipc.SharedState(str(tmp_path / "state.json"))

    crop_calls = []

    def face_crop_cb(frame, target):
        crop_calls.append(target)
        return frame  # stand-in crop

    class DummyPeopleStore:
        def match(self, embedding):
            raise AssertionError("embed_face is a stub and must return None")

        def enroll(self, embedding):
            raise AssertionError("embed_face is a stub and must return None")

    box, clock = _clock_box(0.0)
    app = TrackingApp(camera, detector, transport, state, _calibration(),
                       clock=clock, face_crop_cb=face_crop_cb,
                       people_store=DummyPeopleStore(),
                       recognition_interval_s=1.0)

    app.step()  # t=0: first recognition attempt always fires
    assert len(crop_calls) == 1

    box[0] = 0.5
    app.step()  # within the 1s interval -> no second attempt
    assert len(crop_calls) == 1

    box[0] = 1.1
    app.step()  # interval elapsed -> fires again
    assert len(crop_calls) == 2

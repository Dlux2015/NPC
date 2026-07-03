import threading

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

    # step() only ever calls writer.publish() (in-memory, non-blocking);
    # flush() is the test/shutdown-only way to force the pending publish
    # to disk without waiting on the real background writer thread (see
    # shared/tests/test_ipc.py for the non-blocking/coalescing contract
    # itself).
    app._writer.flush()
    ipc_state = state.read()
    assert ipc_state["person_present"] is True
    assert ipc_state["person_in_range"] is True
    app.close()


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
    app._writer.flush()
    assert state.read()["person_present"] is False
    app.close()


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


def test_step_never_writes_ipc_directly_writer_owns_the_disk_call(tmp_path):
    """F4: the frame loop makes zero filesystem calls. step() must only
    ever reach IPC through writer.publish() (in-memory, same thread);
    SharedState.update (the actual disk write) must only ever run on the
    ThreadedStateWriter's own background thread -- never synchronously on
    the thread that called step(). shared/tests/test_ipc.py covers the
    writer's non-blocking/coalescing contract in isolation; this is the
    integration point proving TrackingApp actually uses it instead of the
    old direct SharedState.update() calls."""
    detections = {"boxes": [(250, 150, 150, 150, 0.9)]}
    camera = FakeCamera()
    detector = SyntheticDetector(lambda f: detections["boxes"])
    transport = FakeTransport()
    state = ipc.SharedState(str(tmp_path / "state.json"))

    caller_thread_ids = []
    orig_update = state.update

    def spy_update(**kwargs):
        caller_thread_ids.append(threading.get_ident())
        orig_update(**kwargs)

    state.update = spy_update

    box, clock = _clock_box(0.0)
    app = TrackingApp(camera, detector, transport, state, _calibration(),
                       clock=clock, ipc_min_interval=0.1, hold_s=0.05)
    assert isinstance(app._writer, ipc.ThreadedStateWriter)
    main_thread_id = threading.get_ident()

    try:
        app.step()  # publishes person_present=True -- no direct disk write
        detections["boxes"] = []
        box[0] = 0.05
        app.step()  # publishes person_present=False

        # Any SharedState.update calls the background thread already made
        # on its own initiative must NOT have run on this (step()'s) thread.
        assert all(tid != main_thread_id for tid in caller_thread_ids), (
            "SharedState.update ran synchronously on step()'s thread -- "
            "the frame loop touched disk directly"
        )

        app._writer.flush()  # force the latest published values to disk
        assert state.read()["person_present"] is False
        assert caller_thread_ids, "writer never reached SharedState.update"
    finally:
        app.close()


def test_recognition_hook_runs_on_worker_thread_and_is_inert_stub(tmp_path):
    """F5: crop_face (cheap, no I/O) still happens synchronously in
    step(), on schedule -- but the actual match/enroll work is handed off
    to the background recognition worker thread, never run inline. We
    detect a (should-never-happen, since embed_face is still a Phase-6
    stub returning None) DummyPeopleStore call via threading.excepthook
    since an AssertionError raised on the worker thread would otherwise
    not fail this (main) thread's test."""
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

    worker_exceptions = []
    orig_hook = threading.excepthook
    threading.excepthook = worker_exceptions.append
    try:
        app.step()  # t=0: first recognition attempt always fires
        app._wait_recognition_idle()
        assert len(crop_calls) == 1

        box[0] = 0.5
        app.step()  # within the 1s interval -> no second attempt
        assert len(crop_calls) == 1

        box[0] = 1.1
        app.step()  # interval elapsed -> fires again
        app._wait_recognition_idle()
        assert len(crop_calls) == 2

        assert worker_exceptions == [], (
            "DummyPeopleStore.match/enroll ran on the worker thread -- "
            "embed_face's Phase-6 stub should still make this dead code"
        )
    finally:
        threading.excepthook = orig_hook
        app.close()

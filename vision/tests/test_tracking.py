import threading

import numpy as np

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


def test_recognition_worker_survives_a_failing_embedder(tmp_path):
    """Survivability: one exception in the recognition path costs one
    tick, never the worker thread -- a dead worker would silently end
    recognition for the process lifetime while tracking keeps running."""
    bbox = (250, 150, 150, 150, 0.9)
    state = ipc.SharedState(str(tmp_path / "state.json"))

    calls = {"n": 0}

    def flaky_embed(crop):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient cv2 failure")
        return np.ones(4, dtype=np.float32)

    class RecordingStore:
        def __init__(self):
            self.matches = []

        def match(self, embedding, threshold=None):
            self.matches.append(embedding)
            return (5, None, 0.9)

        def enroll(self, embedding):
            raise AssertionError("match always hits in this test")

    store = RecordingStore()
    box, clock = _clock_box(0.0)
    app = TrackingApp(FakeCamera(), SyntheticDetector(lambda f: [bbox]),
                       FakeTransport(), state, _calibration(), clock=clock,
                       face_crop_cb=lambda f, t: f, people_store=store,
                       embed_cb=flaky_embed, recognition_interval_s=1.0)
    try:
        app.step()                      # tick 1: embedder raises
        app._wait_recognition_idle()
        assert store.matches == []      # that tick was dropped...
        assert app._recognition_thread.is_alive()  # ...but the worker lives

        box[0] = 1.1
        app.step()                      # tick 2: works normally again
        app._wait_recognition_idle()
        assert len(store.matches) == 1
    finally:
        app.close()


def test_run_forever_relaxes_detection_rate_when_alone(tmp_path, monkeypatch):
    """Power: after idle_after_s with nobody in frame, the frame loop
    paces at idle_hz instead of target_hz -- and snaps back to full rate
    the moment a face appears (the Jetson's always-on detection is the
    system's largest constant power draw; see run_forever docstring)."""
    import vision.tracking as tracking_module

    detections = {"boxes": []}
    camera = FakeCamera()
    detector = SyntheticDetector(lambda f: detections["boxes"])
    state = ipc.SharedState(str(tmp_path / "state.json"))
    box, clock = _clock_box(0.0)
    app = TrackingApp(camera, detector, FakeTransport(), state,
                       _calibration(), clock=clock)

    sleeps = []

    class FakeTime:
        @staticmethod
        def sleep(s):
            sleeps.append(round(s, 4))
            box[0] += s

        monotonic = staticmethod(clock)

    monkeypatch.setattr(tracking_module, "time", FakeTime)

    counter = {"n": 0}

    def stop_flag():
        counter["n"] += 1
        if counter["n"] == 60:
            detections["boxes"] = [(250, 150, 150, 150, 0.9)]  # someone walks up
        return counter["n"] > 70

    try:
        app.run_forever(target_hz=30.0, stop_flag=stop_flag,
                         idle_hz=10.0, idle_after_s=1.0)
    finally:
        app.close()

    full, idle = round(1 / 30.0, 4), round(1 / 10.0, 4)
    assert sleeps[0] == full            # starts at full rate
    assert idle in sleeps               # relaxed once alone long enough
    assert sleeps[-1] == full           # back to full rate with a face
    # ordering: every idle-paced frame sits between full-rate stretches
    first_idle = sleeps.index(idle)
    assert all(s == full for s in sleeps[:first_idle])


def test_recognition_consent_mode_stashes_instead_of_enrolling(tmp_path):
    """auto_enroll=False (consent flows, e.g. demo_friend's "can I be
    your friend?"): an unmatched embedding must NOT be enrolled or
    published -- only stashed for pop_unknown_embedding(); a matched one
    still publishes person_id and clears the stash."""
    bbox = (250, 150, 150, 150, 0.9)
    camera = FakeCamera()
    detector = SyntheticDetector(lambda f: [bbox])
    transport = FakeTransport()
    state = ipc.SharedState(str(tmp_path / "state.json"))

    class ScriptedPeopleStore:
        def __init__(self):
            self.enrolls = []
            self.match_result = None

        def match(self, embedding, threshold=None):
            return self.match_result

        def enroll(self, embedding):
            self.enrolls.append(embedding)
            return 99

    people = ScriptedPeopleStore()
    emb = np.ones(8, dtype=np.float32)

    box, clock = _clock_box(0.0)
    app = TrackingApp(camera, detector, transport, state, _calibration(),
                       clock=clock, face_crop_cb=lambda f, t: f,
                       people_store=people, embed_cb=lambda crop: emb,
                       auto_enroll=False, recognition_interval_s=1.0)
    try:
        app.step()
        app._wait_recognition_idle()
        assert people.enrolls == []          # nothing stored without consent
        app._writer.flush()
        assert state.get("person_id") is None  # nothing published either
        stashed = app.pop_unknown_embedding()
        assert stashed is not None and stashed.size == emb.size
        assert app.pop_unknown_embedding() is None  # pop clears

        # Known person: publishes id, clears any stash
        people.match_result = (7, "Sam", 0.9)
        box[0] = 1.1
        app.step()
        app._wait_recognition_idle()
        app._writer.flush()
        assert state.get("person_id") == "7"
        assert app.pop_unknown_embedding() is None
    finally:
        app.close()


def test_recognition_match_threshold_passed_through(tmp_path):
    """match_threshold (e.g. recognition.SFACE_MATCH_THRESHOLD) must reach
    people.match(); None must keep the store's own default."""
    bbox = (250, 150, 150, 150, 0.9)
    state = ipc.SharedState(str(tmp_path / "state.json"))

    seen_thresholds = []

    class ThresholdSpyStore:
        def match(self, embedding, threshold=None):
            seen_thresholds.append(threshold)
            return (1, None, 0.9)

        def enroll(self, embedding):
            raise AssertionError("must not enroll on a match")

    box, clock = _clock_box(0.0)
    app = TrackingApp(FakeCamera(), SyntheticDetector(lambda f: [bbox]),
                       FakeTransport(), state, _calibration(), clock=clock,
                       face_crop_cb=lambda f, t: f,
                       people_store=ThresholdSpyStore(),
                       embed_cb=lambda crop: np.ones(4, dtype=np.float32),
                       match_threshold=0.363)
    try:
        app.step()
        app._wait_recognition_idle()
        assert seen_thresholds == [0.363]
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

"""Tests for conversation/ambient.py's AmbientBuffer: rolling window,
pause-while-actively_speaking, snapshot, and publish throttling -- all
with fakes, exercised via the real background thread (short, bounded
waits, no audio hardware required).
"""
import time

import numpy as np

from conversation.ambient import AmbientBuffer
from shared.ipc import SharedState


class ScriptedFrameSource:
    """Fake mic: yields (frame, is_speech) pairs from a script, cycling
    "trailing silence" once exhausted so the background thread doesn't
    error out after the script runs dry."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def read(self, n_samples):
        frame = self._frames[self._i] if self._i < len(self._frames) else np.zeros(n_samples, dtype=np.int16)
        self._i = min(self._i + 1, len(self._frames))
        return np.asarray(frame, dtype=np.int16)


class ScriptedVAD:
    def __init__(self, probs):
        self._probs = list(probs)
        self._i = 0

    def speech_prob(self, frame):
        if self._i < len(self._probs):
            p = self._probs[self._i]
            self._i += 1
            return p
        return 0.0


class FakeWhisperModel:
    def __init__(self, text="overheard something"):
        self.text = text
        self.calls = 0

    def transcribe(self, samples, sample_rate=16000):
        self.calls += 1
        return self.text


def make_state(tmp_path, **kwargs):
    state = SharedState(str(tmp_path / "state.json"))
    if kwargs:
        state.update(**kwargs)
    return state


def make_ambient(tmp_path, probs, state=None, **overrides):
    state = state or make_state(tmp_path)
    mic = ScriptedFrameSource([np.full(10, i, dtype=np.int16) for i in range(len(probs))])
    kwargs = dict(
        profile={"name": "sim"},
        state=state,
        mic_source=mic,
        model=FakeWhisperModel(),
        vad=ScriptedVAD(probs),
        duty_cycle_s=0.01,
        publish_interval_s=overrides.pop("publish_interval_s", 0.05),
        window_s=overrides.pop("window_s", 60.0),
    )
    kwargs.update(overrides)
    return AmbientBuffer(**kwargs), state


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_start_stop_lifecycle(tmp_path):
    ambient, state = make_ambient(tmp_path, [0.0] * 20)
    assert ambient._thread is None
    ambient.start()
    assert ambient._thread is not None
    assert ambient._thread.is_alive()
    ambient.start()  # idempotent: calling start() again must not spawn a second thread
    ambient.stop(timeout=2.0)
    assert ambient._thread is None


def test_speech_segment_gets_transcribed_and_appears_in_snapshot(tmp_path):
    # silence, speech, speech, silence (end of speech segment) then quiet
    probs = [0.0, 1.0, 1.0, 0.0] + [0.0] * 20
    ambient, state = make_ambient(tmp_path, probs)
    ambient.start()
    try:
        ok = _wait_until(lambda: ambient.snapshot() == ["overheard something"])
        assert ok, "speech segment was never transcribed into the snapshot"
    finally:
        ambient.stop()


def test_pauses_entirely_while_actively_speaking(tmp_path):
    """Hard rule: the robot must never transcribe itself -- while
    actively_speaking is True, no frame should ever reach the VAD/model,
    even if the (fake) mic is producing "speech"."""
    state = make_state(tmp_path, actively_speaking=True)
    probs = [1.0] * 50  # would be "constant speech" if ambient weren't paused
    ambient, state = make_ambient(tmp_path, probs, state=state)
    ambient.start()
    try:
        time.sleep(0.3)
        assert ambient.snapshot() == []
        assert ambient._model.calls == 0
    finally:
        ambient.stop()


def test_resumes_after_actively_speaking_clears(tmp_path):
    state = make_state(tmp_path, actively_speaking=True)
    probs = [0.0, 1.0, 1.0, 0.0] + [0.0] * 30
    ambient, state = make_ambient(tmp_path, probs, state=state)
    ambient.start()
    try:
        time.sleep(0.1)
        assert ambient.snapshot() == []  # still paused

        state.update(actively_speaking=False)
        ok = _wait_until(lambda: ambient.snapshot() == ["overheard something"])
        assert ok
    finally:
        ambient.stop()


def test_snapshot_trims_to_rolling_window(tmp_path):
    ambient, state = make_ambient(tmp_path, [0.0] * 5, window_s=0.05)
    # Directly seed old + fresh entries -- exercises _trim()/snapshot()
    # without waiting on real transcription timing.
    now = ambient._clock()
    with ambient._lock:
        ambient._entries = [(now - 10.0, "ancient"), (now, "fresh")]
    assert ambient.snapshot() == ["fresh"]


def test_publish_is_throttled_to_configured_interval(tmp_path):
    """Multiple speech segments in quick succession must not each trigger
    an immediate IPC publish -- <=0.5Hz per SS3.2."""
    # Three back-to-back speech segments, each immediately followed by one
    # silence frame so each ends promptly.
    probs = ([1.0, 0.0] * 3) + [0.0] * 20
    state = make_state(tmp_path)
    ambient, state = make_ambient(tmp_path, probs, state=state, publish_interval_s=10.0)
    ambient.start()
    try:
        ok = _wait_until(lambda: len(ambient.snapshot()) >= 1)
        assert ok
        time.sleep(0.2)  # let more segments accumulate in-memory
        # snapshot (in-memory) can have grown, but the published IPC state
        # must reflect at most the throttled cadence -- i.e. it should not
        # have been updated 3 times within publish_interval_s.
        assert len(ambient.snapshot()) >= 1
        published = state.read().get("ambient_transcript")
        # Either nothing published yet, or exactly the first publish's
        # payload -- never spammed with every single segment.
        assert published in (None, [], ["overheard something"])
    finally:
        ambient.stop()


def test_never_persists_beyond_the_ipc_scratch_state(tmp_path):
    """ambient_transcript must never be written anywhere except the shared
    IPC scratch-state file -- no separate on-disk transcript log."""
    probs = [0.0, 1.0, 1.0, 0.0] + [0.0] * 10
    ambient, state = make_ambient(tmp_path, probs, publish_interval_s=0.02)
    ambient.start()
    try:
        _wait_until(lambda: ambient.snapshot() == ["overheard something"])
        time.sleep(0.1)
    finally:
        ambient.stop()
    files = list(tmp_path.iterdir())
    assert [f.name for f in files] == ["state.json"]

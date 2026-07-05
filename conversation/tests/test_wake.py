"""Tests for conversation/wake.py's WakeTrigger: gating logic only, all
backends faked (mic, VAD, wakeword model, PTT poll) -- no audio hardware,
no openWakeWord/Silero, no keyboard library required.
"""
import numpy as np
import pytest

from conversation.wake import WakeTrigger
from conversation.tests._audio_fakes import FakeClock, ScriptedVAD
from shared.ipc import SharedState


def make_state(tmp_path, **kwargs):
    state = SharedState(str(tmp_path / "state.json"))
    if kwargs:
        state.update(**kwargs)
    return state


def make_trigger(tmp_path, **overrides):
    state = overrides.pop("state", None) or make_state(tmp_path)
    clock = overrides.pop("clock", None) or FakeClock()
    kwargs = dict(
        profile={"name": "sim"},
        state=state,
        mic_source=overrides.pop("mic_source", None),
        wakeword_model=overrides.pop("wakeword_model", None),
        vad=overrides.pop("vad", ScriptedVAD([])),
        ptt_poll=overrides.pop("ptt_poll", lambda: False),
        clock=clock.time,
        sleep_fn=clock.sleep,
    )
    kwargs.update(overrides)
    return WakeTrigger(**kwargs), state, clock


def test_ptt_fires_immediately(tmp_path):
    trigger, state, clock = make_trigger(tmp_path, ptt_poll=lambda: True)
    assert trigger.wait(timeout_s=1.0) == "ptt"


def test_times_out_to_none_when_nothing_fires(tmp_path):
    trigger, state, clock = make_trigger(tmp_path, ptt_poll=lambda: False)
    assert trigger.wait(timeout_s=0.2) is None
    assert clock.t >= 0.2


def test_suspended_while_actively_speaking(tmp_path):
    """Hard rule: no source may fire while actively_speaking, even if it
    would otherwise -- the robot must never wake/hear itself. Verified two
    ways: (1) wait() returns None despite a PTT poll that would always say
    yes, and (2) the PTT poll is never even invoked while suspended."""
    calls = []

    def ptt_poll():
        calls.append(True)
        return True

    state = make_state(tmp_path, actively_speaking=True)
    trigger, state, clock = make_trigger(tmp_path, state=state, ptt_poll=ptt_poll)

    assert trigger.wait(timeout_s=0.3) is None
    assert calls == [], "a source was polled while actively_speaking"


def test_becomes_available_again_once_speaking_clears(tmp_path):
    state = make_state(tmp_path, actively_speaking=True)
    trigger, state, clock = make_trigger(tmp_path, state=state, ptt_poll=lambda: True)

    assert trigger.wait(timeout_s=0.1) is None  # still suspended

    state.update(actively_speaking=False)
    assert trigger.wait(timeout_s=1.0) == "ptt"


def test_wakeword_fires_above_threshold(tmp_path):
    class FakeWakeword:
        def __init__(self, score):
            self.score_value = score

        def score(self, frame):
            return self.score_value

    mic = _ConstantFrameSource()
    trigger, state, clock = make_trigger(
        tmp_path, mic_source=mic, wakeword_model=FakeWakeword(0.9),
        ptt_poll=lambda: False,
        audio_config={"wake_threshold": 0.6, "vad_threshold": 0.5, "sample_rate": 16000},
    )
    assert trigger.wait(timeout_s=1.0) == "wakeword"


def test_wakeword_below_threshold_does_not_fire(tmp_path):
    class FakeWakeword:
        def score(self, frame):
            return 0.1

    mic = _ConstantFrameSource()
    trigger, state, clock = make_trigger(
        tmp_path, mic_source=mic, wakeword_model=FakeWakeword(),
        ptt_poll=lambda: False,
        audio_config={"wake_threshold": 0.6, "vad_threshold": 0.5, "sample_rate": 16000},
    )
    assert trigger.wait(timeout_s=0.2) is None


def test_face_speech_requires_both_person_in_range_and_vad_speech(tmp_path):
    mic = _ConstantFrameSource()

    # person not in range: VAD says speech, but source must not fire.
    state = make_state(tmp_path, person_in_range=False)
    trigger, state, clock = make_trigger(
        tmp_path, state=state, mic_source=mic,
        vad=ScriptedVAD([1.0] * 10, threshold=0.5),
        ptt_poll=lambda: False,
    )
    assert trigger.wait(timeout_s=0.2) is None

    # person in range, but VAD says no speech.
    state2 = make_state(tmp_path, person_in_range=True)
    trigger2, _, clock2 = make_trigger(
        tmp_path, state=state2, mic_source=_ConstantFrameSource(),
        vad=ScriptedVAD([0.0] * 10, threshold=0.5),
        ptt_poll=lambda: False,
    )
    assert trigger2.wait(timeout_s=0.2) is None

    # both true -> fires.
    state3 = make_state(tmp_path, person_in_range=True)
    trigger3, _, clock3 = make_trigger(
        tmp_path, state=state3, mic_source=_ConstantFrameSource(),
        vad=ScriptedVAD([1.0] * 10, threshold=0.5),
        ptt_poll=lambda: False,
    )
    assert trigger3.wait(timeout_s=1.0) == "face_speech"


def test_ptt_checked_before_audio_sources(tmp_path):
    """ptt is the cheapest/most deterministic source; WakeTrigger should
    not even need a mic to satisfy a PTT press."""
    trigger, state, clock = make_trigger(
        tmp_path, mic_source=None, ptt_poll=lambda: True,
    )
    assert trigger.wait(timeout_s=1.0) == "ptt"


def test_missing_wakeword_backend_degrades_without_raising(tmp_path):
    """No wakeword_model injected and openwakeword isn't installed on this
    dev PC -- wait() must degrade to "source never fires", not raise."""
    trigger, state, clock = make_trigger(
        tmp_path, mic_source=_ConstantFrameSource(), ptt_poll=lambda: False,
        vad=ScriptedVAD([0.0] * 50, threshold=0.5),
    )
    assert trigger.wait(timeout_s=0.2) is None
    assert "wakeword" in trigger._unavailable


def test_missing_mic_backend_degrades_without_raising(tmp_path, monkeypatch):
    """No mic_source injected and no real MicStream backend available --
    wakeword/face_speech sources should both cleanly disable rather than
    raise. Forced via monkeypatch (rather than relying on sounddevice
    happening to not be pip-installed on whatever machine runs this
    suite -- a dev PC set up for conversation/demo_talk.py's live-audio
    testing has sounddevice installed for real, so the "backend missing"
    case has to be simulated instead) so this test's behavior doesn't
    depend on the local machine's package inventory."""
    import conversation.audio_dev as audio_dev

    def _raise_no_hardware(*args, **kwargs):
        raise RuntimeError("no audio hardware available (forced for test)")

    monkeypatch.setattr(audio_dev, "MicStream", _raise_no_hardware)

    state = make_state(tmp_path, person_in_range=True)
    trigger, state, clock = make_trigger(
        tmp_path, state=state, mic_source=None, ptt_poll=lambda: False,
    )
    assert trigger.wait(timeout_s=0.2) is None
    assert trigger._unavailable == {"wakeword", "face_speech"}


class _ConstantFrameSource:
    def read(self, n_samples):
        return np.zeros(n_samples, dtype=np.int16)

"""Tests for conversation/tts.py's Speaker: actively_speaking lifecycle,
sentence-streaming, and NullAudioSink -- no piper-tts/sounddevice
installed required.
"""
import os
import sys
import types
import wave

import numpy as np
import pytest

from conversation.tts import KokoroSynthesizer, NullAudioSink, Speaker, split_sentences
from conversation.tests._audio_fakes import FakeSynthesizer
from shared.ipc import SharedState


def make_state(tmp_path):
    return SharedState(str(tmp_path / "state.json"))


def make_speaker(tmp_path, synth=None, sink=None):
    state = make_state(tmp_path)
    speaker = Speaker(
        profile={"name": "sim"},
        state=state,
        synthesizer=synth or FakeSynthesizer(),
        sink=sink or NullAudioSink(dir=str(tmp_path / "audio_out")),
    )
    return speaker, state


def test_say_sets_and_clears_actively_speaking(tmp_path):
    speaker, state = make_speaker(tmp_path)
    assert state.get("actively_speaking") is False

    speaker.say("Hello there, welcome to the shop!")

    assert state.get("actively_speaking") is False  # cleared after playback drains


def test_actively_speaking_is_true_during_playback(tmp_path):
    """Ordering matters: actively_speaking must already be True by the
    time the sink is asked to play, and only cleared after say() returns.
    A sink that snapshots the live IPC flag at play()-time proves this."""
    state = make_state(tmp_path)
    seen_during_play = []

    class ObservingSink(NullAudioSink):
        def play(self, samples, sample_rate=16000):
            seen_during_play.append(state.get("actively_speaking"))
            return super().play(samples, sample_rate=sample_rate)

    speaker = Speaker(profile={"name": "sim"}, state=state,
                       synthesizer=FakeSynthesizer(),
                       sink=ObservingSink(dir=str(tmp_path / "out")))

    assert state.get("actively_speaking") is False
    speaker.say("One short sentence.")

    assert seen_during_play == [True]
    assert state.get("actively_speaking") is False  # cleared after playback drains


def test_say_stream_plays_each_sentence_and_writes_a_wav_per_sentence(tmp_path):
    synth = FakeSynthesizer()
    sink = NullAudioSink(dir=str(tmp_path / "out"))
    speaker, state = make_speaker(tmp_path, synth=synth, sink=sink)

    speaker.say_stream(iter(["First sentence.", "Second sentence!", "Third?"]))

    assert synth.calls == ["First sentence.", "Second sentence!", "Third?"]
    assert len(sink.written) == 3
    for path in sink.written:
        assert os.path.isfile(path)
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == synth.sample_rate


def test_say_stream_single_actively_speaking_window_covers_whole_reply(tmp_path):
    """actively_speaking must be one True->False window spanning *all*
    sentences, not flicker True/False between sentences."""
    log = []

    class LoggingState(SharedState):
        def update(self, **kwargs):
            if "actively_speaking" in kwargs:
                log.append(kwargs["actively_speaking"])
            super().update(**kwargs)

    state = LoggingState(str(tmp_path / "state.json"))
    speaker = Speaker(profile={"name": "sim"}, state=state,
                       synthesizer=FakeSynthesizer(),
                       sink=NullAudioSink(dir=str(tmp_path / "out")))

    speaker.say_stream(iter(["Sentence one.", "Sentence two.", "Sentence three."]))

    assert log == [True, False]


def test_say_with_empty_text_is_a_noop(tmp_path):
    synth = FakeSynthesizer()
    speaker, state = make_speaker(tmp_path, synth=synth)
    speaker.say("   ")
    assert synth.calls == []
    assert state.get("actively_speaking") is False


def test_say_stream_skips_blank_sentences_but_still_speaks_real_ones(tmp_path):
    synth = FakeSynthesizer()
    sink = NullAudioSink(dir=str(tmp_path / "out"))
    speaker, state = make_speaker(tmp_path, synth=synth, sink=sink)

    speaker.say_stream(iter(["", "  ", "Actual sentence."]))

    assert synth.calls == ["Actual sentence."]
    assert len(sink.written) == 1


def test_actively_speaking_cleared_even_if_playback_raises(tmp_path):
    """Half-duplex gate must not get stuck on -- a sink error mid-reply
    must still clear actively_speaking, or wake/ambient would stay
    suspended forever."""

    class BoomSink:
        def play(self, samples, sample_rate=16000):
            raise RuntimeError("boom")

    speaker, state = make_speaker(tmp_path, sink=BoomSink())

    with pytest.raises(RuntimeError):
        speaker.say("This will explode.")

    assert state.get("actively_speaking") is False


def test_null_audio_sink_creates_its_own_tempdir_by_default():
    sink = NullAudioSink()
    assert os.path.isdir(sink.dir)


def test_split_sentences():
    assert split_sentences("Hi there! How are you? Fine.") == [
        "Hi there!", "How are you?", "Fine.",
    ]
    assert split_sentences("   ") == []
    assert split_sentences("No terminal punctuation") == ["No terminal punctuation"]


# --- KokoroSynthesizer (kokoro faked -- no real model download) -----------

class _FakeAudioTensor:
    """Stand-in for the torch.FloatTensor KPipeline.Result.audio returns:
    only .numpy() is ever called on it."""

    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


class _FakeResult:
    def __init__(self, audio):
        self.audio = audio


class _FakeKPipeline:
    """Stand-in for kokoro.KPipeline: records construction args and the
    (text, voice) it was called with; yields one fixed-length float32
    "audio" chunk per call, like the real generator does."""

    last_instance = None

    def __init__(self, lang_code, **kwargs):
        self.lang_code = lang_code
        self.kwargs = kwargs
        self.calls = []
        _FakeKPipeline.last_instance = self

    def __call__(self, text, voice=None, **kwargs):
        self.calls.append((text, voice))
        audio = np.full(240, 0.5, dtype=np.float32)  # 10ms @ 24kHz, mid-scale
        yield _FakeResult(_FakeAudioTensor(audio))


@pytest.fixture
def fake_kokoro_module(monkeypatch):
    module = types.ModuleType("kokoro")
    module.KPipeline = _FakeKPipeline
    monkeypatch.setitem(sys.modules, "kokoro", module)
    return module


def test_kokoro_missing_package_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro", None)
    with pytest.raises(RuntimeError, match="kokoro"):
        KokoroSynthesizer()


def test_kokoro_constructs_pipeline_with_lang_code(fake_kokoro_module):
    synth = KokoroSynthesizer(voice="bm_george", lang_code="b")
    assert synth.sample_rate == 24000
    assert _FakeKPipeline.last_instance.lang_code == "b"


def test_kokoro_synthesize_calls_pipeline_with_text_and_voice(fake_kokoro_module):
    synth = KokoroSynthesizer(voice="am_michael", lang_code="a")
    synth.synthesize("Good news, sir.")
    assert _FakeKPipeline.last_instance.calls == [("Good news, sir.", "am_michael")]


def test_kokoro_synthesize_returns_int16_scaled_from_float(fake_kokoro_module):
    synth = KokoroSynthesizer()
    samples = synth.synthesize("hello")
    assert samples.dtype == np.int16
    # fake audio is a constant 0.5 -> scaled to ~16383
    assert samples[0] == int(0.5 * 32767)


def test_kokoro_synthesize_empty_result_is_empty_int16_array(fake_kokoro_module):
    class _EmptyPipeline(_FakeKPipeline):
        def __call__(self, text, voice=None, **kwargs):
            return iter([])  # no results at all

    module = sys.modules["kokoro"]
    module.KPipeline = _EmptyPipeline
    synth = KokoroSynthesizer()
    samples = synth.synthesize("hello")
    assert len(samples) == 0
    assert samples.dtype == np.int16


# --- make_tts_synthesizer engine selection (demo_talk.py) -------------------

def test_make_tts_synthesizer_prefers_kokoro_when_configured(fake_kokoro_module):
    from conversation.demo_talk import make_tts_synthesizer
    profile_yaml = {"tts_engine": "kokoro", "tts_voice": "am_michael",
                     "tts_lang_code": "a"}
    synth, backend = make_tts_synthesizer(profile_yaml)
    assert isinstance(synth, KokoroSynthesizer)
    assert "Kokoro" in backend
    assert "am_michael" in backend


def test_make_tts_synthesizer_falls_back_when_kokoro_not_installed(monkeypatch, capsys):
    from conversation.demo_talk import make_tts_synthesizer
    monkeypatch.setitem(sys.modules, "kokoro", None)
    profile_yaml = {"tts_engine": "kokoro"}  # no piper/tts_model_path either
    synth, backend = make_tts_synthesizer(profile_yaml)
    assert "SAPI" in backend
    assert "Kokoro unavailable" in capsys.readouterr().out

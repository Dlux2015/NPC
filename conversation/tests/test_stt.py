"""Tests for conversation/stt.py's DirectedSTT: fixture-WAV-driven and
scripted-fake tests, no faster-whisper/torch/sounddevice installed."""
import numpy as np
import pytest

from conversation.audio_dev import SAMPLE_RATE, read_wav_int16
from conversation.stt import DirectedSTT
from conversation.tests._audio_fakes import (
    ClockAdvancingMicSource,
    ConstantMicSource,
    EnergyVAD,
    FakeClock,
    FakeWhisperModel,
    WavMicSource,
    fixture_path,
)


def make_stt(**overrides):
    clock = overrides.pop("clock", None) or FakeClock()
    kwargs = dict(
        profile={"name": "sim"},
        mic_source=overrides.pop("mic_source", None),
        model=overrides.pop("model", FakeWhisperModel("hello robot")),
        vad=overrides.pop("vad", EnergyVAD(threshold=0.5)),
        end_silence_s=overrides.pop("end_silence_s", 0.3),
        min_speech_s=overrides.pop("min_speech_s", 0.05),
        clock=clock.time,
        sleep_fn=clock.sleep,
    )
    kwargs.update(overrides)
    return DirectedSTT(**kwargs), clock


def test_transcribe_wav_fixture_reads_and_calls_model():
    model = FakeWhisperModel("hello there robot")
    stt = DirectedSTT(profile={"name": "sim"}, model=model)

    text = stt.transcribe_wav(fixture_path("directed_hello.wav"))

    assert text == "hello there robot"
    assert model.calls  # model actually got the audio
    n_samples, rate = model.calls[0]
    assert rate == SAMPLE_RATE
    assert n_samples == int(SAMPLE_RATE * 1.2)


def test_transcribe_wav_rejects_wrong_sample_rate(tmp_path):
    import wave
    path = tmp_path / "wrong_rate.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(np.zeros(800, dtype=np.int16).tobytes())

    stt = DirectedSTT(profile={"name": "sim"}, model=FakeWhisperModel())
    with pytest.raises(ValueError, match="16000"):
        stt.transcribe_wav(str(path))


def test_listen_utterance_endpoints_on_trailing_silence_from_fixture():
    """Feeds the real directed_hello.wav fixture frame-by-frame through a
    fake mic; EnergyVAD (real RMS on real PCM) should detect the tone
    burst as speech and endpoint once the fixture's trailing silence
    exceeds end_silence_s."""
    samples, rate = read_wav_int16(fixture_path("directed_hello.wav"))
    assert rate == SAMPLE_RATE
    mic = WavMicSource(samples)
    model = FakeWhisperModel("hello there robot")
    stt, clock = make_stt(mic_source=mic, model=model, vad=EnergyVAD(threshold=0.3))

    text = stt.listen_utterance(max_s=10.0)

    assert text == "hello there robot"
    assert model.calls
    n_samples, rate2 = model.calls[0]
    assert rate2 == SAMPLE_RATE
    # Utterance audio should span roughly the tone burst + endpointing
    # silence, but not the whole file's full trailing padding.
    assert 0 < n_samples < len(samples)


def test_listen_utterance_returns_none_when_nobody_speaks():
    samples, _ = read_wav_int16(fixture_path("silence_only.wav"))
    mic = WavMicSource(samples)
    model = FakeWhisperModel("should not be called")
    stt, clock = make_stt(mic_source=mic, model=model, vad=EnergyVAD(threshold=0.5))

    text = stt.listen_utterance(max_s=1.0)

    assert text is None
    assert model.calls == []  # never even asked to transcribe


def test_listen_utterance_stops_at_max_s_for_continuous_speech():
    """A source that never falls silent must still be bounded by max_s."""
    clock = FakeClock()
    frame_ms = 32  # explicit: must match DirectedSTT's own frame_ms below,
    # since ClockAdvancingMicSource's fake-clock advance is decoupled from
    # DirectedSTT's real elapsed-audio bookkeeping and the two need to
    # agree for this test's math to hold.
    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    tone = (8000 * np.sin(np.linspace(0, 6, frame_samples))).astype(np.int16)
    inner = ConstantMicSource(tone)
    mic = ClockAdvancingMicSource(inner, clock, frame_s=frame_samples / SAMPLE_RATE)
    model = FakeWhisperModel("cut off")
    stt, _ = make_stt(mic_source=mic, model=model, frame_ms=frame_ms,
                       vad=EnergyVAD(threshold=0.1), clock=clock)

    text = stt.listen_utterance(max_s=1.0)

    assert text == "cut off"
    assert clock.t >= 1.0


def test_listen_utterance_ignores_blips_shorter_than_min_speech_s():
    """A single short speech-like frame surrounded by silence shouldn't
    count as an utterance -- min_speech_s guards against VAD noise."""
    clock = FakeClock()
    silence = np.zeros(480, dtype=np.int16)
    blip = (8000 * np.sin(np.linspace(0, 3, 480))).astype(np.int16)
    samples = np.concatenate([silence, blip, silence, silence, silence])
    mic = WavMicSource(samples)
    model = FakeWhisperModel("should not fire")
    stt, _ = make_stt(mic_source=mic, model=model, clock=clock,
                       vad=EnergyVAD(threshold=0.3), min_speech_s=1.0,
                       end_silence_s=0.05)

    text = stt.listen_utterance(max_s=0.5)

    assert text is None
    assert model.calls == []


# -- real-model tests: skipped when faster-whisper isn't installed on this
# dev PC (it isn't, by design -- Phase 4 audio-side work must be fully
# testable without model downloads). Left in so a machine with the real
# dependency installed exercises the real pipeline end-to-end.

def _has_faster_whisper():
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_faster_whisper(), reason="faster-whisper not installed")
def test_transcribe_wav_with_real_faster_whisper_tiny():
    from conversation.whisper_model import load_faster_whisper

    stt = DirectedSTT(profile={"name": "sim"},
                       model=load_faster_whisper("tiny"))
    text = stt.transcribe_wav(fixture_path("directed_hello.wav"))
    assert isinstance(text, str)

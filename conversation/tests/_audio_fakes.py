"""Fakes for the audio-side conversation/ tests (wake.py, stt.py,
ambient.py, tts.py, audio_dev.py) -- speech-engineer's half of Phase 4.

Deliberately a plain module, not conftest.py: conversation/tests/conftest.py
is owned by llm-engineer's parallel pipeline.py/persona work (FakeState,
FakeWake, FakeSTT, FakeSpeaker, FakeAmbient there stand in for *this*
module's pinned interfaces from the pipeline side). Keeping these fakes in
a separate importable module avoids two agents editing the same file.
Tests here use the real shared.ipc.SharedState (tmp_path-backed) rather
than a fake, since IPC state itself is a §4 contract module already
covered by shared/tests/test_ipc.py.
"""
import os

import numpy as np

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_path(name):
    return os.path.join(FIXTURES_DIR, name)


class FakeClock:
    """Deterministic, sleep_fn-advanced clock -- no test should depend on
    real wall-clock timing for wait()/listen_utterance() timeouts."""

    def __init__(self, start=0.0):
        self.t = start

    def time(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class ScriptedVAD:
    """Fake VAD driven by a fixed sequence of speech_prob values, one per
    call; holds the last value once the script runs out."""

    def __init__(self, probs, threshold=0.5):
        self._probs = list(probs)
        self._i = 0
        self.threshold = threshold
        self.calls = 0

    def speech_prob(self, frame_int16):
        self.calls += 1
        if self._i < len(self._probs):
            p = self._probs[self._i]
            self._i += 1
        else:
            p = self._probs[-1] if self._probs else 0.0
        return p

    def is_speech(self, frame_int16):
        return self.speech_prob(frame_int16) >= self.threshold


class EnergyVAD:
    """Fake VAD that scores real PCM frames by RMS energy -- lets
    fixture-driven tests (real WAV bytes) exercise VAD-gated logic without
    torch/silero-vad installed."""

    def __init__(self, threshold=0.5, rms_scale=6000.0):
        self.threshold = threshold
        self._scale = rms_scale

    def speech_prob(self, frame_int16):
        frame = np.asarray(frame_int16, dtype=np.float64)
        if frame.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return min(1.0, rms / self._scale)

    def is_speech(self, frame_int16):
        return self.speech_prob(frame_int16) >= self.threshold


class WavMicSource:
    """Fake mic: serves fixed-size frames, in order, from preloaded PCM
    samples (e.g. a WAV fixture read via conversation.audio_dev.
    read_wav_int16). Once exhausted, serves trailing silence forever so
    endpointing/timeout logic in the code under test terminates naturally
    rather than the fake raising StopIteration."""

    def __init__(self, samples):
        self._samples = np.asarray(samples, dtype=np.int16)
        self._pos = 0

    def read(self, n_samples):
        end = self._pos + n_samples
        chunk = self._samples[self._pos:end]
        self._pos = end
        if len(chunk) < n_samples:
            pad = np.zeros(n_samples - len(chunk), dtype=np.int16)
            chunk = np.concatenate([chunk, pad])
        return chunk

    def close(self):
        pass


class ClockAdvancingMicSource:
    """Wraps another fake mic source and advances a shared FakeClock by
    frame_s on every read() -- for tests that need wall-clock-timeout
    behavior (e.g. DirectedSTT's max_s) exercised deterministically
    against a source that never naturally end-points."""

    def __init__(self, inner, clock, frame_s):
        self._inner = inner
        self._clock = clock
        self._frame_s = frame_s

    def read(self, n_samples):
        self._clock.sleep(self._frame_s)
        return self._inner.read(n_samples)


class ConstantMicSource:
    """Fake mic that always returns (a tile of) the same frame -- e.g. a
    constant tone that never naturally end-points."""

    def __init__(self, frame):
        self._frame = np.asarray(frame, dtype=np.int16)

    def read(self, n_samples):
        if len(self._frame) >= n_samples:
            return self._frame[:n_samples]
        reps = n_samples // len(self._frame) + 1
        return np.tile(self._frame, reps)[:n_samples]


class FakeWhisperModel:
    """Fake recognizer: transcribe(samples, sample_rate) -> fixed text,
    recording call args so tests can assert what audio it was fed."""

    def __init__(self, text="hello robot"):
        self.text = text
        self.calls = []

    def transcribe(self, samples, sample_rate=16000):
        self.calls.append((len(samples), sample_rate))
        return self.text


class FakeSynthesizer:
    """Fake Piper synthesizer: synthesize(text) -> a short deterministic
    int16 tone whose length scales with len(text), so say()/say_stream()
    can be exercised without piper-tts installed."""

    sample_rate = 16000

    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        n = max(1, len(text)) * 80
        t = np.arange(n) / self.sample_rate
        return (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)

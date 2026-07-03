"""faster-whisper loader shared by stt.py (small/base, directed) and
ambient.py (tiny, ambient) -- ORCHESTRATION.md SS3.2. One adapter shape,
transcribe(int16_mono_16k_array) -> str, so both consumers -- and any
injected fake recognizer in tests -- look the same. Lazy import: this
module is importable, and FasterWhisperModel is constructible-in-name-only
until you actually call load_faster_whisper(); the ImportError only
becomes a RuntimeError at that point, with a clear fix.
"""
import numpy as np

from conversation.audio_dev import SAMPLE_RATE


class FasterWhisperModel:
    """Wraps faster_whisper.WhisperModel. transcribe() takes a raw int16
    mono array at SAMPLE_RATE (matching everything else in conversation/)
    rather than a file path, so directed/ambient audio never has to round-
    trip through a temp WAV just to get transcribed."""

    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "DirectedSTT/AmbientBuffer need faster-whisper for real "
                "transcription. Install with `pip install faster-whisper` "
                "(model weights download on first use), or inject a fake "
                "model (must implement transcribe(int16_array) -> str) "
                "for tests/sim."
            ) from exc
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, samples_int16, sample_rate=SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                "FasterWhisperModel expects %dHz audio, got %dHz"
                % (SAMPLE_RATE, sample_rate)
            )
        audio = np.asarray(samples_int16, dtype=np.float32) / 32768.0
        segments, _info = self._model.transcribe(audio, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()


def load_faster_whisper(model_size="small", **kwargs):
    """Factory kept separate from the class so callers/tests can patch
    just this one function instead of reaching into DirectedSTT/
    AmbientBuffer internals."""
    return FasterWhisperModel(model_size=model_size, **kwargs)

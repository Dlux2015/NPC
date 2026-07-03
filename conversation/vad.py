"""Silero VAD wrapper shared by wake.py (speech-onset detection for the
face-in-range + speech wake path), stt.py (utterance end-pointing), and
ambient.py (VAD gating). One lazy-loaded model, one interface
(speech_prob()/is_speech()), so every consumer can accept an injected fake
instead -- no torch, no downloaded model, needed for tests/sim.
"""
import numpy as np


class SileroVAD:
    """threshold: speech-probability cutoff used by is_speech(); read from
    the active shell profile's audio.json (vad_threshold) by callers, not
    hardcoded here."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self._model = None
        self._torch = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "SileroVAD requires torch plus the silero-vad model "
                "(downloaded via torch.hub on first use). Install torch, "
                "or inject a fake VAD (must implement speech_prob(frame) "
                "-> float and is_speech(frame) -> bool) for tests/sim."
            ) from exc
        model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        self._model = model
        self._torch = torch

    def speech_prob(self, frame_int16):
        """frame_int16: 1-D numpy int16 array, 16kHz mono. Returns a
        speech-probability float in [0, 1]."""
        self._ensure_loaded()
        audio = self._torch.from_numpy(
            np.asarray(frame_int16, dtype=np.float32) / 32768.0
        )
        with self._torch.no_grad():
            prob = self._model(audio, 16000).item()
        return float(prob)

    def is_speech(self, frame_int16):
        return self.speech_prob(frame_int16) >= self.threshold

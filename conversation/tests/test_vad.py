"""conversation/vad.py: SileroVAD against the REAL downloaded model when
torch is available on this machine (skipped otherwise -- torch is a heavy,
optional dependency, not part of the no-download test baseline).

This exists because the 512-sample-minimum-chunk requirement below was
discovered the hard way: every other test in conversation/ injects a fake
VAD (EnergyVAD or similar), so nothing ever exercised the real Silero
model until a live mic test did. Silero's current hubconf model rejects
any chunk shorter than 512 samples at 16kHz ("Input audio chunk is too
short"), which is why stt.py/wake.py/ambient.py default frame_ms to 32
(-> exactly 512 samples), not 30.
"""
import numpy as np
import pytest

from conversation.audio_dev import SAMPLE_RATE
from conversation.vad import SileroVAD


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_torch(), reason="torch not installed")


def test_speech_prob_accepts_the_minimum_512_sample_chunk():
    vad = SileroVAD()
    frame = np.zeros(512, dtype=np.int16)
    prob = vad.speech_prob(frame)
    assert 0.0 <= prob <= 1.0


def test_speech_prob_rejects_a_shorter_chunk_with_a_clear_error():
    # Documents the constraint rather than silently working around it:
    # anyone tempted to shrink frame_ms below 32 (512 samples @ 16kHz)
    # should see this fail loudly.
    vad = SileroVAD()
    frame = np.zeros(480, dtype=np.int16)
    with pytest.raises(Exception, match="too short"):
        vad.speech_prob(frame)


def test_real_stt_wake_ambient_frame_sizes_meet_silero_minimum():
    """The actual defaults used in production code must stay >=512
    samples at 16kHz -- this is what the 480-vs-512 bug above looked
    like from the call site."""
    from conversation.ambient import AmbientBuffer
    from conversation.stt import DirectedSTT
    from conversation.wake import WakeTrigger

    profile = {"name": "sim"}
    audio_config = {"sample_rate": SAMPLE_RATE}
    stt = DirectedSTT(profile, audio_config=audio_config)
    ambient = AmbientBuffer(profile, state=None, audio_config=audio_config)
    wake = WakeTrigger(profile, state=None, audio_config=audio_config)

    assert stt.frame_samples >= 512
    assert ambient.frame_samples >= 512
    assert wake.frame_samples >= 512

"""SapiSynthesizer: dev-PC-only TTS fallback via pyttsx3 (Windows SAPI5).

DEV-PC ONLY. Production/bench shells use conversation/tts.py's
PiperSynthesizer -- one version-pinned local Piper .onnx voice, per
ORCHESTRATION.md SS3.3. pyttsx3 instead speaks through whatever SAPI
voices happen to be installed on this particular Windows machine, which
is neither local-to-the-repo nor reproducible across machines, so this
class exists purely so conversation/demo_talk.py can still exercise the
real conversation.tts.Speaker lifecycle (actively_speaking, half-duplex
gating) on a dev PC that has not downloaded a Piper voice yet. Never
point a real profile.yaml's `tts_model_path` at this -- it isn't a model
path consumer at all, it's a drop-in synthesizer object.

Matches conversation/tts.py's Speaker synthesizer contract exactly:
synthesize(text) -> 1-D int16 numpy array, and a `sample_rate` attribute
Speaker.say_stream() reads (via getattr) immediately after synthesize()
returns. SAPI voices vary in native output sample rate machine to
machine, so there's no fixed constant to promise up front the way Piper's
voice.config.sample_rate can -- synthesize() updates self.sample_rate to
match the audio it just produced, before returning.

Implemented via save-to-wav + read-back: pyttsx3 has no in-memory PCM
API, so each synthesize() call writes a throwaway WAV under a private
temp dir and reads it straight back with conversation.audio_dev.
read_wav_int16 (which itself enforces mono 16-bit PCM -- exactly what
SAPI5's default output format is).
"""
import atexit
import os
import shutil
import tempfile

import numpy as np

from conversation.audio_dev import read_wav_int16

# Placeholder until the first real synthesize() call reports the actual
# rate of this machine's installed SAPI voice.
DEFAULT_SAMPLE_RATE = 22050


def _import_pyttsx3():
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError(
            "SapiSynthesizer requires pyttsx3 (dev-PC-only TTS fallback, "
            "Windows SAPI5). Install with `pip install pyttsx3`, or "
            "configure a real profile['tts_model_path'] Piper voice "
            "instead (see conversation/tts.py's PiperSynthesizer)."
        ) from exc
    return pyttsx3


class SapiSynthesizer:
    """synthesize(text) -> 1-D int16 numpy array -- the interface Speaker
    expects from any injected synthesizer (see conversation/tts.py)."""

    def __init__(self, rate_wpm=None, volume=None, voice_id=None, engine=None):
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self._tmp_dir = tempfile.mkdtemp(prefix="cbot_sapi_tts_")
        self._n = 0
        atexit.register(self._cleanup)

        if engine is not None:
            self._engine = engine
        else:
            pyttsx3 = _import_pyttsx3()
            self._engine = pyttsx3.init()

        if rate_wpm is not None:
            self._engine.setProperty("rate", rate_wpm)
        if volume is not None:
            self._engine.setProperty("volume", volume)
        if voice_id is not None:
            self._engine.setProperty("voice", voice_id)

    def synthesize(self, text):
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.int16)

        self._n += 1
        path = os.path.join(self._tmp_dir, "utt_%04d.wav" % self._n)
        self._engine.save_to_file(text, path)
        self._engine.runAndWait()

        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            # Some SAPI voices/driver combos occasionally drop a request;
            # treat it as "nothing to say" rather than crashing the
            # conversation loop over a dev-only fallback's flakiness.
            return np.zeros(0, dtype=np.int16)

        samples, rate = read_wav_int16(path)
        self.sample_rate = rate
        try:
            os.remove(path)
        except OSError:
            pass
        return samples

    def _cleanup(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

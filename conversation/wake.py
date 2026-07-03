"""WakeTrigger: enters directed conversation mode (ORCHESTRATION.md SS3.2).
Three sources, polled every cycle while the robot is not
actively_speaking:

  "ptt"         -- push-to-talk (keyboard fallback on the dev PC; real
                   hardware may wire a physical button the same way later)
  "wakeword"    -- openWakeWord score >= profile audio.json's wake_threshold
  "face_speech" -- state.person_in_range True + VAD speech onset

wait(timeout_s) returns the first source name that fires, or None on
timeout. Suspended entirely while actively_speaking (shared/ipc.py) -- the
robot must never wake itself on its own TTS output; both wake and
conversation/ambient.py check the same flag.

Every heavy dependency (openWakeWord, sounddevice, keyboard) is imported
lazily, and every source accepts an injected fake (wakeword_model,
mic_source, ptt_poll, vad) so WakeTrigger is constructible and fully
testable with no audio hardware, no wake-word model file, and no keyboard
library installed. The *factories* below (OpenWakeWordModel, MicStream)
are what raise a clear, actionable RuntimeError -- for whoever explicitly
constructs one on real hardware. WakeTrigger itself never raises out of
wait() for a missing backend: a source with no real dependency and no
injected fake just permanently stops firing for that source (cached in
self._unavailable, so we don't retry an expensive failed import/open every
~50ms poll).
"""
import time

from conversation.audio_dev import load_audio_config
from conversation.vad import SileroVAD

WAKE_SOURCES = ("ptt", "wakeword", "face_speech")


class OpenWakeWordModel:
    """Lazy openWakeWord wrapper. score(frame_int16) -> float in [0, 1],
    the max score across the configured wake word(s) for this chunk --
    the interface any injected fake wakeword_model must also implement."""

    def __init__(self, model_path=None, wakeword_names=None):
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "WakeTrigger's wakeword source requires the openwakeword "
                "package plus its model files. Install with `pip install "
                "openwakeword`, or inject a fake wakeword_model (must "
                "implement score(frame_int16) -> float) for tests/sim."
            ) from exc
        kwargs = {"wakeword_models": [model_path]} if model_path else {}
        self._model = Model(**kwargs)
        self._names = wakeword_names

    def score(self, frame_int16):
        import numpy as np
        predictions = self._model.predict(np.asarray(frame_int16, dtype=np.int16))
        if not predictions:
            return 0.0
        if self._names:
            return max(predictions.get(n, 0.0) for n in self._names)
        return max(predictions.values())


def _default_ptt_poll(key="space"):
    """Keyboard fallback for the dev PC (SS3.2). Returns None (source
    unavailable) if the `keyboard` package isn't installed, rather than
    raising -- PTT is itself a fallback, not the primary path."""
    try:
        import keyboard
    except ImportError:
        return None

    def poll():
        try:
            return bool(keyboard.is_pressed(key))
        except Exception:
            return False

    return poll


class WakeTrigger:
    def __init__(self, profile, state, mic_source=None, wakeword_model=None,
                 vad=None, ptt_poll=None, audio_config=None,
                 frame_ms=30, poll_interval_s=0.05, ptt_key="space",
                 clock=time.monotonic, sleep_fn=time.sleep):
        self.profile = profile
        self.state = state
        self.audio_config = audio_config or load_audio_config(profile.get("name", "sim"))
        self._mic_source = mic_source
        self._wakeword_model = wakeword_model
        self._vad = vad or SileroVAD(threshold=self.audio_config.get("vad_threshold", 0.5))
        self._ptt_poll = ptt_poll
        self._ptt_key = ptt_key
        self.frame_samples = max(
            1, int(self.audio_config.get("sample_rate", 16000) * frame_ms / 1000)
        )
        self.poll_interval_s = poll_interval_s
        self._clock = clock
        self._sleep = sleep_fn
        # Source names whose real backend failed to load/open and have no
        # injected fake -- permanently disabled for this instance.
        self._unavailable = set()

    def wait(self, timeout_s=None):
        """Blocks (polling every poll_interval_s) until a source fires or
        timeout_s elapses. timeout_s=None blocks indefinitely."""
        deadline = None if timeout_s is None else self._clock() + timeout_s
        while True:
            if not self.state.get("actively_speaking"):
                source = self._check_once()
                if source is not None:
                    return source
            if deadline is not None and self._clock() >= deadline:
                return None
            self._sleep(self.poll_interval_s)

    def _check_once(self):
        if self._check_ptt():
            return "ptt"
        frame = self._read_frame() if self._audio_sources_live() else None
        if self._check_wakeword(frame):
            return "wakeword"
        if self._check_face_speech(frame):
            return "face_speech"
        return None

    def _audio_sources_live(self):
        return ("wakeword" not in self._unavailable
                or "face_speech" not in self._unavailable)

    # -- ptt --------------------------------------------------------------

    def _check_ptt(self):
        if "ptt" in self._unavailable:
            return False
        poll = self._ptt_poll
        if poll is None:
            poll = _default_ptt_poll(self._ptt_key)
            if poll is None:
                self._unavailable.add("ptt")
                return False
            self._ptt_poll = poll
        try:
            return bool(poll())
        except Exception:
            return False

    # -- wakeword -----------------------------------------------------------

    def _check_wakeword(self, frame):
        if frame is None or "wakeword" in self._unavailable:
            return False
        model = self._wakeword_model
        if model is None:
            try:
                model = OpenWakeWordModel(
                    model_path=self.profile.get("wakeword_model_path"))
            except RuntimeError:
                self._unavailable.add("wakeword")
                return False
            self._wakeword_model = model
        score = model.score(frame)
        return score >= self.audio_config.get("wake_threshold", 0.5)

    # -- face + speech onset ------------------------------------------------

    def _check_face_speech(self, frame):
        if frame is None or "face_speech" in self._unavailable:
            return False
        if not self.state.get("person_in_range"):
            return False
        try:
            return self._vad.is_speech(frame)
        except RuntimeError:
            self._unavailable.add("face_speech")
            return False

    # -- shared mic frame -----------------------------------------------------

    def _read_frame(self):
        if self._mic_source is None:
            try:
                from conversation.audio_dev import MicStream
                self._mic_source = MicStream(self.audio_config)
            except RuntimeError:
                self._unavailable.update({"wakeword", "face_speech"})
                return None
        return self._mic_source.read(self.frame_samples)

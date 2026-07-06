"""AmbientBuffer: low-priority, VAD-gated ambient listening (ORCHESTRATION.md
SS3.2). Own daemon thread; rolling ~60s **in-memory-only** transcript (never
persisted -- shared/ipc.py's ambient_transcript key is itself scratch
state, not a database); faster-whisper tiny; publishes to shared/ipc.py at
<=0.5Hz; pauses entirely while actively_speaking (never transcribes the
robot's own voice); lowest priority of the whole system -- must never
starve directed listening (conversation/wake.py + stt.py) or the vision
process.

Priority note: CPython has no portable, reliable thread-priority knob, so
"lowest priority" here means a duty-cycle sleep between every mic poll
(duty_cycle_s) rather than a tight loop -- the thread spends most of its
time asleep, yielding the GIL/CPU to the directed-listening and vision
processes instead of contending with them.
"""
import threading
import time

import numpy as np

from conversation.audio_dev import SAMPLE_RATE, load_audio_config
from conversation.vad import SileroVAD

WINDOW_S = 60.0
PUBLISH_INTERVAL_S = 2.0  # <=0.5Hz, per SS3.2
DEFAULT_DUTY_CYCLE_S = 0.2


class AmbientBuffer:
    def __init__(self, profile, state, mic_source=None, model=None, vad=None,
                 audio_config=None, frame_ms=32, window_s=WINDOW_S,
                 publish_interval_s=PUBLISH_INTERVAL_S,
                 duty_cycle_s=DEFAULT_DUTY_CYCLE_S,
                 clock=time.monotonic, sleep_fn=time.sleep):
        self.profile = profile
        self.state = state
        self.audio_config = audio_config or load_audio_config(profile.get("name", "sim"))
        self._mic_source = mic_source
        self._model = model
        self._vad = vad or SileroVAD(threshold=self.audio_config.get("vad_threshold", 0.5))
        self.sample_rate = self.audio_config.get("sample_rate", SAMPLE_RATE)
        # frame_ms=32 -> 512 samples at 16kHz, Silero VAD's real minimum
        # chunk size (see conversation/stt.py's DirectedSTT for detail).
        self.frame_samples = max(1, int(self.sample_rate * frame_ms / 1000))
        self.window_s = window_s
        self.publish_interval_s = publish_interval_s
        self.duty_cycle_s = duty_cycle_s
        self._clock = clock
        self._sleep = sleep_fn

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._entries = []  # list of (timestamp, text), oldest first
        self._last_publish = float("-inf")
        # Sources whose real backend failed to load and no fake was
        # injected -- degrade to idle rather than spinning retries.
        self._mic_unavailable = False
        self._model_unavailable = False

    # -- lifecycle ------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ambient-buffer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout=1.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def snapshot(self):
        """Current rolling transcript, oldest first, trimmed to window_s."""
        with self._lock:
            self._trim(self._clock())
            return [text for _, text in self._entries]

    # -- internals --------------------------------------------------------

    def _trim(self, now):
        cutoff = now - self.window_s
        self._entries = [e for e in self._entries if e[0] >= cutoff]

    def _ensure_model(self):
        if self._model is None:
            from conversation.whisper_model import load_faster_whisper
            self._model = load_faster_whisper("tiny")
        return self._model

    def _ensure_mic(self):
        if self._mic_source is None:
            from conversation.audio_dev import MicStream
            self._mic_source = MicStream(self.audio_config)
        return self._mic_source

    def _run(self):
        threshold = self.audio_config.get("vad_threshold", 0.5)
        speech_buf = []
        while not self._stop.is_set():
            if self.state.get("actively_speaking"):
                speech_buf = []
                self._sleep(self.duty_cycle_s)
                continue

            frame = self._read_frame_safe()
            if frame is None:
                self._sleep(self.duty_cycle_s)
                continue

            try:
                is_speech = self._vad.speech_prob(frame) >= threshold
            except RuntimeError:
                # No VAD backend and no fake injected -- ambient is the
                # lowest-priority feature; degrade silently rather than
                # spamming a background thread's stderr.
                self._sleep(self.duty_cycle_s)
                continue

            if is_speech:
                speech_buf.append(frame)
            elif speech_buf:
                audio = np.concatenate(speech_buf)
                speech_buf = []
                self._transcribe_and_store(audio)

            self._sleep(self.duty_cycle_s)

    def _read_frame_safe(self):
        if self._mic_unavailable:
            return None
        try:
            mic = self._ensure_mic()
        except RuntimeError:
            self._mic_unavailable = True
            return None
        frame = mic.read(self.frame_samples)
        if frame is None or len(frame) == 0:
            return None
        return frame

    def _transcribe_and_store(self, audio):
        if self._model_unavailable:
            return
        try:
            model = self._ensure_model()
        except RuntimeError:
            self._model_unavailable = True
            return
        text = (model.transcribe(audio, sample_rate=self.sample_rate) or "").strip()
        if not text:
            return
        now = self._clock()
        with self._lock:
            self._entries.append((now, text))
            self._trim(now)
        self._maybe_publish(now)

    def _maybe_publish(self, now):
        if now - self._last_publish < self.publish_interval_s:
            return
        self._last_publish = now
        with self._lock:
            transcript = [text for _, text in self._entries]
        self.state.update(ambient_transcript=transcript)

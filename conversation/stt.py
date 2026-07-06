"""DirectedSTT: full speech-to-text for the "someone is addressing the
robot" path (ORCHESTRATION.md SS3.2). faster-whisper small/base, lazy
import; Silero VAD end-pointing decides when the utterance is over.

listen_utterance() is the live-mic entry point pipeline.py calls after
conversation/wake.py fires. transcribe_wav() is the fixture/sim entry
point -- no mic needed, used by conversation/tests/ and by the sim's
canned-utterance scenarios (SS3.6).
"""
import time

import numpy as np

from conversation.audio_dev import SAMPLE_RATE, load_audio_config, read_wav_int16
from conversation.vad import SileroVAD


class DirectedSTT:
    def __init__(self, profile, mic_source=None, model=None, vad=None,
                 audio_config=None, model_size="small", frame_ms=32,
                 end_silence_s=0.8, min_speech_s=0.2,
                 clock=time.monotonic, sleep_fn=time.sleep):
        self.profile = profile
        self.audio_config = audio_config or load_audio_config(profile.get("name", "sim"))
        self._mic_source = mic_source
        self._model = model
        self._vad = vad or SileroVAD(threshold=self.audio_config.get("vad_threshold", 0.5))
        self.model_size = model_size
        self.sample_rate = self.audio_config.get("sample_rate", SAMPLE_RATE)
        # frame_ms=32 -> exactly 512 samples at 16kHz: Silero VAD's real
        # model rejects any chunk shorter than that ("Input audio chunk
        # is too short"). 30ms (480 samples) looked reasonable but was
        # never actually exercised against the real model until a live
        # mic test surfaced it -- every test here injects a fake VAD.
        self.frame_samples = max(1, int(self.sample_rate * frame_ms / 1000))
        self.end_silence_s = end_silence_s
        self.min_speech_s = min_speech_s
        self._clock = clock
        self._sleep = sleep_fn

    def _ensure_model(self):
        if self._model is None:
            from conversation.whisper_model import load_faster_whisper
            self._model = load_faster_whisper(self.model_size)
        return self._model

    def _ensure_mic(self):
        if self._mic_source is None:
            from conversation.audio_dev import MicStream
            self._mic_source = MicStream(self.audio_config)
        return self._mic_source

    def listen_utterance(self, max_s=10.0):
        """Records from the mic until VAD detects `end_silence_s` of
        trailing silence (or `max_s` of audio has been captured), then
        transcribes. Returns None if no speech (>= min_speech_s worth) was
        ever detected -- callers should treat that as "nothing said", not
        an error.

        max_s bounds *audio time captured* (sum of frame durations), not
        wall-clock time: real hardware capture (conversation.audio_dev.
        MicStream.read()) blocks for roughly one frame duration per call,
        so the two coincide in production, but bounding on audio time
        keeps this method's own termination guaranteed even against an
        instantaneous/fake mic_source (as used in tests), independent of
        self._clock()/self._sleep -- no live-audio source should ever be
        able to make this loop spin forever."""
        mic = self._ensure_mic()
        threshold = self.audio_config.get("vad_threshold", 0.5)
        frames = []
        speech_s = 0.0
        trailing_silence_s = 0.0
        started = False
        elapsed_s = 0.0
        while elapsed_s < max_s:
            frame = mic.read(self.frame_samples)
            if frame is None or len(frame) == 0:
                self._sleep(0.01)
                continue
            frame_s = len(frame) / float(self.sample_rate)
            elapsed_s += frame_s
            is_speech = self._vad.speech_prob(frame) >= threshold
            if is_speech:
                started = True
                speech_s += frame_s
                trailing_silence_s = 0.0
                frames.append(frame)
            elif started:
                trailing_silence_s += frame_s
                frames.append(frame)
                if trailing_silence_s >= self.end_silence_s:
                    break

        if not started or speech_s < self.min_speech_s:
            return None

        audio = np.concatenate(frames)
        text = self._ensure_model().transcribe(audio, sample_rate=self.sample_rate)
        text = (text or "").strip()
        return text or None

    def transcribe_wav(self, path):
        """Fixture/sim entry point: transcribes a mono 16-bit PCM WAV file
        directly, no mic/VAD end-pointing involved."""
        samples, rate = read_wav_int16(path)
        if rate != SAMPLE_RATE:
            raise ValueError(
                "%s: expected %dHz mono, got %dHz -- fixtures must match "
                "the hard 16kHz mono capture rate" % (path, SAMPLE_RATE, rate)
            )
        text = self._ensure_model().transcribe(samples, sample_rate=rate)
        return (text or "").strip()

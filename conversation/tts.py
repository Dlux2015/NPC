"""Speaker: local TTS output (ORCHESTRATION.md SS3.3). Piper, lazy import,
one version-pinned voice per profile (profile['tts_model_path']).

Half-duplex hard rule: actively_speaking is published via shared/ipc.py
*before* the first byte of audio plays, and cleared only *after* playback
has fully drained -- conversation/wake.py and conversation/ambient.py both
poll this flag every cycle and must never hear the robot's own voice.

Sentence-streaming (say_stream): synthesizes+plays each sentence as it
arrives from an iterator (pipeline.py can hand this the LLM's own
sentence-chunked output stream) so sentence 1 is already playing while the
LLM is still generating the rest -- perceived latency beats total latency.
say(text) is just say_stream(iter([text])).

NullAudioSink (for tests / sim without a sound card) writes each play()
call's samples to a WAV file under a temp dir instead of a real speaker.
"""
import os
import re
import tempfile

from conversation.audio_dev import SAMPLE_RATE, load_audio_config, write_wav_int16

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    """Simple punctuation-based sentence splitter -- good enough for
    persona replies (1-3 short sentences, SS3.4); not a general NLP
    tokenizer. Useful for callers who have full text up front but still
    want sentence-streamed playback.

    Same name, different input than conversation/pipeline.py's
    split_sentences: this one takes a complete string; pipeline's takes
    an ITERATOR of LLM chunks and yields sentences as they complete
    (that's the one that lets speech start before generation ends)."""
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


class NullAudioSink:
    """Test/sim playback backend: writes each play() call's samples to a
    new WAV file under `dir` instead of touching a sound card. Records the
    written paths (in play() order) so tests can inspect what "played"."""

    def __init__(self, dir=None):
        self.dir = dir or tempfile.mkdtemp(prefix="cbot_null_audio_")
        os.makedirs(self.dir, exist_ok=True)
        self.written = []
        self._n = 0

    def play(self, samples, sample_rate=SAMPLE_RATE):
        path = os.path.join(self.dir, "utterance_%03d.wav" % self._n)
        self._n += 1
        write_wav_int16(path, samples, sample_rate=sample_rate)
        self.written.append(path)
        return path


def _import_piper():
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise RuntimeError(
            "Speaker requires the piper-tts package plus a downloaded, "
            "version-pinned .onnx voice model (profile['tts_model_path']; "
            "pin the exact model file + its .onnx.json config together "
            "with a checksum in the profile so shells stay reproducible). "
            "Install with `pip install piper-tts`, or inject a fake "
            "synthesizer (must implement synthesize(text) -> int16 array) "
            "for tests/sim."
        ) from exc
    return PiperVoice


class PiperSynthesizer:
    """Lazy-loaded Piper wrapper. synthesize(text) -> 1-D int16 numpy
    array at self.sample_rate -- the interface any injected fake
    synthesizer must also implement."""

    def __init__(self, model_path):
        if not model_path:
            raise RuntimeError(
                "Speaker needs profile['tts_model_path'] pointing at a "
                "downloaded, version-pinned Piper .onnx voice."
            )
        PiperVoice = _import_piper()
        self._voice = PiperVoice.load(model_path)

    def synthesize(self, text):
        import numpy as np
        chunks = [chunk.audio_int16_array for chunk in self._voice.synthesize(text)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype="int16")

    @property
    def sample_rate(self):
        return self._voice.config.sample_rate


KOKORO_SAMPLE_RATE = 24000


def _import_kokoro():
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Speaker requires the 'kokoro' package for KokoroSynthesizer "
            "(pip install kokoro soundfile). It bundles its own espeak-ng "
            "phonemizer binaries (espeakng-loader) -- no system install "
            "needed. Or inject a fake synthesizer (must implement "
            "synthesize(text) -> int16 array) for tests/sim."
        ) from exc
    return KPipeline


class KokoroSynthesizer:
    """Lazy-loaded Kokoro-82M wrapper (hexgrad/Kokoro-82M via the `kokoro`
    package, auto-downloaded from Hugging Face on first use).
    synthesize(text) -> 1-D int16 numpy array at self.sample_rate (24kHz)
    -- the same interface PiperSynthesizer implements, so Speaker doesn't
    care which is injected.

    Chosen (2026-07-06, listening comparison) for noticeably more natural
    prosody than Piper's low/medium voices, at a similar (82M parameter)
    model size. Speed depends on the torch build: ~1.1-2.2x realtime on
    CPU-only torch (audibly laggy replies) vs ~41x realtime measured on
    an RTX 4090 with CUDA torch -- the constructor auto-selects cuda when
    available. NOTE for the robot: the Jetson's GPU belongs to the LLM
    (SS3.4), so Kokoro there would be CPU-only -- bench its CPU speed on
    the Orin before considering it over Piper for a hardware profile.

    lang_code selects phonemizer/prosody rules ("a"=American English,
    "b"=British English -- see the kokoro package for the full list);
    voice selects the speaker embedding (e.g. "am_michael", "af_heart",
    "bm_george"). Keep them paired to the voice's own prefix (af_/am_ ->
    "a", bf_/bm_ -> "b") -- mismatching still runs, but sounds off.
    """

    def __init__(self, voice="am_michael", lang_code="a", device=None):
        KPipeline = _import_kokoro()
        self.voice = voice
        self.lang_code = lang_code
        self.sample_rate = KOKORO_SAMPLE_RATE
        if device is None:
            # Prefer the GPU when this interpreter's torch has CUDA --
            # measured ~1.1-2.2x realtime on CPU vs. far faster on GPU;
            # per-sentence synth latency is the audible lag in replies.
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"  # kokoro itself will fail later; tests fake it
        self.device = device
        self._pipeline = KPipeline(lang_code=lang_code, device=device)

    def synthesize(self, text):
        import numpy as np
        chunks = []
        for result in self._pipeline(text, voice=self.voice):
            audio_t = result.audio
            if audio_t is None:
                continue
            # CUDA tensors must come back to host memory before numpy();
            # hasattr guards keep injected test fakes (bare .numpy())
            # working too.
            if hasattr(audio_t, "detach"):
                audio_t = audio_t.detach()
            if hasattr(audio_t, "cpu"):
                audio_t = audio_t.cpu()
            chunks.append(audio_t.numpy())
        if not chunks:
            return np.zeros(0, dtype="int16")
        audio = np.concatenate(chunks)
        return np.clip(audio * 32767.0, -32768, 32767).astype("int16")


class Speaker:
    def __init__(self, profile, state, synthesizer=None, sink=None,
                 audio_config=None):
        self.profile = profile
        self.state = state
        self.audio_config = audio_config or load_audio_config(profile.get("name", "sim"))
        self._synthesizer = synthesizer
        self._sink = sink

    def _ensure_synth(self):
        if self._synthesizer is None:
            self._synthesizer = PiperSynthesizer(self.profile.get("tts_model_path"))
        return self._synthesizer

    def _ensure_sink(self):
        if self._sink is None:
            from conversation.audio_dev import SpeakerSink
            self._sink = SpeakerSink(self.audio_config)
        return self._sink

    def say(self, text):
        """Blocking: synthesizes and plays one utterance."""
        text = (text or "").strip()
        if not text:
            return
        self.say_stream(iter([text]))

    def say_stream(self, sentence_iter):
        """Synthesizes+plays each item pulled from sentence_iter as it
        arrives. actively_speaking is set True right before the *first*
        sentence's audio starts playing, and cleared only after the
        *last* sentence's playback has drained -- one half-duplex window
        covering the whole reply, not one per sentence, so wake/ambient
        stay suspended for the entire utterance rather than flickering
        open in the gaps between synth calls."""
        synth = self._ensure_synth()
        sink = self._ensure_sink()
        spoke = False
        try:
            for sentence in sentence_iter:
                sentence = (sentence or "").strip()
                if not sentence:
                    continue
                samples = synth.synthesize(sentence)
                if samples is None or len(samples) == 0:
                    continue
                if not spoke:
                    self.state.update(actively_speaking=True)
                    spoke = True
                sink.play(samples, sample_rate=getattr(synth, "sample_rate", SAMPLE_RATE))
        finally:
            if spoke:
                self.state.update(actively_speaking=False)

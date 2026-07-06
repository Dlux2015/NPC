"""Device-by-name audio capture/playback helpers (ORCHESTRATION.md SS3.2/
SS3.3 hard rule: select the mic/speaker by name, never index -- enclosure
rebuilds and USB re-enumeration reorder indices, not names). Every heavy
dependency (sounddevice) is imported lazily so this module -- and every
conversation/ module built on it -- is importable and unit-testable on a
dev PC with no audio hardware and no sounddevice install.

profiles/<name>/audio.json (ORCHESTRATION.md SS3.5 step 7) holds the real
gains/thresholds/volume for a shell, written by the speech-engineer's
calibration pass (vision/calibrate.py's step7_audio_stub() documents the
handoff). Until that file exists for a given profile, DEFAULT_AUDIO_CONFIG
below is what's used -- conservative, documented safe defaults, never a
hardcoded number buried in wake.py/stt.py/ambient.py/tts.py.
"""
import json
import os
import wave

import numpy as np

from vision.paths import profile_dir

SAMPLE_RATE = 16000  # hard rule: 16kHz mono capture, everywhere
CHANNELS = 1

DEFAULT_AUDIO_CONFIG = {
    "input_device": None,   # substring match against device name; None = OS default
    "output_device": None,  # substring match against device name; None = OS default
    "sample_rate": SAMPLE_RATE,
    "mic_gain_db": 0.0,
    "vad_threshold": 0.5,   # Silero VAD speech-probability threshold
    "wake_threshold": 0.5,  # openWakeWord score threshold
    "output_volume": 1.0,   # linear, 0.0-1.0
}


def audio_config_path(profile_name, root=None):
    return os.path.join(profile_dir(profile_name, root), "audio.json")


def load_audio_config(profile_name, root=None):
    """Loads profiles/<name>/audio.json merged over DEFAULT_AUDIO_CONFIG,
    so a partially-written file still gets documented defaults for any
    missing key. If the file itself is absent (pre-calibration shell),
    returns DEFAULT_AUDIO_CONFIG unchanged -- never raises, since every
    conversation/ class must be constructible before calibration exists."""
    path = audio_config_path(profile_name, root)
    if not os.path.isfile(path):
        return dict(DEFAULT_AUDIO_CONFIG)
    with open(path, "r") as f:
        data = json.load(f)
    return {**DEFAULT_AUDIO_CONFIG, **data}


def write_audio_config(profile_name, config, root=None):
    """Writes profiles/<name>/audio.json (SS3.5 step 7 output)."""
    path = audio_config_path(profile_name, root)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


# --- WAV helpers (fixtures, NullAudioSink, transcribe_wav) -----------------

def read_wav_int16(path):
    """Reads a mono 16-bit PCM WAV into a 1-D int16 numpy array. Used by
    DirectedSTT.transcribe_wav()/fixture-based tests. Raises ValueError on
    anything that isn't mono 16-bit PCM -- a silent wrong-format read would
    just feed noise into whisper."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("%s: expected 16-bit PCM, got %d-byte samples"
                              % (path, wf.getsampwidth()))
        if wf.getnchannels() != 1:
            raise ValueError("%s: expected mono, got %d channels"
                              % (path, wf.getnchannels()))
        raw = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    samples = np.frombuffer(raw, dtype=np.int16)
    return samples, rate


def write_wav_int16(path, samples, sample_rate=SAMPLE_RATE):
    """Writes a 1-D int16 numpy array as mono 16-bit PCM WAV."""
    samples = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return path


# --- Real hardware backends (lazy sounddevice) ------------------------------

def _import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "This requires sounddevice for real audio hardware. Install "
            "with `pip install sounddevice`, or inject a fake audio "
            "source/sink for tests/sim."
        ) from exc
    return sd


def _wasapi_hostapi_index(sd):
    """Best-effort index of the "Windows WASAPI" host API in
    sd.query_hostapis(), or None if unavailable (non-Windows, an older
    sounddevice/PortAudio build, or a fake `sd` in tests that doesn't
    implement query_hostapis at all).

    Windows exposes the same physical device once per host API (MME,
    DirectSound, WASAPI, WDM-KS) with the SAME NAME, and legacy host APIs
    can deliver a heavily attenuated capture level for the exact device
    that sounds normal over WASAPI -- found 2026-07 debugging a dev-PC
    mic that opened without error and returned real-looking (but ~50x
    quieter than actual speech) samples over the PortAudio-default host
    API (MME), while WASAPI on the same device captured normal levels.
    WASAPI is preferred whenever it's available."""
    query_hostapis = getattr(sd, "query_hostapis", None)
    if query_hostapis is None:
        return None
    for i, api in enumerate(query_hostapis()):
        if "wasapi" in api.get("name", "").lower():
            return i
    return None


def resolve_device_by_name(name, kind="input", sd=None):
    """Returns the sounddevice device index whose name contains `name`
    (case-insensitive substring match) -- preferring the WASAPI host API's
    entry when the name matches more than one (see _wasapi_hostapi_index)
    -- or, if `name` is falsy, the WASAPI host API's own default device
    for `kind` (falling back to None, i.e. "let the OS/PortAudio default
    decide", only if WASAPI isn't available at all). Raises RuntimeError
    if a non-empty name matches nothing: silently falling back to "the
    first device" is exactly how you end up capturing from the wrong mic
    after a USB re-enumeration, which is the hard rule this module exists
    to prevent (ORCHESTRATION.md SS3.2: "device by name, not index").

    kind: "input" or "output".
    """
    sd = sd or _import_sounddevice()
    wasapi_idx = _wasapi_hostapi_index(sd)

    if not name:
        if wasapi_idx is None:
            return None
        key = "default_input_device" if kind == "input" else "default_output_device"
        idx = sd.query_hostapis()[wasapi_idx].get(key)
        return idx if idx is not None and idx >= 0 else None

    devices = sd.query_devices()
    want_key = "max_input_channels" if kind == "input" else "max_output_channels"
    name_lower = name.lower()
    matches = [
        idx for idx, dev in enumerate(devices)
        if dev.get(want_key, 0) > 0 and name_lower in dev.get("name", "").lower()
    ]
    if not matches:
        raise RuntimeError(
            "No %s audio device matching name %r found. Available devices: %s"
            % (kind, name, [d.get("name") for d in devices])
        )
    if wasapi_idx is not None:
        for idx in matches:
            if devices[idx].get("hostapi") == wasapi_idx:
                return idx
    return matches[0]


def _wasapi_extra_settings(sd, device_idx):
    """sd.WasapiSettings(auto_convert=True) if `device_idx` is a WASAPI
    device, else None (nothing extra to pass for other host APIs).

    WASAPI, unlike MME, rejects a samplerate that doesn't match the
    device's native mixer rate unless told to auto-convert -- discovered
    opening a 48kHz-native mic at the hard 16kHz capture rate
    (ORCHESTRATION.md SS3.2) once WASAPI became the preferred host API
    (see _wasapi_hostapi_index): MME silently resampled, WASAPI raised
    "Invalid sample rate" until this flag is set.
    """
    if device_idx is None:
        return None
    wasapi_idx = _wasapi_hostapi_index(sd)
    if wasapi_idx is None:
        return None
    if sd.query_devices()[device_idx].get("hostapi") != wasapi_idx:
        return None
    return sd.WasapiSettings(auto_convert=True)


class MicStream:
    """Real 16kHz mono capture backend from the named input device
    (config['input_device']). Lazy sounddevice import -- constructing this
    is the point at which "no audio hardware" becomes a clear RuntimeError
    (via resolve_device_by_name/_import_sounddevice); wake.py/stt.py/
    ambient.py never construct one unless no fake mic_source was injected.

    read(n_samples) -> 1-D int16 numpy array of exactly n_samples, gain-
    adjusted by config['mic_gain_db']. This is the interface any injected
    fake mic_source must also implement.
    """

    def __init__(self, config, sd=None):
        self._sd = sd or _import_sounddevice()
        self._device = resolve_device_by_name(config.get("input_device"), "input", sd=self._sd)
        self._sample_rate = config.get("sample_rate", SAMPLE_RATE)
        self._gain = 10 ** (config.get("mic_gain_db", 0.0) / 20.0)
        self._stream = self._sd.InputStream(
            samplerate=self._sample_rate, channels=CHANNELS, dtype="int16",
            device=self._device,
            extra_settings=_wasapi_extra_settings(self._sd, self._device),
        )
        self._stream.start()

    def read(self, n_samples):
        data, _overflowed = self._stream.read(n_samples)
        frame = data[:, 0] if data.ndim > 1 else data
        if self._gain != 1.0:
            frame = np.clip(frame.astype(np.float32) * self._gain, -32768, 32767)
            frame = frame.astype(np.int16)
        return frame

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class SpeakerSink:
    """Real playback backend through the named output device
    (config['output_device']), blocking until playback drains (needed so
    conversation/tts.py only clears actively_speaking after the audio has
    actually finished, not after it was merely queued). Volume from
    config['output_volume']. Lazy sounddevice import.

    play(samples, sample_rate) is the interface NullAudioSink also
    implements, for tests/sim.
    """

    def __init__(self, config, sd=None):
        self._sd = sd or _import_sounddevice()
        self._device = resolve_device_by_name(config.get("output_device"), "output", sd=self._sd)
        self.volume = config.get("output_volume", 1.0)

    def play(self, samples, sample_rate=SAMPLE_RATE):
        data = (np.asarray(samples, dtype=np.float32) / 32768.0) * self.volume
        extra_settings = _wasapi_extra_settings(self._sd, self._device)
        self._sd.play(data, samplerate=sample_rate, device=self._device,
                       blocking=True, extra_settings=extra_settings)

"""Tests for conversation/audio_dev.py: device-by-name resolution, audio.json
loading with documented defaults, and the WAV read/write helpers used by
NullAudioSink/DirectedSTT.transcribe_wav. No sounddevice install or audio
hardware required -- resolve_device_by_name takes an injected fake `sd`.
"""
import numpy as np
import pytest

from conversation.audio_dev import (
    DEFAULT_AUDIO_CONFIG,
    SAMPLE_RATE,
    load_audio_config,
    read_wav_int16,
    resolve_device_by_name,
    write_audio_config,
    write_wav_int16,
)
from conversation.tests._audio_fakes import fixture_path


class FakeSoundDevice:
    """Minimal stand-in for the bits of the `sounddevice` module
    resolve_device_by_name touches. Deliberately has NO query_hostapis --
    exercises the "WASAPI unavailable" path (the only one testable
    without a `hostapi` key on every device)."""

    def __init__(self, devices):
        self._devices = devices

    def query_devices(self):
        return self._devices


class FakeSoundDeviceWithHostapis(FakeSoundDevice):
    """Adds query_hostapis(), so tests can exercise the WASAPI-preference
    path added 2026-07 after a real dev-PC mic turned out to be captured
    at ~50x attenuation over MME (the PortAudio-default host API on
    Windows) versus WASAPI for the exact same physical device."""

    def __init__(self, devices, hostapis):
        super().__init__(devices)
        self._hostapis = hostapis

    def query_hostapis(self):
        return self._hostapis


DEVICES = [
    {"name": "Built-in Microphone", "max_input_channels": 2, "max_output_channels": 0},
    {"name": "USB Audio Device", "max_input_channels": 1, "max_output_channels": 2},
    {"name": "HDMI Output", "max_input_channels": 0, "max_output_channels": 2},
]


def test_resolve_device_by_name_matches_case_insensitive_substring():
    sd = FakeSoundDevice(DEVICES)
    idx = resolve_device_by_name("usb audio", "input", sd=sd)
    assert idx == 1


def test_resolve_device_by_name_none_falls_back_to_default():
    sd = FakeSoundDevice(DEVICES)
    assert resolve_device_by_name(None, "input", sd=sd) is None
    assert resolve_device_by_name("", "output", sd=sd) is None


def test_resolve_device_by_name_never_falls_back_to_an_index():
    """Hard rule: device by name, never index -- an unmatched name must
    raise, not silently pick "the first device"."""
    sd = FakeSoundDevice(DEVICES)
    with pytest.raises(RuntimeError, match="No input audio device"):
        resolve_device_by_name("nonexistent mic", "input", sd=sd)


def test_resolve_device_by_name_respects_input_vs_output_channels():
    sd = FakeSoundDevice(DEVICES)
    # HDMI Output has 0 input channels -- must not match as an input device
    # even though the name would otherwise be findable.
    with pytest.raises(RuntimeError):
        resolve_device_by_name("HDMI", "input", sd=sd)
    assert resolve_device_by_name("HDMI", "output", sd=sd) == 2


# --- WASAPI preference (2026-07: same physical device, ~50x quieter over
# MME than WASAPI on a real dev-PC mic) -------------------------------------

def test_resolve_device_by_name_prefers_wasapi_among_matching_candidates():
    """Same mic name appears under two host APIs, MME first in device
    order -- WASAPI's entry must win even though it comes later."""
    devices = [
        {"name": "Microphone (Brio)", "max_input_channels": 2,
         "max_output_channels": 0, "hostapi": 0},
        {"name": "Microphone (Brio)", "max_input_channels": 2,
         "max_output_channels": 0, "hostapi": 1},
    ]
    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    sd = FakeSoundDeviceWithHostapis(devices, hostapis)
    assert resolve_device_by_name("Brio", "input", sd=sd) == 1


def test_resolve_device_by_name_none_prefers_wasapi_default_device():
    devices = [{"name": "x", "max_input_channels": 1, "max_output_channels": 0}]
    hostapis = [
        {"name": "MME", "default_input_device": 0, "default_output_device": 0},
        {"name": "Windows WASAPI", "default_input_device": 5, "default_output_device": 6},
    ]
    sd = FakeSoundDeviceWithHostapis(devices, hostapis)
    assert resolve_device_by_name(None, "input", sd=sd) == 5
    assert resolve_device_by_name(None, "output", sd=sd) == 6


def test_resolve_device_by_name_falls_back_when_no_wasapi_match_among_candidates():
    """Named match exists only under a non-WASAPI host API -- still
    return it; never raise just because WASAPI lacks this device."""
    devices = [{"name": "Weird Legacy Mic", "max_input_channels": 1,
                "max_output_channels": 0, "hostapi": 0}]
    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    sd = FakeSoundDeviceWithHostapis(devices, hostapis)
    assert resolve_device_by_name("Legacy", "input", sd=sd) == 0


def test_load_audio_config_missing_file_returns_documented_defaults(tmp_path):
    profiles_root = tmp_path / "profiles"
    (profiles_root / "no-audio-json-yet").mkdir(parents=True)
    config = load_audio_config("no-audio-json-yet", root=str(profiles_root))
    assert config == DEFAULT_AUDIO_CONFIG
    assert config is not DEFAULT_AUDIO_CONFIG  # must be a copy, not the module dict


def test_load_audio_config_merges_partial_file_over_defaults(tmp_path):
    profiles_root = tmp_path / "profiles"
    prof_dir = profiles_root / "partial"
    prof_dir.mkdir(parents=True)
    write_audio_config("partial", {"input_device": "USB Audio Device",
                                    "vad_threshold": 0.7}, root=str(profiles_root))

    config = load_audio_config("partial", root=str(profiles_root))
    assert config["input_device"] == "USB Audio Device"
    assert config["vad_threshold"] == 0.7
    # untouched keys still fall back to the documented defaults
    assert config["wake_threshold"] == DEFAULT_AUDIO_CONFIG["wake_threshold"]
    assert config["output_volume"] == DEFAULT_AUDIO_CONFIG["output_volume"]


def test_load_audio_config_sim_profile_has_no_audio_json_yet():
    """profiles/sim has no audio.json committed (calibrate.py's audio step
    -- SS3.5 #7 -- writes the real one); until then, sim must still load
    cleanly with defaults, since every conversation/ class has to be
    constructible before calibration exists."""
    config = load_audio_config("sim")
    assert config == DEFAULT_AUDIO_CONFIG


def test_wav_round_trip(tmp_path):
    samples = (np.sin(np.linspace(0, 20, 4000)) * 10000).astype(np.int16)
    path = tmp_path / "roundtrip.wav"
    write_wav_int16(str(path), samples, sample_rate=SAMPLE_RATE)

    read_back, rate = read_wav_int16(str(path))
    assert rate == SAMPLE_RATE
    np.testing.assert_array_equal(read_back, samples)


def test_read_wav_int16_against_committed_fixture():
    samples, rate = read_wav_int16(fixture_path("directed_hello.wav"))
    assert rate == SAMPLE_RATE
    assert samples.dtype == np.int16
    assert len(samples) == int(SAMPLE_RATE * 1.2)  # 0.1 + 0.5 + 0.6s, see generator


def test_read_wav_int16_rejects_stereo(tmp_path):
    import wave
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.zeros(200, dtype=np.int16).tobytes())
    with pytest.raises(ValueError, match="mono"):
        read_wav_int16(str(path))

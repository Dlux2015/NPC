"""Live push-to-talk conversation demo for the developer's own Windows PC
(ORCHESTRATION.md SS4.5: the dev PC is just another shell profile --
`profiles/dev-pc` -- swapped in via CBOT_PROFILE, zero code edits).

    python -m conversation.demo_talk              # live PTT conversation
    python -m conversation.demo_talk --text        # typed chat, no audio at all
    python -m conversation.demo_talk --selfcheck   # non-interactive checks

Pipeline-driving approach
--------------------------
This reuses the REAL conversation.pipeline.ConversationPipeline unmodified
-- persona/history/identity/window-reset logic all run exactly as they
would on the robot -- by injecting two dev-only adapters that satisfy its
pinned interfaces (see pipeline.py's module docstring):

  * `wake`: a real conversation.wake.WakeTrigger, but with its `ptt_poll`
    injected as a closure over a small PttState flag that a tkinter window
    flips on press/release (see PttState/build_ptt_window below), and its
    `mic_source` stubbed with _NullMicSource so it never opens a second,
    competing sounddevice.InputStream just to probe for openWakeWord/
    face+speech (neither of which apply here -- no wake model file, no
    vision process on this profile).

  * `stt`: PttSTT below, NOT conversation.stt.DirectedSTT. DirectedSTT's
    listen_utterance() end-points on Silero VAD trailing silence, which is
    the right call for always-on wake/ambient listening -- but push-to-
    talk's natural end-of-utterance signal is the button's release, not
    silence. PttSTT implements the exact same `listen_utterance(max_s) ->
    str | None` contract pipeline.py calls, just driven by the PttState
    flag instead of a VAD, and reuses the real conversation.audio_dev.
    MicStream + conversation.whisper_model faster-whisper loader for the
    actual capture/transcription.

A GUI window (not the `keyboard` package's global hook) is the PTT
control surface: this script is meant to be launched as a background
process, so it must not depend on console stdin focus, and a global
keyboard hook either silently fails to fire or needs admin rights in a
lot of Windows session configurations. A visible window that the user
clicks (or that a bound spacebar fires while it has focus) is more
reliable and gives visible state feedback (Idle / Listening / Speaking).
`keyboard` is still installed (pre-approved) and importable, but unused
by this script for exactly that reason.

TTS: tries Piper first (profile.yaml's `tts_model_path`, if configured
and present on disk); falls back to conversation.sapi_tts.SapiSynthesizer
(pyttsx3/SAPI5, dev-PC only) otherwise -- printed loudly either way, same
spirit as conversation.llm.make_llm's logging. Playback goes through the
REAL conversation.tts.Speaker + conversation.audio_dev.SpeakerSink (both
already existed as product code; no new sink was needed), so
actively_speaking's publish-before-audio / clear-after-drain lifecycle
runs against a real shared/ipc.py SharedState backed by
run/dev-pc/state.json, exactly like the two-process robot setup.

LLM: conversation.llm.make_llm(profile) -- MockLLM unless CBOT_LLM_MODEL
(or profile.yaml's llm_model_path) points at a real GGUF; loudly printed.
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation.audio_dev import (
    SAMPLE_RATE,
    MicStream,
    SpeakerSink,
    load_audio_config,
    resolve_device_by_name,
)
from conversation.llm import MockLLM, make_llm
from conversation.pipeline import ConversationPipeline
from conversation.tts import Speaker
from conversation.wake import WakeTrigger
from shared.ipc import SharedState
from shared.people import PeopleStore
from vision.paths import load_profile_yaml, profile_dir, repo_root

PROFILE_NAME = os.environ.get("CBOT_PROFILE", "dev-pc")
RUN_DIR = os.path.join(repo_root(), "run", PROFILE_NAME)

# Device names containing these substrings are almost certainly not
# physical hardware -- flag them so a silent/deaf demo run doesn't look
# like a bug when it's really "Windows' default input is a virtual cable".
_VIRTUAL_DEVICE_HINTS = ("cable", "virtual")


# --- device introspection / diagnostics -------------------------------------

def _import_sd():
    import sounddevice as sd
    return sd


def resolve_effective_device_name(kind, audio_config):
    """Returns (name, source_description) for the device that will
    actually be used for `kind` ("input"/"output") -- routed through the
    exact same resolve_device_by_name() that MicStream/SpeakerSink use
    (including its WASAPI-over-MME preference), so what this prints always
    matches what actually gets opened."""
    sd = _import_sd()
    configured = audio_config.get("%s_device" % kind)
    idx = resolve_device_by_name(configured, kind, sd=sd)
    if idx is None:
        return None, "no OS default available"
    source = "configured name %r" % configured if configured else "OS default (WASAPI preferred)"
    return sd.query_devices()[idx]["name"], source


def warn_if_virtual(name, kind):
    if name and any(tok in name.lower() for tok in _VIRTUAL_DEVICE_HINTS):
        print(
            "WARNING: OS-default %s device %r looks like a virtual/loopback "
            "device, not physical hardware. If you don't hear/produce real "
            "audio, open Windows Sound Settings and change the default %s "
            "device, or set profiles/%s/audio.json's %s_device to a "
            "substring of your real device's name."
            % (kind, name, kind, PROFILE_NAME, kind)
        )


def print_devices(audio_config):
    in_name, in_src = resolve_effective_device_name("input", audio_config)
    out_name, out_src = resolve_effective_device_name("output", audio_config)
    print("Mic (input):     %s  [%s]" % (in_name, in_src))
    print("Speaker (output): %s  [%s]" % (out_name, out_src))
    warn_if_virtual(in_name, "input")
    warn_if_virtual(out_name, "output")
    return in_name, out_name


# --- TTS backend selection ---------------------------------------------------

def make_tts_synthesizer(profile_yaml):
    """Try Piper first (profile.yaml's tts_model_path, if configured and
    present on disk); else fall back to the dev-PC-only SAPI synthesizer.
    Returns (synthesizer, backend_description) -- caller prints the
    description loudly, never silently."""
    model_path = profile_yaml.get("tts_model_path")
    if model_path:
        if not os.path.isabs(model_path):
            model_path = os.path.join(profile_dir(PROFILE_NAME), model_path)
        if os.path.isfile(model_path):
            try:
                from conversation.tts import PiperSynthesizer
                synth = PiperSynthesizer(model_path)
                return synth, "Piper (%s)" % model_path
            except RuntimeError as exc:
                print("Piper unavailable (%s) -- falling back to SAPI." % exc)
        else:
            print(
                "Configured tts_model_path %r not found on disk -- "
                "falling back to SAPI." % model_path
            )
    else:
        try:
            import piper  # noqa: F401
        except ImportError:
            print(
                "piper-tts not installed (or no Windows wheel available) "
                "-- falling back to SAPI."
            )
        else:
            print(
                "piper-tts is installed but profile.yaml has no "
                "tts_model_path configured -- falling back to SAPI."
            )

    from conversation.sapi_tts import SapiSynthesizer
    synth = SapiSynthesizer()
    return synth, (
        "SAPI via pyttsx3 (dev-PC-only fallback; production shells must "
        "use a version-pinned Piper voice)"
    )


# --- push-to-talk state + GUI ------------------------------------------------

class PttState:
    """Shared press/release flag: tkinter callbacks (main thread) write
    it, WakeTrigger's injected ptt_poll + PttSTT (pipeline thread) read
    it. A plain bool attribute is fine here -- CPython bool assignment is
    atomic under the GIL and there is exactly one writer."""

    def __init__(self):
        self.pressed = False


class _NullMicSource:
    """Stand-in mic_source for WakeTrigger: always "silence", so
    WakeTrigger never opens a second, competing sounddevice.InputStream
    just to probe for openWakeWord/face+speech, neither of which this
    profile uses (no wake model file, no vision process on dev-pc)."""

    def read(self, n_samples):
        return np.zeros(n_samples, dtype=np.int16)


def build_ptt_window(ptt_state, status_queue, on_close):
    import tkinter as tk

    root = tk.Tk()
    root.title("CBot dev-pc -- push to talk")
    root.geometry("440x240")

    tk.Label(
        root,
        text="Hold SPACE (window focused) or click-and-hold the button to talk.",
        wraplength=400, justify="center",
    ).pack(pady=(14, 6))

    status_var = tk.StringVar(value="Idle. Press and hold to talk.")
    tk.Label(root, textvariable=status_var, fg="#1a5", wraplength=400,
              justify="center", font=("Segoe UI", 10)).pack(pady=6)

    button = tk.Button(
        root, text="HOLD TO TALK", font=("Segoe UI", 14, "bold"),
        bg="#c33", fg="white", width=18, height=3, relief="raised",
    )
    button.pack(pady=10)

    def press(_event=None):
        ptt_state.pressed = True
        button.configure(bg="#2a2")

    def release(_event=None):
        ptt_state.pressed = False
        button.configure(bg="#c33")

    button.bind("<ButtonPress-1>", press)
    button.bind("<ButtonRelease-1>", release)
    root.bind("<KeyPress-space>", press)
    root.bind("<KeyRelease-space>", release)
    root.focus_force()

    def poll_status():
        try:
            while True:
                status_var.set(status_queue.get_nowait())
        except queue.Empty:
            pass
        root.after(100, poll_status)

    root.after(100, poll_status)

    def _on_close():
        on_close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    return root


# --- push-to-talk-driven STT adapter -----------------------------------------

class PttSTT:
    """listen_utterance(max_s) -> str | None, matching the exact contract
    conversation/pipeline.py calls on `self.stt` -- but end-pointed on
    button release instead of conversation.vad.SileroVAD trailing
    silence (see module docstring for why). Uses the real
    conversation.audio_dev.MicStream for capture and the real
    conversation.whisper_model faster-whisper adapter for transcription;
    only the end-pointing signal is swapped out.
    """

    def __init__(self, mic, model, ptt_state, sample_rate=SAMPLE_RATE,
                 frame_ms=30, min_speech_s=0.2, status_cb=None,
                 sleep_fn=time.sleep):
        self._mic = mic
        self._model = model
        self._ptt = ptt_state
        self._sample_rate = sample_rate
        self._frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        self._min_speech_s = min_speech_s
        self._status_cb = status_cb or (lambda msg: None)
        self._sleep = sleep_fn

    def listen_utterance(self, max_s=10.0):
        # The button may already be up again by the time we get here on a
        # later turn in the same conversation (wake.wait() only fired
        # once, on the *first* press) -- wait for the next press, but
        # time out exactly like DirectedSTT's "nothing said" case so
        # ConversationPipeline.run_once() ends the session the same way.
        waited = 0.0
        poll_dt = 0.02
        while not self._ptt.pressed:
            self._sleep(poll_dt)
            waited += poll_dt
            if waited >= max_s:
                return None

        self._status_cb("Listening... (release to finish)")
        frames = []
        elapsed_s = 0.0
        while elapsed_s < max_s and self._ptt.pressed:
            frame = self._mic.read(self._frame_samples)
            if frame is None or len(frame) == 0:
                continue
            frames.append(frame)
            elapsed_s += len(frame) / float(self._sample_rate)

        if not frames or elapsed_s < self._min_speech_s:
            self._status_cb("(too short, ignored) Press and hold to talk.")
            return None

        self._status_cb("Transcribing...")
        audio = np.concatenate(frames)
        text = (self._model.transcribe(audio, sample_rate=self._sample_rate) or "").strip()
        if text:
            self._status_cb('Heard: "%s"' % text)
        else:
            self._status_cb("(heard nothing) Press and hold to talk.")
        return text or None


# --- text-only mode (no audio at all) ----------------------------------------
#
# Same philosophy as PTT mode: drive the REAL ConversationPipeline
# (persona/history/identity/window-reset all unmodified) and swap only the
# transport adapters at its pinned interfaces. Here the "mic" is the
# console and the "speaker" is stdout, so none of sounddevice / faster-
# whisper / TTS is even imported -- useful for exercising the local LLM
# on its own.

class ConsoleWake:
    """wake.wait(timeout_s) -> event | None. Fires exactly once: the user
    sitting at the console *is* the wake event, and after the conversation
    ends (they typed quit / EOF) run_once() returns and the demo exits
    instead of looping on a fresh wake."""

    def __init__(self):
        self._fired = False

    def wait(self, timeout_s=None):
        if self._fired:
            return None
        self._fired = True
        return "text"


class TypedSTT:
    """listen_utterance(max_s) -> str | None, the exact contract
    pipeline.py calls on `self.stt` -- but the "utterance" is a typed
    console line. Returns None (pipeline: end of conversation) on
    quit/exit, EOF, or Ctrl-C; re-prompts on empty input. max_s is
    ignored: a human at a keyboard has no trailing-silence endpoint."""

    QUIT_WORDS = ("quit", "exit", "/quit", "/exit")

    def __init__(self, prompt="You: ", input_fn=input):
        self._prompt = prompt
        self._input = input_fn

    def listen_utterance(self, max_s=None):
        while True:
            try:
                text = self._input(self._prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if text.lower() in self.QUIT_WORDS:
                return None
            if text:
                return text


class ConsoleSpeaker:
    """say(text) / say_stream(sentence_iter) -> stdout. Prints each
    sentence as it arrives from the LLM stream (one line per sentence),
    so the sentence-streaming behavior the robot's TTS relies on stays
    visible in text mode."""

    def say(self, text):
        print("CBot: %s" % text, flush=True)

    def say_stream(self, sentence_iter):
        first = True
        for sentence in sentence_iter:
            print(("CBot: " if first else "      ") + sentence, flush=True)
            first = False


def run_text():
    ensure_run_dir()
    profile_yaml = load_profile_yaml(PROFILE_NAME)

    print("== CBot dev-pc text conversation demo (no audio) ==")
    print("Profile: %s (%s)"
          % (PROFILE_NAME, os.path.join(profile_dir(PROFILE_NAME), "profile.yaml")))

    state = SharedState(os.path.join(RUN_DIR, "state.json"))
    people = PeopleStore(os.path.join(RUN_DIR, "people.db"))

    llm = make_llm(profile_yaml)
    if isinstance(llm, MockLLM):
        print(
            "LLM backend: MockLLM -- replies are scripted. For the real "
            "model run this with the repo venv's interpreter "
            "(.venv\\Scripts\\python.exe) so llama-cpp-python is available."
        )
    else:
        print("LLM backend: LocalLLM (%s)" % llm.model_path)

    pipeline = ConversationPipeline(
        PROFILE_NAME, state, ConsoleWake(), TypedSTT(), llm, ConsoleSpeaker(),
        people,
    )

    print()
    print("Type to talk; 'quit' (or Ctrl-C) to leave.")
    pipeline.run_once()
    print("Conversation over.")
    return 0


# --- wiring -------------------------------------------------------------

def ensure_run_dir():
    os.makedirs(RUN_DIR, exist_ok=True)
    return RUN_DIR


def build_pipeline(status_cb, stt_model_size="base"):
    ensure_run_dir()
    profile_yaml = load_profile_yaml(PROFILE_NAME)
    audio_config = load_audio_config(PROFILE_NAME)

    print("== CBot dev-pc live conversation demo ==")
    print("Profile: %s (%s)"
          % (PROFILE_NAME, os.path.join(profile_dir(PROFILE_NAME), "profile.yaml")))
    print_devices(audio_config)

    state_path = os.path.join(RUN_DIR, "state.json")
    people_path = os.path.join(RUN_DIR, "people.db")
    state = SharedState(state_path)
    people = PeopleStore(people_path)
    print("Shared state:  %s" % state_path)
    print("People store:  %s" % people_path)

    print("Loading faster-whisper %r (int8, CPU)... first run downloads "
          "weights, that's expected." % stt_model_size)
    from conversation.whisper_model import load_faster_whisper
    whisper_model = load_faster_whisper(stt_model_size, device="cpu", compute_type="int8")
    print("faster-whisper %r ready." % stt_model_size)

    mic = MicStream(audio_config)

    llm = make_llm(profile_yaml)
    if isinstance(llm, MockLLM):
        print(
            "LLM backend: MockLLM -- replies are scripted until a GGUF is "
            "configured via CBOT_LLM_MODEL (or profile.yaml's "
            "llm_model_path)."
        )
    else:
        print("LLM backend: LocalLLM (%s)" % llm.model_path)

    synthesizer, tts_backend = make_tts_synthesizer(profile_yaml)
    print("TTS backend: %s" % tts_backend)
    sink = SpeakerSink(audio_config)
    speaker = Speaker(profile_yaml, state, synthesizer=synthesizer, sink=sink,
                       audio_config=audio_config)

    ptt_state = PttState()
    wake = WakeTrigger(
        profile_yaml, state, mic_source=_NullMicSource(),
        ptt_poll=lambda: ptt_state.pressed, audio_config=audio_config,
    )
    stt = PttSTT(mic, whisper_model, ptt_state,
                 sample_rate=audio_config.get("sample_rate", SAMPLE_RATE),
                 status_cb=status_cb)

    pipeline = ConversationPipeline(PROFILE_NAME, state, wake, stt, llm, speaker, people)
    return pipeline, ptt_state


def run_live(stt_model_size="base"):
    status_queue = queue.Queue()

    def status_cb(msg):
        status_queue.put(msg)
        print(msg)

    pipeline, ptt_state = build_pipeline(status_cb, stt_model_size=stt_model_size)

    print()
    print("Ready. Hold SPACE (with the window focused) or click-and-hold ")
    print("the on-screen button to talk; release the button/key to end ")
    print("your turn (or after 10s, whichever comes first).")
    print("Close the window to quit.")

    def worker():
        try:
            pipeline.run_forever()
        except Exception:
            import traceback
            traceback.print_exc()

    t = threading.Thread(target=worker, name="cbot-pipeline", daemon=True)
    t.start()

    root = build_ptt_window(ptt_state, status_queue, on_close=pipeline.stop)
    try:
        root.mainloop()
    finally:
        pipeline.stop()
        t.join(timeout=3.0)
    return 0


# --- non-interactive verification -------------------------------------------

def run_selfcheck():
    ensure_run_dir()
    profile_yaml = load_profile_yaml(PROFILE_NAME)
    audio_config = load_audio_config(PROFILE_NAME)

    print("=== (a) audio devices ===")
    sd = _import_sd()
    idx_in, idx_out = sd.default.device
    for i, d in enumerate(sd.query_devices()):
        marker = ""
        if i == idx_in:
            marker += " [default input]"
        if i == idx_out:
            marker += " [default output]"
        print("%3d in=%d out=%d  %s%s"
              % (i, d["max_input_channels"], d["max_output_channels"], d["name"], marker))
    print()
    in_name, out_name = print_devices(audio_config)

    print()
    print('=== (b) TTS test phrase "CBot audio test, one two three" ===')
    state = SharedState(os.path.join(RUN_DIR, "state.json"))
    synthesizer, tts_backend = make_tts_synthesizer(profile_yaml)
    print("TTS backend: %s" % tts_backend)
    sink = SpeakerSink(audio_config)
    speaker = Speaker(profile_yaml, state, synthesizer=synthesizer, sink=sink,
                       audio_config=audio_config)
    print("actively_speaking before: %s" % state.get("actively_speaking"))
    speaker.say("CBot audio test, one two three.")
    print("actively_speaking after:  %s" % state.get("actively_speaking"))
    print("Played through: %s -- you should have heard it." % out_name)

    print()
    print("=== (c) 1.0s mic capture: peak/RMS ===")
    mic = MicStream(audio_config)
    n = int(audio_config.get("sample_rate", SAMPLE_RATE) * 1.0)
    frame = mic.read(n)
    mic.close()
    if len(frame):
        peak = int(np.max(np.abs(frame)))
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    else:
        peak, rms = 0, 0.0
    print("captured %d samples from %s" % (len(frame), in_name))
    print("peak=%d rms=%.1f (int16 full scale=32767)" % (peak, rms))

    print()
    print("=== (d) transcribe conversation/tests/fixtures/directed_hello.wav ===")
    fixture = os.path.join(repo_root(), "conversation", "tests", "fixtures",
                            "directed_hello.wav")
    print("Loading faster-whisper 'base' (int8, CPU)...")
    from conversation.stt import DirectedSTT
    from conversation.whisper_model import load_faster_whisper
    whisper_model = load_faster_whisper("base", device="cpu", compute_type="int8")
    stt = DirectedSTT(profile_yaml, model=whisper_model, audio_config=audio_config)
    text = stt.transcribe_wav(fixture)
    print("fixture: %s" % fixture)
    print("transcribed text: %r" % text)

    print()
    print("Selfcheck complete.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="Run non-interactive verification (devices, TTS, mic, STT) and exit -- no GUI.",
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Typed console conversation -- no mic, no TTS, no audio stack at all.",
    )
    parser.add_argument(
        "--stt-model", default="base",
        help="faster-whisper model size for the live loop (default: base).",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        return run_selfcheck()
    if args.text:
        return run_text()
    return run_live(stt_model_size=args.stt_model)


if __name__ == "__main__":
    sys.exit(main() or 0)

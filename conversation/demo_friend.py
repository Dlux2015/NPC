"""Live "friend" demo: webcam face recognition + mic conversation with
CONSENT-GATED enrollment, on the dev PC (profile: dev-pc).

    python -m conversation.demo_friend               # the full experience
    python -m conversation.demo_friend --camera 1    # other webcam index
    python -m conversation.demo_friend --selfcheck   # headless checks, no mic/LLM

The experience:
  1. A webcam window opens; YuNet tracks your face (green box), SFace
     embeds it ~1Hz and matches against run/dev-pc/people.db.
  2. You're in range + you start talking -> "face_speech" wake fires
     (walk-up-and-talk; no wake word, no button).
  3. If the robot doesn't recognize you, it ASKS: "Can I be your
     friend?" -- only a yes enrolls your face (auto_enroll=False;
     spec's silent auto-enroll is deliberately disabled here). Say
     "my name is <name>" and it remembers that too (people.set_name via
     the pipeline's existing name capture).
  4. Quit, run it again, walk up: "Welcome back, <name>!" -- recognition
     across sessions is the whole point.
  Privacy: embeddings only (never images), local only; press "p" in the
  window to purge everyone (people.purge()).

Wiring philosophy (same as demo_talk.py): the REAL product objects --
TrackingApp (vision), ConversationPipeline, WakeTrigger, DirectedSTT,
Speaker, PeopleStore, SharedState -- with dev-only adapters only at the
physical edges (webcam wrapper, null serial transport since there are no
servos, and FriendWake, which inserts the consent conversation between a
wake event and the pipeline). The consent flow itself is a candidate to
graduate into the product pipeline at Phase 6 -- see FriendWake's
docstring.

Run with the repo venv's interpreter (llama-cpp-python + torch live
there): A:\\code\\CBot\\.venv\\Scripts\\python.exe -m conversation.demo_friend
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from conversation.audio_dev import SAMPLE_RATE, MicStream, SpeakerSink, load_audio_config
from conversation.llm import MockLLM, make_llm
from conversation.pipeline import ConversationPipeline
from conversation.stt import DirectedSTT
from conversation.tts import Speaker
from conversation.wake import WakeTrigger
from shared.ipc import SharedState
from shared.people import PeopleStore
from vision.paths import load_profile_yaml, profile_dir, repo_root
from vision.recognition import SFACE_MATCH_THRESHOLD, make_embedder
from vision.tracking import TrackingApp, crop_face, crop_face_padded

PROFILE_NAME = os.environ.get("CBOT_PROFILE", "dev-pc")
RUN_DIR = os.path.join(repo_root(), "run", PROFILE_NAME)

# The dev PC has no servos: PID outputs go to a null transport, so the
# pixel->degree numbers below only need to be sane, not measured. A real
# shell gets these from vision/calibrate.py -- never copy these there.
WEBCAM_CALIBRATION = {
    "version": 1,
    "axes": {
        "pan": {"sign": 1, "center": 0.0, "min": -60.0, "max": 60.0},
        "tilt": {"sign": 1, "center": 0.0, "min": -30.0, "max": 30.0},
    },
    "deg_per_px": {"pan": 0.1, "tilt": 0.1},
    "deadband_deg": 0.5,
}

# Desk webcams see faces a bit smaller than the robot's in-your-face
# camera will; slightly laxer than TrackingApp's 0.25 default so sitting
# at a normal distance still counts as "in range" for face_speech wake.
IN_RANGE_FRAC = 0.18

AFFIRMATIVE_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "absolutely",
    "definitely", "please", "course",  # "of course"
}


def is_affirmative(text):
    if not text:
        return False
    words = {w.strip(".,!?'\"").lower() for w in text.split()}
    return bool(words & AFFIRMATIVE_WORDS)


class _BufferedLLM:
    """generate_stream that drains the model FULLY before yielding.

    Why: in this demo the LLM (llama.cpp, its own bundled CUDA runtime)
    and Kokoro TTS (torch CUDA) share one GPU in one process, and the
    pipeline's sentence-streaming deliberately overlaps them (speak
    sentence 1 while generating sentence 2). Two live runs died in
    native access violations (exit code 0xC0000005, reported as 5)
    mid-reply -- concurrent use of two independent CUDA runtimes is the
    prime suspect. Serializing them costs <1s at ~110 tok/s.
    The robot proper doesn't have this problem: its TTS (Piper) is CPU,
    and the GPU belongs to the LLM alone (SS3.4)."""

    def __init__(self, inner):
        self._inner = inner

    def generate_stream(self, messages):
        return iter(list(self._inner.generate_stream(messages)))


class _NullTransport:
    """No servos on the dev PC; TrackingApp still writes target lines."""

    def write_line(self, line):
        pass

    def read_lines(self):
        return []

    def close(self):
        pass


class EchoGuardSTT:
    """DirectedSTT wrapper that drains the mic's buffered backlog before
    each listen. While the robot speaks, nobody reads the mic, so the
    input stream buffers audio -- including the tail of the robot's OWN
    voice; without this, listen_utterance() can end up transcribing the
    robot. Drained frames are detected by wall-clock: buffered frames
    return near-instantly, live frames take ~frame duration to arrive.
    (The real robot solves self-hearing at calibration step 7; this is
    the dev-PC equivalent, kept demo-side on purpose.)

    `listen_cue` (optional callable) fires after the drain, right before
    real listening starts -- wired to a short beep so the user KNOWS
    when to talk. Without it, anything said during the robot's own
    speech or the drain is silently discarded and the user can't tell
    (exactly how the first two live runs lost the consent "yes")."""

    def __init__(self, stt, mic, sample_rate=SAMPLE_RATE, frame_ms=32,
                 listen_cue=None):
        self._stt = stt
        self._mic = mic
        self._frame_samples = int(sample_rate * frame_ms / 1000)
        self._frame_s = frame_ms / 1000.0
        self._listen_cue = listen_cue

    def _drain(self, max_frames=240):
        for _ in range(max_frames):
            t0 = time.perf_counter()
            self._mic.read(self._frame_samples)
            if time.perf_counter() - t0 > self._frame_s * 0.5:
                return  # took ~real time -> live audio now, backlog gone

    def listen_utterance(self, max_s=10.0):
        self._drain()
        if self._listen_cue is not None:
            self._listen_cue()
        print("[friend] listening...", flush=True)
        text = self._stt.listen_utterance(max_s=max_s)
        print("[friend] heard: %r" % text, flush=True)
        return text


def make_listen_beep(sink, sample_rate=SAMPLE_RATE):
    """Short soft two-tone chirp -> 'your turn to talk'. Tonal, not
    speech-like, so Silero VAD won't mistake its echo for an utterance."""
    def tone(freq, dur_s):
        t = np.arange(int(sample_rate * dur_s)) / sample_rate
        w = np.sin(2 * np.pi * freq * t)
        fade = min(len(w) // 4, int(sample_rate * 0.01))
        if fade:
            w[:fade] *= np.linspace(0, 1, fade)
            w[-fade:] *= np.linspace(1, 0, fade)
        return w
    chirp = np.concatenate([tone(660, 0.07), tone(990, 0.09)])
    samples = (0.3 * 32767 * chirp).astype(np.int16)

    def play():
        sink.play(samples, sample_rate=sample_rate)
    return play


class FriendWake:
    """WakeTrigger wrapper that runs the consent-to-enroll conversation
    between the wake event and the pipeline taking over.

    On wake, if vision has NOT matched the person (person_id is None)
    but has stashed an unknown embedding (TrackingApp auto_enroll=False),
    the robot asks to be their friend:
      - yes -> enroll the stashed embedding, publish person_id +
        new_person_seq bump; the pipeline then greets them as new and its
        existing name capture ("my name is <name>") attaches the name.
      - no/silence -> nothing is stored; don't ask again until they
        leave (person_present False resets the declined flag).

    Phase 6 note: if this consent behavior graduates to the robot, it
    belongs in ConversationPipeline (contract change: auto_enroll off in
    the vision process + an enroll handshake over IPC, since pipeline
    and tracking run in separate processes there -- orchestrator owns
    that). Here vision and conversation share one process, so the
    TrackingApp reference is honest and simple.
    """

    def __init__(self, inner, tracking_app, state, people, speaker, stt,
                 persona_name="your robot friend"):
        self._inner = inner
        self._app = tracking_app
        self._state = state
        self._people = people
        self._speaker = speaker
        self._stt = stt
        self._persona_name = persona_name
        self._declined = False

    def wait(self, timeout_s=None):
        if not self._state.get("person_present"):
            self._declined = False  # new visitor -> consent question again
        event = self._inner.wait(timeout_s)
        if event is None:
            return None
        if self._state.get("person_id") is None and not self._declined:
            self._consent_flow()
        return event

    def _consent_flow(self):
        embedding = self._app.pop_unknown_embedding()
        if embedding is None:
            return  # recognition hasn't seen a stable face yet -- chat anonymously
        # Deliberately short: people answer WHILE the robot is still
        # talking, and anything said before listen_utterance() starts is
        # discarded with the echo-guard's mic drain (first live run
        # 2026-07-06 lost the user's "yes" exactly this way). The privacy
        # detail moved to the yes-branch, after the decision.
        self._speaker.say(
            "Hi! I don't think we've met. Can I be your friend?"
        )
        reply = self._stt.listen_utterance(max_s=10.0)
        print('[friend] consent reply: %r' % reply)
        if is_affirmative(reply):
            pid = self._people.enroll(embedding)
            seq = self._state.get("new_person_seq") + 1
            self._state.update(person_id=str(pid), new_person_seq=seq)
            self._speaker.say(
                "Great! I'll remember your face -- never a picture. "
                "To tell me your name, say: my name is, then your name."
            )
        else:
            self._declined = True
            if reply:
                self._speaker.say(
                    "No worries, I won't remember your face. "
                    "Happy to chat anyway!"
                )


# --- assembly ---------------------------------------------------------------

def _persona_display_name(profile_yaml):
    """First word of the persona's title line if available, else a
    generic name -- persona text itself stays profile-owned (SS3.4)."""
    try:
        path = os.path.join(profile_dir(PROFILE_NAME),
                             profile_yaml.get("persona", "persona.md"))
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
        name = first.replace("#", "").replace("Persona:", "").strip()
        return name.split()[0] if name else "your robot friend"
    except OSError:
        return "your robot friend"


def build_vision(camera_index, state, people):
    from sim.demo_visual import _Webcam, _pick_real_detector

    camera = _Webcam(camera_index)
    detector = _pick_real_detector()
    embedder = make_embedder()
    if embedder is None:
        print("WARNING: SFace model missing -- recognition inert, consent "
              "flow will never trigger (download: see vision/recognition.py).")
    app = TrackingApp(
        camera, detector, _NullTransport(), state, WEBCAM_CALIBRATION,
        in_range_frac=IN_RANGE_FRAC,
        face_crop_cb=crop_face_padded if embedder else crop_face,
        people_store=people, embed_cb=embedder,
        match_threshold=SFACE_MATCH_THRESHOLD if embedder else None,
        auto_enroll=False,
    )
    return app, camera, detector, embedder


def build_conversation(state, people, tracking_app, stt_model_size="base"):
    profile_yaml = load_profile_yaml(PROFILE_NAME)
    audio_config = load_audio_config(PROFILE_NAME)

    print("Loading faster-whisper %r (int8, CPU)..." % stt_model_size)
    from conversation.whisper_model import load_faster_whisper
    whisper_model = load_faster_whisper(stt_model_size, device="cpu",
                                         compute_type="int8")

    mic = MicStream(audio_config)

    llm = make_llm(profile_yaml)
    print("LLM backend: %s"
          % ("MockLLM (scripted -- run under .venv for the real model)"
             if isinstance(llm, MockLLM) else "LocalLLM"))
    llm = _BufferedLLM(llm)  # serialize GPU: LLM finishes before TTS starts

    from conversation.demo_talk import make_tts_synthesizer
    synthesizer, tts_backend = make_tts_synthesizer(profile_yaml)
    print("TTS backend: %s" % tts_backend)
    sink = SpeakerSink(audio_config)
    speaker = Speaker(profile_yaml, state, synthesizer=synthesizer,
                       sink=sink, audio_config=audio_config)

    # One shared MicStream: wake's VAD polls it, then STT listens on it --
    # strictly sequential (same pipeline thread), never concurrent.
    inner_wake = WakeTrigger(profile_yaml, state, mic_source=mic,
                              ptt_poll=lambda: False, audio_config=audio_config)
    # min_speech_s below DirectedSTT's 0.2 default: one-word answers
    # ("yes") can be under 0.2s of VAD-positive frames and were being
    # discarded as blips in live runs.
    stt = EchoGuardSTT(
        DirectedSTT(profile_yaml, mic_source=mic, model=whisper_model,
                     audio_config=audio_config, min_speech_s=0.1),
        mic,
        listen_cue=make_listen_beep(sink),
    )
    wake = FriendWake(inner_wake, tracking_app, state, people, speaker, stt,
                       persona_name=_persona_display_name(profile_yaml))

    pipeline = ConversationPipeline(PROFILE_NAME, state, wake, stt, llm,
                                     speaker, people)
    return pipeline


def _draw_hud(cv2, frame, status, state, people):
    if status:
        for det in status.get("detections") or []:
            x, y, w, h = (int(v) for v in det[:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)
        target = status.get("target")
        if target is not None:
            x, y, w, h = (int(v) for v in target[:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            pid = state.get("person_id")
            label = "unknown (will ask to be friends)"
            if pid is not None:
                rec = people.get(pid)
                name = rec.get("name") if rec else None
                label = name or ("friend #%s" % pid)
            cv2.putText(frame, label, (x, max(14, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    lines = [
        "talk to start (in range + speech)",
        "speaking..." if state.get("actively_speaking") else
        ("in conversation" if state.get("conversation_active") else "idle"),
        "q quit | p purge remembered faces",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 20 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def run_live(camera_index, stt_model_size="base"):
    import cv2
    import logging

    # Pipeline/wake/llm modules log turn-by-turn detail (wake events,
    # utterance timeouts, name capture) -- surface it for live debugging.
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    os.makedirs(RUN_DIR, exist_ok=True)
    state = SharedState(os.path.join(RUN_DIR, "state.json"))
    # Stale identity from a previous run must not skip the consent flow
    # or mis-greet: start every session unseen.
    state.update(person_present=False, person_in_range=False,
                 person_id=None, conversation_active=False,
                 actively_speaking=False)
    people = PeopleStore(os.path.join(RUN_DIR, "people.db"))
    print("== CBot friend demo (profile: %s) ==" % PROFILE_NAME)
    print("People remembered so far: %d" % people.count())

    app, camera, detector, embedder = build_vision(camera_index, state, people)
    print("Detector: %s | Recognition: %s"
          % (getattr(detector, "name", "yunet"),
             "SFace" if embedder else "OFF"))

    pipeline = build_conversation(state, people, app,
                                   stt_model_size=stt_model_size)

    t = threading.Thread(target=pipeline.run_forever,
                          name="cbot-conversation", daemon=True)
    t.start()
    print()
    print("Window open. Get in frame, then just start talking.")

    try:
        while True:
            status = app.step()
            frame = camera.last_frame
            if frame is not None:
                hud = _draw_hud(cv2, frame.copy(), status, state, people)
                cv2.imshow("CBot friend demo", hud)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                people.purge()
                state.update(person_id=None)
                print("Purged all remembered faces.")
    finally:
        pipeline.stop()
        app.close()
        camera.release()
        cv2.destroyAllWindows()
        t.join(timeout=3.0)
    return 0


def run_selfcheck(camera_index):
    """Headless: vision half only (no mic/LLM/TTS) -- proves webcam,
    detector, embedder, consent stash, and IPC publishing all work."""
    import tempfile

    run_dir = tempfile.mkdtemp(prefix="cbot_friend_check_")
    state = SharedState(os.path.join(run_dir, "state.json"))
    people = PeopleStore(os.path.join(run_dir, "people.db"))

    app, camera, detector, embedder = build_vision(camera_index, state, people)
    print("detector: %s" % getattr(detector, "name", "yunet"))
    print("embedder: %s" % ("SFace (%s)" % embedder.model_path if embedder else "MISSING"))

    faces_seen = 0
    t0 = time.time()
    while time.time() - t0 < 4.0:
        status = app.step()
        if status and status.get("target") is not None:
            faces_seen += 1
        time.sleep(0.03)
    app._wait_recognition_idle()
    stashed = app.pop_unknown_embedding()

    app.close()
    camera.release()

    print("frames with a locked face: %d" % faces_seen)
    print("person_present (IPC): %s" % state.get("person_present"))
    print("unknown embedding stashed: %s"
          % ("yes (dim=%d)" % stashed.size if stashed is not None else "no"))
    if faces_seen and embedder and stashed is None:
        print("note: face seen but nothing stashed -- either you're already")
        print("enrolled in this throwaway DB (impossible) or recognition")
        print("hasn't ticked yet; try sitting still for the 4s window.")
    print("Selfcheck complete.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0,
                         help="webcam index (default 0)")
    parser.add_argument("--stt-model", default="base",
                         help="faster-whisper size (default: base)")
    parser.add_argument("--selfcheck", action="store_true",
                         help="headless vision-half checks; no mic/LLM/TTS")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return run_selfcheck(args.camera)
    return run_live(args.camera, stt_model_size=args.stt_model)


if __name__ == "__main__":
    sys.exit(main() or 0)

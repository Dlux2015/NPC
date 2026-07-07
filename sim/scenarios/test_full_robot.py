"""Whole-robot end-to-end proof (ORCHESTRATION.md SS3.6): the REAL
vision.tracking.TrackingApp and the REAL conversation.pipeline.
ConversationPipeline, running concurrently against the digital twin,
sharing one real shared/ipc.py SharedState *file* and one real
shared/people.py PeopleStore *file* -- exactly like the two real
processes would on-device, just interleaved in one Python process for
deterministic sim-time stepping instead of real threads/wall-clock.

Product code (vision/, conversation/, shared/, firmware/) runs
UNMODIFIED. Fakes/hooks are only at the documented hardware seams:
  - camera/detector: SimCamera + SyntheticDetector on world ground truth
    (the same swap point test_product_loop.py uses).
  - serial transport: in-process ServoSim adapter (ditto).
  - TTS synthesizer + audio sink: conversation/tests/_audio_fakes.py's
    FakeSynthesizer + conversation.tts.NullAudioSink (both are already
    the product's documented injection points -- Speaker's constructor
    takes synthesizer=/sink=).
  - LLM: conversation.llm.MockLLM (the product's own stand-in for
    LocalLLM, used by tests and by any profile with no model configured).
  - wake / STT: small scripted fakes mirroring conversation/tests/
    conftest.py's FakeWake/FakeSTT shape (pipeline.py has no import-time
    dependency on a concrete wake/STT implementation -- these are exactly
    the pinned wait()/listen_utterance() interfaces it calls).

ONE MISSING SEAM, reported rather than silently worked around (see the
module docstring of _fake_embed_face below): vision/tracking.py's
recognition hook has a face_crop_cb constructor param (used as-is here)
but embed_face() itself -- the thing that turns a crop into a vector for
shared/people.py's cosine match -- is a hardcoded module-level function,
not an injected callable, and is a permanent Phase-6 stub that always
returns None (vision/tests/test_tracking.py's
test_recognition_hook_runs_on_worker_thread_and_is_inert_stub pins this
exact behavior). There is no constructor seam to give it a real
implementation. This test monkeypatches vision.tracking.embed_face for
its own duration -- pytest's standard, file-untouched substitution
mechanism, not an edit to vision/tracking.py -- to exercise
enrollment/matching against the REAL shared/people.py cosine path. A
proper fix upstream would add an `embed_cb=embed_face` constructor
parameter to TrackingApp (mirroring the existing face_crop_cb/
people_store pattern) so recognition becomes injectable without
monkeypatching once Phase 6 lands.

A SECOND, more serious finding surfaced only once the above got a real
(non-None) embedding flowing end to end for the first time ever: see
ThreadSafePeopleStore below. This is a genuine PRE-EXISTING product bug
(shared/people.py + vision/tracking.py's threading contract), not
something introduced by the sim -- reported here, not silently patched
in shared/people.py.
"""
import os

import numpy as np
import pytest

from shared import ipc
from shared.people import PeopleStore
from shared.serial_protocol import HEARTBEAT_TIMEOUT_S
from sim.servo_sim import ServoSim
from sim.world import SimCamera, SimWorld, face_color
from vision import calibrate, tracking
from vision.detector import SyntheticDetector
from vision.paths import profile_dir, load_profile_yaml
from vision.tracking import TrackingApp, crop_face

from conversation.llm import MockLLM
from conversation.pipeline import ConversationPipeline
from conversation.persona import load_persona_text
from conversation.tts import NullAudioSink, Speaker
from conversation.tests._audio_fakes import FakeSynthesizer

FRAME_HZ = 30.0
FRAME_DT = 1.0 / FRAME_HZ
PHYSICS_DT = 0.02  # 50Hz, matches firmware/servo_sim's real tick rate

FACE_ID = 0
FACE_SIZE_PX = 140          # >= 0.25 * frame_h(480) bbox-height fraction
CONVERGED_PX = 30.0         # same "settled" bound test_product_loop.py uses
IN_RANGE_FRAC = 0.25
HOLD_S = 0.5                # snappier than the 3.0s product default, still
                             # a normal documented TrackingApp kwarg (no
                             # product-code change)
RECOGNITION_INTERVAL_S = 0.2
EMBED_DIM = 128


def _flush_writer_with_retry(writer, attempts=8, delay_s=0.005):
    """Retry wrapper around ThreadedStateWriter.flush() for Windows'
    os.replace() sharing-violation race (PermissionError when another
    handle briefly holds the destination at the instant of replace).
    HISTORY: this scenario is what originally surfaced the race as a
    product gap; SharedState._write() has since gained its own
    retry-with-backoff (50 x 2ms -- see shared/ipc.py), so product code
    is covered upstream. This wrapper remains as belt-and-braces for the
    test's own tight flush() loops, which hammer the file far harder
    than the coalesced <=10Hz production write cadence ever does."""
    import time
    for attempt in range(attempts):
        try:
            writer.flush()
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)


# ---------------------------------------------------------------------------
# TrackingApp wiring -- identical swap points to test_product_loop.py
# ---------------------------------------------------------------------------

class _ServoSimTransport:
    """write_line()/read_lines() adapter over an in-process ServoSim --
    same contract vision/transport.py's real transports implement (see
    test_product_loop.py's identical helper)."""

    def __init__(self, servo_sim):
        self._sim = servo_sim

    def write_line(self, line):
        self._sim.inject_line(line)

    def read_lines(self):
        return self._sim.read_lines()


def _synthetic_detector(world, servo_sim):
    def _ground_truth(frame_bgr):
        pan = servo_sim.head.pan.current
        tilt = servo_sim.head.tilt.current
        return [(x, y, w, h, 0.95)
                for (_face_id, x, y, w, h) in world.ground_truth(pan, tilt)]
    return SyntheticDetector(_ground_truth)


class _SimClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# ---------------------------------------------------------------------------
# Deterministic fake embedder -- see module docstring for the seam gap.
# ---------------------------------------------------------------------------

def _embedding_for_face_id(face_id):
    """world face_id -> a fixed unit vector (one-hot, padded to 128 dims),
    so shared/people.py's real cosine match (threshold 0.55) sees a clean
    1.0 on a repeat visit and a clean mismatch against any other id."""
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    v[int(face_id) % EMBED_DIM] = 1.0
    return v


def _make_fake_embed_face(known_face_ids):
    """embed_face(face_crop_bgr) -> embedding | None, matching the real
    hook's signature. Recovers *which* world face_id a face_crop_cb crop
    came from via sim/world.py's per-face fill color (crop_face slices
    strictly inside the sprite's painted rectangle, so every pixel --
    including the center one sampled here -- is a pure, unblended fill
    color; see sim/world.py's face_color())."""
    palette = {face_color(fid): fid for fid in known_face_ids}

    def _embed(face_crop_bgr):
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        h, w = face_crop_bgr.shape[:2]
        px = tuple(int(c) for c in face_crop_bgr[h // 2, w // 2])
        face_id = palette.get(px)
        if face_id is None:
            return None
        return _embedding_for_face_id(face_id)

    return _embed


# HISTORY: this test originally needed a ThreadSafePeopleStore workaround
# subclass here -- the sim rig surfaced that shared/people.py's sqlite
# connection was check_same_thread=True while vision/tracking.py uses the
# store from its recognition worker thread (a crash that would have hit
# the real robot at Phase 6). The upstream fix (check_same_thread=False +
# an internal lock in shared/people.py) landed, so the real, unmodified
# PeopleStore is now used everywhere in this test.


# ---------------------------------------------------------------------------
# Scripted conversation-side fakes (mirror conversation/tests/conftest.py's
# FakeWake/FakeSTT shape, but state-driven / real-SharedState-aware since
# they sit against the real ipc.SharedState file here, not a FakeState).
# ---------------------------------------------------------------------------

class RangeGatedWake:
    """wait(timeout_s) -> "face_speech" whenever person_in_range is
    currently True in the (real, file-backed) shared state, else None --
    a deterministic stand-in for the real face-in-range + speech-onset
    directed-mode wake path (SS3.2), with no audio involved."""

    def __init__(self, state):
        self.state = state
        self.calls = 0

    def wait(self, timeout_s):
        self.calls += 1
        if self.state.get("person_in_range"):
            return "face_speech"
        return None


class ScriptedSTT:
    """listen_utterance(max_s) pops the next scripted (utterance, mutate)
    pair, calling mutate() (no args -- callers close over whatever sim
    state they need) right after "hearing" it. Returns None (timeout) once
    the script is exhausted, ending the conversation exactly like a real
    STT timeout would."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def listen_utterance(self, max_s=10.0):
        self.calls += 1
        if not self.script:
            return None
        entry = self.script.pop(0)
        if isinstance(entry, tuple):
            utterance, mutate = entry
        else:
            utterance, mutate = entry, None
        if mutate is not None:
            mutate()
        return utterance


class RecordingState(ipc.SharedState):
    """The REAL SharedState, plus an update_log so the test can assert on
    the conversation_active on/off sequence (mirrors conversation/tests/
    conftest.py's FakeState.update_log, but built on the real class -- not
    a reimplementation -- since this test wants the real file I/O too)."""

    def __init__(self, path):
        super().__init__(path)
        self.update_log = []

    def update(self, **kwargs):
        super().update(**kwargs)
        self.update_log.append(dict(kwargs))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_calibration():
    """A genuinely measured profiles/sim/calibration.json (generated via
    `vision.calibrate --auto` against the real sim world/servo twin if this
    checkout doesn't already have one) -- same fixture shape as
    test_product_loop.py's, so this whole-robot proof also runs against
    real calibration, not a hand-typed fixture."""
    calib_path = os.path.join(profile_dir("sim"), "calibration.json")
    if not os.path.isfile(calib_path):
        calibrate.run("sim", auto=True)
    import json
    with open(calib_path) as f:
        return json.load(f)


class Rig:
    """Owns the digital twin (world/servo/camera/detector/transport) plus
    the REAL TrackingApp, and advances both on a shared sim clock. Frames
    and physics ticks are interleaved (same two-rate structure as
    test_product_loop.py's _run_product_loop), just wrapped so a scenario
    can call step_frames() more than once across a running story instead
    of one fixed-duration pass."""

    def __init__(self, world, calibration, state_path, people_db_path,
                 known_face_ids):
        self.world = world
        self.servo = ServoSim()
        self.camera = SimCamera(world, self.servo)
        self.detector = _synthetic_detector(world, self.servo)
        self.transport = _ServoSimTransport(self.servo)
        self.state = ipc.SharedState(state_path)
        self.people_store = PeopleStore(people_db_path)
        self.clock = _SimClock()
        self.t = 0.0
        self._next_frame = 0.0

        self.app = TrackingApp(
            self.camera, self.detector, self.transport, self.state,
            calibration, clock=self.clock, hold_s=HOLD_S,
            in_range_frac=IN_RANGE_FRAC,
            recognition_interval_s=RECOGNITION_INTERVAL_S,
            face_crop_cb=crop_face, people_store=self.people_store,
            # Deliberately the product's own default (SS3.1: "<=10Hz"),
            # not a tighter test-only value: a tighter interval measurably
            # raised the frequency of the shared/ipc.py Windows race
            # documented on _flush_writer_with_retry below, by giving the
            # writer thread more chances to collide with our flush() (or
            # with the pipeline's own state.update() calls) mid-story.
        )

    def step_frames(self, duration_s, move=None):
        end = self.t + duration_s
        while self.t < end:
            if move is not None:
                move(self.t)
            if self.t >= self._next_frame:
                self.clock.now = self.t
                self.app.step()
                self._next_frame += FRAME_DT
            self.servo.step(PHYSICS_DT)
            self.t += PHYSICS_DT
        # Deterministic visibility: wait out any in-flight recognition
        # worker job, then force the coalescing IPC writer's latest
        # publish() to disk -- instead of racing the writer's own
        # background thread on freshness.
        self.app._wait_recognition_idle(timeout=1.0)
        _flush_writer_with_retry(self.app._writer)

    def pixel_error(self):
        pan, tilt = self.servo.head.pan.current, self.servo.head.tilt.current
        boxes = self.world.ground_truth(pan, tilt)
        if not boxes:
            return None
        _face_id, x, y, w, h = boxes[0]
        ex = (x + w / 2.0) - self.world.frame_w / 2.0
        ey = (y + h / 2.0) - self.world.frame_h / 2.0
        return (ex ** 2 + ey ** 2) ** 0.5

    def close(self):
        self.app.close()


@pytest.fixture
def rig(tmp_path, sim_calibration, monkeypatch):
    world = SimWorld()  # starts empty -- "person enters" adds the face
    state_path = str(tmp_path / "state.json")
    people_db_path = str(tmp_path / "people.db")

    monkeypatch.setattr(tracking, "embed_face", _make_fake_embed_face([FACE_ID]))

    r = Rig(world, sim_calibration, state_path, people_db_path, [FACE_ID])
    try:
        yield r
    finally:
        r.close()


def _make_pipeline(state_path, people_db_path):
    """Builds ONE long-lived REAL ConversationPipeline, exactly as the real
    conversation process would be constructed once at boot -- BEFORE any
    person has ever been seen, so its new_person_seq baseline (SS4.2:
    "only seq increases *after* this point count as 'just met them'")
    starts at 0 like the real process's does. It sees rig's state/people.db
    only through their *file* paths (its own SharedState instance / its
    own PeopleStore sqlite3 connection), like a second real OS process
    would, not by sharing rig's Python objects -- proving the file-backed
    IPC/identity contracts round-trip, not just in-process object sharing.
    Each "session" reuses the same pipeline/wake/speaker/sink; only the
    STT script and LLM replies are swapped between run_once() calls."""
    pstate = RecordingState(state_path)
    ppeople = PeopleStore(people_db_path)
    speaker_profile = load_profile_yaml("sim")
    sink = NullAudioSink(dir=None)
    speaker = Speaker(speaker_profile, pstate,
                       synthesizer=FakeSynthesizer(), sink=sink)
    llm = MockLLM(replies=["Okay!"], chunk_words=3)
    wake = RangeGatedWake(pstate)
    stt = ScriptedSTT([])
    pipeline = ConversationPipeline(
        "sim", pstate, wake, stt, llm, speaker, ppeople,
    )
    return pipeline, pstate, ppeople, sink, llm, stt


def _run_session(pipeline, llm, stt, stt_script, llm_replies):
    """Scripts one conversation session onto the shared pipeline/llm/stt
    objects, then drives it -- mirrors one wake->...->timeout cycle of the
    real process's run_forever() loop."""
    stt.script = list(stt_script)
    llm.replies = list(llm_replies)
    llm._i = 0
    return pipeline.run_once()


def _enter_face(rig, azimuth_deg=30.0, elevation_deg=0.0):
    rig.world.add_face(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
                        size_px=FACE_SIZE_PX, face_id=FACE_ID)


# ---------------------------------------------------------------------------
# The end-to-end story
# ---------------------------------------------------------------------------

def test_full_robot_closed_loop_scenarios(tmp_path, rig):
    state_path = str(tmp_path / "state.json")
    people_db_path = str(tmp_path / "people.db")

    # Persona text must load for real (proves profiles/sim/persona.md is a
    # real, usable profile -- not a hand-typed fixture string).
    assert "NPC" in load_persona_text("sim")

    # Conversation process "boots" before anyone has ever been seen --
    # same ordering the real two-process deployment has (both start at
    # system boot, long before a person walks up).
    pipeline, pstate, ppeople, sink, llm, stt = _make_pipeline(
        state_path, people_db_path)
    assert pstate.get("new_person_seq") == 0

    # --- (a) person enters -> tracked to center -> person_in_range -------
    assert rig.state.get("person_present") is False
    assert rig.state.get("person_in_range") is False

    _enter_face(rig)
    rig.step_frames(5.0)

    err = rig.pixel_error()
    assert err is not None and err <= CONVERGED_PX, (
        "TrackingApp never converged the entering face to within %spx"
        % CONVERGED_PX)
    assert rig.state.get("person_present") is True
    assert rig.state.get("person_in_range") is True, (
        "face never registered in_range (bbox-height-fraction check)")

    person_id_1 = rig.state.get("person_id")
    assert person_id_1 is not None, (
        "recognition never enrolled a person_id -- see module docstring's "
        "seam-gap note about embed_face not being constructor-injectable")
    assert rig.people_store.get(int(person_id_1)) is not None
    assert pstate.get("new_person_seq") == 1  # the pipeline sees this bump

    # --- pipeline session 1: wake -> greet (new person) -> 2 utterances --
    assert pstate.get("conversation_active") is False
    happened = _run_session(
        pipeline, llm, stt,
        stt_script=["hi there", "my name is Alex"],
        llm_replies=["Hi there, nice to meet you!", "Good to know, Alex!"],
    )
    assert happened is True
    assert pstate.get("conversation_active") is False  # cleared on exit
    assert pstate.update_log[0] == {"conversation_active": True}
    assert pstate.update_log[-1] == {"conversation_active": False}

    # New-person greeting ("Nice to meet you!") plus both LLM replies were
    # all played through the (fake) speaker.
    assert sink.written, "no audio was played for the greeting/replies"
    assert len(sink.written) >= 3, (
        "expected >=3 played utterances: greeting + 2 LLM replies, got %d"
        % len(sink.written))
    for path in sink.written:
        assert os.path.isfile(path) and os.path.getsize(path) > 0
    assert pipeline._history[0]["role"] == "assistant"
    assert "meet" in pipeline._history[0]["content"].lower(), (
        "first-encounter greeting did not read as a 'nice to meet you'")

    # --- (b) "my name is Alex" -> people.db has the name ------------------
    record = rig.people_store.get(int(person_id_1))
    assert record["name"] == "Alex", (
        "ConversationPipeline._maybe_capture_name never wrote the name "
        "back through the REAL shared/people.py PeopleStore")
    # Also visible through the pipeline's own (file-sharing) PeopleStore
    # connection -- proves this round-tripped through the DB file, not an
    # in-process object.
    assert ppeople.get(int(person_id_1))["name"] == "Alex"

    # --- (c) person leaves mid-conversation -------------------------------
    def _depart():
        rig.world.move_face(FACE_ID, azimuth_deg=500.0, elevation_deg=0.0)
        rig.step_frames(HOLD_S + 1.0)  # past hold_s -> target drops to None

    sink_before = len(sink.written)
    happened2 = _run_session(
        pipeline, llm, stt,
        stt_script=[("what's the weather like", None), ("bye then", _depart)],
        llm_replies=["Sunny and warm today!"],
    )
    assert happened2 is True
    assert rig.state.get("person_present") is False, (
        "TrackingApp never dropped person_present after the face left")
    assert pstate.get("conversation_active") is False
    assert pstate.update_log[-1] == {"conversation_active": False}
    # Window reset: proven directly (no crash / no stale person_id) and by
    # the next session starting a fresh window rather than continuing.
    assert pipeline._history_person_id is None
    # Only one reply was played (the pre-departure question); "bye then"
    # never got a reply because the departure was detected before the LLM
    # was asked to respond to it.
    assert len(sink.written) - sink_before == 1

    # --- idle scan after heartbeat silence ---------------------------------
    rig.step_frames(HEARTBEAT_TIMEOUT_S + 1.0)
    assert rig.servo.head.is_idle(rig.servo.now), (
        "ServoSim never entered idle scan after heartbeat silence")

    # --- (d) same face returns -> recognized (same person_id) ------------
    # Generous budget: the head is mid idle-scan-sweep (ORCHESTRATION.md
    # SS4 rule 3: "ESP32 owns it on serial silence") when the face
    # reappears, so re-detection first requires the sweep to happen to
    # cross the face's azimuth (worst case ~9s of a 60deg-span, 20deg/s
    # sweep) before PID convergence even starts.
    rig.world.move_face(FACE_ID, azimuth_deg=-15.0, elevation_deg=0.0)
    rig.step_frames(20.0)

    err2 = rig.pixel_error()
    assert err2 is not None and err2 <= CONVERGED_PX
    assert rig.state.get("person_present") is True
    assert rig.state.get("person_in_range") is True

    person_id_2 = rig.state.get("person_id")
    assert person_id_2 == person_id_1, (
        "returning face was NOT matched to the same person_id -- "
        "shared/people.py cosine match failed to recognize a repeat visit")
    assert rig.people_store.count() == 1, (
        "returning face was mistakenly auto-enrolled as a second person"
    )
    assert pstate.get("new_person_seq") == 1  # no re-enroll bump on return

    sink_before = len(sink.written)
    happened3 = _run_session(
        pipeline, llm, stt,
        stt_script=["good to see you again"],
        llm_replies=["Good to see you again!"],
    )
    assert happened3 is True
    assert len(sink.written) > sink_before, "no greeting/reply audio played on return"

    # The very first thing said on return must be the known-person
    # greeting (spoken via speaker.say(), i.e. it is NOT one of the
    # scripted LLM replies) referencing the now-known name.
    assert pipeline._history[0]["role"] == "assistant"
    assert "Alex" in pipeline._history[0]["content"], (
        "returning-visitor greeting did not reference the known name")
    assert "back" in pipeline._history[0]["content"].lower(), (
        "greeting text did not read as a 'returning visitor' greeting"
    )

    # And it was NOT treated as a brand-new person again (no re-enroll):
    assert rig.people_store.count() == 1
    assert pstate.get("new_person_seq") == 1

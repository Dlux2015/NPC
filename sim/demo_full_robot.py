"""Runnable whole-robot demo: python sim/demo_full_robot.py

Walks the exact story sim/scenarios/test_full_robot.py proves
(ORCHESTRATION.md SS3.6): a person enters, gets tracked to center, wakes
the REAL conversation.pipeline.ConversationPipeline, gets greeted,
auto-enrolled, and named; leaves mid-conversation (window reset, idle
scan on heartbeat silence); then returns and is recognized as the same,
now-named person -- both the REAL vision.tracking.TrackingApp and the
REAL ConversationPipeline running against the digital twin, sharing one
real state.json + people.db file pair, exactly as the two real processes
would on-device.

Reuses sim/scenarios/test_full_robot.py's rig/fakes directly (single
source of truth for the wiring -- see that module's docstring for the
full swap-point/finding notes) rather than re-deriving them here.

Prints a timeline as it runs; exits 0 if every milestone checks out
(same assertions as the pytest scenario, as plain checks here), 1 with a
FAILED line otherwise.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.serial_protocol import HEARTBEAT_TIMEOUT_S
from sim.scenarios import test_full_robot as scenario
from sim.world import SimWorld
from vision import calibrate, tracking
from vision.paths import profile_dir


def log(t, msg):
    print("t=%6.1fs: %s" % (t, msg))


def _print_new_history(pipeline, start_idx, t):
    for entry in pipeline._history[start_idx:]:
        if entry["role"] == "user":
            log(t, 'heard "%s"' % entry["content"])
        else:
            log(t, 'replied "%s"' % entry["content"])
    return len(pipeline._history)


def _load_sim_calibration():
    import json
    calib_path = os.path.join(profile_dir("sim"), "calibration.json")
    if not os.path.isfile(calib_path):
        calibrate.run("sim", auto=True)
    with open(calib_path) as f:
        return json.load(f)


def run():
    tmp_dir = tempfile.mkdtemp(prefix="cbot_demo_full_robot_")
    state_path = os.path.join(tmp_dir, "state.json")
    people_db_path = os.path.join(tmp_dir, "people.db")

    calibration = _load_sim_calibration()
    world = SimWorld()  # starts empty -- "person enters" adds the face

    orig_embed_face = tracking.embed_face
    tracking.embed_face = scenario._make_fake_embed_face([scenario.FACE_ID])
    rig = scenario.Rig(world, calibration, state_path, people_db_path,
                        [scenario.FACE_ID])
    try:
        pipeline, pstate, ppeople, sink, llm, stt = scenario._make_pipeline(
            state_path, people_db_path)
        history_idx = 0

        # --- person enters -> tracked to center -> person_in_range -------
        log(rig.t, "world empty; person_present=%s person_in_range=%s"
            % (rig.state.get("person_present"), rig.state.get("person_in_range")))
        scenario._enter_face(rig)
        log(rig.t, "person enters (face at azimuth=30.0deg)")
        rig.step_frames(5.0)

        err = rig.pixel_error()
        assert err is not None and err <= scenario.CONVERGED_PX
        log(rig.t, "tracked to %.1fpx of center" % err)
        assert rig.state.get("person_in_range") is True
        log(rig.t, "in range")

        person_id_1 = rig.state.get("person_id")
        assert person_id_1 is not None
        log(rig.t, "enrolled person %s" % person_id_1)

        # --- session 1: wake -> greet (new person) -> 2 utterances --------
        assert rig.state.get("person_in_range") is True
        log(rig.t, "wake (face_speech)")
        happened = scenario._run_session(
            pipeline, llm, stt,
            stt_script=["hi there", "my name is Alex"],
            llm_replies=["Hi there, nice to meet you!", "Good to know, Alex!"],
        )
        assert happened is True
        history_idx = _print_new_history(pipeline, history_idx, rig.t)

        record = rig.people_store.get(int(person_id_1))
        assert record["name"] == "Alex"
        log(rig.t, "named Alex (person %s)" % person_id_1)
        assert sink.written and all(
            os.path.isfile(p) and os.path.getsize(p) > 0 for p in sink.written)
        assert pstate.get("conversation_active") is False

        # --- continue the conversation, then leave mid-conversation --------
        # Two separate run_once() sessions (same window -- STT exhaustion
        # alone doesn't reset it, only person-absence/person-id changes do,
        # per pipeline.py's _maybe_reset_window): the weather Q&A first, so
        # its history is printed before the second session's departure
        # clears pipeline._history via the real window-reset path.
        happened2a = scenario._run_session(
            pipeline, llm, stt,
            stt_script=["what's the weather like"],
            llm_replies=["Sunny and warm today!"],
        )
        assert happened2a is True
        history_idx = _print_new_history(pipeline, history_idx, rig.t)

        def _depart():
            rig.world.move_face(scenario.FACE_ID, azimuth_deg=500.0,
                                 elevation_deg=0.0)
            rig.step_frames(scenario.HOLD_S + 1.0)

        happened2b = scenario._run_session(
            pipeline, llm, stt,
            stt_script=[("bye then", _depart)],
            llm_replies=["(unused)"],
        )
        assert happened2b is True
        log(rig.t, 'heard "bye then" -> person left mid-utterance (no reply)')

        assert rig.state.get("person_present") is False
        assert pipeline._history_person_id is None
        history_idx = 0  # the window reset just cleared pipeline._history
        log(rig.t, "person_present=False, conversation window reset")

        # --- idle scan after heartbeat silence -----------------------------
        rig.step_frames(HEARTBEAT_TIMEOUT_S + 1.0)
        assert rig.servo.head.is_idle(rig.servo.now)
        log(rig.t, "idle scan engaged (heartbeat silence > %.1fs)"
            % HEARTBEAT_TIMEOUT_S)

        # --- same face returns -> recognized --------------------------------
        rig.world.move_face(scenario.FACE_ID, azimuth_deg=-15.0, elevation_deg=0.0)
        log(rig.t, "face returns (azimuth=-15.0deg)")
        rig.step_frames(20.0)

        err2 = rig.pixel_error()
        assert err2 is not None and err2 <= scenario.CONVERGED_PX
        log(rig.t, "tracked to %.1fpx of center" % err2)
        assert rig.state.get("person_in_range") is True
        log(rig.t, "in range")

        person_id_2 = rig.state.get("person_id")
        assert person_id_2 == person_id_1
        assert rig.people_store.count() == 1
        log(rig.t, "returned -> recognized person %s ('%s'), no re-enroll"
            % (person_id_2, rig.people_store.get(int(person_id_2))["name"]))

        log(rig.t, "wake (face_speech)")
        happened3 = scenario._run_session(
            pipeline, llm, stt,
            stt_script=["good to see you again"],
            llm_replies=["Good to see you again!"],
        )
        assert happened3 is True
        history_idx = _print_new_history(pipeline, history_idx, rig.t)

        assert "Alex" in pipeline._history[0]["content"]
        assert "back" in pipeline._history[0]["content"].lower()
        assert rig.people_store.count() == 1

        print()
        print("ALL GOOD -- whole-robot closed loop verified end to end "
              "(detect -> PID -> serial -> virtual servo -> new view, "
              "wake -> STT -> LLM -> TTS, and recognition/identity, "
              "all against the digital twin).")
        return 0
    except AssertionError as exc:
        print()
        print("FAILED: %s" % exc)
        return 1
    finally:
        rig.close()
        tracking.embed_face = orig_embed_face


if __name__ == "__main__":
    sys.exit(run())

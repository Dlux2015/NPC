"""display/expressions.py: the emote state machine, driven by scripted
IPC sequences exactly as the real renderer drives it -- this is the
feature's sim scenario per the sim-first rule (the renderer itself is
eyeballed via `python -m display.emote --demo`; only geometry lives
there)."""
import numpy as np
import pytest

from display.expressions import (
    ALERT, IDLE, LISTENING, NEUTRAL, SURPRISED, TALKING,
    ExpressionStateMachine,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_state(**kwargs):
    base = {"person_present": False, "conversation_active": False,
            "actively_speaking": False, "new_person_seq": 0}
    base.update(kwargs)
    return base


def test_nobody_around_is_idle():
    m = ExpressionStateMachine(clock=FakeClock())
    assert m.update(make_state()) == IDLE


def test_person_present_is_alert():
    m = ExpressionStateMachine(clock=FakeClock())
    assert m.update(make_state(person_present=True)) == ALERT


def test_conversation_beats_alert():
    m = ExpressionStateMachine(clock=FakeClock())
    assert m.update(make_state(person_present=True,
                                conversation_active=True)) == LISTENING


def test_speaking_beats_listening():
    m = ExpressionStateMachine(clock=FakeClock())
    assert m.update(make_state(person_present=True, conversation_active=True,
                                actively_speaking=True)) == TALKING


def test_new_person_bump_surprises_then_decays():
    clock = FakeClock()
    m = ExpressionStateMachine(clock=clock)
    assert m.update(make_state(person_present=True)) == ALERT  # baseline seq=0

    # vision auto-enrolls someone -> seq bumps -> one-shot surprise...
    s = make_state(person_present=True, conversation_active=True,
                   new_person_seq=1)
    assert m.update(s) == SURPRISED
    clock.t = 0.5
    assert m.update(s) == SURPRISED       # still inside the hold window
    clock.t = 2.0
    assert m.update(s) == LISTENING       # ...decays to the live state


def test_preexisting_seq_at_boot_does_not_surprise():
    """First observation just sets the baseline -- people enrolled before
    this process started must not trigger a boot-time surprise."""
    m = ExpressionStateMachine(clock=FakeClock())
    assert m.update(make_state(person_present=True,
                                new_person_seq=7)) == ALERT


def test_unreadable_state_degrades_to_neutral():
    class BoomState:
        def get(self, key):
            raise OSError("ipc gone")

    m = ExpressionStateMachine(clock=FakeClock())
    assert m.update(BoomState()) == NEUTRAL


def test_full_visit_scenario_transitions():
    """One visitor's arc end-to-end: idle -> walks up -> enrolled
    (surprise) -> conversation -> robot replies -> leaves -> idle."""
    clock = FakeClock()
    m = ExpressionStateMachine(clock=clock)

    assert m.update(make_state()) == IDLE
    clock.t = 1.0
    assert m.update(make_state(person_present=True)) == ALERT
    clock.t = 2.0
    assert m.update(make_state(person_present=True,
                                new_person_seq=1)) == SURPRISED
    clock.t = 4.0
    assert m.update(make_state(person_present=True,
                                conversation_active=True,
                                new_person_seq=1)) == LISTENING
    clock.t = 5.0
    assert m.update(make_state(person_present=True,
                                conversation_active=True,
                                actively_speaking=True,
                                new_person_seq=1)) == TALKING
    clock.t = 6.0
    assert m.update(make_state(new_person_seq=1)) == IDLE


def test_renderer_produces_frames_that_differ_by_expression():
    from display.emote import EyeRenderer, PANEL

    clock = FakeClock()
    r = EyeRenderer(clock=clock)
    # Let the eased parameters settle on each pose before comparing.
    for _ in range(90):
        clock.t += 1 / 30
        idle = r.draw(IDLE)
    for _ in range(90):
        clock.t += 1 / 30
        surprised = r.draw(SURPRISED)

    assert idle.shape == surprised.shape == (PANEL, PANEL, 3)
    assert idle.dtype == np.uint8
    assert np.abs(idle.astype(int) - surprised.astype(int)).mean() > 1.0


def test_renderer_is_red_dominant_and_round():
    """The eye is RED (user direction: robot eye, red core + red halo,
    not a human eye) and confined to the round panel."""
    from display.emote import EyeRenderer, PANEL

    clock = FakeClock()
    r = EyeRenderer(clock=clock)
    for _ in range(90):
        clock.t += 1 / 30
        frame = r.draw(LISTENING)

    b, g, red = (frame[:, :, i].astype(int) for i in range(3))
    assert red.sum() > 5 * g.sum()      # overwhelmingly red
    assert red.sum() > 5 * b.sum()
    assert red.max() > 200              # actually bright
    # corners are outside the round panel -> dark
    assert frame[:8, :8].max() == 0
    assert frame[-8:, -8:].max() == 0

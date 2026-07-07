"""Expression state machine for the emote display
(hardware/emote_display.md v1): maps the IPC state the robot ALREADY
publishes to a named expression. Pure logic, no cv2/hardware -- the
renderer (display/emote.py) and the scenario tests both drive this same
class, per the sim-first rule.

v1 mapping (proposal table; no contract changes -- reads existing keys
only):

    person_present False        -> IDLE      (sleepy; matches idle scan)
    person_present True         -> ALERT     (eyes open)
    conversation_active True    -> LISTENING (attentive)
    actively_speaking True      -> TALKING   (animated)
    new_person_seq bump         -> SURPRISED (one-shot, decays)
    stale/absent IPC            -> NEUTRAL   (degrade gracefully)

Priority when several are true: SURPRISED > TALKING > LISTENING >
ALERT > IDLE.
"""
import time

IDLE = "idle"
ALERT = "alert"
LISTENING = "listening"
TALKING = "talking"
SURPRISED = "surprised"
NEUTRAL = "neutral"

SURPRISE_HOLD_S = 1.4  # how long the one-shot surprise overrides the rest


class ExpressionStateMachine:
    def __init__(self, clock=time.monotonic, surprise_hold_s=SURPRISE_HOLD_S):
        self._clock = clock
        self._surprise_hold_s = surprise_hold_s
        self._last_seq = None  # unknown until the first update
        self._surprised_until = float("-inf")

    def update(self, state):
        """state: dict-like with .get(key) (shared/ipc.py SharedState or
        a plain dict via .get) -> expression name for right now."""
        now = self._clock()
        get = state.get

        try:
            seq = get("new_person_seq")
            person_present = get("person_present")
            conversation_active = get("conversation_active")
            actively_speaking = get("actively_speaking")
        except Exception:
            return NEUTRAL  # unreadable IPC: neutral face, never crash

        if seq is not None:
            if self._last_seq is not None and seq > self._last_seq:
                self._surprised_until = now + self._surprise_hold_s
            self._last_seq = seq

        if now < self._surprised_until:
            return SURPRISED
        if actively_speaking:
            return TALKING
        if conversation_active:
            return LISTENING
        if person_present:
            return ALERT
        return IDLE

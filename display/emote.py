"""Emote display process (hardware/emote_display.md v1): renders the
robot's expression as a pair of round GC9A01-style eyes, driven ONLY by
what shared/ipc.py already publishes -- run it next to any demo (or,
later, next to the real robot processes) and the eyes react as you
interact:

    python -m display.emote                # eyes window, dev-pc profile
    python -m display.emote --demo         # cycle all expressions, no IPC
    python -m display.emote --profile sim  # another profile's run dir

On the robot this same renderer targets two SPI panels instead of a
window (device swap per profile, SS3.6 style); the expression logic
(display/expressions.py) is identical either way. Window backend uses
cv2 (already a project dependency) -- each eye is drawn inside a round
bezel at 240x240, the physical panels' resolution, so what you see is
what the hardware will show.
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from display.expressions import (
    ALERT, IDLE, LISTENING, NEUTRAL, SURPRISED, TALKING,
    ExpressionStateMachine,
)
from shared.ipc import SharedState
from vision.paths import repo_root

PANEL = 240          # GC9A01 resolution (240x240, round)
GAP = 24             # px between the two eyes in the window
BG = (12, 10, 8)     # near-black bezel background
IRIS = (255, 190, 60)   # BGR: warm cyan-ish? no -- friendly amber-blue
PUPIL = (30, 20, 10)

# Per-expression targets: (lid_open 0..1, pupil_scale, look_y -1..1)
_EXPRESSION_POSE = {
    IDLE:      (0.30, 1.00, +0.35),   # heavy lids, gaze low
    NEUTRAL:   (0.70, 1.00, 0.00),
    ALERT:     (0.85, 1.00, 0.00),
    LISTENING: (1.00, 1.15, -0.10),   # wide, big pupils, slightly up
    TALKING:   (0.90, 1.00, 0.00),
    SURPRISED: (1.00, 0.70, -0.20),   # wide lids, small pupils
}


class EyesRenderer:
    """Draws both eyes into one BGR frame. Continuous parameters ease
    toward each expression's pose so transitions read as motion, not
    cuts; blinks and idle drift keep it alive."""

    def __init__(self, clock=time.monotonic, rng=None):
        self._clock = clock
        self._rng = rng or random.Random()
        self._lid = 0.7
        self._pupil = 1.0
        self._look_y = 0.0
        self._last_t = None
        self._blink_until = 0.0
        self._next_blink = self._clock() + self._rng.uniform(2.0, 5.0)

    def _ease(self, current, target, dt, rate=8.0):
        a = min(1.0, rate * dt)
        return current + (target - current) * a

    def draw(self, expression, talk_level=None):
        now = self._clock()
        dt = 0.0 if self._last_t is None else min(0.1, now - self._last_t)
        self._last_t = now

        lid_t, pupil_t, look_t = _EXPRESSION_POSE.get(
            expression, _EXPRESSION_POSE[NEUTRAL])

        # Natural blinking (not while surprised -- wide eyes sell it).
        if expression != SURPRISED and now >= self._next_blink:
            self._blink_until = now + 0.12
            self._next_blink = now + self._rng.uniform(
                2.0, 5.0 if expression != IDLE else 8.0)
        if now < self._blink_until:
            lid_t = 0.05

        # Talking: rhythmic lid/pupil energy so the face visibly "speaks".
        if expression == TALKING:
            wobble = 0.5 + 0.5 * math.sin(now * 12.0)
            lid_t = 0.75 + 0.2 * wobble
            pupil_t = 1.0 + 0.08 * math.sin(now * 9.0)

        # Idle: slow sleepy drift.
        if expression == IDLE:
            lid_t += 0.06 * math.sin(now * 0.8)

        self._lid = self._ease(self._lid, lid_t, dt)
        self._pupil = self._ease(self._pupil, pupil_t, dt)
        self._look_y = self._ease(self._look_y, look_t, dt)

        frame = np.zeros((PANEL, 2 * PANEL + GAP, 3), dtype=np.uint8)
        frame[:] = BG
        self._draw_eye(frame, PANEL // 2)
        self._draw_eye(frame, PANEL + GAP + PANEL // 2)
        return frame

    def _draw_eye(self, frame, cx):
        import cv2
        cy = PANEL // 2
        r_bezel = PANEL // 2 - 4
        r_iris = int(56 * 1.0)
        r_pupil = int(26 * self._pupil)
        pupil_dy = int(self._look_y * 26)

        cv2.circle(frame, (cx, cy), r_bezel, (28, 24, 20), -1)      # screen
        cv2.circle(frame, (cx, cy), r_bezel, (70, 60, 50), 2)       # bezel ring
        cv2.circle(frame, (cx, cy + pupil_dy), r_iris, IRIS, -1)    # iris
        cv2.circle(frame, (cx, cy + pupil_dy), r_pupil, PUPIL, -1)  # pupil
        cv2.circle(frame, (cx - 18, cy + pupil_dy - 18), 10,
                   (255, 255, 255), -1)                              # glint

        # Eyelids: two dark arcs closing over the eye; lid=1 fully open.
        closed = 1.0 - max(0.0, min(1.0, self._lid))
        lid_px = int(closed * r_bezel * 1.05)
        if lid_px > 0:
            cv2.ellipse(frame, (cx, cy - r_bezel + lid_px // 2),
                        (r_bezel, lid_px // 2 + 6), 0, 0, 360, BG, -1)
            cv2.ellipse(frame, (cx, cy + r_bezel - lid_px // 2),
                        (r_bezel, lid_px // 2 + 6), 0, 0, 360, BG, -1)
        # Re-punch the bezel ring over the lids so the screen edge stays.
        cv2.circle(frame, (cx, cy), r_bezel, (70, 60, 50), 2)


def run(state_path, demo=False, fps=30.0):
    import cv2

    machine = ExpressionStateMachine()
    renderer = EyesRenderer()
    state = None if demo else SharedState(state_path)

    demo_cycle = [IDLE, ALERT, LISTENING, TALKING, SURPRISED, NEUTRAL]
    t0 = time.monotonic()
    period = 1.0 / fps

    print("NPC eyes up%s -- q/Esc to close."
          % (" (demo cycle)" if demo else ", following " + state_path))
    expression = NEUTRAL
    while True:
        loop_t = time.monotonic()
        if demo:
            expression = demo_cycle[int((loop_t - t0) / 2.5) % len(demo_cycle)]
        else:
            expression = machine.update(state)

        frame = renderer.draw(expression)
        cv2.putText(frame, expression, (8, PANEL - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 100, 90), 1)
        cv2.imshow("NPC eyes", frame)
        key = cv2.waitKey(max(1, int(period * 1000) -
                               int((time.monotonic() - loop_t) * 1000))) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.environ.get("CBOT_PROFILE", "dev-pc"),
                         help="whose run/<profile>/state.json to follow")
    parser.add_argument("--demo", action="store_true",
                         help="cycle through all expressions; no IPC needed")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)

    state_path = os.path.join(repo_root(), "run", args.profile, "state.json")
    if not args.demo:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
    return run(state_path, demo=args.demo, fps=args.fps)


if __name__ == "__main__":
    sys.exit(main() or 0)

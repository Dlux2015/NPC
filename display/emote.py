"""Emote display process (hardware/emote_display.md v1): renders the
robot's expression as ONE round, red robot eye -- a glowing red core
(the "pupil", it looks around) inside a red halo ring, on a virtual
240x240 GC9A01-style panel. Driven ONLY by what shared/ipc.py already
publishes -- run it next to any demo (or, later, next to the real robot
processes) and the eye reacts as you interact:

    python -m display.emote                # eye window, dev-pc profile
    python -m display.emote --demo         # cycle all expressions, no IPC
    python -m display.emote --profile sim  # another profile's run dir

Design (user-directed, 2026-07-06): deliberately NOT a human eye -- no
whites/iris/eyelids/blinks. Expression is carried by brightness, core
size, halo radius, pulse rhythm, and how the core wanders:

    idle       dim, slow deep breathing pulse, core drifts around lazily
    alert      bright, core mostly centered, small wander
    listening  brightest, halo expanded, steady, core locked center
    talking    intensity and core pulse in speech rhythm
    surprised  flare: big halo, pinpoint core
    neutral    medium everything

On the robot this same renderer targets an SPI panel instead of a
window (device swap per profile, SS3.6 style); the expression logic
(display/expressions.py) is identical either way.
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

PANEL = 240  # GC9A01 resolution (240x240, round)

# Per-expression pose:
# (brightness 0..1, core_radius px, halo_radius px, wander_radius px)
_EXPRESSION_POSE = {
    IDLE:      (0.35, 16, 68, 42),
    NEUTRAL:   (0.60, 20, 76, 18),
    ALERT:     (0.85, 22, 84, 12),
    LISTENING: (1.00, 24, 94, 4),
    TALKING:   (0.90, 22, 88, 8),
    SURPRISED: (1.00, 12, 108, 0),
}


class EyeRenderer:
    """Draws the single red robot eye into one BGR frame. Continuous
    parameters ease toward each expression's pose so transitions read
    as motion, not cuts; the core smoothly wanders ("looks around")
    within the expression's wander radius."""

    def __init__(self, clock=time.monotonic, rng=None):
        self._clock = clock
        self._rng = rng or random.Random()
        # eased state
        self._brightness = 0.5
        self._core_r = 20.0
        self._halo_r = 76.0
        self._look = np.array([0.0, 0.0])         # current core offset
        self._look_target = np.array([0.0, 0.0])
        self._next_look = self._clock()
        self._last_t = None
        # precomputed pixel grid for the radial fields
        yy, xx = np.mgrid[0:PANEL, 0:PANEL].astype(np.float32)
        self._xx = xx - PANEL / 2.0
        self._yy = yy - PANEL / 2.0
        self._r_center = np.sqrt(self._xx ** 2 + self._yy ** 2)

    def _ease(self, current, target, dt, rate=6.0):
        a = min(1.0, rate * dt)
        return current + (target - current) * a

    def _retarget_look(self, now, wander_r):
        """Pick a new spot for the core to drift toward."""
        if wander_r <= 0:
            self._look_target = np.array([0.0, 0.0])
        else:
            ang = self._rng.uniform(0, 2 * math.pi)
            rad = self._rng.uniform(0, wander_r)
            self._look_target = np.array(
                [math.cos(ang) * rad, math.sin(ang) * rad])
        self._next_look = now + self._rng.uniform(1.2, 3.5)

    def draw(self, expression):
        now = self._clock()
        dt = 0.0 if self._last_t is None else min(0.1, now - self._last_t)
        self._last_t = now

        brightness, core_r, halo_r, wander_r = _EXPRESSION_POSE.get(
            expression, _EXPRESSION_POSE[NEUTRAL])

        if expression == IDLE:
            brightness *= 0.85 + 0.15 * math.sin(now * 1.2)  # slow breathing
        elif expression == TALKING:
            brightness *= 0.80 + 0.20 * math.sin(now * 11.0)  # speech rhythm
            core_r *= 1.0 + 0.15 * math.sin(now * 8.0)
        elif expression == LISTENING:
            brightness *= 0.95 + 0.05 * math.sin(now * 2.0)  # calm shimmer

        if now >= self._next_look:
            self._retarget_look(now, wander_r)
        look_rate = 3.0 if expression == IDLE else 7.0
        self._look = self._look + (self._look_target - self._look) * min(
            1.0, look_rate * dt)

        self._brightness = self._ease(self._brightness, brightness, dt)
        self._core_r = self._ease(self._core_r, core_r, dt)
        self._halo_r = self._ease(self._halo_r, halo_r, dt)

        # --- compose the radial fields -----------------------------------
        px, py = self._look
        d_core = np.sqrt((self._xx - px) ** 2 + (self._yy - py) ** 2)
        core = np.exp(-(d_core / max(6.0, self._core_r)) ** 2)
        halo = np.exp(-(((self._r_center - self._halo_r) /
                          max(6.0, self._halo_r * 0.16)) ** 2))
        # faint inner wash so the "screen" looks lit between core and halo
        wash = np.exp(-(self._r_center / (self._halo_r * 1.1)) ** 2) * 0.12

        intensity = np.clip(
            (core * 1.0 + halo * 0.55 + wash) * self._brightness, 0.0, 1.0)

        # round panel mask (it's a circular screen)
        mask = (self._r_center <= PANEL / 2.0 - 3).astype(np.float32)
        intensity *= mask

        frame = np.zeros((PANEL, PANEL, 3), dtype=np.uint8)
        frame[:, :, 2] = (intensity * 255).astype(np.uint8)          # red
        frame[:, :, 1] = (np.clip(intensity - 0.55, 0, 1) * 120
                           ).astype(np.uint8)   # hot core tints orange
        frame[:, :, 0] = (np.clip(intensity - 0.85, 0, 1) * 80
                           ).astype(np.uint8)   # only the very center
        return frame


def run(state_path, demo=False, fps=30.0):
    import cv2

    machine = ExpressionStateMachine()
    renderer = EyeRenderer()
    state = None if demo else SharedState(state_path)

    demo_cycle = [IDLE, ALERT, LISTENING, TALKING, SURPRISED, NEUTRAL]
    t0 = time.monotonic()
    period = 1.0 / fps

    print("NPC eye up%s -- q/Esc to close."
          % (" (demo cycle)" if demo else ", following " + state_path))
    while True:
        loop_t = time.monotonic()
        if demo:
            expression = demo_cycle[int((loop_t - t0) / 2.5) % len(demo_cycle)]
        else:
            expression = machine.update(state)

        frame = renderer.draw(expression)
        cv2.putText(frame, expression, (8, PANEL - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 50, 90), 1)
        cv2.imshow("NPC eye", frame)
        # Power: idle is a slow breathing pulse -- half the frame rate is
        # visually identical and halves this process's steady-state CPU
        # (matters on the battery-powered robot, harmless on the dev PC).
        wanted = period * 2 if expression == IDLE else period
        key = cv2.waitKey(max(1, int(wanted * 1000) -
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

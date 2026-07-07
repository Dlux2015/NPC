# Emote display — round LCD "eyes" proposal

**Status: PROPOSAL (2026-07-06); software v1 IMPLEMENTED same day** —
`display/expressions.py` (state machine) + `display/emote.py` (window
renderer, two virtual GC9A01s) + `display/tests/`. Run
`python -m display.emote` beside any demo and the eyes follow the live
IPC state; `--demo` cycles expressions. Hardware (real panels, BOM
rows) still unpurchased/not adopted into the build spec. Adoption would
touch ORCHESTRATION §4.2 (one new IPC key, v2 only) and §4.5 (shell
constraints below).

Goal: the head visibly *emotes* — idle/alert/listening/thinking/talking
expressions on one or two round displays behind the shell's eye
apertures. The robot already publishes every state needed to drive this;
the feature is mostly a renderer.

## Hardware choice: GC9A01 1.28" round SPI LCD (240×240)

The de-facto "robot eye" module, ~$7–9 each. Two = a pair of eyes
(full-spec build), one = a single cyclops eye/face (cost-reduced).

Considered and rejected:
- **Round HDMI display (4–5", ~$50+)**: one big expressive face, but
  heavier, pricier, eats the Jetson's display output, and a sealed-shell
  HDMI cable run is clumsy. Revisit only if a shell design demands it.
- **NeoPixel ring**: glow aesthetic, no real expressions; and WS2812
  timing is easiest from the ESP32 — which we specifically don't want to
  load (below).

## Who drives it: the Jetson, NOT the ESP32

The firmware's 50Hz servo loop is allocation-free by contract (GC pauses
must never stutter servos). SPI framebuffer blits from MicroPython on the
same chip is exactly the load that rule exists to prevent. The Jetson's
40-pin header has SPI; a small Python emote process pushing 10–15fps is
negligible CPU (vision keeps its budget: renderer sleeps between frames,
never in the vision hot loop — it's a separate process) and zero GPU.

Both displays share the SPI bus (SCLK/MOSI), separate CS + DC/RST pins.
Enable SPI on the header once via `sudo /opt/nvidia/jetson-io/jetson-io.py`
(document in the profile's setup notes). 240×240 @ 16bpp = ~115KB/frame;
SPI at 33MHz supports ~30fps ceiling — 10–15fps animation is comfortable.

## Architecture (fits existing contracts)

New process `display/emote.py` (working name), started by systemd like
the other two:

```
shared/ipc.py state --> expression state machine --> renderer --> device
                                                       device = SPI panel
                                                       (robot) or a cv2/
                                                       pygame window (sim)
```

- **State mapping (v1 — no contract changes at all):**

  | IPC state | Expression |
  |---|---|
  | `person_present` false | idle/sleepy (matches ESP32 idle scan) |
  | `person_present` true | eyes open, alert |
  | `conversation_active` true | attentive/listening |
  | `actively_speaking` true | talking animation |
  | `new_person_seq` bump | surprised/happy blink (one-shot) |
  | stale IPC heartbeat | neutral face (degrade gracefully) |

- **Profile keys (§4.5 — all shell-dependent values in the profile):**
  display count, SPI bus/CS/DC/RST pins, rotation, brightness, fps,
  eye style. A shell with no screens omits the block; emote.py exits
  cleanly ("this shell has no face").
- **Sim-first:** the expression state machine is pure logic — scenario
  tests script IPC sequences and assert expression transitions; the
  renderer draws into a window for `sim/demo_visual.py`. Same product
  code, device swapped by profile, per §3.6 — no `if simulation:`.

## v2 niceties (each needs a contract addition — orchestrator-owned)

- `emote` IPC key: persona instructs the LLM to lead replies with a mood
  tag (e.g. `[happy]`); pipeline strips it before TTS and publishes it.
- `gaze_offset` IPC key: vision's current target pixel offset, so pupils
  glance toward a person a beat before the servos swing — very lifelike,
  ~10 lines in the renderer. (Vision writes IPC outside the frame loop.)

## Power / memory budget

- Logic rail, 3.3V from the Jetson header: ~40–60mA per panel + backlight
  — negligible; nowhere near the servo rail, no new bucks.
- Process footprint: one more CPython process (~30–50MB) + ~115KB
  framebuffers. Counted, like everything, at the Phase 5 tegrastats
  audit; not expected to matter.

## Shell design constraints (feeds ORCHESTRATION §4.5 list)

- Eye aperture(s) ≥ panel active area, with viewing-angle relief (panel
  recessed behind the shell face reads as "dead eyes" — keep it shallow).
- Panel mounting bosses + ribbon/jumper routing from head to Jetson that
  survives full pan/tilt travel (flex loop, like the camera cable).
- If two eyes: interpupillary spacing fixed per shell in the 3D model;
  renderer must not care (it just gets two CS pins).

## Phase placement

Expression state machine + sim renderer: any time after Phase 0 (it's
software-only, sim-provable today). Real panels: bench alongside Phase 5
integration (needs the assembled head + jetson-io SPI setup), before the
Phase 7 shell-swap drill so eye apertures make it into the first printed
shell.

## Shopping list

| Item | Qty | ~Price (Jul 2026, verify before ordering) |
|---|---|---|
| GC9A01 1.28" round LCD, 240×240 SPI | 2 (full) / 1 (reduced) | ~$8 ea, ~$14/2-pack |
| Jumper/ribbon wiring | — | covered by existing Misc row |

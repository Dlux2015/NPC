---
name: sim-rig
description: Load before any work under /sim or on simulation/testing of robot behavior without hardware.
---
# Sim rig: digital twin on the dev PC

Product code runs **unmodified**; only transport/devices swap via the
`sim` shell profile. No `if simulation:` branches — swap points are the
§4 contract seams (serial transport, audio device, camera source).

- **Virtual ESP32** (`sim/servo_sim.py`): speaks the exact wire format
  from `shared/serial_protocol.py` over a local TCP socket. Imports the
  **real** `firmware/easing.py` — never re-implement easing/limits.
  Models slew rate, command latency, heartbeat timeout → idle scan.
- **Virtual world** (`sim/world.py`): camera frame = pan/tilt-driven crop
  of a panorama (video with faces, or scripted synthetic face sprites).
  The full detect→PID→serial→virtual-servo→new-view loop closes in
  software.
- **Audio:** dev-PC devices or WAV in/out through the same by-name device
  selection; canned utterance fixtures for repeatable STT tests.
- **LLM:** same llama.cpp + model on the PC; slow is fine — sim validates
  correctness, not latency.
- **Scenarios** (`sim/scenarios/`, pytest, deterministic/seeded):
  enter→track→greet; two faces without snapping; known face recognized on
  return; person leaves mid-conversation → context reset; serial silence →
  idle scan; wake word in noise. Every new feature ships with a scenario.
- `calibrate.py` runs against the sim; `/profiles/sim` is a real profile
  produced the real way.
- **Sim cannot prove:** real optics/lighting, enclosure acoustics, servo
  load/binding, Jetson memory/thermal — hardware phases stay mandatory.

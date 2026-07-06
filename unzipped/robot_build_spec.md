# Convention face-tracking robot — build spec

A reusable robot head that uses machine vision to track faces/objects with a
pan/tilt camera, and a locally-run LLM for voice conversation with people at
cosplay conventions. This document is the handoff spec for implementation.

## Goals

- Camera continuously scans for people; when a face is found, pan/tilt servos
  smoothly keep it centered in frame.
- When someone is close and talking, the robot has a real spoken conversation
  using a fully local speech-to-text -> LLM -> text-to-speech pipeline (no
  cloud dependency, since convention WiFi is unreliable).
- Tracking must keep running while the LLM is "thinking" — the two systems
  run as independent processes, not a single blocking loop.
- Reusable: the architecture should not be hard-wired to one camera/servo/LLM
  choice. Swapping a component should not require a redesign.

## System architecture

Two-tier compute, connected over USB serial:

```
Camera (CSI) ---> Jetson Orin Nano (vision + LLM + speech) ---serial---> ESP32-S3 (servo PWM) ---> pan/tilt servos
Mic/speaker  ---> Jetson (USB sound card)
```

- **Jetson Orin Nano Super, 8GB (67 TOPS)** runs face/object detection,
  tracking logic, wake-word detection, STT, local LLM inference, and TTS.
- **ESP32-S3** is a dedicated real-time servo controller. It receives target
  angle commands over USB serial and owns the smoothing/acceleration curve
  itself, so servo motion stays fluid even if the Jetson stalls or is busy
  with LLM inference. It also enforces hard angle limits and runs an idle
  "look around" scan pattern if it stops hearing from the Jetson.
- Vision/tracking and the conversation pipeline run as **separate processes**
  on the Jetson so the robot keeps visually tracking a person while it is
  composing/speaking a reply.

## Finalized hardware (bill of materials)

All prices are user-confirmed as of July 2026 unless noted "estimate."
See `robot_bom_tracker_v2.html` in this same folder for the interactive version
with links.

| Category | Component | Price | Notes |
|---|---|---|---|
| Compute | NVIDIA Jetson Orin Nano Super Dev Kit, 8GB, 67 TOPS | $320.00 | Runs JetPack, CUDA, TensorRT |
| Storage | KingSpec 256GB NVMe M.2 2280, PCIe Gen3 x4, up to 3500MB/s | $56.99 | Boot/OS drive |
| Camera | Arducam IMX219 wide-angle CSI module | $15.99 | ~120 degree FOV, native Jetson CSI connector |
| MCU | ESP32-S3-DevKitC-1, WROOM-1-N16R8, 16MB flash / 8MB PSRAM | $24.99 | 3-pack — 1 in use, 2 spares |
| Servos | DS3218MG 20kg digital servo, metal gear, standard footprint, 2-pack | $26.99 | Pan + tilt. No position feedback (standard PWM, not bus servo) — smoothing is handled in ESP32-S3 firmware |
| Mount | Pan/tilt bracket for standard-size servos | $24.69 | Standard MG995/MG996/DS3218 footprint |
| Audio | USB sound card w/ onboard mic + 8-ohm 5W speaker, driver-free | $19.79 | Single forward-facing mic — camera/servo aiming handles directionality, no beamforming needed |
| Power | DeWalt DCB205 5Ah 20V MAX pack, qty 2 | ~$115 (estimate) | One pack per rail — hot-swappable |
| Power | Fused DeWalt adapter dock, 12AWG leads, 30A fuse, qty 2 | ~$40 (estimate) | One per pack |
| Power | Buck converter, 19V/5A output (Jetson rail) | ~$12 (estimate) | Steps pack voltage down for the Jetson's 19V barrel input |
| Power | Buck converter, 6.5V/5A+ output (servo rail) | ~$12 (estimate) | Must be a genuine 5A-class part — two DS3218MG stall at ~2.5A each |
| Power | Low-voltage alarm/cutoff module, ~15V, qty 2 | ~$10 (estimate) | Mandatory — see safety note below |
| Misc | Wiring, connectors, standoffs, enclosure | $30.00 (estimate) | JST/Dupont wiring, mounting hardware |

**Power chain (v2 — DeWalt 20V MAX tool-pack rework, proposed 2026-07-05,
not yet purchased/confirmed):**
```
DeWalt pack A --fused adapter--> buck 19V/5A   --barrel--> Jetson Orin Nano
DeWalt pack B --fused adapter--> buck 6.5V/5A+ ----------> DS3218MG servos
```

This replaces an earlier NOBIS USB-C power bank (Jetson) + Talentcell 12V
pack + Seloky LM2596 buck converter (servos) design. Reasons for the switch:
hot-swappable packs mean a full convention day across 2-3 packs with zero
downtime, ~100Wh per DCB205 pack (more than either the NOBIS's 74Wh or the
Talentcell's 84Wh), and tool packs handle dual-servo stall current loads
that stress USB power banks. DeWalt was chosen over Milwaukee M18 (better
cells, pricier, smaller adapter market) and Ryobi ONE+ (cheapest, bulkier
packs, spottier adapter supply) mainly for its large third-party robotics
adapter market — brand choice here is about ecosystem, not electrical
differences, so if packs from another brand are already owned, the
downstream electronics (buck converters, cutoff modules) are identical.

**Critical safety notes for this power system:**
- **Never wire a pack directly to the Jetson.** A fresh pack sits at ~21V,
  over the Jetson barrel jack's 19V spec — always go through the 19V/5A
  buck converter first.
- **Low-voltage cutoff is mandatory, not optional.** Tool packs have no
  internal over-discharge protection (the tool body normally provides
  that). Fit a ~15V alarm/cutoff module per pack, or have the tracking/
  conversation software watchdog monitor rail voltage and alert well before
  cutoff. Over-discharging a tool pack causes permanent damage.
- **Fuse every pack.** Use adapter docks with a built-in 30A fuse, or add
  inline fuses — these packs can deliver very high fault current.
- **The old Seloky LM2596 buck converter is undersized regardless of which
  battery system is used** — its ~2A continuous rating is below the ~2.5A
  stall current of each DS3218MG servo. Replace it with a genuine 5A-class
  buck converter even if this DeWalt rework is not adopted.

**Runtime estimate:** Jetson at 25W power mode plus servos averaging a few
watts is roughly 30W total draw, giving ~3 hours per 5Ah pack per rail, with
hot-swapping extending that indefinitely. Confirm with measured draw during
the Phase 1 bench test (see implementation milestones below) and record
real numbers before finalizing pack count for a full convention day.

**Shell/enclosure constraints this introduces:**
- Battery bay sized to the specific pack + adapter dock, with an external
  hatch for tool-free hot-swap mid-event.
- Bay positioned so pack weight sits low and central in the shell — the
  packs are the heaviest single components in the whole build.
- Cutoff/alarm module should be audible or visible from outside the shell
  so a low-battery state is noticeable during an event, not just in logs.

## Software stack

### 1. Servo firmware (ESP32-S3, C++/Arduino or MicroPython)

Responsibilities:
- Listen on USB serial for target angle commands, e.g. `P:1520 T:1480`
  (pan/tilt in microseconds or degrees — pick one convention and stick to it).
- Run a ~50Hz control loop that eases toward the target angle rather than
  snapping to it (since the DS3218MG has no onboard position feedback, all
  smoothing must happen here).
- Enforce hard min/max angle limits per axis so the mechanism can never bind.
- If no serial command is received for N seconds, fall back to a slow
  "idle scan" sweep pattern (search-for-a-face behavior) rather than freezing
  in place.
- Report current angle back over serial periodically for debugging.

Suggested libraries: ESP32Servo (Arduino) or standard PWM control in
MicroPython.

### 2. Vision + tracking (Python, on the Jetson)

- Face detection: OpenCV's YuNet or MediaPipe Face Detection (both run
  30fps+ on this hardware). Optionally add YOLOv8n for general object
  detection when no face is present.
- Tracking loop: compute the offset between detected-face-center and
  frame-center, run it through a PID controller, send corrected angles to
  the ESP32-S3 over serial.
- Target persistence: stick with one detected person for a few seconds
  rather than snapping between multiple faces in a crowd.
- Idle behavior: if no face is detected for N seconds, hand control back to
  the ESP32-S3's own idle scan (or command a scan pattern from this side —
  pick one owner for this behavior and document it).
- This loop must run continuously and never block on LLM/speech work.

### 3. Conversation pipeline (Python, on the Jetson, separate process)

- **Wake trigger:** face-within-range (from the tracking process) plus a
  wake word via openWakeWord, or a physical push-to-talk button as a
  fallback for noisy environments.
- **STT:** whisper.cpp or faster-whisper (small/base model).
- **LLM:** llama.cpp or Ollama running a quantized 3B-class instruct model
  (e.g. Qwen 2.5 3B or Llama 3.2 3B, Q4 quantization) with a system prompt
  defining the robot's convention persona. Keep responses short — this is a
  convention interaction, not a long-form chat.
- **TTS:** Piper (fast, fully local, decent voice quality).
- Audio I/O goes through the USB sound card (single mic, single speaker) —
  no diarization/direction-of-arrival needed since the camera/servo system
  already keeps the mic pointed at whoever is being tracked.

### Inter-process communication

Recommend a lightweight local message bus or simple shared state (e.g. a
small Redis instance, or even a file-based/socket IPC) between the tracking
process and the conversation process, so tracking can flag "person present /
absent" and the conversation process can flag "actively speaking" (which
tracking may want to know, e.g. to reduce movement while talking).

## Known tradeoffs to keep in mind

- **DS3218MG servos have no closed-loop feedback.** All motion smoothness
  is a firmware problem on the ESP32-S3, not a hardware guarantee. Budget
  real tuning time for the easing curve.
- **Single-direction mic.** No beamforming or multi-mic localization. The
  interaction design should assume people approach and speak roughly toward
  the front of the robot (which is where the camera is already aiming them).
- **Local LLM is the source of truth, not a cloud fallback.** No network
  dependency by design — convention WiFi is unreliable. If a future version
  adds an optional cloud LLM path, it must have a short timeout and fail
  silently back to the local model; it should never block or degrade the
  baseline experience.
- **8GB Jetson budget.** Vision (YuNet/MediaPipe) + a 3B Q4 LLM + Piper TTS
  need to coexist in memory alongside JetPack overhead. Profile memory
  early; if it's tight, the fallback is a smaller (1.5B) LLM.

## Suggested repo structure

```
/firmware
  /esp32_servo_controller     # Arduino/PlatformIO or MicroPython project
/vision
  tracking.py                 # face detection + PID + serial output to ESP32
  idle_scan.py
/conversation
  wake.py                     # wake word / push-to-talk
  stt.py                      # whisper.cpp wrapper
  llm.py                      # llama.cpp/Ollama wrapper + persona prompt
  tts.py                      # Piper wrapper
  pipeline.py                 # orchestrates wake -> stt -> llm -> tts
/shared
  ipc.py                      # shared state between vision and conversation processes
  serial_protocol.py          # shared angle-command format used by both vision.py and firmware
robot_bom_tracker_v2.html        # hardware bill of materials (reference only)
robot_build_spec.md           # this file
```

## Suggested first implementation milestones

1. ESP32-S3 firmware: accept a hardcoded angle over serial, verify smooth
   motion on the bench with servos powered from the Talentcell/buck-converter
   rail (not the Jetson).
2. Jetson: face detection running standalone, printing offset values (no
   servo connection yet).
3. Wire tracking.py to firmware: closed-loop face tracking working end to
   end.
4. Conversation pipeline standalone (wake word -> STT -> LLM -> TTS) on the
   bench, no tracking involved.
5. Combine: both processes running concurrently, sharing "person present"
   state, robot tracks continuously while conversing.

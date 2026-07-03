---
name: vision-face-tracking
description: Load before any work under /vision or touching face detection, recognition, tracking, or calibration.
---
# Vision: face tracking + recognition

## Detection
- OpenCV YuNet (`cv2.FaceDetectorYN`) primary; MediaPipe Face Detection
  alternate; optional YOLOv8n person-detect when no face visible.
- ≥30fps on **CPU** — the GPU belongs to the LLM. Verify with `tegrastats`
  on the Jetson.

## Recognition (core goal)
- SFace/ArcFace embeddings → `people.db` via `shared/people.py` ONLY
  (fields: embedding, ID, first/last-seen, optional name).
- Known face → publish `person_id` over IPC. Unknown face stable for a few
  seconds → auto-enroll + publish `new_person_seq` (counter, increments
  per auto-enroll).
- Runs low-frequency (~1Hz) in the tracking process; never in the
  per-frame hot loop.
- Privacy: embeddings only (no images), local-only, provide a purge
  command.

## Tracking loop (`vision/tracking.py`)
- Error = face-center − frame-center, converted to degrees via the
  **measured** deg/px in the active profile's `calibration.json`. Refuse
  to start without a calibration file — no hardcoded fallbacks.
- PID with anti-windup and the calibrated deadband; output target angles,
  never PWM.
- Target persistence: hold one face N seconds; no crowd-snapping.
- **The frame loop never blocks** — no LLM, disk, or network calls.
- Serial only via `shared/serial_protocol.py`. When no face: stop sending;
  the ESP32 owns idle scan.

## Calibration (`vision/calibrate.py`)
Interactive CLI; writes the shell profile. Steps in order:
1. Axis direction/sign (small moves, watch image shift).
2. Mechanical center offsets (keyboard jog; bracket ≠ 1500µs neutral).
3. Soft limits (jog to safe extremes) → provisions the ESP32's enforced
   limits — one set of numbers for firmware and Python.
4. Measured deg/px: step a known angle, measure pixel shift of a
   stationary face, average over positions.
5. Deadband/backlash: smallest command with visible motion → PID deadband.
6. Command→motion latency → PID gain ceiling.
7. Audio (see audio skills): speaker level, mic gain + VAD threshold,
   wake threshold, self-hearing check.
8. Verify: live crosshair test with P-only control.
Output: one versioned `calibration.json` per profile. Re-run on any
hardware or shell change. ESP32 access only via `shared/serial_protocol.py`.

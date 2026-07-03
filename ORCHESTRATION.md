# CBot Orchestration — Claude Code + Fable 5

Builds the robot in `unzipped/robot_build_spec.md` (BOM:
`unzipped/robot_bom_tracker.html`). Plan of record; nothing built yet.

**Goals:** (1) pan/tilt head keeps people in frame; (2) recognizes
returning people, notices new ones — persistent identity; (3) hears
directed + ambient speech, replies via local LLM; (4) swappable 3D-printed
shells — recalibrate via config, never recode; (5) everything testable in
a local sim before hardware (§3.6).

**Language:** Python everywhere. ESP32-S3 firmware = MicroPython (C++ only
if the 50Hz loop measurably fails).

## 1. Orchestration + token economy

- Fable 5 main session orchestrates. Subagents start cold — delegations
  name exact files + acceptance criteria. Skills carry domain decisions.
- Delegate only parallel or isolated-heavy work; small edits stay here.
- Token rules: point at paths, never paste contents; agents read only
  their skill + touched files; detail lives in skills, this plan is an
  index; escalate stalled Sonnet work to Fable 5 instead of retrying.

## 2. Agents (`.claude/agents/*.md`)

| Agent | Model | Scope |
|---|---|---|
| orchestrator (main session) | Fable 5 | Breakdown, delegation, integration, §4 contracts, sign-off |
| `vision-engineer` | Sonnet | `/vision`: detection, recognition, PID, calibrate.py |
| `firmware-engineer` | Sonnet | `/firmware`: 50Hz easing loop, limits, idle scan, serial parser |
| `speech-engineer` | Sonnet | Wake word, STT, TTS, USB audio |
| `llm-engineer` | Sonnet | `llm.py`, `pipeline.py`: serving, persona, conversation flow |
| `sim-engineer` | Sonnet | `/sim`: virtual head, virtual world, scenarios |
| `integration-reviewer` | Opus | Cross-module review, 8GB audit, tracking-never-blocks |
| `docs-scribe` | Haiku | Docs/BOM sync at phase boundaries |

Sonnet for skill-guided implementation; Opus for cross-process subtlety;
Haiku for diligence; Fable 5 for anything genuinely hard.

## 3. Skills (`.claude/skills/<name>/SKILL.md`)

### 3.1 `vision-face-tracking` — `/vision`
- YuNet primary, MediaPipe alt; optional YOLOv8n person-detect. ≥30fps on
  **CPU** (GPU is the LLM's, §3.4); verify via `tegrastats`.
- Recognition (core goal): SFace/ArcFace embeddings → `people.db`
  (embedding, ID, first/last-seen, name). Known → publish `person_id`;
  unknown stable a few s → auto-enroll + bump `new_person_seq`. Runs
  low-frequency — never in the per-frame hot loop. Embeddings only (no
  images), local, purge command.
- PID: pixel error → degrees via **measured** deg/px from
  `calibration.json`; outputs angles, never PWM; anti-windup + calibrated
  deadband. `tracking.py` refuses to start without calibration.
- Hold one target N s (no crowd-snapping). Frame loop never blocks — no
  LLM/disk/network. Serial only via `shared/serial_protocol.py`.

### 3.2 `audio-capture` — mic, wake, STT
- Single USB mic (servo aiming = directionality); device by name, not
  index; 16kHz mono.
- Directed mode (openWakeWord / push-to-talk / face-in-range + speech
  onset): full STT → LLM.
- Ambient mode: VAD-gated low-duty STT → rolling ~60s in-memory buffer
  (never persisted); scene context + robot-name-mention detection; lowest
  priority, never starves directed or vision.
- faster-whisper: small/base directed, tiny ambient; Silero VAD;
  wake→transcript <2s. Gains/thresholds from shell profile (§4.5).

### 3.3 `speech-output` — TTS, speaker
- Piper, one version-pinned voice. Sentence-streaming (speak sentence 1
  while LLM writes the rest).
- Publish `actively_speaking` via IPC: gates mic/ambient (never hears
  itself); vision damps motion. Volume/EQ from shell profile.

### 3.4 `local-llm-conversation` — `llm.py`, `pipeline.py`, persona
- Decided (Jul 2026, benchmarked): llama.cpp + CUDA,
  **Llama-3.2-3B-Instruct Q4_K_M** (~2.0GB, ~29 tok/s); alt Qwen 2.5 3B
  Q4; fallback 1–1.5B Q4. Board at 25W (`nvpmodel -m 1`). Context ≤2K.
- **GPU = LLM; vision = CPU.** Whisper GPU bursts OK (sequential with
  generation). Run headless (+~800MB).
- Local-only; any future cloud path: short timeout, silent local fallback.
- Persona: convention character, 1–3 sentence replies, all-ages; text
  lives in shell profile.
- Inject `person_id` (greet returns, react to `new_person`); learned names
  written only via `shared/people.py`. Ambient transcript = overheard
  scene context, never direct address.
- Window resets on person-absent. Model stays resident. Profile memory
  with the full stack running.

### 3.5 Calibration — `/vision/calibrate.py` (vision-engineer; reviewer signs off)
Steps: (1) axis direction/sign; (2) mechanical center offsets (bracket ≠
1500µs neutral); (3) soft limits → provisions the ESP32's enforced limits
(one set of numbers); (4) measured deg/px — step known angle, measure
pixel shift, average (beats FOV math on a nonlinear wide lens);
(5) deadband/backlash → PID deadband; (6) command→motion latency → PID
gain cap; (7) audio — speaker level, mic gain + VAD threshold (room tone,
speech at 1m/3m), wake threshold (N utterances through the shell),
self-hearing check; (8) verify — live crosshair, P-only control.
Rules: one versioned `calibration.json` per profile; consumers refuse to
run without it; re-run on any hardware/shell change; ESP32 access only via
`shared/serial_protocol.py`.

### 3.6 `sim-rig` — `/sim`: digital twin on the dev PC
Product code runs unmodified; only transport/devices swap via the `sim`
profile — no `if simulation:` branches.
- Virtual ESP32 (`sim/servo_sim.py`): real serial protocol over a local
  socket; **imports the actual firmware logic** — easing/limits live in
  pure-Python `firmware/easing.py`, imported by both MicroPython `main.py`
  and the sim. Models slew rate, latency, idle-scan.
- Virtual world (`sim/world.py`): camera view = pan/tilt-driven crop of a
  panorama (video faces or scripted sprites) — the full
  detect→PID→serial→servo→new-view loop closes in software.
- Audio: PC devices or WAV in/out; canned utterances for repeatable STT.
  LLM: same model on the PC (slow OK — sim validates correctness).
- Scenarios (`sim/scenarios/`, pytest): enter→track→greet; two faces, no
  snapping; known face recognized; leave mid-conversation → reset; serial
  silence → idle scan; wake word in noise. New features ship with one.
- `calibrate.py` runs against the sim; `/profiles/sim` is a real profile.
- Sim can't model optics/lighting, enclosure acoustics, servo load, Jetson
  memory/thermal — hardware Phases 2.5 and 5 stay mandatory.

## 4. Contracts (orchestrator-owned, defined first)

1. **Serial** (`shared/serial_protocol.py`, imported by firmware too):
   `P:<deg> T:<deg>\n`, degrees only, limits, heartbeat/timeout, angle
   reports. Pluggable transport: USB serial or sim socket, same bytes.
2. **IPC** (`shared/ipc.py`): `person_present`, `person_in_range`,
   `person_id`, `new_person_seq` (counter, increments per auto-enroll),
   `actively_speaking`, `conversation_active`, ambient-transcript handoff.
   File/socket-simple first; Redis only if it earns its footprint.
3. **Idle scan:** ESP32 owns it (on serial silence); vision just stops
   sending.
4. **Identity** (`shared/people.py`): sole reader/writer of `people.db`.

## 4.5 Shell profiles

All shell-dependent values live in a boot-selected profile
(`CBOT_PROFILE=<name>`); none in code, ever.

```
/profiles/<name>/
  profile.yaml       # name, persona file, flags
  calibration.json   # §3.5 steps 1–6
  audio.json         # §3.5 step 7
  persona.md         # this shell's character
```

Swap (<30 min, zero code edits): re-house → `calibrate.py --profile <new>`
full suite → optional new persona → reboot → verify.
3D-modeling constraints: camera aperture ≥ lens FOV, mic port + speaker
grille, **Jetson vent/fan duct (sealed head = throttle)**, USB/power
access, clearance for full pan/tilt range.

## 5. Phases

| Phase | Agents | Done when |
|---|---|---|
| 0 Scaffold | orchestrator | Repo tree (spec layout + `/profiles` + `/sim`), agents, skills, §4 stubs |
| 0.5 Sim rig | sim + reviewer | Closed-loop tracking of a scripted face entirely in sim; scenario suite green |
| 1 Firmware bench | firmware | Smooth motion from serial angles; limits + idle scan verified (servo rail power) |
| 2 Vision standalone | vision | Detection printing offsets ≥30fps, no servos |
| 2.5 Calibration | vision + reviewer | Full §3.5 on real hardware; profile committed |
| 3 Closed loop | vision + firmware + reviewer | Smooth tracking, no oscillation, PID on measured calibration |
| 4 Conversation bench | speech + llm | Wake→STT→LLM→TTS standalone, latency OK |
| 5 Full integration | all + reviewer | Concurrent processes; no tracking stutter during inference; fits 8GB (tegrastats) |
| 6 Recognition + ambient | vision + llm + speech | Greets returning person; auto-enrolls new face; ambient context informs a reply |
| 7 Shell-swap drill | user + docs-scribe | §4.5 on a real printed shell, zero code edits |

0.5 ∥ 1 ∥ 2 after 0; 4 anytime after 0; 2.5 needs 1+2, user at bench.
**Sim-first:** every feature passes its sim scenario before touching
hardware.

## 6. Agreements + ops

- Read the domain skill before coding in it, every time.
- Firmware: `machine.PWM` @50Hz, `mpremote` deploy, allocation-free loop
  (pre-built buffers — GC must not stutter servos).
- Cross-module changes → `integration-reviewer`. Hardware-in-the-loop
  steps: agents prep code + commands, user runs and reports.
- Standing constraints, checked each phase: tracking non-blocking, two
  isolated power rails, local-only LLM, 8GB budget.
- Boot: systemd starts both processes with the profile, from battery, no
  keyboard. Health: IPC heartbeats + watchdog; ESP32 idle-scan covers
  "Jetson died". Thermal: log `tegrastats` (throttle = dropped FPS before
  errors). Battery: no telemetry — record observed runtime, plan swaps.
  Kill switch: physical, on the servo rail.

## 7. Next actions

1. `git init`; commit spec, BOM, this plan.
2. Create agents (§2) + skills (§3).
3. Scaffold tree + `/profiles` + `/sim`; stub the §4 contract modules.
4. Kick off Phases 0.5, 1, 2 in parallel.

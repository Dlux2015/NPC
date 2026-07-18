# NPC (Nomadic. Personal. Companion.) — System Reference

Top-to-bottom documentation of what the system is, how the pieces fit,
what the hardware can and cannot do, and the guarantees that keep the
robot functional and responsive when things go wrong. Written 2026-07-06
after the full software stack was live-verified on the dev PC.

**Reading map** — this document vs. the others:

| Doc | What it's for |
|---|---|
| `ORCHESTRATION.md` | Plan of record: phases, contracts, agents. Deliberately token-lean. |
| `STATUS.md` | Session handoff: current state, decisions log, env quirks, open items. |
| **`docs/SYSTEM.md`** (this) | The system explained: architecture, capability envelope, safety/responsiveness design. |
| `hardware/budget_analysis.md` | Power + RAM sufficiency numbers for the robot build. |
| `hardware/power_system.md` | DeWalt battery rework rationale + shopping list. |
| `hardware/emote_display.md` | The eye: hardware proposal + implemented software v1. |
| `hardware/chassis_design.md` | Chassis material + base/neck/head structural split, weight/torque budget. |

---

## 1. What the system is

NPC is a convention robot head: a pan/tilt camera platform that finds
and follows faces, recognizes people it has met (with their consent),
holds spoken conversations through a local LLM, and shows its mood on a
round red robot eye. Everything runs locally — no cloud, ever.

Two computers share the work:

- **Jetson Orin Nano Super 8GB** ("the brain"): vision, speech, LLM,
  identity, the eye. Runs at 25W (`nvpmodel -m 1`), headless.
- **ESP32-S3** ("the spine"): the 50Hz servo control loop. Speaks a
  tiny line-based serial protocol; owns its own safety behaviors
  (soft limits, idle scan) so it stays sane even if the brain dies.

The **dev PC** (Windows, RTX 4090) is not a third computer in the
design — it is a *stand-in shell* for the brain, selected purely by
profile (`CBOT_PROFILE=dev-pc`). Same product code, different devices.

## 2. Process model and dataflow

Three independent OS processes on the brain + the microcontroller:

```
             ┌─────────────────────────────  Jetson / dev PC  ─────────────────────────────┐
             │                                                                             │
  camera ──▶ │  VISION PROCESS                     CONVERSATION PROCESS                    │
             │  vision/tracking.py                 conversation/pipeline.py                │
             │  ┌──────────────────────┐           ┌────────────────────────────┐          │
             │  │ 30fps frame loop     │           │ wake (VAD/wakeword/PTT)    │ ◀── mic  │
             │  │  YuNet detect (CPU)  │           │  └▶ DirectedSTT (whisper)  │          │
             │  │  TargetTracker       │           │      └▶ LLM (GPU)          │          │
             │  │  PID → serial target │           │          └▶ TTS → speaker  │ ──▶ spkr │
             │  ├──────────────────────┤           │ ambient (duty-cycled)      │          │
             │  │ ~1Hz recognition     │           └────────────┬───────────────┘          │
             │  │ worker thread:       │                        │                          │
             │  │  SFace embed (CPU)   │                        │                          │
             │  │  people.db match     │                        │                          │
             │  └──────────┬───────────┘                        │                          │
             │             │      run/<profile>/state.json      │                          │
             │             └────────────▶  shared/ipc.py  ◀─────┘                          │
             │                                  ▲                                          │
             │  DISPLAY PROCESS                 │                                          │
             │  display/emote.py  ──────────────┘   (read-only consumer)                   │
             │   expression state machine → red eye renderer → window / SPI panel          │
             └────────────────────────────────────────────┬──────────────────────────────┘
                                                          │ USB serial, "P:<deg> T:<deg>\n"
                                                   ┌──────▼──────┐
                                                   │  ESP32-S3   │  50Hz easing loop,
                                                   │ firmware/   │  enforced limits,
                                                   └─────────────┘  idle scan on silence
```

Processes never import each other. They meet only at four contracts
(§3). Any of them can die and restart without corrupting the others —
that isolation is a safety feature, not an accident.

## 3. The four contracts (ORCHESTRATION §4)

1. **Serial** (`shared/serial_protocol.py`) — the ONLY way anything
   talks to the ESP32. Plain ASCII lines: `P:<deg> T:<deg>`, degrees
   only (never PWM), plus ping/pong and angle reports. Transport is
   pluggable (USB serial or the sim's socket) — same bytes either way.
2. **IPC** (`shared/ipc.py`) — one JSON file per profile
   (`run/<profile>/state.json`), atomic-replace writes with retry,
   mtime-cached reads. Keys: `person_present`, `person_in_range`,
   `person_id`, `new_person_seq` (counter; a bump means "someone new
   was just enrolled"), `actively_speaking`, `conversation_active`,
   `ambient_transcript`. Writers own disjoint key sets (vision owns
   `person_*`; conversation owns the rest). `ThreadedStateWriter`
   fronts the hot loop: publish() is an in-memory handoff, a background
   thread coalesces disk writes to ≤10Hz.
3. **Identity** (`shared/people.py`) — sole reader/writer of
   `people.db` (SQLite). Embeddings only, never images; `purge()` for
   privacy. Cosine matching with per-embedder thresholds, EMA refresh
   on confident matches so enrollments track appearance drift, and
   near-miss score logging so thresholds get tuned on data.
4. **Idle scan** — a behavioral contract: when vision has no face it
   simply stops sending serial targets; the ESP32 notices the silence
   and runs its own gentle scan pattern. "Jetson died" and "nobody
   here" degrade to the same safe behavior with zero coordination.

**Shell profiles** (`profiles/<name>/`) are the fifth, softer contract:
every shell-dependent value — camera, serial transport, calibration,
audio gains, TTS engine/voice, LLM path, persona text — lives in the
boot-selected profile. Code never branches on "which shell is this";
swapping shells (or the dev PC standing in for the robot) is a config
change, never a code change.

## 4. Module tour

### `vision/` — find, follow, recognize
- `detector.py` — pinned API: `detect(frame_bgr) -> [(x,y,w,h,score)]`.
  YuNet (`cv2.FaceDetectorYN`) for real cameras; `SyntheticDetector`
  wraps ground truth for sim/tests. CPU-only by rule — the GPU belongs
  to the LLM.
- `tracking.py` — the frame loop (`TrackingApp.step()`): detect →
  `TargetTracker` (holds one face ≥3s; no crowd-snapping) → PID (on
  *measured* deg/px from calibration.json; refuses to run without it)
  → serial target. Hard rules enforced here: the frame loop never
  blocks, makes zero filesystem calls, and never touches people.db —
  recognition runs on a dedicated worker thread at ~1Hz, fed crops via
  a non-blocking handoff that drops frames rather than ever queuing.
  `run_forever(idle_hz=...)` relaxes the frame rate when nobody has
  been seen for a while (power; §7).
- `recognition.py` — `SFaceEmbedder`: landmark-aligned SFace features
  (internal YuNet pass on a padded crop → `alignCrop()`; resize
  fallback if no landmarks). `SFACE_MATCH_THRESHOLD` documents its own
  tuning history; `make_embedder()` degrades loudly to None (tracking
  still works, recognition inert) when the model file is absent.
- `calibrate.py` — the §3.5 interactive calibration suite; writes the
  profile's `calibration.json`. Every consumer refuses to run without
  one; there are no hardcoded pixel→degree fallbacks anywhere.

### `conversation/` — listen, think, speak
- `wake.py` — three wake paths, all optional per profile: wake word
  (openWakeWord), push-to-talk, and `face_speech` (person in range +
  speech onset — the "walk up and talk" mode). Skips all mic work
  while `actively_speaking` (never hears the robot itself); disables
  paths whose backends are unavailable rather than crashing.
- `stt.py` — `DirectedSTT`: Silero-VAD-endpointed capture → faster-
  whisper. Bounded by `max_s` of *audio time* so no fake or hung mic
  can spin it forever. Silero's real model needs ≥512-sample chunks —
  `frame_ms` defaults to 32 everywhere for exactly that reason.
- `llm.py` — `LocalLLM` (llama.cpp, model resident for process
  lifetime, GPU-offloaded, `MAX_REPLY_TOKENS` cap on every generation)
  and `MockLLM` (scripted, for tests/sim/missing-model). `make_llm()`
  picks loudly, never silently.
- `pipeline.py` — `ConversationPipeline`: the turn-taking loop that
  owns persona/history/identity. Everything it touches is an injected
  pinned interface, which is why demos can swap mics for consoles and
  the sim can swap everything. Rolling history trimmed to a
  conservative char budget under the 2K-token context cap; window
  resets when the person leaves; name capture ("my name is X") writes
  through people.py only. `run_forever()` survives failed turns (§6).
- `tts.py` — `Speaker` + synthesizers (Kokoro-82M primary on dev-pc,
  Piper fallback, SAPI last resort — the chain in
  `demo_talk.make_tts_synthesizer`). `say_stream()` publishes
  `actively_speaking` before the first sample and clears it only after
  the last drains — one half-duplex window per reply, exception-safe.
- `ambient.py` — duty-cycled background listening into a rolling
  in-memory 60s transcript (never persisted); pauses entirely while
  the robot speaks; lowest priority by design.

### `display/` — the eye
- `expressions.py` — pure-logic state machine: existing IPC keys →
  `idle / alert / listening / talking / surprised / neutral`, with
  one-shot surprise on `new_person_seq` bumps and `neutral` on
  unreadable IPC (degrade, never crash).
- `emote.py` — the red robot eye: glowing core-as-pupil (wanders within
  a per-expression radius) + halo ring, rendered on a virtual 240×240
  round panel. Window backend today; the robot swaps in an SPI panel
  backend with the same expression logic. Renders at half rate while
  idle (power).

### `shared/` — the contracts (see §3)

### `firmware/` — the spine
MicroPython on the ESP32-S3. The 50Hz loop is allocation-free (pre-built
buffers; GC must never stutter a servo), wrap-safe on `ticks_ms`, sleeps
out each tick's remainder. Easing/limits live in pure-Python
`firmware/easing.py`, imported by BOTH the board and the sim — one
implementation, never two.

### `sim/` — the digital twin
Virtual ESP32 (real protocol over a socket, real easing code), virtual
world (camera view = pan/tilt-driven crop of a panorama), scenario
suite in `sim/scenarios/` that closes the full detect→PID→serial→servo
loop in software. Every feature passes a sim scenario before hardware —
this is the project's most load-bearing rule, and it has caught real
bugs (sign inversion, PID gains, a threading contract violation, the
Windows IPC race).

### Demos (`conversation/demo_*.py`, `sim/demo_*.py`)
Dev-PC front-ends that drive the REAL pipeline with adapters only at
physical edges (console instead of mic, PTT window, webcam wrapper).
`demo_friend.py` is the full experience: webcam recognition + walk-up
wake + consent-gated enrollment + LLM chat. Its dev-only adapters
(`EchoGuardSTT` mic-backlog drain, listen-cue chirp, `_BufferedLLM` GPU
serialization) each document which real-robot mechanism replaces them.

## 5. Hardware capability envelope

### The two machines

| | Dev PC | Jetson Orin Nano Super 8GB |
|---|---|---|
| GPU | RTX 4090, 24GB dedicated VRAM | integrated, **shares the 8GB with everything** |
| Role | Development twin of the robot brain | The actual robot brain |
| LLM (Llama-3.2-3B Q4) | ~109 tok/s measured, first token 0.13s | ~29 tok/s (benchmarked figure of record) |
| TTS | Kokoro-82M on GPU: ~41× realtime (0.13s/sentence) | Piper on CPU (Kokoro would be CPU-only there — bench before adopting; GPU belongs to the LLM) |
| STT | whisper base int8, CPU | whisper base/tiny int8, CPU (GPU bursts allowed only because STT and generation never overlap) |
| Vision | YuNet + SFace, CPU | same, CPU — non-negotiable (GPU = LLM) |
| Headroom | Vast (~10% utilized) | **~1.5–2GB RAM margin with everything resident** (see budget_analysis.md) |

### What is sized to what — and why it matters

Every model choice in this repo is sized to the *Jetson*, not the dev
PC: 3B LLM, whisper base, SFace, Kokoro-82M/Piper, 2K context. The dev
PC could trivially run an 8–14B LLM, whisper-medium, and heavier face
models — but then the dev PC would stop *predicting the robot*. If that
divergence is ever wanted, the profile system is the sanctioned
mechanism: a `dev-pc-max` profile with different model paths, leaving
`dev-pc` as the honest twin. Zero code changes either way.

### The robot's ceiling, honestly

The Orin Nano 8GB design is close to its ceiling by intent:
- RAM is unified — the LLM's GPU allocations and every CPU process
  drain one pool. The full-stack estimate is 4.8–6.0GB of ~7.4GB
  usable. The mitigation ladder if reality lands high: whisper
  base→tiny, KV cache f16→q8, and ultimately the sanctioned 1–1.5B
  fallback model. Steady-state swap/zram use = budget failed.
- The GPU-belongs-to-the-LLM rule is what keeps replies fast; it is
  also why TTS stays on CPU and vision never gets CUDA. Breaking it
  for a "nicer" component makes everything else worse. (The dev PC
  demo proved the point involuntarily: two CUDA runtimes overlapping
  in one process caused hard native crashes until serialized.)
- Meaningfully more capability on the robot = different hardware
  (Orin NX 16GB class), not tuning.

## 6. Responsiveness and survivability guarantees

The design principle, in one line: **the robot must keep looking at
you, keep answering, and keep failing loudly-but-partially — a dead
subsystem may cost a feature, never the robot.**

### Hard latency budgets (what bounds each interaction step)

| Step | Bound | Mechanism |
|---|---|---|
| Face lost → servo silence | 1 frame | vision stops sending; ESP32 owns idle scan |
| Frame loop iteration | never blocks | no disk/LLM/network in step(); IPC via in-memory publish; recognition crops dropped if worker busy |
| Wake poll | ~50ms cycle | bounded mic frame reads; VAD only when in range |
| Utterance capture | `max_s` of audio (10s default) | counted in audio time, immune to clock/mic tricks |
| LLM reply | `MAX_REPLY_TOKENS` (220) | explicit cap on every generation — a runaway costs seconds, not minutes |
| TTS half-duplex window | exactly the reply duration | `actively_speaking` set before first sample, cleared in `finally` |
| IPC write | ≤10Hz, off-thread | ThreadedStateWriter coalescing; event-driven (no idle polling) |
| Eye reaction to state | ≤~100ms | 10Hz IPC cadence + 24–30fps render |

### Degradation ladder (what happens when a piece is missing/broken)

| Failure | Behavior — by design |
|---|---|
| Jetson dies / vision stops sending | ESP32 idle-scans on serial silence; servos stay inside enforced limits |
| Calibration file missing | tracking REFUSES to start (loud) — never guesses pixel→degree |
| SFace/YuNet model file missing | recognition inert / detector fallback, tracking + conversation unaffected, warned loudly |
| One recognition tick throws | that crop is dropped, worker thread survives, logged (`tracking._recognition_worker`) |
| One conversation turn throws | logged with backoff, next wake starts fresh; `conversation_active` cleared in `finally` (`pipeline.run_forever`) |
| TTS sink dies mid-reply | `actively_speaking` still cleared (exception-safe window) so wake/ambient never deadlock |
| LLM package/model missing | loud fallback to MockLLM (dev) — never a crash, never silent |
| TTS engine missing | Kokoro → Piper → SAPI chain, each step printed |
| IPC file unreadable/corrupt | readers serve last good cache; eye shows `neutral`; writers retry the Windows replace race (50×2ms) |
| Person leaves mid-conversation | history window resets; `person_id` cleared on departure so the next visitor is never mis-greeted |
| Whisper hears nothing / gibberish | turn ends on VAD timeout; `min_speech_s` guards blips; nothing stored |
| Consent answer missed | nothing enrolled (fail-closed on privacy); asks again only after the person leaves |

### The "never" list (invariants, all test-pinned)

- The vision frame loop never blocks, never touches disk, never holds
  the recognition lock while iterating.
- The firmware loop never allocates in steady state.
- On the robot, the GPU is never contended: STT bursts, generation,
  and (CPU) TTS are strictly sequential by construction.
- `people.db` is never written by anything but `shared/people.py`;
  images are never stored, only embeddings; enrollment without consent
  never happens in the friend flow.
- The robot never hears itself: one `actively_speaking` window spans
  every reply, and mic backlog from the speaking period is drained.
- Failures are never silent: every fallback logs or prints which
  backend it chose and why.

## 7. Power posture (summary — numbers in budget_analysis.md)

Biggest-to-smallest levers, and where each stands:
1. **LLM residency + GPU exclusivity** — model loads once; the 25W
   nvpmodel cap bounds the SoC by construction.
2. **Always-on detection** — the largest *constant* draw; `--idle-fps`
   relaxes detection to 10fps after 10s alone, snapping back on the
   first frame with a face.
3. **Inference gating** — VAD only when someone is in range; ambient
   duty-cycled; recognition at 1Hz off the hot path.
4. **No idle polling** — the IPC writer sleeps until woken; the eye
   halves its frame rate when idle; firmware sleeps out each tick.
5. Servo rail excursions (stall) are a hardware-phase concern — see
   budget_analysis §2 (8A-class buck recommendation).

## 8. Testing strategy

- **Sim-first**: every feature lands with a scripted scenario before it
  meets hardware. The digital twin closes the full control loop in
  software (`sim/scenarios/`), and the whole-robot e2e test runs the
  REAL TrackingApp + REAL ConversationPipeline against one real IPC
  file and one real people.db.
- **Fakes only at physical seams**: cameras, mics, speakers, serial,
  and the LLM have documented injection points; contract-owning code
  (people.py, ipc.py, serial protocol, easing) is always exercised for
  real.
- **Live verification** is its own tier: the suite passing does not
  close a feature that has a physical edge (the mic privacy toggle,
  MME-vs-WASAPI attenuation, Silero's 512-sample minimum, and the CUDA
  runtime clash were ALL invisible to the suite and found live).
  STATUS.md records what has actually been proven on real devices.
- Suite size at this writing: 235 passed, 2 skipped (the skips are
  real-model tests that need gitignored weights). Lint bar:
  `ruff check --select F,E7,E9` clean.

## 9. Runbook

Dev PC (all commands from repo root; venv interpreter mandatory for
real LLM/TTS/VAD):

```
A:\code\NPC\.venv\Scripts\python.exe -m pytest -q          # the suite
...python.exe -m conversation.demo_friend                    # the full experience
...python.exe -m display.emote                               # the eye (2nd window)
...python.exe -m conversation.demo_talk --text               # typed chat
...python.exe -m conversation.demo_talk --selfcheck          # audio sanity
...python.exe -m display.emote --demo                        # eye expression reel
python -m vision.calibrate --profile sim --auto              # sim calibration
```

Robot (Phase 5+): systemd starts the vision and conversation processes
with `CBOT_PROFILE=<shell>`; the display process joins them; health is
IPC heartbeats + the ESP32's autonomous idle scan. Environment quirks
and device gotchas live in STATUS.md ("Dev-PC environment facts").

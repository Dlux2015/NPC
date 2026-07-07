# CBot — session handoff / status

Last updated: 2026-07-06 (v1.1 spec era). Read this + `ORCHESTRATION.md`
(plan of record) before resuming work. Spec: `unzipped/robot_build_spec.md`;
BOM: `unzipped/robot_bom_tracker_v2.html`; power detail:
`hardware/power_system.md`.

## Where the project stands

All software-provable work is DONE and proven in the digital twin.
Suite: **157 passed, 2 skipped** (`python -m pytest -q`). Phases 0, 0.5,
1(logic), 2, 4 complete + integration-reviewed + e2e-proven. Remaining
work needs hardware: Phase 1 bench (flash ESP32, measure rail current
incl. servo stall — v1.1 exit criterion), 2.5 (real calibration),
3 (closed loop), 5 (Jetson memory/thermal), 6 (recognition — the SFace
embedder is now REAL and wired: `vision/recognition.py`, model at
`vision/models/face_recognition_sface_2021dec.onnx` (gitignored, 38MB,
opencv_zoo), `tracking.main()` uses it when present, threshold 0.363
for unaligned crops; remaining Phase 6 work = Jetson bench + ambient),
7 (shell-swap drill).

## Decisions log (don't re-litigate)

- LLM: llama.cpp + CUDA, Llama-3.2-3B-Instruct Q4_K_M, 25W mode, ctx ≤2K.
  GPU=LLM, vision=CPU (YuNet).
- Firmware: MicroPython; easing/limits in pure-Python `firmware/easing.py`
  imported by BOTH board and sim (never re-implement).
- IPC event is `new_person_seq` (counter), not `new_person`.
- Power (spec v1.1, adopted 2026-07-05): DeWalt 20V DCB205 ×2, one per
  rail, fused adapter docks, 19V/5A + 6.5V/5A bucks, ~15V low-voltage
  alarms MANDATORY (packs have no cutoff). Prices unconfirmed until
  purchase.
- Detection: YuNet primary (model at
  `vision/models/face_detection_yunet_2023mar.onnx`, needs cv2 ≥ 4.8);
  demo falls back to Haar automatically if YuNet can't load.
- Tracker locks ONE face by design (anti crowd-snapping); detector sees
  all — demo shows gray boxes for all, green for locked.

## What runs right now (dev PC)

```
python -m pytest -q                          # 213 tests
python sim/demo_full_robot.py                # whole-robot e2e story
python sim/demo_visual.py                    # interactive virtual world
python sim/demo_visual.py --camera 0         # webcam: real YuNet tracking
python -m conversation.demo_talk             # live PTT voice chat (PC audio)
python -m conversation.demo_talk --text      # typed chat, no audio stack
python -m conversation.demo_friend           # webcam+mic: consent-gated
                                             # face enrollment + LLM chat
python -m vision.calibrate --profile sim --auto
```

For the REAL LLM (not MockLLM), use the repo venv's interpreter:
`A:\code\CBot\.venv\Scripts\python.exe -m conversation.demo_talk` —
llama-cpp-python lives only in `.venv` (see env facts below). Verified
2026-07-06: Llama-3.2-3B Q4_K_M on the RTX 4090 via the dev-pc profile,
~109 tok/s, first token 0.13s. Under system python, make_llm now falls
back loudly to MockLLM (no crash).

## Dev-PC environment facts (Windows 11, Python 3.10 @ C:\Python310)

- **cv2 pinned 4.10.0.84** — do NOT `pip install --upgrade opencv-python`:
  it drags numpy→2.x (breaks faster-whisper risk) and the numpy uninstall
  fails on a locked f2py.exe anyway. 4.10 = YuNet-capable + numpy 1.26.
- Audio installed: sounddevice, faster-whisper, **piper-tts (works on
  Windows!)** with pinned voice `profiles/dev-pc/voices/en_US-amy-low`,
  pyttsx3 (SAPI fallback), keyboard.
- **User's default Windows mic is a VB-Audio virtual cable (silence!)** —
  set `input_device` in `profiles/dev-pc/audio.json` to the real mic's
  name substring, or change the Windows default, before demo_talk works.
- C: drive was near-full (pip cache already purged once; ~2.9GB free).
  Put big files (GGUF) on A:, e.g. `A:\code\CBot\models\`.
- LLM replies are REAL under the venv interpreter (GGUF wired in the
  dev-pc profile, see below); MockLLM only under system python or if
  the GGUF file is removed.
- Webcam capture: CAP_DSHOW + 640x480 + BUFFERSIZE=1 (in `_Webcam`) —
  default MSMF buffering caused perceived lag.
- **LLM integrated (2026-07-06).** GGUF at
  `A:\code\CBot\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~1.9GB, from
  HF bartowski/Llama-3.2-3B-Instruct-GGUF; gitignored). Wired via
  `profiles/dev-pc/profile.yaml` `llm_model_path` (relative paths now
  resolve against repo root) + `llm_gpu_layers: -1`.
- **Repo venv `A:\code\CBot\.venv`** (`--system-site-packages`, so cv2/
  whisper/piper come from C:\Python310). Exists because C: has <1GB free
  and the user restricts changes to the CBot tree. Holds
  llama-cpp-python 0.3.4 (prebuilt cu121 wheel) + nvidia-cuda-runtime/
  cublas cu12 wheels; their DLLs (cudart64_12, cublas64_12,
  cublasLt64_12) were COPIED into `.venv\...\llama_cpp\lib\` — the wheel
  doesn't bundle them and there's no system CUDA toolkit. Re-copy if
  llama-cpp-python is ever reinstalled/upgraded. Dev PC GPU: RTX 4090.
- Shell PATH quirk: `python`/`git`/`curl` aren't on this machine's
  PowerShell PATH — use `C:\Python310\python.exe`, the venv's python,
  or Git Bash (which has git/curl).
- **TTS engine: Kokoro-82M, voice am_michael (2026-07-06)**, not Piper.
  `conversation/tts.py`'s `KokoroSynthesizer` wraps the `kokoro` pip
  package (hexgrad/Kokoro-82M, auto-downloaded from HF on first use into
  the default HF cache — not repo-local, unlike the LLM GGUF). Bundles
  its own espeak-ng via `espeakng-loader` — no system install needed.
  Chosen after a listening comparison against several Piper voices
  (including en_GB-alan-medium, tried for the JARVIS persona but read as
  flat/"depressed"): Kokoro sounds materially more natural at a similar
  (82M) parameter size. Speed: `.venv` torch is now **2.12.1+cu126**
  (CUDA; installed 2026-07-06 because CPU synthesis at ~1.1-2.2x
  realtime made replies audibly laggy) — Kokoro auto-selects cuda,
  measured ~41x realtime (0.13s/sentence) on the 4090. pip couldn't
  reach download.pytorch.org (old pip 21.2.4 TLS handshake failure) but
  curl could: wheels downloaded manually to `.venv/tmp/wheels/` and
  pip-installed from file. Jetson note: GPU belongs to the LLM there,
  so Kokoro-on-robot would be CPU-only — bench before adopting over
  Piper. `profiles/dev-pc/profile.yaml`: `tts_engine: kokoro` +
  `tts_voice`/`tts_lang_code`; Piper (en_GB-alan-medium) stays configured
  as the automatic fallback if `kokoro` isn't installed in whatever
  interpreter runs the profile (`make_tts_synthesizer` tries Kokoro
  first, then Piper, then SAPI — loud fallback, never silent).
- **Live audio input+output verified working end-to-end (2026-07-06)**:
  real mic -> Silero VAD end-pointing -> faster-whisper -> Piper TTS
  playback, through the actual product classes (MicStream/DirectedSTT/
  Speaker). Getting there required three real fixes, not just config:
  1. **Silero VAD's real model needs >=512-sample chunks at 16kHz**
     ("Input audio chunk is too short" otherwise) — every conversation/
     test injects a fake VAD, so this was never exercised until a live
     mic test hit it. Fixed: `frame_ms` default 30->32 in stt.py/wake.py/
     ambient.py (32ms = exactly 512 samples). Needs torch + torchaudio
     in `.venv` (installed from plain PyPI, not download.pytorch.org --
     that CDN TLS-handshake-fails from this network; PyPI's `torch`
     wheel is CPU-only here anyway, ~123MB).
  2. **Windows Settings -> Privacy & security -> Microphone -> "Let
     desktop apps access your microphone"** was OFF. Symptom was
     maximally confusing: MicStream opened with no error and returned
     correctly-shaped samples that were just ~all zero/noise-floor --
     looks exactly like a code bug, isn't one. Check this FIRST on any
     "mic captures silence" report.
  3. **MME (PortAudio's Windows default host API) delivered ~50x
     quieter capture than WASAPI for the identical physical device**
     (Brio 101) even after fixing #2. `resolve_device_by_name()` now
     prefers a WASAPI-hosted match/default over MME/DirectSound
     (`conversation/audio_dev.py`'s `_wasapi_hostapi_index`). WASAPI is
     stricter about sample rate than MME though (rejects 16kHz on a
     48kHz-native device outright) -- fixed by passing
     `extra_settings=sd.WasapiSettings(auto_convert=True)`
     (`_wasapi_extra_settings`) on every real stream open/play.
  New tests: `conversation/tests/test_vad.py` (real-model, skipped
  without torch) + WASAPI-preference cases in `test_audio_dev.py`.

## Recognition accuracy (2026-07-06, after first live demo_friend run)

Live testing confirmed the unaligned-crop SFace path **mixes people
up** (false matches) and misses returning faces. Fixes shipped:
- `SFaceEmbedder` now landmark-aligns via its own internal YuNet pass
  on the crop + `alignCrop()` (self-sufficient — the pinned detect()
  API is unchanged); falls back to resize-112 only when no face is
  found in the crop.
- Crops for the real embedder are padded 40% (`crop_face_padded`) so
  the landmark pass has context; sim/fake embedders keep tight
  `crop_face` (their deterministic embeddings depend on those pixels).
- `SFACE_MATCH_THRESHOLD` raised 0.363 → 0.45: false-accept (greeting
  the wrong person by name) is much worse than false-reject (re-asking
  the consent question). Tune with real faces at Phase 6 bench.
- `run/dev-pc/people.db` purged — pre-alignment embeddings have a
  different score distribution and must not be matched against.
Also from that run: consent question shortened (users answer WHILE the
robot talks; the echo-guard drain was eating early answers).

**Live-verified by the user (5 rounds, 2026-07-06), all green by round
5:** consent flow (heard "Sure", enrolled), name capture ("my name is
David" → people.set_name), cross-session recognition (fresh process
recognized him, skipped consent, greeted by name), multi-conversation
wake/timeout cycling, clean q exit. Fixes that got it there, in order:
listen-cue chirp after the echo-guard drain (users can't know when
listening starts otherwise — two runs lost the "yes" to this),
min_speech_s 0.2→0.1 for one-word answers, and _BufferedLLM. That last
one solved the mystery exit code 5 = low byte of 0xC0000005 native
access violation: llama.cpp's bundled CUDA runtime and torch-CUDA
(Kokoro) overlapping on one GPU in one process via sentence-streaming.
Demo now drains LLM generation fully before TTS (<1s at ~110 tok/s).
Robot proper is immune by design (Piper=CPU, GPU=LLM only) — but this
is a hard warning for any future GPU-TTS-on-Jetson idea.

## Consent-gated enrollment (demo_friend, 2026-07-06)

`conversation/demo_friend.py`: webcam recognition + mic conversation in
one process — walk up, talk (face_speech wake), and if unrecognized the
robot ASKS "can I be your friend?" before enrolling (auto_enroll=False
+ `TrackingApp.pop_unknown_embedding()`; a yes enrolls + bumps
new_person_seq so the pipeline greets-as-new; "my name is X" then names
the record; declines stick until the person leaves). This intentionally
softens the spec's silent auto-enroll — if adopted for the robot it
must move into ConversationPipeline with an IPC enroll handshake
(vision/conversation are separate processes there; orchestrator owns
that contract change). EchoGuardSTT (demo-side) drains mic backlog
after TTS so the robot doesn't transcribe its own voice tail — real
robot solves this properly at calibration step 7.

## Known open items (priority order)

1. CI: pytest on push (private remote EXISTS as of 2026-07-06:
   https://github.com/Dlux2015/CBot, origin/master, push via gh auth).
   Note: CI runners have no GPU/GGUF — suite already mock-based, fine.
2. User hasn't yet run demo_talk (live PTT mode) against the REAL LLM
   with real voice -- LLM integration and live mic/TTS are each verified
   working (2026-07-06) but not yet exercised together in one session.
   Needs the venv interpreter + torch/torchaudio (see mic fixes above).
3. User hasn't yet tested multi-face display in webcam demo (built,
   committed, untested by user).
4. When DeWalt parts are purchased: docs-scribe flips BOM v2 power rows
   to confirmed with real prices.
5. Optional Phase 3+: ESP32 ADC monitors servo rail voltage (spec v1.1
   battery ops).
6. Emote display proposal (2026-07-06): round GC9A01 LCD "eyes" driven
   from the Jetson SPI header — `hardware/emote_display.md`, BOM rows
   added unconfirmed. Expression state machine is sim-provable now;
   panels bench at Phase 5. Not yet adopted into the spec.
7. Power+RAM budget analysis done (2026-07-06):
   `hardware/budget_analysis.md`. Verdicts: Jetson rail fine (~33W peak
   vs 95W); RAM fits with ~1.5-2GB headroom (CUDA context = biggest
   unknown, unified memory!); ONE flag — servo buck should be 8A-class,
   not 5A (dual stall 5-6A), Phase 1 measurement decides. BOM note
   updated. Doc lists the exact Phase 1/5 bench measurements to take.
8. Per-person memory (facts table in people.db via shared/people.py,
   extract at conversation end, inject at greeting) — discussed
   2026-07-06, deliberately NOT a graph DB (Neo4j rejected: ~1-2GB JVM
   vs 8GB budget). Phase 6+ enhancement, not yet designed in detail.

## Session history (git log has detail)

Phase 0 scaffold `3f32a44` → phases 0.5/1/2 `179449e` → review fixes
(8 findings + sign-inversion & PID-gain bugs the sim caught) `d2c0852` →
Phase 4 conversation `c214e2c` → full-robot e2e + 4 upstream fixes
`ad6af74` → visual viewer `ec05ec1` → webcam mode `f70072b`/`5140a6c` →
dev-PC live audio `80df349` → power rework doc `f9fcfa5` → spec v1.1
adopted `c426fef` → webcam smoothness/multi-face `1ecc2c4`.

## Working agreements that carried the project

Sim-first (features pass a sim scenario before hardware); contracts in
`shared/` are orchestrator-owned; delegations name exact files +
acceptance criteria; integration-reviewer (Opus) on cross-module changes;
hardware-in-the-loop steps are prepared by agents, run by the user.
User permission scope: full permission INSIDE A:\code\CBot only; asks
before env changes outside it.

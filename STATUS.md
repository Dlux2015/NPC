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
3 (closed loop), 5 (Jetson memory/thermal), 6 (real SFace embedder →
`TrackingApp(embed_cb=...)` seam), 7 (shell-swap drill).

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
python -m pytest -q                          # 173 tests
python sim/demo_full_robot.py                # whole-robot e2e story
python sim/demo_visual.py                    # interactive virtual world
python sim/demo_visual.py --camera 0         # webcam: real YuNet tracking
python -m conversation.demo_talk             # live PTT voice chat (PC audio)
python -m conversation.demo_talk --text      # typed chat, no audio stack
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

## Known open items (priority order)

1. CI: pytest on push (private remote EXISTS as of 2026-07-06:
   https://github.com/Dlux2015/CBot, origin/master, push via gh auth).
   Note: CI runners have no GPU/GGUF — suite already mock-based, fine.
2. User hasn't yet run demo_talk against the REAL LLM (integrated +
   smoke-tested 2026-07-06, but live voice chat needs the mic fix below
   and the venv interpreter).
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
7. Per-person memory (facts table in people.db via shared/people.py,
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

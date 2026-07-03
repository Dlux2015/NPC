---
name: audio-capture
description: Load before any work on mic input, wake word, VAD, or speech-to-text.
---
# Audio capture: mic, wake, STT

- Single forward-facing USB sound-card mic — no beamforming; servo/camera
  aiming provides directionality. Select device **by name**, never index.
  16kHz mono.
- **Directed mode** (someone addresses the robot): entered by openWakeWord
  wake word, push-to-talk fallback, or face-in-range + speech onset.
  Full STT → LLM reply. Latency: wake→transcript <2s for short utterances.
- **Ambient mode** (not addressed): VAD-gated, low-duty STT into a rolling
  ~60s **in-memory** buffer — never persisted. Used for LLM scene context
  and robot-name-mention detection. Lowest priority; must never starve
  directed mode or the vision process.
- STT: faster-whisper — small/base for directed, tiny for ambient.
  Silero VAD for utterance end-pointing.
- Mic gain, VAD threshold, wake threshold: read from the active shell
  profile's `audio.json` — never hardcoded (enclosure acoustics change
  them).
- Respect `actively_speaking` from IPC: suspend wake + ambient while the
  robot speaks (it must never transcribe itself).

---
name: speech-output
description: Load before any work on TTS or speaker output.
---
# Speech output: TTS

- Piper, fully local, one voice, version-pinned model file.
- **Sentence-streaming:** synthesize and play sentence 1 while the LLM is
  still writing — perceived latency beats total latency.
- Half-duplex reality (mic + speaker on one USB card): publish
  `actively_speaking` via `shared/ipc.py` before audio starts, clear after
  playback ends. This gates the mic path and lets vision damp head motion
  while talking.
- Output volume / EQ compensation from the active shell profile's
  `audio.json` — a closed printed head sounds different from the bench.

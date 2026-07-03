---
name: local-llm-conversation
description: Load before any work on llm.py, pipeline.py, or persona behavior.
---
# Local LLM conversation

- **Serving (decided Jul 2026, benchmarked):** llama.cpp built with CUDA
  (not Ollama), **Llama-3.2-3B-Instruct Q4_K_M** (~2.0GB, ~29 tok/s on the
  Orin Nano Super). Alternate: Qwen 2.5 3B Q4. Memory-tight fallback:
  1B–1.5B Q4. Jetson at 25W (`nvpmodel -m 1`), headless.
- Context cap ~2K tokens. Model loads once and stays resident — never
  reload per utterance.
- **GPU belongs to the LLM; vision runs on CPU.** Whisper GPU bursts are
  fine (STT and generation are sequential).
- **Local-only.** Any future cloud path: short timeout, silent fallback to
  local, never degrades the baseline.
- Persona: loaded from the active shell profile's `persona.md` — a
  convention character, 1–3 sentence replies, all-ages, graceful
  "didn't catch that". Never hardcode persona text.
- Identity-aware: read `person_id`/`new_person_seq` (counter, increments
  per auto-enroll) from IPC — greet returning people, notice new ones.
  Names learned in conversation are written back **only** via
  `shared/people.py`.
- Ambient transcript (from audio-capture) may be summarized into the
  prompt as *overheard* scene context — never treated as direct address.
- Conversation window resets when IPC reports person-absent.
- Test flow logic against a mocked model interface; profile real memory
  with the full stack running before sign-off.

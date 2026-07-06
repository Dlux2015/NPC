# Power + RAM budget analysis — full-spec build, all functionality

**Status: ANALYSIS (2026-07-06), datasheet/benchmark estimates.** Nothing
here is measured on real hardware yet — every number marked (est.) gets
replaced at the Phase 1 / Phase 5 bench gates, which this doc feeds.
Scope: spec v1.1 power (DeWalt packs, two rails) + 8GB Orin Nano Super,
with EVERYTHING running: tracking + recognition + directed STT + LLM
generation + TTS + ambient + emote displays (proposal).

**Verdicts up front**
- Jetson rail: **fine, large margin** (peak ~33W vs 95W buck capacity).
- Jetson pack runtime: ~3.5–5h per DCB205, hot-swap covers a full day.
- Servo rail: **energy fine, PEAK current is the one flagged risk** —
  dual hard stall can exceed a "5A" buck (details below). Fix is cheap.
- RAM: **fits with ~1.5–2GB headroom** on paper; the one big unknown is
  CUDA context overhead on Jetson (unified memory). Mitigation ladder
  exists; Phase 5 tegrastats audit remains the binding gate.

## 1. Power — Jetson rail (pack A → 19V/5A buck)

| Load | Sustained (est.) | Peak (est.) |
|---|---|---|
| Orin Nano Super, 25W mode (`nvpmodel -m 1`), SoC+carrier | 12–20W | 25W |
| NVMe (KingSpec 256GB) | 0.5–1W | 3W (burst writes) |
| CSI camera (IMX219) | 0.3W | 0.4W |
| USB sound card + 5W speaker | 0.3W | 3W (TTS at volume) |
| 2× GC9A01 displays + backlight (proposal, 3.3V header) | 0.4W | 0.6W |
| ESP32-S3 (if USB-powered from Jetson; serial link) | 0.3W | 0.8W |
| **Rail total** | **~14–22W** | **~33W** |

- Buck capacity 19V × 5A = **95W → ~3× headroom over peak**. No issue.
- Worst sustained case is LLM generation + Whisper burst + tracking:
  the 25W nvpmodel cap bounds the SoC by construction — the power mode
  is doing the budgeting for us.
- Runtime: 100Wh pack × ~88% buck efficiency ≈ 88Wh ÷ 18–22W ≈
  **~4–5h typical, ~3.5h worst case** per pack. 2–3 packs + hot-swap =
  full convention day. (Refines power_system.md's combined ~3h guess.)

## 2. Power — servo rail (pack B → 6.5V/5A buck)

| Load | Sustained (est.) | Peak (est.) |
|---|---|---|
| 2× DS3218MG, tracking motion (intermittent) | 3–8W (0.5–1.2A) | — |
| 1× DS3218MG hard stall | — | ~2.5–3A |
| 2× DS3218MG simultaneous hard stall | — | **~5–6A** |

- **⚠ THE flagged risk:** DS3218MG stall spec varies 2.5–3A across
  datasheets/clones. Dual simultaneous stall = 5–6A at the buck's exact
  5A rating → brownout/foldback on the rail that also feeds nothing
  else (good — isolation contains it, servos just sag). Mitigations,
  cheapest first:
  1. Buy an **8A-class buck** for the servo rail instead (~$2–5 more).
     Recommended; update BOM row note at purchase time.
  2. Firmware already helps: easing limits slew, soft limits prevent
     driving into hard stops (the usual stall cause), and idle scan is
     low-torque. Sustained dual stall should never happen in operation.
  3. Phase 1 exit criterion already requires measuring rail current
     **including stall** — this analysis makes that measurement decide
     the buck, not just record it.
- Energy: tracking duty averages a few W → 88Wh ÷ ~5W ≈ **servo pack
  outlasts the day** (>10h); swap it opportunistically or not at all.
- Transients: servo direction reversals cause current spikes;
  the standard 470–1000µF electrolytic at the servo power header is
  cheap insurance (add to Misc wiring at purchase).

## 3. RAM — 8GB Orin Nano, unified memory, everything resident

Physical 8GB LPDDR5; ~7.4GB visible/usable to Linux (est., firmware
carveout). **Unified memory: GPU allocations and CPU processes drain the
same pool** — "GPU belongs to the LLM" partitions *compute*, not RAM.

| Consumer | Est. resident | Notes |
|---|---|---|
| Ubuntu headless + systemd + drivers | 700–900MB | headless already saves ~800MB vs GUI (decided) |
| CUDA driver context (first CUDA process) | 600–900MB | **biggest unknown**; Jetson CUDA contexts are notoriously heavy |
| LLM weights, Llama-3.2-3B Q4_K_M | ~2,000MB | resident for process lifetime (contract) |
| LLM KV cache, 2K ctx, f16 | ~230MB | 28 layers × 8 KV heads × 128 dim × 2B × 2 (K+V) × 2048 tok |
| llama.cpp compute buffers | 150–300MB | |
| faster-whisper base int8 + CT2 runtime | 400–600MB | inside conversation process |
| Piper TTS + voice | 150–250MB | inside conversation process |
| Conversation process Python overhead | 150–250MB | pipeline, wake, VAD, IPC |
| Vision process (cv2 + YuNet + SFace + numpy) | 350–500MB | CPU-only by contract |
| Emote process (proposal) | 50–100MB | framebuffers are ~115KB each, noise |
| **Total** | **~4.8–6.0GB** | vs ~7.4GB usable |

**Headroom: ~1.4–2.6GB** — enough for page cache + spikes, *if* the
estimates hold. What decides which end of the range we land on:

1. **CUDA context size** (±300MB swing). Measure first on-device thing.
2. Whisper size: `base` → `tiny` for directed too saves ~300MB (skill
   already specs tiny for ambient).
3. KV cache q8_0 instead of f16 halves it (~115MB saved, quality cost
   negligible at this scale).
4. The decided fallback ladder if it genuinely doesn't fit: Qwen 2.5
   3B Q4 (similar size) → **1–1.5B Q4 (saves ~1GB+)** per §3.4. Losing
   some reply quality beats swapping/OOM — tracking must never stutter.
5. zram is on by default on Jetson — treat it as spike absorber, not
   budget. If steady-state dips into zram, we've failed the budget.

**What does NOT fit and was already excluded, confirming those calls:**
GUI desktop (~800MB), Neo4j/JVM memory graph (1–2GB, rejected
2026-07-06), a 7–8B model (~4.5GB+ weights alone), Redis "until it earns
its footprint".

## 4. Worst-case concurrency scenario (the Phase 5 acceptance moment)

Person mid-conversation: tracking at 30fps (CPU) + LLM generating (GPU)
+ TTS streaming sentence 1 (CPU) + `actively_speaking` gating mic +
emote "talking" animation (CPU, 15fps SPI) + recognition embed deferred
(low-frequency, off hot loop). Whisper is sequential with generation by
design (GPU bursts, decided) — STT and LLM never peak together.

- Power: bounded by 25W nvpmodel cap + ~8W peripherals → within rail.
- RAM: the table above IS this scenario (everything resident — nothing
  loads per-utterance by contract).
- CPU: the real contention point (vision 30fps + CT2 int8 whisper
  bursts + Piper on 6 cores) — not this doc's scope, but tegrastats at
  Phase 5 captures it in the same session.

## 5. Bench measurements this analysis needs (feeds Phase 1 / 5 gates)

Phase 1 (servo bench): [ ] single-servo stall current @6.5V
[ ] dual-stall current [ ] buck behavior at/over rating (sag? foldback?
thermal?) [ ] reversal transient with/without bulk cap
→ then buy/keep the right buck (see §2).

Phase 5 (Jetson integration): [ ] `free -m` after boot headless
[ ] after CUDA init [ ] after full stack up [ ] tegrastats RAM/power
logged through a scripted worst-case conversation (§4) [ ] zram usage
must be ~0 steady-state [ ] record real pack runtime both rails
→ replace every (est.) above with the measured number.

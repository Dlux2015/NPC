# Chassis design — material + structural layout

**Status: PROPOSAL (2026-07-18).** First pass at "what holds what" and
what to print it from. Nothing built yet; numbers below are datasheet/
estimate-grade like the rest of `hardware/`, confirmed at the Phase 2.5/7
bench gates (real shell, real weigh-in). Printer assumed: Bambu Lab X1
Carbon (enclosed, stock hardened nozzle, AMS optional).

## 1. Material

**PETG is the default for the whole chassis.** Tougher than PLA (won't
crack when a con-goer bumps it), enough heat resistance (~75°C HDT) for
a shell that sits next to the Jetson's vent duct, prints clean on the
X1C's textured plate without fighting warping the way ABS does.

| Part | Material | Why |
|---|---|---|
| Structural: neck bracket mounts, standoff bosses, battery-bay hatch, any screw boss that gets re-threaded on shell swaps | **PETG**, or **PETG-CF** if the pan/tilt mount plate needs extra stiffness | Toughness against drops/repeated hatch cycles; CF adds stiffness so the camera doesn't oscillate under servo torque — X1C's stock hardened nozzle handles CF, no upgrade needed |
| Cosmetic shell skin / snap-fit panels (non-load-bearing) | PETG or PLA | Not load-bearing; PLA prints crisper cosmetic detail faster if that matters more than toughness here |
| Any shell stored in a car/sun between events | **ASA** instead of PETG | Better UV stability, similar toughness, prints fine in the X1C's enclosure |
| Avoid: PLA on anything load-bearing or near the Jetson vent | — | Softens well below con-hall + Jetson-exhaust temps, brittle at repeatedly-swapped screw bosses |

Dry the PETG spool before structural prints (AMS drying or a dry box) —
it's hygroscopic, and layer adhesion on the parts actually holding servo
torque is exactly what wet filament wrecks first.

## 2. Architecture: stationary base + neck + swappable head shell

The robot reads as "a head on a stand," but mechanically it's three
sub-assemblies. This split — not everything in one sealed shell — is
the load-bearing decision this doc makes:

```
   ┌─────────────────────┐
   │   HEAD (swappable)   │  pans + tilts as one rigid unit
   │  Jetson + NVMe        │
   │  CSI camera            │
   │  ESP32-S3               │
   │  USB sound card + spkr   │
   │  eye display(s), optional │
   └──────────┬───────────────┘
              │  neck: DS3218MG pan/tilt bracket kit (BOM)
              │  crossing wires: Jetson power pair (2) +
              │  Jetson↔ESP32 serial/USB cable (1) — service loop
   ┌──────────┴───────────────┐
   │   BASE (stationary)         │
   │  2× DeWalt DCB205 packs (~500g ea) │
   │  2× buck converters (19V/5A, 6.5V/5A+) │
   │  2× low-voltage cutoff/alarm modules │
   │  physical kill switch (servo rail)    │
   └────────────────────────────┘
```

**Why the Jetson rides in the head instead of staying in the base:**
keeping camera, mic/speaker, and (optional) eye displays all wired
directly to the Jetson means those connections stay fully internal and
rigid — no ribbon or SPI cable has to flex across the pan/tilt joint.
Only two things cross the neck: the Jetson's DC power pair, and one
serial/USB cable to the ESP32. Both are thin, flexible wire types well
suited to a service loop through the bracket. *(This supersedes the
"ribbon routing from head to Jetson" line in `hardware/emote_display.md`
§"Architecture" — that was written before this split was settled. The
GC9A01 eye ribbons are now fully internal to the head, alongside the
Jetson they connect to.)*

**Why the batteries stay in the base, not the head:** a DCB205 pack is
~500g — two of them would add ~1kg to a tilting payload for no
mechanical benefit, and would fight the "pack weight low/central"
constraint already in `hardware/power_system.md`. Keeping them
stationary in the base does double duty: it's the natural low-ballast
that keeps the whole prop stable on a table at a crowded convention,
no separate weighted foot needed.

**ESP32 placement:** lives in the head (short, direct PWM wire runs to
both servos matter less than keeping the Jetson↔ESP32 link to one
crossing cable). Its own servo signal+power wires still have to reach
down through the neck to the pan servo body (stationary, mounted to the
base's top plate) and out to the tilt servo (mounted on the pan
sub-plate) — bundle these with the Jetson power pair in the same
service loop.

## 3. Weight/torque budget (confirms DS3218MG is still the right servo)

Head payload estimate (PETG shell walls, all internal hardware, no
batteries):

| Component | Est. mass |
|---|---|
| Jetson Orin Nano Super devkit (board + heatsink/fan) | ~200g |
| Arducam IMX219 + NVMe | ~20g |
| ESP32-S3 devkit | ~10g |
| USB sound card + 5W speaker | ~60g |
| Wiring, standoffs, fasteners | ~30g |
| PETG shell walls (mid-size head, ~2–3mm walls) | ~250g |
| GC9A01 eye(s), optional | ~20g |
| **Total** | **~590g, budget to ~700g** |

Required tilt-axis holding torque ≈ weight × horizontal offset from the
tilt pivot to the head's center of mass, ×2 for a dynamic-load safety
margin: 0.7kg × 5cm (compact-head design target) × 2 ≈ **7 kg·cm**.
DS3218MG is rated ~17–20 kg·cm stall (varies by clone/voltage) — **>2×
margin even after the safety factor**, so no BOM change from this. Pan
axis needs far less torque (no gravity moment, just inertia/friction);
the firmware's eased slew limiting already bounds acceleration loads.

Design rule this implies for every future shell: **keep head payload
under ~700–800g and its center of mass within ~5cm of the tilt pivot.**
A shell that blows this budget (e.g. two GC9A01 eyes plus a much bigger
cosmetic shell) needs the torque math re-checked before printing, not
after.

## 4. Constraints per sub-assembly (feeds ORCHESTRATION §4.5)

**Head:**
- Camera aperture ≥ IMX219's ~120° FOV cone; no shell rim or paint
  intruding on the lens's view.
- Jetson vent duct: intake low/rear, exhaust matched to the stock
  heatsink fan's flow direction — sealed head throttles the SoC, and
  since the head is the swappable part, **every new shell design has to
  re-verify this duct**, not inherit it from the reference shell.
- Mic port + speaker grille, forward-facing (single mic, no
  beamforming — directionality comes from the pan/tilt aim, per
  `unzipped/robot_build_spec.md`).
- Eye aperture(s) (if adopted): panel active area + viewing-angle
  relief per `hardware/emote_display.md`; interpupillary spacing fixed
  per shell, renderer doesn't care.
- Mounting bosses to the neck's head-plate; this is the one interface
  every swappable shell must keep identical.

**Neck:**
- Standard pan/tilt bracket kit (BOM), sized for DS3218MG.
- Cable service loop (power pair + serial cable + servo wires) with
  enough slack to survive full ±90° pan / ±45° tilt (calibrated soft
  limits, `profiles/*/calibration.json`) without binding or fatiguing.
- Clearance so the head shell never contacts the base shell at any
  angle within those limits.

**Base:**
- Battery bay(s) sized to DCB205 + adapter dock, external tool-free
  hatch per pack, per `hardware/power_system.md`.
- Buck converters mounted with airflow clearance (they run warm under
  servo stall current).
- Low-voltage alarm/cutoff indicator audible or visible from outside
  the shell.
- Physical kill switch on the servo rail, reachable without opening the
  shell.
- Footprint wide enough that the base+battery ballast keeps the whole
  assembly stable at max tilt (no added weighted foot needed per §2,
  but the footprint geometry still has to be checked once a real shell
  exists).

## 5. Open questions

1. **Mount context assumed but not confirmed:** tabletop/floor pedestal
   stand. If this is meant to be worn (backpack, cart) instead, the base
   design changes — say so before the first shell is modeled.
2. First real shell CAD + weigh-in is what turns §3's estimates into
   measured numbers (Phase 2.5/7 gate, same pattern as
   `hardware/budget_analysis.md`).
3. Fastener/standoff hardware (M2/M3 heat-set inserts for repeated shell
   swaps) isn't costed yet — small addition to the Misc BOM row.
4. Print settings for structural parts (wall count, infill %) not
   pinned down — defer to the first bench print, tune from there.

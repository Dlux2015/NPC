# Power system — DeWalt 20V tool-pack rework

**Status: ADOPTED into build spec v1.1 (2026-07-05)** — see
`unzipped/robot_build_spec.md` power section and
`unzipped/robot_bom_tracker_v2.html`. Prices remain estimates until
purchase. This doc keeps the detailed rationale and the bench numbers to
confirm.

Replaced (BOM v1): NOBIS USB-C power bank (Jetson) + Talentcell 12V pack
+ Seloky LM2596 buck (servos).

## Why tool packs

- **Hot-swap**: packs click in/out in seconds — a full convention day on
  2–3 packs with zero downtime, one charger ecosystem.
- **Capacity**: 20V 5Ah (DCB205) ≈ 100Wh — more than the NOBIS (74Wh) or
  Talentcell (84Wh).
- **Current delivery**: tool packs handle dual-servo stall loads that
  stress USB power banks.

## Why DeWalt 20V MAX (vs Milwaukee M18 / Ryobi ONE+)

Largest third-party adapter market: fused adapter docks with 12AWG leads
marketed for robotics run ~$15–35 with 30A fuse + switch built in; deep
supply of 3D-printable mounts for shell bay design. Milwaukee = better
cells, pricier, smaller adapter market. Ryobi = cheapest, bulkier stem
packs, spottier adapter supply. If packs of another brand are already
owned, downstream electronics are identical — brand choice is ecosystem,
not electrical.

## Architecture (keeps the spec's two isolated rails)

```
DeWalt pack A --fused adapter--> buck 19V/5A  --barrel--> Jetson Orin Nano
DeWalt pack B --fused adapter--> buck 6.5V/5A+ --------> DS3218MG servos
```

- Two packs = full rail isolation + swap the servo pack while the Jetson
  keeps running (robot never fully reboots during a day).
- **Never wire a pack direct to the Jetson**: fresh pack sits ~21V, over
  the barrel jack's 19V spec — always through the buck.
- Servo buck must be a genuine 5A-class part. (The BOM's LM2596 is
  ~2A continuous — undersized for two DS3218MG at stall, ~2.5A each.
  Replace it in this rework regardless of battery choice.)
- **Low-voltage cutoff is mandatory**: tool packs have no internal
  discharge cutoff (the tool normally provides it). Fit a ~15V
  alarm/cutoff module per pack, or have the watchdog monitor rail
  voltage. Over-discharge permanently damages packs.
- Fusing: use adapter docks with built-in 30A fuse, or add inline fuses —
  these packs deliver enormous current into a wiring fault.

## Shopping list

| Item | Qty | ~Price (Jul 2026, verify before ordering) |
|---|---|---|
| DeWalt DCB205 5Ah 20V MAX pack | 2 | ~$100–130/pair on sale |
| Fused adapter dock, 12AWG leads, 30A fuse | 2 | ~$15–25 ea |
| Buck converter 19V/5A out (Jetson rail) | 1 | ~$12 |
| Buck converter 6.5V/5A+ out (servo rail) | 1 | ~$12 |
| Low-voltage alarm/cutoff module (~15V) | 2 | ~$5 ea |

Net vs BOM: replaces NOBIS ($35.97) + Talentcell ($49.99) + LM2596
($7.99); ~$100–150 up if starting from zero packs, ~free if packs are
already owned.

## Runtime math

Jetson at 25W mode + servos averaging a few W ≈ ~30W total draw →
~3h per 5Ah pack per rail; hot-swap extends indefinitely. (Compare: NOBIS
gave ~2.5h with no swap option.) Confirm with measured draw at Phase 1/5
bench tests and record real numbers here.

## Shell design constraints (feeds ORCHESTRATION §4.5 list)

- Battery bay sized to the specific pack + adapter dock, with an
  external hatch for tool-free hot-swap.
- Bay orientation so pack weight sits low/central (packs are the
  heaviest single components).
- Cutoff/alarm module audible or visible from outside the shell.

## Sources

- DAIER DeWalt adapter kit: https://www.daierswitches.com/products/dewalt-20v-to-power-wheels-battery-adapter-kit
- Amazon fused DeWalt adapter (robotics): https://www.amazon.com/Adapter-Conversion-Terminals-Connector-Robotics/dp/B0B9NPZM3M
- Amazon 2-pack w/ switch+fuse: https://www.amazon.com/Conversion-Terminals-Connector-Converter-Robotics/dp/B09T39SYQR
- Surebonder Ryobi/Milwaukee adapter (backordered as of Jul 2026): https://surebonder.com/products/mil-18v-ryobi%C2%AE-to-milwaukee%C2%AE-battery-adapter
- Walmart Surebonder listing: https://www.walmart.com/ip/Ryobi-to-Milwaukee-Battery-Adapter/976192793

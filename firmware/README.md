# CBot Firmware — ESP32-S3-DevKitC-1 (MicroPython)

Phase 1: pan/tilt servo controller. 50Hz easing loop, protocol-enforced
limits, idle scan on heartbeat silence, USB-CDC serial (the native USB
port — same cable flashes and talks).

## Wiring

| Signal | Pin |
|---|---|
| Pan servo signal | GPIO4 |
| Tilt servo signal | GPIO5 |
| Servo V+ (both) | **buck converter rail** — never the DevKit's 5V pin |
| Ground | buck GND + servo GND + DevKit GND **tied common** |

**Safety: set the buck output to 6.0–6.8V and confirm with a multimeter
BEFORE connecting the servos.** An unadjusted buck can ship at 12V+ and
will destroy them. Kill switch sits on the servo rail.

## Deploy

From the repo root (`A:\code\CBot`), board on USB:

```
mpremote cp shared/serial_protocol.py :serial_protocol.py
mpremote cp firmware/easing.py :easing.py
mpremote cp firmware/main.py :main.py
mpremote reset
```

`main.py` auto-runs on boot. serial_protocol/easing live flat at the board
root; `main.py` imports them flat (with a fallback to the repo package
paths so the same file imports on CPython for the tests).

## Bench test

```
mpremote repl
```

Then type (newline sends):

```
P:10 T:0
```

Head eases to pan 10°. You'll see `A:<pan> <tilt>` reports at 2Hz; send
`PING` to get `PONG` and hold off the idle scan. Go silent >2s and the
head starts its slow idle sweep — that's the heartbeat timeout working,
not a bug. Ctrl-] exits the repl.

## Tests (dev PC, no hardware)

```
python -m pytest firmware/ -q
```

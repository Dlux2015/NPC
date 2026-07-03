"""CPython pytest for firmware/main.py logic (Phase 1).

Runs on the dev PC: imports main.py WITHOUT executing any hardware code
(machine/select/sys hardware bits live inside main.start(), which is only
called from main.py's __main__ guard). Covers:
  - angle -> pulse-width math (endpoints, midpoint, clamping)
  - serial command handling (target / ping / garbage) via CommandHandler
  - LineBuffer byte-at-a-time framing + overflow discard
  - a simulated 50Hz loop converging through easing.HeadController
  - idle scan engaging on heartbeat silence (owned by HeadController)

Run from the repo root:  python -m pytest firmware/ -q
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main  # noqa: E402  (firmware/main.py — hardware guarded in start())

from shared import serial_protocol  # noqa: E402
from firmware import easing  # noqa: E402


# ------------------------------------------------------------- pulse math

def test_pulse_endpoints_and_center():
    assert main.angle_to_pulse_us(-90.0) == 500
    assert main.angle_to_pulse_us(0.0) == 1500
    assert main.angle_to_pulse_us(90.0) == 2500


def test_pulse_intermediate_values():
    assert main.angle_to_pulse_us(45.0) == 2000
    assert main.angle_to_pulse_us(-45.0) == 1000
    assert main.angle_to_pulse_us(9.0) == 1600


def test_pulse_clamps_out_of_range():
    assert main.angle_to_pulse_us(-180.0) == 500
    assert main.angle_to_pulse_us(180.0) == 2500
    assert main.angle_to_pulse_us(-90.0001) == 500
    assert main.angle_to_pulse_us(90.0001) == 2500


def test_pulse_returns_int():
    assert isinstance(main.angle_to_pulse_us(12.34), int)


# -------------------------------------------------------- command handling

def _handler():
    head = main.make_head()
    return main.CommandHandler(head), head


def test_target_command_sets_targets_and_heartbeat():
    handler, head = _handler()
    reply = handler.handle_line("P:30.00 T:-10.00", 5.0)
    assert reply is None
    assert head.pan.target == 30.0
    assert head.tilt.target == -10.0
    assert head.last_cmd_s == 5.0
    assert not head.is_idle(5.5)


def test_target_command_clamped_to_limits():
    handler, head = _handler()
    handler.handle_line("P:500 T:-500", 1.0)
    assert head.pan.target == serial_protocol.PAN_MAX
    assert head.tilt.target == serial_protocol.TILT_MIN


def test_ping_replies_pong_and_heartbeats():
    handler, head = _handler()
    reply = handler.handle_line("PING", 7.0)
    assert reply == serial_protocol.encode_pong()
    assert head.last_cmd_s == 7.0
    # a PING alone must not move the targets
    assert head.pan.target == 0.0
    assert head.tilt.target == 0.0


def test_garbage_ignored():
    handler, head = _handler()
    for junk in ("", "hello", "P:abc T:def", "P:1.0", "P:1 X:2",
                 "T:5 P:5", "\x00\xff", "PINGG"):
        assert handler.handle_line(junk, 1.0) is None
    assert head.pan.target == 0.0
    assert head.last_cmd_s == 0.0       # garbage is not a heartbeat


def test_mcu_to_host_messages_not_echoed():
    handler, head = _handler()
    assert handler.handle_line("A:10.00 5.00", 1.0) is None
    assert handler.handle_line("PONG", 1.0) is None
    assert head.pan.target == 0.0


def test_roundtrip_with_protocol_encoder():
    handler, head = _handler()
    handler.handle_line(serial_protocol.encode_target(12.5, -7.25), 2.0)
    assert head.pan.target == 12.5
    assert head.tilt.target == -7.25


# ------------------------------------------------------------- line buffer

def _feed(lb, text):
    out = []
    for ch in text:
        line = lb.feed(ch)
        if line is not None:
            out.append(line)
    return out


def test_linebuffer_frames_lines():
    lb = main.LineBuffer()
    assert _feed(lb, "P:10.00 T:0.00\nPING\n") == ["P:10.00 T:0.00", "PING"]


def test_linebuffer_partial_then_complete():
    lb = main.LineBuffer()
    assert _feed(lb, "P:1.0 T:") == []
    assert _feed(lb, "2.0\n") == ["P:1.0 T:2.0"]


def test_linebuffer_overflow_discards_line_then_recovers():
    lb = main.LineBuffer(size=8)
    got = _feed(lb, "X" * 100 + "\n" + "PING\n")
    assert got == ["PING"]


def test_linebuffer_carriage_return_tolerated():
    lb = main.LineBuffer()
    # parse_line strips, so "PING\r" framed from "PING\r\n" must still work
    (line,) = _feed(lb, "PING\r\n")
    assert serial_protocol.parse_line(line) == ("ping",)


# --------------------------------------------------- simulated 50Hz loop

DT = 1.0 / 50.0


def test_loop_converges_on_target():
    head = main.make_head()
    handler = main.CommandHandler(head)
    now = 0.0
    handler.handle_line("P:40.00 T:20.00", now)
    for _ in range(50 * 4):             # 4 simulated seconds at 50Hz
        now += DT
        if int(now * 50) % 25 == 0:     # host heartbeat keeps idle away
            handler.handle_line("PING", now)
        pan, tilt = head.step(DT, now)
    assert abs(pan - 40.0) <= head.pan.deadband + 0.6
    assert abs(tilt - 20.0) <= head.tilt.deadband + 0.6
    assert not head.is_idle(now)


def test_loop_motion_is_slew_limited():
    head = main.make_head()
    handler = main.CommandHandler(head)
    handler.handle_line("P:90.00 T:0.00", 0.0)
    now, prev = 0.0, head.pan.current
    max_step = 0.0
    for _ in range(50):
        now += DT
        handler.handle_line("PING", now)
        pan, _tilt = head.step(DT, now)
        if abs(pan - prev) > max_step:
            max_step = abs(pan - prev)
        prev = pan
    # never faster than max_dps (plus float slack)
    assert max_step <= head.pan.max_dps * DT + 1e-6


def test_idle_scan_engages_on_heartbeat_silence():
    head = main.make_head()
    handler = main.CommandHandler(head)
    handler.handle_line("P:0.00 T:0.00", 0.0)
    now = 0.0
    # silence past the heartbeat timeout -> HeadController starts sweeping
    for _ in range(50 * 5):
        now += DT
        head.step(DT, now)
    assert head.is_idle(now)
    assert abs(head.pan.target) > 1.0   # sweep has pulled pan off center
    assert head.tilt.target == easing.HeadController.IDLE_TILT
    # a fresh command must immediately reclaim control
    handler.handle_line("P:5.00 T:5.00", now)
    assert not head.is_idle(now)
    assert head.pan.target == 5.0


# --------------------------------------------------------- import hygiene

def test_no_hardware_modules_needed_at_import():
    # main imported fine on CPython (no machine module here) and its
    # constants document the wiring.
    assert main.PAN_PIN == 4
    assert main.TILT_PIN == 5
    assert main.PWM_FREQ_HZ == 50
    assert main.LOOP_HZ == 50
    assert "machine" not in sys.modules

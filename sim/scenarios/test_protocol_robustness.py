"""Garbage in -> ignored; out-of-range targets -> clamped; PING -> PONG."""
from shared import serial_protocol as sp
from sim.servo_sim import ServoSim

GARBAGE = ["", "hello", "P:1.0", "P:a T:b", "T:5 P:5", "P: T:", "A:1 2",
           "PONG", "\x00\xffP:9 T:9junk", "P:1 T:2 X:3"]


def test_garbage_lines_ignored():
    servo = ServoSim(latency_s=0.0)
    for line in GARBAGE:
        servo.inject_line(line)
    for _ in range(25):  # 0.5s
        servo.step(0.02)
    # No crash, targets untouched (garbage is not a command or heartbeat).
    assert servo.head.pan.target == 0.0
    assert servo.head.tilt.target == 0.0


def test_ping_gets_pong_and_counts_as_heartbeat():
    servo = ServoSim(latency_s=0.0)
    servo.step(0.02)
    servo.read_lines()  # drop initial angle report
    servo.inject_line(sp.encode_ping())
    servo.step(0.02)
    replies = [sp.parse_line(l) for l in servo.read_lines()]
    assert ("pong",) in replies
    assert not servo.head.is_idle(servo.now)


def test_out_of_range_targets_clamped():
    servo = ServoSim(latency_s=0.0)
    reports = []
    dt = 0.02
    for i in range(int(4.0 / dt)):
        if i % 25 == 0:
            # Hand-formatted on purpose: simulates a buggy host bypassing
            # encode_target()'s clamp. Firmware-side limits must still hold.
            servo.inject_line("P:9999.0 T:-9999.0\n")
        servo.step(dt)
        for line in servo.read_lines():
            parsed = sp.parse_line(line)
            if parsed and parsed[0] == "angles":
                reports.append((parsed[1], parsed[2]))
    assert reports
    for pan, tilt in reports:
        assert sp.PAN_MIN <= pan <= sp.PAN_MAX
        assert sp.TILT_MIN <= tilt <= sp.TILT_MAX
    # It did actually drive toward the (clamped) extreme, not ignore it.
    assert reports[-1][0] > sp.PAN_MAX - 2.0
    assert reports[-1][1] < sp.TILT_MIN + 2.0


def test_encode_target_clamps_at_source():
    assert sp.parse_line(sp.encode_target(500.0, 200.0)) == \
        ("target", sp.PAN_MAX, sp.TILT_MAX)

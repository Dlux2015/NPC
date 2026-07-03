"""Serial silence > heartbeat timeout -> ESP32-owned idle scan kicks in."""
from firmware.easing import HeadController
from shared.serial_protocol import parse_line, encode_target
from sim.servo_sim import ServoSim


def test_silence_starts_pan_sweep_with_reversal():
    servo = ServoSim()
    servo.inject_line(encode_target(0.0, 0.0))  # one command, then silence

    pans = []  # (sim_time, reported_pan) from the 2Hz A: reports
    dt = 0.02
    for i in range(int(14.0 / dt)):  # 14 sim-seconds of silence
        servo.step(dt)
        for line in servo.read_lines():
            parsed = parse_line(line)
            if parsed and parsed[0] == "angles":
                pans.append((servo.now, parsed[1]))

    assert servo.head.is_idle(servo.now)
    idle_pans = [p for t, p in pans if t > 3.0]  # well past the 2s timeout
    span = HeadController.IDLE_PAN_SPAN
    assert max(abs(p) for p in idle_pans) <= span + 1.0, \
        "scan exceeded +/-%s deg" % span
    assert max(idle_pans) > span * 0.5, "pan never swept away from center"

    # Direction reverses: consecutive-report deltas of both signs.
    deltas = [b - a for a, b in zip(idle_pans, idle_pans[1:])]
    assert any(d > 0.5 for d in deltas) and any(d < -0.5 for d in deltas), \
        "sweep never changed direction"

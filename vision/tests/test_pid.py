from vision.pid import PID


def test_deadband_zero_output_and_no_state_buildup():
    pid = PID(kp=1.0, ki=1.0, kd=1.0, deadband=2.0)
    assert pid.update(1.5, dt=0.1) == 0.0
    assert pid.update(-2.0, dt=0.1) == 0.0  # boundary is inclusive
    assert pid._integral == 0.0


def test_output_is_clamped():
    pid = PID(kp=10.0, output_limits=(-5.0, 5.0))
    assert pid.update(100.0, dt=0.1) == 5.0
    assert pid.update(-100.0, dt=0.1) == -5.0


def test_anti_windup_recovers_faster_than_unclamped_integral():
    # Long saturation without anti-windup lets the integral balloon far
    # past the output clamp; on a sign reversal it stays pinned to the
    # rail for a long time. The (default) integral clamp should let the
    # controller come off the rail immediately instead.
    guarded = PID(kp=0.0, ki=1.0, output_limits=(-10.0, 10.0))
    unguarded = PID(kp=0.0, ki=1.0, output_limits=(-10.0, 10.0),
                     integral_limits=(-1e6, 1e6))

    for _ in range(200):
        guarded.update(50.0, dt=0.1)
        unguarded.update(50.0, dt=0.1)

    assert guarded._integral <= 10.0 + 1e-9
    assert unguarded._integral > 500.0  # windup: nowhere near the clamp

    guarded_out = guarded.update(-50.0, dt=0.1)
    unguarded_out = unguarded.update(-50.0, dt=0.1)

    assert guarded_out < 10.0        # reacts to the reversal right away
    assert unguarded_out == 10.0     # still pinned to the rail by windup


def test_reset_clears_integral_and_derivative_history():
    pid = PID(kp=1.0, ki=1.0, kd=1.0)
    pid.update(5.0, dt=0.1)
    pid.update(5.0, dt=0.1)
    assert pid._integral != 0.0
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error is None
    assert pid._prev_time is None


def test_convergence_on_a_simple_plant():
    # position += output * dt; PID should drive error -> ~0 over time.
    pid = PID(kp=1.5, ki=0.3, kd=0.05, output_limits=(-50.0, 50.0),
              deadband=0.01)
    position = 10.0
    target = 0.0
    dt = 0.02
    for _ in range(1000):
        error = target - position
        output = pid.update(error, dt=dt)
        position += output * dt
    assert abs(target - position) < 0.05


def test_now_based_dt_matches_explicit_dt():
    pid_now = PID(kp=1.0, ki=1.0, kd=1.0, deadband=0.0)
    pid_dt = PID(kp=1.0, ki=1.0, kd=1.0, deadband=0.0)

    pid_now.update(4.0, now=0.0)
    out_now = pid_now.update(4.0, now=0.5)

    pid_dt.update(4.0, dt=0.0)
    out_dt = pid_dt.update(4.0, dt=0.5)

    assert out_now == out_dt

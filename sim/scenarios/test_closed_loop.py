"""Static face at azimuth 20°: P-control converges, no oscillation."""
from sim.servo_sim import ServoSim
from sim.world import SimWorld


def test_converges_within_5s_no_blowup(track):
    world = SimWorld()
    world.add_face(azimuth_deg=20.0, elevation_deg=0.0, face_id=0)
    servo = ServoSim()

    samples = track(world, servo, duration_s=5.0, gain=0.05)

    errs = [(t, e[0]) for t, e in samples if e is not None]
    assert errs, "face never detected"
    converged_at = None
    for t, ex in errs:
        if converged_at is None and abs(ex) <= 30.0:
            converged_at = t
    assert converged_at is not None, "never within 30px of center in 5s"

    # No oscillation blowup: once converged, error stays converged.
    after = [ex for t, ex in errs if t >= converged_at]
    assert max(abs(ex) for ex in after) <= 30.0, \
        "error grew again after convergence (oscillation)"

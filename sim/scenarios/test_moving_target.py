"""Face drifting at 5 deg/s: tracking lag stays within 80px of center."""
from sim.servo_sim import ServoSim
from sim.world import SimWorld


def test_tracks_slow_mover_within_80px(track):
    world = SimWorld()
    world.add_face(azimuth_deg=0.0, elevation_deg=0.0, face_id=0)
    servo = ServoSim()

    def move(t):
        world.move_face(0, azimuth_deg=5.0 * t, elevation_deg=0.0)

    samples = track(world, servo, duration_s=10.0, gain=0.05, move=move)

    errs = [(t, e[0]) for t, e in samples if e is not None]
    assert errs and errs[-1][0] > 9.0, "lost the face before the end"
    settled = [ex for t, ex in errs if t >= 1.0]  # 1s to acquire
    assert max(abs(ex) for ex in settled) <= 80.0, \
        "tracking lag exceeded 80px"

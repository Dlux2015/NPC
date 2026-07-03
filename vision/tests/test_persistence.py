from vision.tracking import TargetTracker


def _clock_box(start=0.0):
    box = [start]
    return box, (lambda: box[0])


def test_first_detection_locks_highest_score():
    box, clock = _clock_box()
    tracker = TargetTracker(hold_s=3.0, clock=clock)
    low = (10, 10, 40, 40, 0.4)
    high = (400, 10, 40, 40, 0.9)
    chosen = tracker.update([low, high])
    assert chosen == high


def test_two_simultaneous_faces_no_snapping():
    box, clock = _clock_box()
    tracker = TargetTracker(hold_s=3.0, clock=clock)

    face_a = (100, 100, 50, 50, 0.9)
    face_b = (450, 100, 50, 50, 0.9)  # far away, distinct face

    chosen = tracker.update([face_a, face_b])
    assert chosen == face_a  # first max-score pick establishes the lock

    # Face A drifts a little each frame while face B stays put nearby in
    # the same detection list -- nearest-bbox association must keep
    # following A, never snap to B.
    for i in range(1, 21):
        box[0] += 0.05
        face_a = (100 + i, 100, 50, 50, 0.9)
        chosen = tracker.update([face_a, face_b])
        assert chosen[0] == 100 + i
        assert chosen != face_b


def test_holds_last_known_position_when_lost_before_hold_elapses():
    box, clock = _clock_box()
    tracker = TargetTracker(hold_s=1.0, clock=clock)

    face_a = (100, 100, 50, 50, 0.9)
    face_b = (450, 100, 50, 50, 0.9)

    tracker.update([face_a])
    box[0] += 0.1  # well under hold_s

    # A is gone; only a distinct face B remains -- must not switch yet.
    chosen = tracker.update([face_b])
    assert chosen == face_a


def test_switches_once_hold_elapses_after_target_lost():
    box, clock = _clock_box()
    tracker = TargetTracker(hold_s=1.0, clock=clock)

    face_a = (100, 100, 50, 50, 0.9)
    face_b = (450, 100, 50, 50, 0.9)

    tracker.update([face_a])
    box[0] += 0.1
    assert tracker.update([face_b]) == face_a  # still holding

    box[0] += 2.0  # now past hold_s since the lock started
    assert tracker.update([face_b]) == face_b  # switch permitted


def test_current_cleared_when_nothing_detected_past_hold():
    box, clock = _clock_box()
    tracker = TargetTracker(hold_s=0.5, clock=clock)

    face_a = (100, 100, 50, 50, 0.9)
    tracker.update([face_a])
    box[0] += 1.0
    assert tracker.update([]) is None


def test_no_detections_ever_returns_none():
    box, clock = _clock_box()
    tracker = TargetTracker(hold_s=1.0, clock=clock)
    assert tracker.update([]) is None

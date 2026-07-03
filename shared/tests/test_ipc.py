"""Unit tests for shared/ipc.py's ThreadedStateWriter (F4): the frame loop
(vision/tracking.py) must make zero filesystem calls, so publish() has to be
a pure in-memory hand-off, with a background thread owning the real
SharedState.update() disk write, coalesced to <=1/interval_s Hz.
"""
import time

from shared.ipc import SharedState, ThreadedStateWriter


def _spy_state(tmp_path):
    state = SharedState(str(tmp_path / "state.json"))
    calls = []
    orig_update = state.update

    def spy(**kwargs):
        calls.append(kwargs)
        orig_update(**kwargs)

    state.update = spy
    return state, calls


def test_publish_never_calls_state_update_directly(tmp_path):
    """start=False: no background thread running at all, so if publish()
    itself touched disk, we'd see it immediately -- deterministic, no
    thread-timing race."""
    state, calls = _spy_state(tmp_path)
    writer = ThreadedStateWriter(state, start=False)

    writer.publish(person_present=True)
    writer.publish(person_in_range=True)

    assert calls == []  # publish() is pure in-memory; never touches disk
    assert writer._pending == {"person_present": True, "person_in_range": True}


def test_flush_coalesces_multiple_publishes_into_one_write(tmp_path):
    state, calls = _spy_state(tmp_path)
    writer = ThreadedStateWriter(state, start=False)

    for i in range(20):
        writer.publish(person_present=True, person_in_range=(i % 2 == 0))

    assert calls == []  # still nothing written
    writer.flush()

    assert len(calls) == 1  # 20 publishes -> exactly one write
    assert calls[0] == {"person_present": True, "person_in_range": False}
    assert state.read()["person_present"] is True


def test_flush_is_a_noop_when_nothing_pending(tmp_path):
    state, calls = _spy_state(tmp_path)
    writer = ThreadedStateWriter(state, start=False)
    writer.flush()
    assert calls == []


def test_background_thread_eventually_writes_without_explicit_flush(tmp_path):
    """Integration-style: with the real daemon thread running, a publish()
    shows up on disk within a bounded wait -- no explicit flush() needed
    (that's the whole point: the frame loop never calls flush() either)."""
    state, calls = _spy_state(tmp_path)
    writer = ThreadedStateWriter(state, interval_s=0.05)
    try:
        writer.publish(person_present=True)
        deadline = time.time() + 2.0
        while time.time() < deadline and not calls:
            time.sleep(0.01)
        assert calls, "background writer never wrote the published state"
        assert state.read()["person_present"] is True
    finally:
        writer.stop()


def test_rapid_publishes_do_not_flood_disk_writes(tmp_path):
    """Bursting publish() calls faster than interval_s must not translate
    into one disk write per call -- the background thread coalesces."""
    state, calls = _spy_state(tmp_path)
    writer = ThreadedStateWriter(state, interval_s=0.2)
    try:
        for i in range(50):
            writer.publish(person_present=True, new_person_seq=i)
        time.sleep(0.05)  # well under interval_s
    finally:
        writer.stop()
    # 50 rapid publishes must not become 50 writes.
    assert 1 <= len(calls) <= 3
    assert calls[-1]["new_person_seq"] == 49  # latest value wins


def test_stop_flushes_pending_publish(tmp_path):
    state, calls = _spy_state(tmp_path)
    writer = ThreadedStateWriter(state, interval_s=60.0)
    writer.publish(person_present=True)
    writer.stop()
    assert calls  # stop() flushed the pending publish
    assert state.read()["person_present"] is True

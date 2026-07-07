"""shared/people.py: enroll/match/refresh semantics on a tmp sqlite db.
(The store also gets exercised end-to-end by sim/scenarios and
conversation tests; this file pins the matching math itself.)
"""
import numpy as np

from shared.people import PeopleStore


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_emb(direction, dim=8):
    e = np.zeros(dim, dtype=np.float32)
    e[direction] = 1.0
    return e


def test_enroll_then_exact_match(tmp_path):
    store = PeopleStore(tmp_path / "people.db")
    pid = store.enroll(make_emb(0), name="Ada")
    hit = store.match(make_emb(0), threshold=0.5)
    assert hit is not None
    assert hit[0] == pid and hit[1] == "Ada"
    assert hit[2] > 0.99


def test_no_match_below_threshold_logs_best_score(tmp_path, caplog):
    import logging
    store = PeopleStore(tmp_path / "people.db")
    store.enroll(make_emb(0))
    # orthogonal query: score ~0 -> miss, and the near-miss gets logged
    with caplog.at_level(logging.INFO):
        assert store.match(make_emb(1), threshold=0.5) is None
    assert any("best score" in r.message for r in caplog.records)


def test_match_refreshes_embedding_toward_query(tmp_path):
    """Appearance drift: repeated matches with a slightly-rotated query
    must pull the stored embedding along, keeping the score high instead
    of letting it decay (the live regression this feature fixes)."""
    store = PeopleStore(tmp_path / "people.db")
    pid = store.enroll(make_emb(0))

    drifted = unit([0.9, 0.435, 0, 0, 0, 0, 0, 0])  # ~0.9 cosine vs stored
    first = store.match(drifted, threshold=0.5, refresh_alpha=0.3)
    assert first is not None and first[0] == pid
    second = store.match(drifted, threshold=0.5, refresh_alpha=0.3)
    assert second is not None
    assert second[2] > first[2]  # stored embedding moved toward the query


def test_refresh_alpha_zero_leaves_embedding_frozen(tmp_path):
    store = PeopleStore(tmp_path / "people.db")
    store.enroll(make_emb(0))
    drifted = unit([0.9, 0.435, 0, 0, 0, 0, 0, 0])
    first = store.match(drifted, threshold=0.5, refresh_alpha=0.0)
    second = store.match(drifted, threshold=0.5, refresh_alpha=0.0)
    assert abs(first[2] - second[2]) < 1e-6


def test_refreshed_embedding_stays_unit_norm(tmp_path):
    store = PeopleStore(tmp_path / "people.db")
    pid = store.enroll(make_emb(0) * 5.0)  # enroll un-normalized on purpose
    drifted = unit([0.8, 0.6, 0, 0, 0, 0, 0, 0])
    store.match(drifted, threshold=0.5, refresh_alpha=0.5)
    row = store.db.execute(
        "SELECT embedding FROM people WHERE id=?", (pid,)).fetchone()
    stored = np.frombuffer(row[0], dtype=np.float32)
    assert abs(float(np.linalg.norm(stored)) - 1.0) < 1e-5


def test_set_name_and_purge(tmp_path):
    store = PeopleStore(tmp_path / "people.db")
    pid = store.enroll(make_emb(2))
    store.set_name(pid, "Grace")
    assert store.get(pid)["name"] == "Grace"
    store.purge()
    assert store.count() == 0

"""Identity store — the SOLE reader/writer of people.db (contract §4.4).

Vision enrolls embeddings and matches faces; conversation writes learned
names. Nobody else touches the DB file. Embeddings only — never images.
SQLite, stdlib-only except numpy for cosine matching.
"""
import logging
import sqlite3
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY,
    name TEXT,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL
);
"""

MATCH_THRESHOLD = 0.55  # generic default; SFace callers pass
                        # vision.recognition.SFACE_MATCH_THRESHOLD

# On a confident match, the stored embedding is blended toward the query
# by this factor (exponential moving average): people's appearance drifts
# across sessions (lighting, angle, hair), and a single frozen enrollment
# embedding demonstrably decays -- live testing 2026-07-06 had the same
# person match in afternoon light and miss in evening light. Small alpha
# so an (unlikely, above-threshold) wrong match poisons slowly enough to
# notice; 0 disables refresh entirely.
DEFAULT_REFRESH_ALPHA = 0.1


class PeopleStore:
    def __init__(self, path):
        # check_same_thread=False + _lock: TrackingApp constructs this on
        # the main thread but uses it from its recognition worker thread.
        # The lock serializes all access — sqlite objects aren't otherwise
        # safe to share across threads.
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.db.execute(_SCHEMA)
            self.db.commit()

    def enroll(self, embedding, name=None):
        """Add a new person; returns their id."""
        emb = np.asarray(embedding, dtype=np.float32)
        now = time.time()
        with self._lock:
            return self._enroll_locked(emb, name, now)

    def _enroll_locked(self, emb, name, now):
        cur = self.db.execute(
            "INSERT INTO people (name, embedding, dim, first_seen, last_seen)"
            " VALUES (?, ?, ?, ?, ?)",
            (name, emb.tobytes(), emb.size, now, now),
        )
        self.db.commit()
        return cur.lastrowid

    def match(self, embedding, threshold=MATCH_THRESHOLD,
              refresh_alpha=DEFAULT_REFRESH_ALPHA):
        """Best cosine match above threshold → (id, name, score) or None.
        On a hit: bumps last_seen and (unless refresh_alpha=0) blends the
        stored embedding toward the query so recognition tracks gradual
        appearance drift (see DEFAULT_REFRESH_ALPHA). On a miss with
        candidates present, logs the best below-threshold score --
        that number is what threshold tuning needs (Phase 6 bench)."""
        emb = np.asarray(embedding, dtype=np.float32)
        n = np.linalg.norm(emb)
        if n == 0:
            return None
        emb = emb / n
        best = None
        best_below = None  # (score, pid) of the nearest miss, for tuning
        with self._lock:
            for pid, name, blob, dim in self.db.execute(
                "SELECT id, name, embedding, dim FROM people"
            ):
                other = np.frombuffer(blob, dtype=np.float32)
                if other.size != emb.size:
                    continue
                on = np.linalg.norm(other)
                if on == 0:
                    continue
                score = float(np.dot(emb, other / on))
                if score >= threshold and (best is None or score > best[2]):
                    best = (pid, name, score)
                elif score < threshold and (
                        best_below is None or score > best_below[0]):
                    best_below = (score, pid)
            if best:
                self.db.execute(
                    "UPDATE people SET last_seen=? WHERE id=?",
                    (time.time(), best[0]),
                )
                if refresh_alpha:
                    self._refresh_embedding_locked(best[0], emb, refresh_alpha)
                self.db.commit()
        if best is None and best_below is not None:
            logger.info(
                "people.match miss: best score %.3f (person %s) below "
                "threshold %.3f", best_below[0], best_below[1], threshold,
            )
        return best

    def _refresh_embedding_locked(self, person_id, query_unit, alpha):
        """EMA-blend the stored embedding toward a freshly-matched query
        (both unit-normalized; result re-normalized). Caller holds _lock
        and commits."""
        row = self.db.execute(
            "SELECT embedding FROM people WHERE id=?", (person_id,)
        ).fetchone()
        if not row:
            return
        stored = np.frombuffer(row[0], dtype=np.float32)
        norm = np.linalg.norm(stored)
        if norm == 0 or stored.size != query_unit.size:
            return
        blended = (1.0 - alpha) * (stored / norm) + alpha * query_unit
        bnorm = np.linalg.norm(blended)
        if bnorm == 0:
            return
        blended = (blended / bnorm).astype(np.float32)
        self.db.execute(
            "UPDATE people SET embedding=? WHERE id=?",
            (blended.tobytes(), person_id),
        )

    def set_name(self, person_id, name):
        with self._lock:
            self.db.execute(
                "UPDATE people SET name=? WHERE id=?", (name, person_id)
            )
            self.db.commit()

    def get(self, person_id):
        with self._lock:
            row = self.db.execute(
                "SELECT id, name, first_seen, last_seen FROM people WHERE id=?",
                (person_id,),
            ).fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1],
                "first_seen": row[2], "last_seen": row[3]}

    def purge(self):
        """Privacy: forget everyone."""
        with self._lock:
            self.db.execute("DELETE FROM people")
            self.db.commit()

    def count(self):
        with self._lock:
            return self.db.execute(
                "SELECT COUNT(*) FROM people").fetchone()[0]

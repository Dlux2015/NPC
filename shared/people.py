"""Identity store — the SOLE reader/writer of people.db (contract §4.4).

Vision enrolls embeddings and matches faces; conversation writes learned
names. Nobody else touches the DB file. Embeddings only — never images.
SQLite, stdlib-only except numpy for cosine matching.
"""
import sqlite3
import time

import numpy as np

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

MATCH_THRESHOLD = 0.55  # cosine similarity; tune against SFace in Phase 6


class PeopleStore:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute(_SCHEMA)
        self.db.commit()

    def enroll(self, embedding, name=None):
        """Add a new person; returns their id."""
        emb = np.asarray(embedding, dtype=np.float32)
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO people (name, embedding, dim, first_seen, last_seen)"
            " VALUES (?, ?, ?, ?, ?)",
            (name, emb.tobytes(), emb.size, now, now),
        )
        self.db.commit()
        return cur.lastrowid

    def match(self, embedding, threshold=MATCH_THRESHOLD):
        """Best cosine match above threshold → (id, name, score) or None.
        Also bumps last_seen on a hit."""
        emb = np.asarray(embedding, dtype=np.float32)
        n = np.linalg.norm(emb)
        if n == 0:
            return None
        emb = emb / n
        best = None
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
        if best:
            self.db.execute(
                "UPDATE people SET last_seen=? WHERE id=?",
                (time.time(), best[0]),
            )
            self.db.commit()
        return best

    def set_name(self, person_id, name):
        self.db.execute(
            "UPDATE people SET name=? WHERE id=?", (name, person_id)
        )
        self.db.commit()

    def get(self, person_id):
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
        self.db.execute("DELETE FROM people")
        self.db.commit()

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM people").fetchone()[0]

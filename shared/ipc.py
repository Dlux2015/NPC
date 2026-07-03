"""Shared state between the vision and conversation processes (Jetson-side,
CPython only). File-backed JSON with atomic replace — simple, debuggable,
fast enough at ~10Hz. Upgrade to Redis only if this measurably falls short.

Canonical keys (contract §4.2):
  person_present      bool
  person_in_range     bool
  person_id           str | None   ("unknown" while enrolling)
  new_person_seq      int          (increment on each auto-enroll event)
  actively_speaking   bool
  conversation_active bool
  ambient_transcript  list[str]    (rolling, in-memory semantics: writer
                                    trims to ~60s worth; never persisted
                                    beyond this scratch state file)
"""
import json
import os
import tempfile
import time

DEFAULTS = {
    "person_present": False,
    "person_in_range": False,
    "person_id": None,
    "new_person_seq": 0,
    "actively_speaking": False,
    "conversation_active": False,
    "ambient_transcript": [],
}


class SharedState:
    def __init__(self, path):
        self.path = str(path)
        self._mtime = -1.0
        self._cache = dict(DEFAULTS)
        if not os.path.exists(self.path):
            self._write(self._cache)

    def _write(self, data):
        d = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)  # atomic on POSIX and Windows
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def read(self):
        """Full state dict; cached by mtime so polling is cheap."""
        try:
            m = os.stat(self.path).st_mtime
            if m != self._mtime:
                with open(self.path) as f:
                    data = json.load(f)
                self._cache = {**DEFAULTS, **data}
                self._mtime = m
        except (OSError, ValueError):
            pass  # writer mid-replace or corrupt: serve last good cache
        return dict(self._cache)

    def get(self, key):
        return self.read()[key]

    def update(self, **kwargs):
        """Read-modify-replace. Single-writer-per-key by convention:
        vision owns person_*; conversation owns *_speaking/_active and
        ambient_transcript."""
        data = self.read()
        unknown = set(kwargs) - set(DEFAULTS)
        if unknown:
            raise KeyError("unknown IPC keys: %s" % sorted(unknown))
        data.update(kwargs)
        data["_ts"] = time.time()
        self._write(data)
        self._cache = data

    def heartbeat_age(self):
        """Seconds since last write by anyone (watchdog input)."""
        return time.time() - self.read().get("_ts", 0)

"""Shared state between the vision and conversation processes (Jetson-side,
CPython only). File-backed JSON with atomic replace — simple, debuggable,
fast enough at ~10Hz. Upgrade to Redis only if this measurably falls short.

Canonical keys (contract §4.2):
  person_present      bool
  person_in_range     bool
  person_id           int | None   (None = nobody / not yet recognized)
  new_person_seq      int          (counter, increments per auto-enroll)
  actively_speaking   bool
  conversation_active bool
  ambient_transcript  list[str]    (rolling, in-memory semantics: writer
                                    trims to ~60s worth; never persisted
                                    beyond this scratch state file)

ThreadedStateWriter (SS3.1 hard rule: the tracking frame loop never touches
disk) sits in front of SharedState for the vision process: publish(**kw) is
a non-blocking, lock-protected dict merge -- no filesystem call -- and a
daemon thread owns the actual SharedState.update() (mkstemp+json+os.replace)
at <=10Hz, coalescing bursts of publish() calls down to their latest values.
"""
import json
import os
import tempfile
import threading
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


class ThreadedStateWriter:
    """Coalescing, non-blocking front end for SharedState.update(), so a
    hot loop (vision/tracking.py's frame loop -- SS3.1: "no disk in the
    hot path") can publish state without ever making a filesystem call
    itself.

    publish(**kwargs) just merges into an in-memory "latest values" dict
    under a short-held lock and wakes the writer thread -- no I/O. A single
    daemon thread owns the real SharedState.update() disk write, at most
    every `interval_s` seconds (<=10Hz at the default), always writing the
    most recent values published since its last write (bursts of publish()
    calls between writes collapse to one write of the latest state, not
    one write per call).
    """

    def __init__(self, state, interval_s=0.1, start=True):
        self.state = state
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._pending = {}
        self._dirty = False
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        if start:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def publish(self, **kwargs):
        """Non-blocking hand-off: merge kwargs into the pending snapshot
        and wake the writer. Never touches disk; safe to call every frame.
        """
        with self._lock:
            self._pending.update(kwargs)
            self._dirty = True
        self._wake.set()

    def _drain(self):
        with self._lock:
            data = dict(self._pending) if self._dirty else None
            self._dirty = False
        if data:
            self.state.update(**data)

    def flush(self):
        """Synchronous immediate write of whatever is pending. Not for the
        frame loop -- for tests and clean shutdown only."""
        self._drain()

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            self._drain()
            if self._stop.is_set():
                break
            # Enforce the <=interval_s write cadence: further publish()
            # calls during this sleep just keep coalescing into _pending.
            time.sleep(self.interval_s)
        self._drain()  # flush anything published right before stop()

    def stop(self, timeout=1.0):
        """Stop the writer thread, flushing any pending publish() first.
        Safe to call on a writer constructed with start=False (no thread
        to join) -- just flushes."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        else:
            self._drain()

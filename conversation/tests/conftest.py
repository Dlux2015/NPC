"""Shared fixtures for conversation/ tests.

No audio, no model download: everything here is either a tmp-path-backed
real component (profile files, shared/people.py's PeopleStore -- both
cheap and worth exercising for real) or an in-memory fake standing in for
a pinned interface that speech-engineer is building in parallel (wake,
stt, speaker) or for shared/ipc.py's SharedState (kept as a lightweight
fake here per the task brief; shared/ipc.py itself is covered by its own
tests elsewhere).
"""
import os

import pytest

from shared.people import PeopleStore

PERSONA_TEXT = (
    "# Persona: TestBot\n\n"
    "You are TestBot, a friendly test robot. Reply in 1-3 short "
    "sentences. All-ages tone. If you didn't understand, say so "
    "cheerfully and ask them to repeat.\n"
)

PROFILE_NAME = "testprof"


@pytest.fixture
def profile_root(tmp_path):
    """A tmp /profiles-style root with one profile: testprof."""
    prof_dir = tmp_path / PROFILE_NAME
    prof_dir.mkdir()
    (prof_dir / "persona.md").write_text(PERSONA_TEXT, encoding="utf-8")
    (prof_dir / "profile.yaml").write_text(
        "name: %s\npersona: persona.md\n" % PROFILE_NAME, encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def people_store(tmp_path):
    return PeopleStore(tmp_path / "people.db")


class FakeState:
    """Minimal stand-in for shared/ipc.py's SharedState: .get(key) /
    .update(**kwargs), plus an update_log so tests can assert on the
    conversation_active on/off sequence."""

    def __init__(self, **kwargs):
        self.data = {
            "person_present": False,
            "person_in_range": False,
            "person_id": None,
            "new_person_seq": 0,
            "actively_speaking": False,
            "conversation_active": False,
            "ambient_transcript": [],
        }
        self.data.update(kwargs)
        self.update_log = []

    def get(self, key):
        return self.data[key]

    def update(self, **kwargs):
        self.data.update(kwargs)
        self.update_log.append(dict(kwargs))


class FakeWake:
    """wait(timeout_s) pops the next scripted event; returns None (and
    stops popping) once the script is exhausted, mimicking a real
    wake.wait() timing out with nothing to do."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def wait(self, timeout_s):
        self.calls += 1
        if self.script:
            return self.script.pop(0)
        return None


class FakeSTT:
    """listen_utterance(max_s) pops the next scripted (utterance, mutate)
    pair. `mutate`, if given, is called with the FakeState right after
    "hearing" the utterance -- lets a test simulate the person walking
    off (or a different person stepping in) mid-conversation, since real
    IPC state can change between STT calls. Returns None (timeout) once
    the script is exhausted."""

    def __init__(self, script, state=None):
        # script entries are either a str, None, or (str|None, mutate_fn)
        self.script = list(script)
        self.state = state
        self.calls = 0

    def listen_utterance(self, max_s=10.0):
        self.calls += 1
        if not self.script:
            return None
        entry = self.script.pop(0)
        if isinstance(entry, tuple):
            utterance, mutate = entry
        else:
            utterance, mutate = entry, None
        if mutate is not None and self.state is not None:
            mutate(self.state)
        return utterance


class FakeSpeaker:
    """Records everything said. say_stream() fully drains its sentence
    iterator (as a real TTS engine would, synthesizing/playing each
    sentence as it arrives) so tests can assert on ordering."""

    def __init__(self):
        self.say_stream_calls = []  # list[list[str]]
        self.say_calls = []  # list[str]

    def say_stream(self, sentence_iter):
        sentences = list(sentence_iter)
        self.say_stream_calls.append(sentences)

    def say(self, text):
        self.say_calls.append(text)


class FakeAmbient:
    def __init__(self, lines=None):
        self.lines = list(lines or [])

    def snapshot(self):
        return list(self.lines)


@pytest.fixture
def fake_state():
    return FakeState()


@pytest.fixture
def fake_speaker():
    return FakeSpeaker()

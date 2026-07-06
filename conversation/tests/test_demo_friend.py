"""demo_friend's consent-gated enrollment (FriendWake) + is_affirmative:
fakes at every physical edge, real shared/people.py PeopleStore on tmp
sqlite (the contract-owning code worth exercising for real).
"""
import numpy as np
import pytest

from conversation.demo_friend import FriendWake, is_affirmative
from shared.people import PeopleStore

from conftest import FakeState


# --- is_affirmative ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "yes", "Yes!", "yeah sure", "OK", "okay then", "yep.",
    "of course you can", "absolutely",
])
def test_affirmative_variants(text):
    assert is_affirmative(text) is True


@pytest.mark.parametrize("text", [None, "", "no", "no way", "nope",
                                    "I don't think so", "maybe later"])
def test_non_affirmative_variants(text):
    assert is_affirmative(text) is False


# --- FriendWake --------------------------------------------------------------

class FakeInnerWake:
    def __init__(self, events):
        self.events = list(events)

    def wait(self, timeout_s=None):
        return self.events.pop(0) if self.events else None


class FakeApp:
    def __init__(self, embedding=None):
        self.embedding = embedding

    def pop_unknown_embedding(self):
        emb, self.embedding = self.embedding, None
        return emb


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class FakeSTT:
    def __init__(self, replies):
        self.replies = list(replies)

    def listen_utterance(self, max_s=10.0):
        return self.replies.pop(0) if self.replies else None


def make_wake(tmp_path, events=("face_speech",), reply="yes",
               embedding=..., state=None):
    if embedding is ...:
        embedding = np.ones(128, dtype=np.float32)
    state = state or FakeState(person_present=True)
    people = PeopleStore(tmp_path / "people.db")
    speaker = FakeSpeaker()
    wake = FriendWake(FakeInnerWake(events), FakeApp(embedding), state,
                       people, speaker, FakeSTT([reply]),
                       persona_name="TestBot")
    return wake, state, people, speaker


def test_yes_enrolls_publishes_id_and_bumps_seq(tmp_path):
    wake, state, people, speaker = make_wake(tmp_path, reply="yes I would love that")

    event = wake.wait(1.0)

    assert event == "face_speech"          # wake event passes through
    assert people.count() == 1             # enrolled exactly once
    assert state.get("person_id") is not None
    assert state.get("new_person_seq") == 1
    assert any("friend" in s.lower() for s in speaker.said)
    # invites the name in a NAME_RE-compatible phrasing
    assert any("my name is" in s.lower() for s in speaker.said)


def test_no_stores_nothing_and_declines_stick_while_present(tmp_path):
    state = FakeState(person_present=True)
    people = PeopleStore(tmp_path / "people.db")
    speaker = FakeSpeaker()
    app = FakeApp(np.ones(128, dtype=np.float32))
    wake = FriendWake(FakeInnerWake(["face_speech", "face_speech"]), app,
                       state, people, speaker,
                       FakeSTT(["no thanks", "still no"]),
                       persona_name="TestBot")

    assert wake.wait(1.0) == "face_speech"
    assert people.count() == 0
    assert state.get("person_id") is None
    asked_once = len(speaker.said)
    assert asked_once >= 1

    # Same person still present: must NOT be asked again.
    app.embedding = np.ones(128, dtype=np.float32)  # vision stashed again
    assert wake.wait(1.0) == "face_speech"
    assert len(speaker.said) == asked_once
    assert people.count() == 0


def test_decline_resets_when_person_leaves(tmp_path):
    state = FakeState(person_present=True)
    people = PeopleStore(tmp_path / "people.db")
    speaker = FakeSpeaker()
    app = FakeApp(np.ones(128, dtype=np.float32))
    wake = FriendWake(
        FakeInnerWake(["face_speech", None, "face_speech"]), app, state,
        people, speaker, FakeSTT(["nope", "yes please"]),
        persona_name="TestBot")

    wake.wait(1.0)          # asks; declined
    assert people.count() == 0
    assert wake._declined is True

    # Person leaves; a wait() poll observes the absence (inner times out).
    state.data["person_present"] = False
    assert wake.wait(1.0) is None
    assert wake._declined is False  # decline forgotten with the departure

    # Somebody (same or new) walks up again -> consent question again,
    # and this time they say yes.
    state.data["person_present"] = True
    app.embedding = np.ones(128, dtype=np.float32)
    assert wake.wait(1.0) == "face_speech"
    assert people.count() == 1


def test_recognized_person_skips_consent_entirely(tmp_path):
    wake, state, people, speaker = make_wake(
        tmp_path, state=FakeState(person_present=True, person_id="3"))

    assert wake.wait(1.0) == "face_speech"
    assert speaker.said == []              # no consent question
    assert people.count() == 0


def test_no_stashed_embedding_chats_anonymously_without_asking(tmp_path):
    wake, state, people, speaker = make_wake(tmp_path, embedding=None)

    assert wake.wait(1.0) == "face_speech"
    assert speaker.said == []
    assert people.count() == 0
    assert state.get("person_id") is None

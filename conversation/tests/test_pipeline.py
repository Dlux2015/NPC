"""conversation/pipeline.py: turn-taking flow logic, with MockLLM and
fake wake/stt/speaker/state -- no audio, no model download. Name
capture is checked against a real shared/people.py PeopleStore on a tmp
sqlite db (the one place we want the real contract-owning code exercised).
"""
from conversation.llm import MockLLM
from conversation.pipeline import ConversationPipeline, split_sentences

from conftest import PROFILE_NAME, FakeAmbient, FakeState, FakeSTT, FakeSpeaker, FakeWake


def make_pipeline(
    profile_root,
    state,
    stt_script,
    llm_replies=("Okay, got it."),
    people=None,
    ambient=None,
    wake_script=("wakeword",),
):
    wake = FakeWake(wake_script)
    stt = FakeSTT(stt_script, state=state)
    speaker = FakeSpeaker()
    llm = MockLLM(replies=list(llm_replies) if not isinstance(llm_replies, str) else [llm_replies])
    pipeline = ConversationPipeline(
        profile=PROFILE_NAME,
        state=state,
        wake=wake,
        stt=stt,
        llm=llm,
        speaker=speaker,
        people=people,
        ambient=ambient,
        profile_root=profile_root,
    )
    return pipeline, wake, stt, speaker, llm


# --- sentence splitter ---------------------------------------------------

def test_split_sentences_splits_on_terminal_punctuation_across_chunks():
    chunks = ["Hi there", ". How ", "are you", "? Great!"]
    sentences = list(split_sentences(iter(chunks)))
    assert sentences == ["Hi there.", "How are you?", "Great!"]


def test_split_sentences_flushes_trailing_text_without_punctuation():
    chunks = ["No terminal punctuation here"]
    sentences = list(split_sentences(iter(chunks)))
    assert sentences == ["No terminal punctuation here"]


def test_split_sentences_skips_empty_chunks():
    chunks = ["Hello.", "", " World."]
    assert list(split_sentences(iter(chunks))) == ["Hello.", "World."]


# --- full turn flow --------------------------------------------------------

def test_full_turn_flow_speaks_reply_and_updates_history(profile_root):
    state = FakeState(person_present=True, person_id=1, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root,
        state,
        stt_script=["Hello robot"],
        llm_replies=["Hi there! Nice to see you."],
    )

    happened = pipeline.run_once()

    assert happened is True
    assert wake.calls == 1
    # one turn -> one say_stream call; "Hi there!" and "Nice to see you."
    # are two sentences (boundary after the "!"), streamed in order
    assert speaker.say_stream_calls == [["Hi there!", "Nice to see you."]]
    # rolling history now has the user turn + assistant reply
    contents = [m["content"] for m in pipeline._history]
    assert "Hello robot" in contents
    assert "Hi there! Nice to see you." in contents
    # the system prompt handed to the LLM carries the persona
    system_msg = llm.calls[0][0]
    assert system_msg["role"] == "system"
    assert "TestBot" in system_msg["content"]


def test_conversation_active_lifecycle(profile_root):
    state = FakeState(person_present=True, person_id=1)
    pipeline, *_ = make_pipeline(profile_root, state, stt_script=["hi"])
    pipeline.run_once()
    # toggled True then False, and ends False
    assert {"conversation_active": True} in state.update_log
    assert {"conversation_active": False} in state.update_log
    assert state.data["conversation_active"] is False
    true_idx = state.update_log.index({"conversation_active": True})
    false_idx = state.update_log.index({"conversation_active": False})
    assert true_idx < false_idx


def test_wake_timeout_returns_false_without_conversation(profile_root):
    state = FakeState(person_present=False)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=[], wake_script=[None]
    )
    happened = pipeline.run_once()
    assert happened is False
    assert speaker.say_stream_calls == []
    assert speaker.say_calls == []
    # never even flips conversation_active
    assert state.update_log == []


def test_utterance_timeout_ends_conversation(profile_root):
    state = FakeState(person_present=True, person_id=1)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=[]  # listen_utterance -> None immediately
    )
    happened = pipeline.run_once()
    assert happened is True
    assert speaker.say_stream_calls == []  # no turns actually happened
    assert state.data["conversation_active"] is False


# --- multi-turn within one conversation ------------------------------------

def test_multiple_turns_within_one_wake_event(profile_root):
    state = FakeState(person_present=True, person_id=1)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root,
        state,
        stt_script=["First question", "Second question"],
        llm_replies=["First reply.", "Second reply."],
    )
    pipeline.run_once()
    assert wake.calls == 1  # only waited for wake once
    assert speaker.say_stream_calls == [["First reply."], ["Second reply."]]
    contents = [m["content"] for m in pipeline._history]
    assert contents == [
        "First question",
        "First reply.",
        "Second question",
        "Second reply.",
    ]


# --- greeting: new person ---------------------------------------------------

def test_greets_new_person_on_new_person_seq_bump(profile_root):
    state = FakeState(person_present=True, person_id=5, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=["hello"]
    )
    # baseline seq captured at construction (0); simulate vision bumping
    # the counter for this brand-new auto-enrolled person before wake.
    state.data["new_person_seq"] = 1

    pipeline.run_once()

    assert speaker.say_calls == ["Nice to meet you!"]
    assert pipeline._history[0] == {"role": "assistant", "content": "Nice to meet you!"}


def test_no_new_person_greeting_when_seq_unchanged(profile_root):
    state = FakeState(person_present=True, person_id=5, new_person_seq=3)
    # pipeline constructed with this seq as baseline -> not "new"
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=["hello"]
    )
    pipeline.run_once()
    assert speaker.say_calls == []  # no name known either -> no greeting at all


# --- greeting: returning person --------------------------------------------

def test_greets_returning_person_with_known_name(profile_root, people_store):
    pid = people_store.enroll([0.1, 0.2, 0.3], name="Alice")
    state = FakeState(person_present=True, person_id=pid, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=["hi"], people=people_store
    )
    pipeline.run_once()
    assert speaker.say_calls == ["Welcome back, Alice!"]


def test_no_greeting_for_returning_person_without_name(profile_root, people_store):
    pid = people_store.enroll([0.1, 0.2, 0.3], name=None)
    state = FakeState(person_present=True, person_id=pid, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=["hi"], people=people_store
    )
    pipeline.run_once()
    assert speaker.say_calls == []


# --- name capture -----------------------------------------------------------

def test_name_capture_writes_back_via_people_store(profile_root, people_store):
    pid = people_store.enroll([0.1, 0.2, 0.3], name=None)
    state = FakeState(person_present=True, person_id=pid, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root,
        state,
        stt_script=["Hi, my name is Charlie"],
        people=people_store,
    )
    pipeline.run_once()
    assert people_store.get(pid)["name"] == "Charlie"


def test_name_capture_contraction_form(profile_root, people_store):
    pid = people_store.enroll([0.1, 0.2, 0.3], name=None)
    state = FakeState(person_present=True, person_id=pid, new_person_seq=0)
    pipeline, *_ = make_pipeline(
        profile_root, state, stt_script=["my name's Dana"], people=people_store
    )
    pipeline.run_once()
    assert people_store.get(pid)["name"] == "Dana"


def test_name_capture_with_no_person_id_does_not_crash(profile_root, people_store):
    state = FakeState(person_present=True, person_id=None, new_person_seq=0)
    pipeline, *_ = make_pipeline(
        profile_root,
        state,
        stt_script=["my name is Eve"],
        people=people_store,
    )
    # should not raise even though there's no person_id to attach the name to
    pipeline.run_once()


# --- window reset ------------------------------------------------------------

def test_window_resets_when_person_goes_absent_mid_conversation(profile_root):
    state = FakeState(person_present=True, person_id=1, new_person_seq=0)

    def make_absent(s):
        s.data["person_present"] = False

    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root,
        state,
        stt_script=["Hello", ("I have to go", make_absent)],
    )
    pipeline.run_once()

    # the second utterance triggered absence -> reset before it was
    # ever added to history or replied to
    assert pipeline._history == []
    assert pipeline._history_person_id is None
    assert speaker.say_stream_calls == [["Okay, got it."]]  # only 1st turn spoke


def test_window_resets_when_person_id_changes_mid_conversation(profile_root):
    state = FakeState(person_present=True, person_id=1, new_person_seq=0)

    def swap_person(s):
        s.data["person_id"] = 999

    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root,
        state,
        stt_script=["Hello", ("New person talking", swap_person)],
    )
    pipeline.run_once()

    # window reset to the new person; the first turn's content is gone
    assert pipeline._history_person_id == 999
    contents = [m["content"] for m in pipeline._history]
    assert "Hello" not in contents
    assert "Okay, got it." not in contents or contents.count("Okay, got it.") == 1


def test_window_reset_across_separate_run_once_calls(profile_root):
    state = FakeState(person_present=True, person_id=1, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=["First conversation"]
    )
    pipeline.run_once()
    assert pipeline._history  # has turns now

    # person leaves, then a *different* person is present for the next
    # wake event
    state.data["person_present"] = False
    state.data["person_id"] = None

    wake.script = ["wakeword"]
    stt.script = ["Second conversation"]
    state.data["person_present"] = True
    state.data["person_id"] = 2

    pipeline.run_once()
    contents = [m["content"] for m in pipeline._history]
    assert "First conversation" not in contents
    assert "Second conversation" in contents


# --- ambient context ---------------------------------------------------------

def test_ambient_snapshot_appears_marked_overheard_in_prompt(profile_root):
    state = FakeState(person_present=True, person_id=1, new_person_seq=0)
    ambient = FakeAmbient(["a dog barked outside"])
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=["hi"], ambient=ambient
    )
    pipeline.run_once()
    system_msg = llm.calls[0][0]["content"]
    assert "Overheard nearby" in system_msg
    assert "a dog barked outside" in system_msg
    assert "not addressing you" in system_msg or "NOT said to you" in system_msg


def test_no_ambient_block_when_ambient_not_wired_up(profile_root):
    state = FakeState(person_present=True, person_id=1, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root, state, stt_script=["hi"], ambient=None
    )
    pipeline.run_once()
    system_msg = llm.calls[0][0]["content"]
    assert "Overheard nearby" not in system_msg


# --- sentence streaming order ------------------------------------------------

def test_sentence_streaming_order_matches_generation_order(profile_root):
    state = FakeState(person_present=True, person_id=1, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root,
        state,
        stt_script=["tell me things"],
        llm_replies=["Sentence one. Sentence two! Sentence three?"],
    )
    llm.chunk_words = 1  # word-by-word streaming, worst case for ordering
    pipeline.run_once()
    assert speaker.say_stream_calls == [
        ["Sentence one.", "Sentence two!", "Sentence three?"]
    ]


# --- run_forever / stop ------------------------------------------------------

def test_run_forever_stops_cleanly(profile_root):
    state = FakeState(person_present=True, person_id=1, new_person_seq=0)
    pipeline, wake, stt, speaker, llm = make_pipeline(
        profile_root,
        state,
        stt_script=["hi", "bye"] + [None] * 5,
        wake_script=["wakeword", None, None, "wakeword", None],
    )

    calls = {"n": 0}
    real_wait = wake.wait

    def counting_wait(timeout_s):
        calls["n"] += 1
        if calls["n"] >= 4:
            pipeline.stop()
        return real_wait(timeout_s)

    wake.wait = counting_wait
    pipeline.run_forever()
    assert pipeline._stop is True

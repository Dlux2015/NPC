"""demo_talk --text mode: the console adapters (ConsoleWake / TypedSTT /
ConsoleSpeaker) driving the REAL ConversationPipeline with MockLLM -- no
audio stack, no model download, exactly what `--text` wires up minus the
on-disk state/people stores (covered by their own tests).
"""
import pytest

from conversation.demo_talk import ConsoleSpeaker, ConsoleWake, TypedSTT
from conversation.llm import MockLLM
from conversation.pipeline import ConversationPipeline

from conftest import PROFILE_NAME, FakeState


def make_input_fn(lines):
    """input()-alike that pops scripted lines, then EOFs (Ctrl-Z/D)."""
    it = iter(lines)

    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return _input


# --- ConsoleWake ---------------------------------------------------------

def test_console_wake_fires_exactly_once():
    wake = ConsoleWake()
    assert wake.wait(1.0) == "text"
    assert wake.wait(1.0) is None
    assert wake.wait(1.0) is None


# --- TypedSTT ------------------------------------------------------------

def test_typed_stt_returns_typed_line_stripped():
    stt = TypedSTT(input_fn=make_input_fn(["  hello robot  "]))
    assert stt.listen_utterance(max_s=5) == "hello robot"


def test_typed_stt_reprompts_on_empty_input():
    stt = TypedSTT(input_fn=make_input_fn(["", "   ", "hi"]))
    assert stt.listen_utterance() == "hi"


@pytest.mark.parametrize("word", ["quit", "QUIT", "exit", "/quit", "/exit"])
def test_typed_stt_quit_words_end_conversation(word):
    stt = TypedSTT(input_fn=make_input_fn([word]))
    assert stt.listen_utterance() is None


def test_typed_stt_eof_ends_conversation():
    stt = TypedSTT(input_fn=make_input_fn([]))
    assert stt.listen_utterance() is None


# --- ConsoleSpeaker --------------------------------------------------------

def test_console_speaker_streams_sentences_as_lines(capsys):
    speaker = ConsoleSpeaker()
    speaker.say_stream(iter(["Hi there!", "Nice to see you."]))
    assert capsys.readouterr().out == (
        "CBot: Hi there!\n"
        "      Nice to see you.\n"
    )


def test_console_speaker_say_prints_one_line(capsys):
    ConsoleSpeaker().say("Welcome back, Sam!")
    assert capsys.readouterr().out == "CBot: Welcome back, Sam!\n"


# --- full text-mode conversation through the real pipeline ----------------

def test_text_mode_full_conversation_via_real_pipeline(profile_root, capsys):
    state = FakeState()
    pipeline = ConversationPipeline(
        profile=PROFILE_NAME,
        state=state,
        wake=ConsoleWake(),
        stt=TypedSTT(input_fn=make_input_fn(["Hi robot", "quit"])),
        llm=MockLLM(replies=["Hello! What brings you by?"]),
        speaker=ConsoleSpeaker(),
        people=None,
        profile_root=profile_root,
    )

    assert pipeline.run_once() is True
    out = capsys.readouterr().out
    assert "CBot: Hello!" in out
    assert "      What brings you by?" in out
    # history recorded the exchange like any other transport would
    contents = [m["content"] for m in pipeline._history]
    assert "Hi robot" in contents

    # wake is exhausted: a second run_once() finds nothing to do, which
    # is what lets run_text() exit instead of looping forever
    assert pipeline.run_once() is False

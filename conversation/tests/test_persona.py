"""conversation/persona.py: persona.md loading + system prompt assembly."""
import pytest

from conversation.persona import (
    HARD_RULES,
    build_ambient_context,
    build_identity_context,
    build_system_prompt,
    build_system_prompt_for_profile,
    load_persona_text,
)

PROFILE_NAME = "testprof"
PERSONA_TEXT = (
    "# Persona: TestBot\n\n"
    "You are TestBot, a friendly test robot. Reply in 1-3 short "
    "sentences. All-ages tone. If you didn't understand, say so "
    "cheerfully and ask them to repeat.\n"
)


def test_load_persona_text_reads_active_profiles_file(profile_root):
    text = load_persona_text(PROFILE_NAME, root=profile_root)
    assert "TestBot" in text
    assert text == PERSONA_TEXT.strip()


def test_load_persona_text_missing_profile_raises(profile_root):
    with pytest.raises(FileNotFoundError):
        load_persona_text("no-such-profile", root=profile_root)


def test_identity_context_unknown_person():
    ctx = build_identity_context(None, None, is_new=False)
    assert "do not currently have a confirmed identity" in ctx


def test_identity_context_new_person():
    ctx = build_identity_context(7, None, is_new=True)
    assert "brand new" in ctx
    assert "7" in ctx


def test_identity_context_returning_with_name():
    ctx = build_identity_context(3, "Alice", is_new=False)
    assert "Alice" in ctx
    assert "returning visitor" in ctx


def test_identity_context_returning_without_name():
    ctx = build_identity_context(3, None, is_new=False)
    assert "don't know yet" in ctx


def test_ambient_context_empty_is_blank():
    assert build_ambient_context([]) == ""
    assert build_ambient_context(None) == ""


def test_ambient_context_marks_overheard_never_direct_address():
    ctx = build_ambient_context(["someone mentioned pizza", "a dog barked"])
    assert "Overheard nearby" in ctx
    assert "NOT said to you" in ctx or "not addressing you" in ctx
    assert "someone mentioned pizza" in ctx
    assert "a dog barked" in ctx


def test_build_system_prompt_includes_persona_and_hard_rules():
    prompt = build_system_prompt("You are Bot.", "", "")
    assert "You are Bot." in prompt
    assert HARD_RULES in prompt


def test_build_system_prompt_omits_empty_blocks():
    prompt = build_system_prompt("You are Bot.", "", "")
    # No stray blank identity/ambient sections when nothing was passed.
    assert prompt.count("\n\n") == 1  # persona, then hard rules -- that's it


def test_build_system_prompt_includes_identity_and_ambient_when_present():
    prompt = build_system_prompt(
        "You are Bot.",
        identity_context="You are speaking with Alice.",
        ambient_context="Overheard nearby:\n- dogs barking",
    )
    assert "Alice" in prompt
    assert "Overheard nearby" in prompt


def test_build_system_prompt_for_profile_end_to_end(profile_root):
    prompt = build_system_prompt_for_profile(
        PROFILE_NAME,
        person_id=1,
        name="Bob",
        is_new=False,
        ambient_lines=["the weather is nice"],
        root=profile_root,
    )
    assert "TestBot" in prompt
    assert HARD_RULES in prompt
    assert "Bob" in prompt
    assert "the weather is nice" in prompt
    assert "Overheard nearby" in prompt

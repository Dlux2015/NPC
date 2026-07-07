"""Persona + system-prompt assembly (contract §3.4, §4.5).

Persona *text* lives only in the active shell profile's `persona.md` --
never hardcoded here. This module loads that text and stitches it
together with:

  1. a short set of hard behavioral rules that apply regardless of
     persona (reply length, all-ages tone, graceful mishear recovery),
  2. an identity block built from `person_id` / `people.py` lookups
     (new person vs. recognized returning person vs. unknown), and
  3. an ambient-context block that is explicitly labelled as *overheard*
     background chatter -- the model must never treat it as something
     said directly to it.

Nothing here talks to the LLM or to IPC directly; `pipeline.py` gathers
the inputs (profile name, person info, ambient snapshot) and calls
`build_system_prompt`.
"""
import os

from vision.paths import profile_dir, load_profile_yaml

# Applies on top of whatever character the active profile's persona.md
# describes. Keep this short -- it competes for context budget with the
# persona text and the rolling conversation window (see pipeline.py's
# MAX_HISTORY_CHARS heuristic).
HARD_RULES = (
    "Hard rules, regardless of character: reply in 1 to 3 short sentences. "
    "Keep everything appropriate for all ages. If you did not understand "
    "what the person said, say so cheerfully and ask them to repeat "
    "themselves -- never guess or make something up."
)


def load_persona_text(profile_name, root=None):
    """Read the active profile's persona.md (path taken from its
    profile.yaml `persona` key, default "persona.md")."""
    cfg = load_profile_yaml(profile_name, root=root)
    persona_file = cfg.get("persona", "persona.md")
    path = os.path.join(profile_dir(profile_name, root=root), persona_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "No persona file for profile %r at %s" % (profile_name, path)
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_identity_context(person_id, name, is_new):
    """Short block telling the model who it's (probably) talking to.

    `person_id` / `name` come from IPC + shared/people.py; `is_new`
    means a new_person_seq bump just happened for this person (i.e.
    vision auto-enrolled them this session).
    """
    if person_id is None:
        return (
            "You do not currently have a confirmed identity for the "
            "person in front of you."
        )
    if is_new:
        return (
            "This person is brand new -- you are meeting them for the "
            "first time just now (internal person_id=%s)." % person_id
        )
    if name:
        return (
            "You are speaking with %s, a returning visitor you already "
            "know (internal person_id=%s)." % (name, person_id)
        )
    return (
        "You are speaking with a returning visitor you've met before but "
        "whose name you don't know yet (internal person_id=%s). If it "
        "comes up naturally, you can ask their name." % person_id
    )


def build_ambient_context(ambient_lines):
    """Block of recently-overheard (not-directed-at-the-robot) speech.

    Explicitly marked as overheard so the model never mistakes ambient
    chatter for something said to it directly. Returns "" when there's
    nothing to report (omit the block entirely).
    """
    lines = [line for line in (ambient_lines or []) if line and line.strip()]
    if not lines:
        return ""
    joined = "\n".join("- %s" % line.strip() for line in lines)
    return (
        "Overheard nearby (ambient background chatter, NOT said to you -- "
        "these people are not addressing you; never reply to this as if "
        "it were direct speech, use it only as optional scene context if "
        "it's relevant):\n%s" % joined
    )


def build_system_prompt(persona_text, identity_context="", ambient_context=""):
    """Combine persona text + hard rules + identity + ambient into the
    single system-role message content."""
    parts = [persona_text.strip(), HARD_RULES]
    if identity_context:
        parts.append(identity_context)
    if ambient_context:
        parts.append(ambient_context)
    return "\n\n".join(p for p in parts if p)


def build_system_prompt_for_profile(
    profile_name, person_id, name, is_new, ambient_lines, root=None
):
    """Convenience wrapper: load persona.md for `profile_name` and build
    the full system prompt in one call."""
    persona_text = load_persona_text(profile_name, root=root)
    identity_context = build_identity_context(person_id, name, is_new)
    ambient_context = build_ambient_context(ambient_lines)
    return build_system_prompt(persona_text, identity_context, ambient_context)

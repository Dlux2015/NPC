"""Conversation pipeline (contract §3.4, §4.2, §4.4).

Wires together wake / STT / LLM / TTS / identity into one turn-taking
loop. Everything it talks to is injected via the constructor as a
pinned-interface object (see the module docstring items below) --
pipeline.py has **no import-time dependency** on the concrete
wake/STT/ambient/TTS implementations, since those are being built in
parallel by speech-engineer. It only ever calls the methods on whatever
objects it's handed:

  wake.wait(timeout_s)        -> "wakeword" | "ptt" | "face_speech" | None
  stt.listen_utterance(max_s) -> str | None
  ambient.snapshot()          -> list[str]                 (optional)
  speaker.say_stream(sentence_iter)
  speaker.say(text)

`llm` is conversation/llm.py's LocalLLM or MockLLM (generate_stream).
`state` is shared/ipc.py's SharedState (or anything with the same
.get(key)/.update(**kwargs) surface). `people` is shared/people.py's
PeopleStore (or anything with the same match/get/set_name surface).
"""
import logging
import re
import time

from conversation.persona import (
    build_ambient_context,
    build_identity_context,
    build_system_prompt,
    load_persona_text,
)

logger = logging.getLogger(__name__)

# --- rolling window sizing -------------------------------------------------
# n_ctx is capped at 2048 tokens (ORCHESTRATION.md §3.4). We don't have a
# real tokenizer handy in the pipeline (and don't want one just to size a
# history buffer), so we approximate with a conservative
# chars-per-token ratio and reserve a chunk of the budget for the system
# prompt (persona + hard rules + identity + ambient block) and for the
# model's own output, leaving the remainder for the rolling
# user/assistant history:
#
#   2048 tokens total
#   -  600 tokens reserved for system prompt (persona can be verbose)
#   -  450 tokens reserved for the model's reply
#   = ~1000 tokens left for history, at ~4 chars/token (English-ish
#     average for this tokenizer family) => ~4000 chars.
#
# This is a heuristic, not a token-exact budget -- it's deliberately
# conservative so we trim before actually hitting the model's hard cap.
CHARS_PER_TOKEN_APPROX = 4
MAX_CONTEXT_TOKENS = 2048
RESERVED_SYSTEM_TOKENS = 600
RESERVED_OUTPUT_TOKENS = 450
MAX_HISTORY_TOKENS = MAX_CONTEXT_TOKENS - RESERVED_SYSTEM_TOKENS - RESERVED_OUTPUT_TOKENS
MAX_HISTORY_CHARS = MAX_HISTORY_TOKENS * CHARS_PER_TOKEN_APPROX

# Utterance timeout: if listen_utterance() returns None (no speech within
# max_s), the conversation ends.
DEFAULT_UTTERANCE_MAX_S = 10.0
# How long wake.wait() blocks per poll before we re-check the outer loop
# (only matters for run_forever's shutdown responsiveness).
DEFAULT_WAKE_TIMEOUT_S = 1.0

NAME_RE = re.compile(
    r"\bmy name(?:'s| is)\s+([A-Za-z][A-Za-z'\-]{0,30})\b", re.IGNORECASE
)

_SENTENCE_BOUNDARY_RE = re.compile(r"([.!?])(\s+)")


def split_sentences(chunk_iter):
    """Turn an iterator of raw text chunks (as produced by
    llm.generate_stream) into an iterator of complete sentences, so the
    speaker can start talking before the model has finished generating.

    A sentence boundary is recognized as sentence-ending punctuation
    (. ! ?) followed by whitespace; whatever's left in the buffer when
    the source iterator is exhausted is flushed as a final "sentence"
    even without trailing punctuation (handles clipped replies).
    """
    buf = ""
    for chunk in chunk_iter:
        if not chunk:
            continue
        buf += chunk
        while True:
            m = _SENTENCE_BOUNDARY_RE.search(buf)
            if not m:
                break
            end = m.end()
            sentence = buf[:end].strip()
            buf = buf[end:]
            if sentence:
                yield sentence
    tail = buf.strip()
    if tail:
        yield tail


class ConversationPipeline:
    def __init__(
        self,
        profile,
        state,
        wake,
        stt,
        llm,
        speaker,
        people,
        ambient=None,
        profile_root=None,
        utterance_max_s=DEFAULT_UTTERANCE_MAX_S,
        wake_timeout_s=DEFAULT_WAKE_TIMEOUT_S,
    ):
        self.profile = profile  # profile *name* (str), e.g. "sim"
        self.state = state
        self.wake = wake
        self.stt = stt
        self.llm = llm
        self.speaker = speaker
        self.people = people
        self.ambient = ambient
        self.profile_root = profile_root
        self.utterance_max_s = utterance_max_s
        self.wake_timeout_s = wake_timeout_s

        self._history = []  # list[{"role": "user"|"assistant", "content": str}]
        self._history_person_id = None  # person_id the current window belongs to
        # Baseline for "new_person_seq bumped" detection: only seq
        # increases *after* this point count as "just met them", so we
        # don't greet everyone who was already enrolled before this
        # process started.
        self._last_seq = self.state.get("new_person_seq")

        self._stop = False

    # -- window / history -----------------------------------------------

    def _reset_window(self, person_id):
        self._history = []
        self._history_person_id = person_id

    def _maybe_reset_window(self, person_id, person_present):
        """Window resets when person_present goes False or person_id
        changes (contract requirement). Returns True if a reset just
        happened."""
        current = person_id if person_present else None
        if current != self._history_person_id:
            self._reset_window(current)
            return True
        return False

    def _append_history(self, role, content):
        self._history.append({"role": role, "content": content})
        self._trim_history()

    def _trim_history(self):
        """Drop oldest turns until the approximate char budget is met.
        See MAX_HISTORY_CHARS heuristic above."""
        total = sum(len(m["content"]) for m in self._history)
        while total > MAX_HISTORY_CHARS and len(self._history) > 1:
            dropped = self._history.pop(0)
            total -= len(dropped["content"])

    # -- identity / greeting ----------------------------------------------

    def _person_name(self, person_id):
        if person_id is None or self.people is None:
            return None
        rec = self.people.get(person_id)
        return rec.get("name") if rec else None

    def _greeting_for(self, person_id, is_new, name):
        if person_id is None:
            return None
        if is_new:
            return "Nice to meet you!"
        if name:
            return "Welcome back, %s!" % name
        return None  # known face, no name yet -- no canned greeting

    def _check_new_person(self, person_id):
        """Consume the new_person_seq counter: True if it has bumped
        since we last looked (i.e. vision just auto-enrolled someone)."""
        seq = self.state.get("new_person_seq")
        is_new = person_id is not None and seq > self._last_seq
        self._last_seq = seq
        return is_new

    # -- name capture -------------------------------------------------------

    def _maybe_capture_name(self, person_id, utterance):
        m = NAME_RE.search(utterance or "")
        if not m:
            return
        name = m.group(1).strip().capitalize()
        if person_id is None:
            logger.info(
                "Heard a name (%r) but no person_id to attach it to -- "
                "skipping write-back.",
                name,
            )
            return
        self.people.set_name(person_id, name)
        logger.info("people.set_name(%s, %r)", person_id, name)

    # -- prompt building ------------------------------------------------

    def _build_messages(self, person_id, name, is_new):
        persona_text = load_persona_text(self.profile, root=self.profile_root)
        identity_context = build_identity_context(person_id, name, is_new)
        ambient_lines = self.ambient.snapshot() if self.ambient else []
        ambient_context = build_ambient_context(ambient_lines)
        system_prompt = build_system_prompt(
            persona_text, identity_context, ambient_context
        )
        return [{"role": "system", "content": system_prompt}] + list(self._history)

    # -- turn / streaming -------------------------------------------------

    def _speak_reply(self, messages):
        """Stream the LLM's reply, sentence-splitting it into the
        speaker, and return the full reply text for history."""
        chunks = self.llm.generate_stream(messages)
        parts = []

        def sentence_gen():
            for sentence in split_sentences(chunks):
                parts.append(sentence)
                yield sentence

        self.speaker.say_stream(sentence_gen())
        return " ".join(parts)

    # -- main loop ----------------------------------------------------------

    def run_once(self):
        """Wait for one wake event, then hold a conversation (possibly
        many turns) until the person leaves or an utterance times out.
        Returns True if a conversation happened, False if wake.wait()
        just timed out with nothing to do."""
        event = self.wake.wait(self.wake_timeout_s)
        if not event:
            return False

        logger.info("Wake event: %s", event)
        self.state.update(conversation_active=True)
        try:
            person_id = self.state.get("person_id")
            person_present = self.state.get("person_present")
            just_reset = self._maybe_reset_window(person_id, person_present)

            if just_reset and person_present:
                is_new = self._check_new_person(person_id)
                name = self._person_name(person_id)
                greeting = self._greeting_for(person_id, is_new, name)
                if greeting:
                    self.speaker.say(greeting)
                    self._append_history("assistant", greeting)
            else:
                # Not a fresh window (continuing an existing one) --
                # still drain the new_person_seq counter so a later
                # reset doesn't misfire on a stale bump.
                self._check_new_person(person_id)

            while True:
                utterance = self.stt.listen_utterance(max_s=self.utterance_max_s)
                if utterance is None:
                    logger.info("Utterance timeout -- ending conversation.")
                    break

                person_id = self.state.get("person_id")
                person_present = self.state.get("person_present")
                if self._maybe_reset_window(person_id, person_present):
                    # Person changed mid-loop (shouldn't normally happen
                    # without an absence in between, but handle it).
                    if not person_present:
                        break

                self._maybe_capture_name(person_id, utterance)
                self._append_history("user", utterance)

                name = self._person_name(person_id)
                is_new = self._check_new_person(person_id)
                messages = self._build_messages(person_id, name, is_new)
                reply = self._speak_reply(messages)
                self._append_history("assistant", reply)

                person_present = self.state.get("person_present")
                person_id_after = self.state.get("person_id")
                if self._maybe_reset_window(person_id_after, person_present):
                    break
        finally:
            self.state.update(conversation_active=False)
        return True

    def run_forever(self):
        """Loop run_once() until stop() is called. Cleanly checks a stop
        flag between conversations (and between wake.wait() polls, since
        wake.wait(timeout_s) is expected to return None periodically)."""
        while not self._stop:
            self.run_once()

    def stop(self):
        """Request run_forever() to exit after the current run_once()
        (or wake.wait() poll) returns."""
        self._stop = True

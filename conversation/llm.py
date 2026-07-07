"""Local LLM serving (contract §3.4).

Decided (Jul 2026, benchmarked): llama.cpp + CUDA,
Llama-3.2-3B-Instruct Q4_K_M (~2.0GB, ~29 tok/s on the Orin Nano Super).
Context capped at ~2K tokens. The model loads once, in the constructor,
and stays resident for the life of the process -- never reload per
utterance.

`llama_cpp` is imported lazily (inside LocalLLM.__init__), not at module
scope, so that:
  - this module imports cleanly on a dev machine without the package or
    a model file installed,
  - `pipeline.py` never gains an import-time dependency on this class
    just because it type-hints/constructs one elsewhere,
  - the error you get when the package or the model file is missing is
    an actionable one raised at construction time, not an ImportError
    stack trace from deep inside module import.

MockLLM below is the interface's other implementation: scripted replies,
no llama_cpp, no model file -- used by tests and the `sim` profile.
"""
import logging
import os

logger = logging.getLogger(__name__)

# What we tell people to go get when no model is configured/found.
EXPECTED_MODEL_DESC = (
    "Llama-3.2-3B-Instruct Q4_K_M GGUF "
    "(e.g. Llama-3.2-3B-Instruct-Q4_K_M.gguf)"
)

N_CTX = 2048  # hard cap per ORCHESTRATION.md §3.4 / skill

# Responsiveness guard: hard ceiling on tokens per reply. The persona
# contract is 1-3 short sentences (~<80 tokens); without an explicit cap
# a derailed generation can run on for many seconds, freezing the whole
# conversation turn (and, on the robot, holding the GPU the entire
# time). 220 leaves comfortable headroom over any legitimate reply and
# bounds the runaway worst case to ~7.5s on the Orin's ~29 tok/s
# (~2s on the dev PC) instead of "until the context fills". Normal
# replies never come near it; tune down if field replies never do.
MAX_REPLY_TOKENS = 220


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_model_path(profile):
    """Where to find the GGUF file: env var CBOT_LLM_MODEL wins (handy for
    local dev/override without touching profile.yaml); otherwise the
    active profile's `llm_model_path` key. A relative profile path is
    resolved against the repo root (so committed profiles can say
    `models/foo.gguf` regardless of CWD); the env var is taken as-is.
    Returns None if neither is set -- that's a normal, expected state on
    a dev machine."""
    env_path = os.environ.get("CBOT_LLM_MODEL")
    if env_path:
        return env_path
    if isinstance(profile, dict):
        path = profile.get("llm_model_path")
        if path and not os.path.isabs(path):
            return os.path.join(_repo_root(), path)
        return path
    return None


class LocalLLM:
    """llama.cpp-backed LLM, resident for the process lifetime."""

    def __init__(self, profile, model_path=None, n_ctx=N_CTX, **llama_kwargs):
        self.profile = profile
        self.n_ctx = n_ctx
        self.model_path = model_path or resolve_model_path(profile)

        if not self.model_path:
            raise RuntimeError(
                "LocalLLM: no model path configured. Set profile.yaml key "
                "'llm_model_path' or env CBOT_LLM_MODEL to a local "
                + EXPECTED_MODEL_DESC
            )
        if not os.path.isfile(self.model_path):
            raise RuntimeError(
                "LocalLLM: configured model path does not exist: %r. "
                "Expected a local %s at that path."
                % (self.model_path, EXPECTED_MODEL_DESC)
            )

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "LocalLLM: the 'llama-cpp-python' package is not "
                "installed (pip install llama-cpp-python, built with "
                "CUDA support on-device). Needed to serve the local %s."
                % EXPECTED_MODEL_DESC
            ) from e

        # GPU belongs to the LLM (§3.4): offload every layer by default,
        # overridable per-profile via `llm_gpu_layers` (0 = pure CPU).
        # Ignored gracefully by CPU-only llama.cpp builds.
        if "n_gpu_layers" not in llama_kwargs:
            n_gpu_layers = -1
            if isinstance(profile, dict):
                n_gpu_layers = profile.get("llm_gpu_layers", -1)
            llama_kwargs["n_gpu_layers"] = n_gpu_layers
        # llama.cpp's own load/perf chatter would drown the demo console.
        llama_kwargs.setdefault("verbose", False)

        logger.info(
            "LocalLLM: loading %s (n_ctx=%d, n_gpu_layers=%s) -- resident "
            "for process lifetime, not reloaded per utterance.",
            self.model_path,
            self.n_ctx,
            llama_kwargs["n_gpu_layers"],
        )
        self._llama = Llama(
            model_path=self.model_path, n_ctx=self.n_ctx, **llama_kwargs
        )

    def generate_stream(self, messages, max_tokens=MAX_REPLY_TOKENS):
        """messages: list[{"role": ..., "content": ...}] (system/user/
        assistant, OpenAI-style). Yields text chunks as they're produced.
        max_tokens bounds a runaway generation (see MAX_REPLY_TOKENS)."""
        stream = self._llama.create_chat_completion(
            messages=messages, stream=True, max_tokens=max_tokens
        )
        for chunk in stream:
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            text = delta.get("content")
            if text:
                yield text


class MockLLM:
    """Scripted-reply stand-in for LocalLLM. Same interface
    (`generate_stream(messages) -> iterator[str]`), no model file, no
    llama_cpp dependency. Used by tests and by profiles with no
    llm_model_path configured (e.g. `sim` until a GGUF is wired up).

    `replies` is a list of canned full-sentence strings; each call to
    generate_stream() consumes the next one (cycling once exhausted), so
    a test can script a short back-and-forth deterministically. Replies
    are chunked word-by-word to exercise streaming/sentence-splitting
    logic the same way a real model would.
    """

    def __init__(self, profile=None, replies=None, chunk_words=1):
        self.profile = profile
        self.replies = list(replies) if replies else ["Okay!"]
        self.chunk_words = max(1, chunk_words)
        self.calls = []  # recorded (messages,) for assertions in tests
        self._i = 0

    def _next_reply(self):
        reply = self.replies[self._i % len(self.replies)]
        self._i += 1
        return reply

    def generate_stream(self, messages):
        self.calls.append(messages)
        reply = self._next_reply()
        words = reply.split(" ")
        for start in range(0, len(words), self.chunk_words):
            group = words[start:start + self.chunk_words]
            text = " ".join(group)
            # Preserve a trailing space between chunks (except the very
            # last one) so "".join(chunks) reconstructs the original
            # sentence exactly -- the sentence-splitter relies on that.
            if start + self.chunk_words < len(words):
                text += " "
            yield text


def make_llm(profile):
    """Factory: LocalLLM if a model file is configured and present on
    disk, else MockLLM -- and always log which one, loudly, never
    silently fall back."""
    model_path = resolve_model_path(profile)
    if model_path and os.path.isfile(model_path):
        try:
            llm = LocalLLM(profile, model_path=model_path)
        except RuntimeError as e:
            # Model file present but llama-cpp-python isn't importable in
            # this interpreter (e.g. system Python instead of the project
            # venv). Loud fallback, never silent.
            logger.warning(
                "make_llm: model file %r is present but LocalLLM could "
                "not start (%s) -- falling back to MockLLM.",
                model_path,
                e,
            )
            return MockLLM(profile)
        logger.info("make_llm: using LocalLLM (%s)", model_path)
        return llm

    if model_path:
        logger.warning(
            "make_llm: llm_model_path/CBOT_LLM_MODEL is set to %r but "
            "that file does not exist -- falling back to MockLLM. "
            "Expected a local %s.",
            model_path,
            EXPECTED_MODEL_DESC,
        )
    else:
        logger.warning(
            "make_llm: no llm_model_path configured (profile.yaml key or "
            "CBOT_LLM_MODEL env) -- falling back to MockLLM. Expected a "
            "local %s.",
            EXPECTED_MODEL_DESC,
        )
    return MockLLM(profile)

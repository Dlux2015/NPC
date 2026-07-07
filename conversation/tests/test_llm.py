"""conversation/llm.py: MockLLM interface + make_llm() fallback/selection
logic. No real model download needed -- LocalLLM's dependency on
llama_cpp is exercised with a fake module injected into sys.modules, and
the one real-model test is skipif'd when no GGUF is configured.
"""
import logging
import os
import sys
import types

import pytest

from conversation.llm import LocalLLM, MockLLM, make_llm, resolve_model_path


# --- MockLLM -----------------------------------------------------------

def test_mock_llm_reconstructs_reply_exactly_via_join():
    llm = MockLLM(replies=["Hello there. How are you?"], chunk_words=2)
    chunks = list(llm.generate_stream(messages=[]))
    assert len(chunks) > 1  # actually streamed in more than one piece
    assert "".join(chunks) == "Hello there. How are you?"


def test_mock_llm_cycles_scripted_replies_per_call():
    llm = MockLLM(replies=["First reply.", "Second reply."])
    first = "".join(llm.generate_stream(messages=[]))
    second = "".join(llm.generate_stream(messages=[]))
    third = "".join(llm.generate_stream(messages=[]))  # cycles back
    assert first == "First reply."
    assert second == "Second reply."
    assert third == "First reply."


def test_mock_llm_records_messages_it_was_called_with():
    llm = MockLLM(replies=["Hi."])
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    list(llm.generate_stream(msgs))
    assert llm.calls == [msgs]


def test_mock_llm_default_reply_when_none_scripted():
    llm = MockLLM()
    assert "".join(llm.generate_stream([])) == "Okay!"


# --- make_llm() fallback / selection -----------------------------------

def test_make_llm_falls_back_to_mock_with_no_model_configured(monkeypatch, caplog):
    monkeypatch.delenv("CBOT_LLM_MODEL", raising=False)
    profile = {"name": "testprof"}
    with caplog.at_level(logging.WARNING):
        llm = make_llm(profile)
    assert isinstance(llm, MockLLM)
    assert any("MockLLM" in r.message for r in caplog.records)


def test_make_llm_falls_back_to_mock_when_configured_path_missing(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.delenv("CBOT_LLM_MODEL", raising=False)
    missing = tmp_path / "does-not-exist.gguf"
    profile = {"llm_model_path": str(missing)}
    with caplog.at_level(logging.WARNING):
        llm = make_llm(profile)
    assert isinstance(llm, MockLLM)
    assert any("does not exist" in r.message for r in caplog.records)


def test_resolve_model_path_env_overrides_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("CBOT_LLM_MODEL", "/env/path.gguf")
    profile = {"llm_model_path": "/profile/path.gguf"}
    assert resolve_model_path(profile) == "/env/path.gguf"


def test_resolve_model_path_falls_back_to_profile_key(monkeypatch):
    monkeypatch.delenv("CBOT_LLM_MODEL", raising=False)
    profile = {"llm_model_path": "/profile/path.gguf"}
    assert resolve_model_path(profile) == "/profile/path.gguf"


def test_resolve_model_path_anchors_relative_profile_path_at_repo_root(
    monkeypatch,
):
    monkeypatch.delenv("CBOT_LLM_MODEL", raising=False)
    profile = {"llm_model_path": os.path.join("models", "foo.gguf")}
    resolved = resolve_model_path(profile)
    assert os.path.isabs(resolved)
    # repo root = parent of the conversation/ package (namespace pkg, so
    # anchor off a real module file in it)
    import conversation.llm as llm_module
    repo_root = os.path.dirname(
        os.path.dirname(os.path.abspath(llm_module.__file__))
    )
    assert resolved == os.path.join(repo_root, "models", "foo.gguf")


# --- LocalLLM (llama_cpp faked -- no real download) ---------------------

class _FakeLlama:
    """Stand-in for llama_cpp.Llama: records construction args and
    returns a scripted streamed chat completion."""

    last_instance = None

    def __init__(self, model_path, n_ctx, **kwargs):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.kwargs = kwargs
        self.chat_calls = []
        self.chat_kwargs = []
        _FakeLlama.last_instance = self

    def create_chat_completion(self, messages, stream=True, **kwargs):
        self.chat_calls.append(messages)
        self.chat_kwargs.append(kwargs)
        for word in ["Hi", " there", "."]:
            yield {"choices": [{"delta": {"content": word}}]}
        yield {"choices": [{"delta": {}}]}  # trailing empty delta, like real streams


@pytest.fixture
def fake_llama_cpp_module(monkeypatch):
    module = types.ModuleType("llama_cpp")
    module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return module


def test_local_llm_raises_clear_error_with_no_model_path(monkeypatch):
    monkeypatch.delenv("CBOT_LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="Llama-3.2-3B-Instruct"):
        LocalLLM(profile={})


def test_local_llm_raises_clear_error_when_file_missing(tmp_path):
    missing = tmp_path / "nope.gguf"
    with pytest.raises(RuntimeError, match="Llama-3.2-3B-Instruct"):
        LocalLLM(profile={}, model_path=str(missing))


def test_local_llm_loads_once_and_stays_resident(
    tmp_path, fake_llama_cpp_module, monkeypatch
):
    model_file = tmp_path / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    model_file.write_bytes(b"not a real gguf")

    llm = LocalLLM(profile={}, model_path=str(model_file))
    first_llama = llm._llama

    # generate twice -- must reuse the same resident instance, never
    # reconstruct Llama() per utterance.
    list(llm.generate_stream([{"role": "user", "content": "hi"}]))
    list(llm.generate_stream([{"role": "user", "content": "again"}]))

    assert llm._llama is first_llama
    assert llm.n_ctx == 2048
    assert len(first_llama.chat_calls) == 2


def test_local_llm_generate_stream_yields_text_chunks(
    tmp_path, fake_llama_cpp_module
):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"stub")
    llm = LocalLLM(profile={}, model_path=str(model_file))
    chunks = list(llm.generate_stream([{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "Hi there."


def test_local_llm_bounds_reply_length_with_max_tokens(
    tmp_path, fake_llama_cpp_module
):
    """Responsiveness guard: every generation must carry an explicit
    max_tokens ceiling (MAX_REPLY_TOKENS) so a runaway reply can't
    freeze the conversation turn indefinitely."""
    from conversation.llm import MAX_REPLY_TOKENS

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"stub")
    llm = LocalLLM(profile={}, model_path=str(model_file))
    list(llm.generate_stream([{"role": "user", "content": "hi"}]))
    assert _FakeLlama.last_instance.chat_kwargs == [
        {"max_tokens": MAX_REPLY_TOKENS}
    ]


def test_make_llm_uses_local_llm_when_model_file_present(
    tmp_path, fake_llama_cpp_module, monkeypatch
):
    monkeypatch.delenv("CBOT_LLM_MODEL", raising=False)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"stub")
    profile = {"llm_model_path": str(model_file)}
    llm = make_llm(profile)
    assert isinstance(llm, LocalLLM)


def test_local_llm_offloads_all_gpu_layers_by_default(
    tmp_path, fake_llama_cpp_module
):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"stub")
    LocalLLM(profile={}, model_path=str(model_file))
    assert _FakeLlama.last_instance.kwargs["n_gpu_layers"] == -1
    assert _FakeLlama.last_instance.kwargs["verbose"] is False


def test_local_llm_gpu_layers_overridable_via_profile(
    tmp_path, fake_llama_cpp_module
):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"stub")
    LocalLLM(profile={"llm_gpu_layers": 0}, model_path=str(model_file))
    assert _FakeLlama.last_instance.kwargs["n_gpu_layers"] == 0


def test_make_llm_falls_back_to_mock_when_llama_cpp_not_installed(
    tmp_path, monkeypatch, caplog
):
    # Model file exists (as on the dev PC), but this interpreter has no
    # llama_cpp (e.g. system Python instead of the repo venv): make_llm
    # must warn loudly and hand back a MockLLM, not crash.
    monkeypatch.delenv("CBOT_LLM_MODEL", raising=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"stub")
    profile = {"llm_model_path": str(model_file)}
    with caplog.at_level(logging.WARNING):
        llm = make_llm(profile)
    assert isinstance(llm, MockLLM)
    assert any("MockLLM" in r.message for r in caplog.records)


def test_local_llm_missing_llama_cpp_package_gives_clear_error(tmp_path, monkeypatch):
    # Simulate the package genuinely not being installed.
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"stub")
    with pytest.raises(RuntimeError, match="llama-cpp-python"):
        LocalLLM(profile={}, model_path=str(model_file))


# --- real-model smoke test, only when a GGUF is actually configured -----

_REAL_MODEL_PATH = os.environ.get("CBOT_LLM_MODEL")


@pytest.mark.skipif(
    not _REAL_MODEL_PATH or not os.path.isfile(_REAL_MODEL_PATH),
    reason="No local Llama-3.2-3B-Instruct Q4_K_M GGUF configured via CBOT_LLM_MODEL",
)
def test_real_local_llm_generates_something():
    llm = LocalLLM(profile={}, model_path=_REAL_MODEL_PATH)
    messages = [
        {"role": "system", "content": "Reply in one short sentence."},
        {"role": "user", "content": "Say hello."},
    ]
    text = "".join(llm.generate_stream(messages))
    assert text.strip()

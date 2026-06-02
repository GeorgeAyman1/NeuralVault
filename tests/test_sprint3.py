"""
Sprint 3 tests: context builder, conversation memory, LLM client, chat().

The LLM is mocked end-to-end (FakeAnthropic) so these run fully offline —
no anthropic package, API key, or network required. The fake echoes back the
system blocks and messages it received so tests can assert that the right
context and conversation history were passed through.
"""
from types import SimpleNamespace

import pytest

from core.llm.context_builder import ContextBuilder, SYSTEM_INSTRUCTIONS
from core.llm.conversation import ConversationMemory
from core.llm.llm_client import LLMClient


# --------------------------------------------------------------------------- #
# Fake Anthropic client (mimics messages.stream(...).get_final_message())     #
# --------------------------------------------------------------------------- #

class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, responder):
        self._responder = responder
        self.last_kwargs = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        text = self._responder(kwargs)
        message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=8,
            ),
        )
        return _FakeStream(message)


class FakeAnthropic:
    def __init__(self, responder=None):
        self.messages = _FakeMessages(responder or (lambda kw: "FAKE ANSWER"))


# --------------------------------------------------------------------------- #
# ContextBuilder                                                              #
# --------------------------------------------------------------------------- #

def test_context_builder_empty_memories():
    ctx = ContextBuilder().build_context([])
    assert "No relevant memories" in ctx


def test_context_builder_numbers_and_includes_text():
    memories = [
        {"text": "Paris is the capital of France", "score": 0.9, "metadata": {}},
        {"text": "The Eiffel Tower is in Paris", "score": 0.8, "metadata": {}},
    ]
    ctx = ContextBuilder().build_context(memories)
    assert "[0]" in ctx and "[1]" in ctx
    assert "capital of France" in ctx
    assert "Eiffel Tower" in ctx
    assert "0.900" in ctx  # score formatted


def test_context_builder_respects_char_budget():
    memories = [{"text": "x" * 500, "score": 0.5, "metadata": {}} for _ in range(10)]
    ctx = ContextBuilder(max_context_chars=600).build_context(memories)
    # Only the first memory fits within 600 chars; second would overflow
    assert ctx.count("(source:") == 1


def test_context_builder_always_includes_at_least_one():
    memories = [{"text": "y" * 5000, "score": 0.5, "metadata": {}}]
    ctx = ContextBuilder(max_context_chars=100).build_context(memories)
    assert "yyyy" in ctx  # included despite exceeding budget


def test_system_blocks_have_cached_frozen_prefix():
    blocks = ContextBuilder().build_system_blocks(
        [{"text": "fact", "score": 0.5, "metadata": {}}]
    )
    assert len(blocks) == 2
    assert blocks[0]["text"] == SYSTEM_INSTRUCTIONS
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    # Volatile memory context is the second block, with no cache_control
    assert "cache_control" not in blocks[1]
    assert "fact" in blocks[1]["text"]


# --------------------------------------------------------------------------- #
# ConversationMemory                                                          #
# --------------------------------------------------------------------------- #

def test_conversation_records_turns():
    conv = ConversationMemory()
    conv.add_user("hello")
    conv.add_assistant("hi there")
    conv.add_user("how are you")
    msgs = conv.get_messages()
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1]["role"] == "assistant"
    assert conv.turn_count() == 2


def test_conversation_clear():
    conv = ConversationMemory()
    conv.add_user("x")
    conv.clear()
    assert conv.get_messages() == []


def test_conversation_trims_and_starts_with_user():
    conv = ConversationMemory(max_turns=2)  # keep at most 4 messages
    for i in range(5):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")
    msgs = conv.get_messages()
    assert len(msgs) <= 4
    assert msgs[0]["role"] == "user"  # never starts with assistant


# --------------------------------------------------------------------------- #
# LLMClient (with injected fake)                                              #
# --------------------------------------------------------------------------- #

def test_llm_client_returns_text_and_usage():
    fake = FakeAnthropic(responder=lambda kw: "grounded answer")
    client = LLMClient(client=fake)
    result = client.complete(
        system_blocks=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "q"}],
    )
    assert result["text"] == "grounded answer"
    assert result["usage"]["output_tokens"] == 5


def test_llm_client_passes_model_and_adaptive_thinking():
    fake = FakeAnthropic()
    LLMClient(client=fake, model="claude-opus-4-8").complete(
        system_blocks=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "q"}],
    )
    kw = fake.messages.last_kwargs
    assert kw["model"] == "claude-opus-4-8"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["system"][0]["text"] == "sys"


# --------------------------------------------------------------------------- #
# MemoryService.chat (integration with mocked LLM)                            #
# --------------------------------------------------------------------------- #

@pytest.fixture
def service(tmp_path, monkeypatch):
    """A MemoryService with isolated storage paths and a fake LLM."""
    from core.memory.service import MemoryService
    from core.storage.vecdb_store import VecDBStore

    # Echo responder: returns the memory-context block so we can assert on it
    def responder(kw):
        ctx_block = kw["system"][1]["text"] if len(kw["system"]) > 1 else ""
        return f"ANSWER using: {ctx_block}"

    fake = FakeAnthropic(responder=responder)

    svc = MemoryService(auto_load=False, llm_client=LLMClient(client=fake))
    # Point storage at tmp_path so we don't touch real data
    svc.vector_store = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    # Rebuild retriever against the swapped store
    from core.indexing.retrieval import SemanticRetriever
    svc.retriever = SemanticRetriever(
        encoder=svc.encoder,
        vector_store=svc.vector_store,
        metadata_store=svc.metadata_store,
    )
    return svc


def test_chat_grounds_answer_in_memories(service):
    service.add_memory("The project deadline is June 15th.")
    service.add_memory("The team lead is Alice.")

    result = service.chat("When is the deadline?", top_k=2)

    assert "answer" in result
    assert len(result["memories_used"]) == 2
    # The fake echoed the context block back — confirm a memory reached the LLM
    assert "deadline" in result["answer"] or "Alice" in result["answer"]


def test_chat_records_conversation(service):
    service.add_memory("Fact one.")
    service.chat("first question?")
    service.chat("follow up?")
    # 2 user + 2 assistant messages retained
    assert service.conversation.turn_count() == 2
    assert len(service.conversation.get_messages()) == 4


def test_chat_use_conversation_false_is_stateless(service):
    service.add_memory("Fact one.")
    service.chat("q1", use_conversation=False)
    service.chat("q2", use_conversation=False)
    assert service.conversation.turn_count() == 0


def test_chat_empty_query_raises(service):
    with pytest.raises(ValueError):
        service.chat("   ")


def test_reset_conversation(service):
    service.add_memory("Fact.")
    service.chat("q")
    assert service.conversation.turn_count() == 1
    service.reset_conversation()
    assert service.conversation.turn_count() == 0

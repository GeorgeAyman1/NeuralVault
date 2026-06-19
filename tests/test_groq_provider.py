"""
Tests for the Groq / OpenAI-compatible LLM provider path.

Fully offline: the HTTP call is replaced by an injected callable that captures
the request payload and returns a canned OpenAI-style response.
"""
import pytest

from core.llm.llm_client import LLMClient, LLMUnavailableError
from core.utils.config import Settings


def test_groq_translates_blocks_and_parses_response():
    captured = {}

    def fake_post(payload):
        captured["payload"] = payload
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Alice is the team lead."}}
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        }

    client = LLMClient(
        provider="groq",
        model="llama-3.3-70b-versatile",
        client=fake_post,
    )
    result = client.complete(
        system_blocks=[
            {"type": "text", "text": "INSTRUCTIONS", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Retrieved memories:\n\n[0] Alice is the team lead."},
        ],
        messages=[{"role": "user", "content": "Who is the team lead?"}],
    )

    # Response parsing
    assert result["text"] == "Alice is the team lead."
    assert result["usage"]["input_tokens"] == 42
    assert result["usage"]["output_tokens"] == 7
    assert result["usage"]["cache_read_input_tokens"] is None

    # Request translation: content blocks -> single system message; caching dropped
    msgs = captured["payload"]["messages"]
    assert msgs[0]["role"] == "system"
    assert "INSTRUCTIONS" in msgs[0]["content"]
    assert "Retrieved memories" in msgs[0]["content"]
    assert "cache_control" not in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "Who is the team lead?"}
    assert captured["payload"]["model"] == "llama-3.3-70b-versatile"


def test_groq_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = LLMClient(provider="groq", api_key=None)
    with pytest.raises(LLMUnavailableError):
        client.complete(
            system_blocks=[{"type": "text", "text": "x"}],
            messages=[{"role": "user", "content": "hi"}],
        )


def test_settings_groq_provider_defaults(monkeypatch):
    for var in ("NEURALVAULT_LLM_MODEL", "NEURALVAULT_LLM_BASE_URL", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NEURALVAULT_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    s = Settings.from_env()
    assert s.llm_provider == "groq"
    assert s.llm_model == "llama-3.3-70b-versatile"
    assert s.llm_base_url == "https://api.groq.com/openai/v1"
    assert s.active_api_key == "gsk-test"
    assert s.llm_available is True
